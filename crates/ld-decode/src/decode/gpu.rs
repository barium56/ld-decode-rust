//! OpenCL f64 FFT: the bit-exact device replica of ducc's plan, the probes that
//! price it, and the `--gpu` late group that production uses.
//!
//! Two questions, in order. First, **could a GPU decode path be worth
//! building?** That is what the probes measure: the two parts of the pipeline a
//! GPU could plausibly accelerate -- the demod FFT layer (~90% of that kernel's
//! CPU) and the sinc resampler (the largest serial step) -- and the resulting
//! end-to-end arithmetic against the pipeline's measured per-field budget.
//! Second, **is it bit-exact?** Yes, and `gpu_late_group_f64_gate` (the one that
//! matters: it compares in f64, the width the reference computes in) plus
//! `gpu_late_group_matches_cpu` and `gpu_bitexact_probe` are the gates; `--gpu`
//! is therefore a speed knob, not a second decoder (see the late-group section
//! at the bottom of this comment for the measured thread-count crossover).
//!
//! Usage (release profile, this box):
//!
//! ```text
//! cargo test -p ld-decode --release gpu_fft_probe -- --ignored --nocapture
//! ```
//!
//! Optional env: `LD_GPU_ROUNDS` (timed rounds, default 7). The first GPU
//! device the ICD reports is used; without one the probe prints `SKIP` and
//! returns, so it is safe to run anywhere.
//!
//! The loader is resolved with `LoadLibraryA("OpenCL.dll")` on Windows and
//! `dlopen("libOpenCL.so.1")` on unix. `dlopen` lives in glibc itself from
//! 2.34 (the repo's Linux floor is ubuntu-22.04 / 2.35), so no `-ldl` is
//! needed.
//!
//! What it measures:
//!
//! 1. **Correctness** — a batched inverse Stockham FFT on the device vs
//!    `ffi_ducc::ifft` on the same data (max relative error), plus a
//!    forward->inverse round trip. Without this the timing means nothing: a
//!    kernel that is fast and wrong is not evidence of anything.
//! 2. **Device FFT cost for one field** — 22 blocks x (6 inverse + 2 forward)
//!    transforms of n=32768, i.e. the exact transform count and size the demod
//!    kernel issues per field (`dcalls` in `LD_TIMING`). Reported as device
//!    time (per-launch event deltas), wall time including launches, and the
//!    host->device bandwidth measured on the same buffers.
//! 3. **CPU baselines, same shapes** — the FFT layer alone through `ffi_ducc`
//!    (the kernel's own batching) and the whole `demod_block_cpu` kernel, per
//!    field, single-threaded.
//! 4. **Device sinc gather** — a 16-tap LUT gather kernel over one field of
//!    output pixels (`linesout x outwidth`), validated sample-by-sample
//!    against the real `scale_field_sinc` on the same inputs, timed against
//!    that function running on the production tail pool.
//! 5. **The verdict** — the per-field model below, printed with its inputs, so
//!    the conclusion is checkable rather than asserted.
//!
//! Why the model looks the way it does (`work/bench_log.md`, `LD_TIMING`):
//! the demod already runs on 12 pool threads *underneath* the serial tail via
//! the prefetch spawn, so its CPU cost is not the critical path — the exposed
//! cost is the `pffold` wait. A GPU therefore only helps if the device kernel
//! beats the demod **wall** time, and it only helps *a lot* if it also gets
//! under the serial tail, which is why both scenarios are printed: FFT only
//! (capped by the tail), and FFT + sinc (which is the only one that can move
//! the wall).
//!
//! Non-parity by construction: nothing here is byte-comparable to the Python
//! reference. The probe exists to price that decision, not to take it.
//!
//! # The `--gpu` late group (`LateGroup`)
//!
//! `LateGroup` is the one production user of this module. It runs the demod
//! kernel's *late group* on the device -- `demod_fft` (one forward real
//! transform) and the three video channels (half-spectrum product + real
//! inverse) -- for a whole field's blocks in one batch, and returns the three
//! cut f32 channels per block. Everything else in the block kernel (the input
//! spectrum, hilbert, EFM, rfhpf, the audio slices, unwrap and clamp) stays on
//! the CPU, because moving it would cost more in transfers than it saves: see
//! the `[2b]` transfer table in `work/bench_log.md`.
//!
//! It is bit-exact by the same argument the probe establishes: the same ducc
//! pass tree, the same twiddle tables, `FP_CONTRACT OFF`, and the video
//! products evaluated with `fma` exactly as `np_cmul` does.
//!
//! Measured on gfx1201 (32 CU), interleaved against the CPU path: **+63% at
//! `-t 1`, +52% at `-t 2`, +31% at `-t 4`, +16% at `-t 6`, +12% at the default
//! 10 threads** (48.59 vs 43.37 FPS over 1000 frames). It pays at every thread
//! count, and the reason is worth keeping: two of the first version's costs were
//! on the *host*, not the device. The upload repacked each block's `n` doubles
//! into `n/2` tuples (`Vec<Complex>`) before a pageable `clEnqueueWriteBuffer`,
//! and the readback went through a second staging buffer -- together ~0.057 of
//! the 0.206 ms/block the late group cost. A `double2` *is* two consecutive
//! `f64`, so the block's doubles can be mapped and copied straight in, and
//! page-locked buffers (`CL_MEM_ALLOC_HOST_PTR` + map/unmap) lift the download
//! from 7.7 to 20.4 GB/s (the probe's `[2b]` prints both). In situ the stage is
//! now **0.149 ms/block = 3.3 ms/field** against the ~4.5 ms the same work costs
//! the pool. `LD_GPU_STATS=1` and the `GPUTIME` line in `LD_TIMING` are the
//! instruments.
//!
//! # Where the remaining 0.15 ms/block sits, and why overlapping it loses
//!
//! `LD_GPU_PROF=1` splits a batch into its host and device halves. Steady state
//! (21 blocks/batch) is **upload 21 us + submit 2 us + finish 92 us + download
//! 35 us per block**, and `finish - kernels` puts the kernel sum at 64 us with
//! the remaining 28 us being the upload's DMA -- i.e. 256 KB at ~9 GB/s, the
//! same number the probe's `[2b]` table reports. So the host copies are 37% of
//! the stage and the device 63%.
//!
//! Two things follow, and both were measured rather than argued:
//!
//! 1. **The stage is exposed 1:1 on the wall clock.** `LD_GPU_X2=1` runs the
//!    same batch twice and throws the second pass away, which lengthens the
//!    stage by 3.05 ms/field; the run goes 50.4 -> 38.6 FPS, i.e. +2.98
//!    ms/field of wall. Slope ~1, so a ms saved here is a ms of FPS.
//! 2. **Chunking the batch to overlap host and device still loses.** Three
//!    variants were built and measured against the 0.152 single-batch path:
//!    per-chunk non-blocking map/unmap on the compute queue (0.186), the same
//!    on a second transfer queue with barriers (0.163), and a per-chunk DMA
//!    `clEnqueueReadBuffer` (0.184). The reason is the driver, not the design:
//!    `clEnqueueMapBuffer` synchronizes with every pending write to that buffer
//!    object (and in practice with the queue), so a per-chunk map costs ~0.5 ms
//!    -- an order of magnitude more than the whole-batch map it replaced. Do
//!    not retry this without first re-measuring per-chunk map cost.
//!
//! What *did* change: the three channels' inverse transforms run as one
//! row-major chain (`c2r_pre_r`, rows `block*3 + channel`) instead of three
//! per-channel chains, which removes two thirds of the launches in that phase
//! for identical arithmetic (0.153 -> 0.152 ms/block, i.e. neutral in time and
//! simpler). An explicit work-group size (`reqd_work_group_size`-style local
//! size of 64/128/256) was also measured and is neutral -- the driver's own
//! choice is already right.

// Most of these are only reached from the probes; the module is
// `#[allow(dead_code)]` and this keeps the import list one list.
#[allow(unused_imports)]
use std::ffi::c_void;
#[allow(unused_imports)]
use std::sync::Arc;
#[allow(unused_imports)]
use std::time::Instant;

#[allow(unused_imports)]
use rustfft::num_complex::Complex64;

#[allow(unused_imports)]
use super::ducc_plan::{self, Body, C2cPlan, Kind, C};
#[allow(unused_imports)]
use crate::decode::{compute_mtf_pow, demod_block_cpu, DemodSpecRef};
#[allow(unused_imports)]
use crate::ffi_ducc;
#[allow(unused_imports)]
use crate::optimized::sinc::{SINC_PHASE_COUNT, SINC_TAP_COUNT};
#[allow(unused_imports)]
use crate::optimized::{scale_field_sinc, sinc_lut, SincScaleParams};
#[allow(unused_imports)]
use crate::request::DecodeRequest;
#[allow(unused_imports)]
use crate::spec::{CalibLevels, DecoderSpec};

/// Demod blocks whose output advances one field, taken from `LD_TIMING`
/// (`pfblk`/`dcalls` median 22 on a real capture). The per-field figures below
/// multiply this by the measured per-block cost.
const BLOCKS_PER_FIELD: usize = 22;

/// Pipeline stages in `--gpu`'s batch loop: the batch is split into this many
/// chunks so one chunk's host copy overlaps another's device work. Each stage
/// needs its own pinned upload buffer, so this is also the buffer count.
const NCHUNK: usize = 4;

/// Pipeline budget as measured by `LD_TIMING` / `LD_DEMODTIME` (medians over
/// 100 fields, Pioneer GGV1069 ldf, `-t 12`, `-s 1000 -l 200`, 2026-10-25).
/// These are context for the verdict, not something this probe can re-measure
/// without running a full decode.
const ITER_NOW_MS: f64 = 11.74;
const DEMOD_WALL_MS: f64 = 11.50;
const PF_WAIT_MS: f64 = 1.51;
/// Everything serial per field except the demod wait: `down` 5.67 (the scaler,
/// the wow spline, dropout detect, audio and the encode — the probe measures
/// the scaler itself at ~2.1 ms) + `asm` 2.53 + `proc` 2.26 + `vits` 0.55 +
/// `out` 0.10.
const TAIL_MS: f64 = 11.10;
/// Demod pool threads the wire measurement used; the CPU figures are divided by
/// this to get the wall the current schedule needs.
const POOL_THREADS: f64 = 12.0;
/// Measured parallel efficiency of the pool (`pfwork`/`pfspan` = 128/11.5 ~ 11x
/// on 12 threads — the split scales almost perfectly).
const POOL_EFFICIENCY: f64 = 11.0;

// ---------------------------------------------------------------------------
// OpenCL, hand-rolled and dynamically loaded
// ---------------------------------------------------------------------------
//
// No new dependency on purpose. The crate already vendors ducc and binds the
// platform C runtime by hand, and a `--gpu` path would have to probe for the
// ICD at run time anyway (Windows ships `OpenCL.dll`, Linux distros do not ship
// `libOpenCL.so` without a vendor driver). `opencl-sys` would add a build
// dependency and a link requirement to *every* target for a feature that may
// not survive this probe.
mod ffi {
    use std::ffi::c_void;

    pub type Handle = *mut c_void;
    pub type PlatformId = Handle;
    pub type DeviceId = Handle;
    pub type Context = Handle;
    pub type Queue = Handle;
    pub type Mem = Handle;
    pub type Program = Handle;
    pub type Kernel = Handle;
    pub type Event = Handle;

    pub const CL_SUCCESS: i32 = 0;
    pub const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
    pub const CL_DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;

    pub const CL_DEVICE_NAME: u32 = 0x102B;
    pub const CL_DEVICE_VENDOR: u32 = 0x102C;
    pub const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
    pub const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
    pub const CL_DEVICE_DOUBLE_FP_CONFIG: u32 = 0x1033;
    pub const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
    pub const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;

    pub const CL_MEM_READ_WRITE: u64 = 1 << 0;
    pub const CL_MEM_READ_ONLY: u64 = 1 << 2;
    /// Ask the driver for page-locked host memory, so a `clEnqueueWriteBuffer`
    /// does not have to copy through a staging buffer on every transfer.
    pub const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
    pub const CL_MAP_READ: u64 = 1 << 0;
    pub const CL_MAP_WRITE: u64 = 1 << 1;

    pub const CL_QUEUE_PROFILING_ENABLE: u64 = 1 << 1;
    pub const CL_PROFILING_COMMAND_START: u32 = 0x1282;
    pub const CL_PROFILING_COMMAND_END: u32 = 0x1283;

    /// The ICD loader, as a private handle plus its entry points.
    ///
    /// Every function pointer is `extern "C"`: on x86-64 Windows and SysV the
    /// `CL_API_CALL` convention is the platform one, so one declaration covers
    /// both.
    pub struct Api {
        pub get_platform_ids:
            unsafe extern "C" fn(u32, *mut PlatformId, *mut u32) -> i32,
        pub get_device_ids:
            unsafe extern "C" fn(PlatformId, u64, u32, *mut DeviceId, *mut u32) -> i32,
        pub get_device_info:
            unsafe extern "C" fn(DeviceId, u32, usize, *mut c_void, *mut usize) -> i32,
        pub create_context: unsafe extern "C" fn(
            *const isize,
            u32,
            *const DeviceId,
            *const c_void,
            *mut c_void,
            *mut i32,
        ) -> Context,
        pub create_queue: unsafe extern "C" fn(Context, DeviceId, u64, *mut i32) -> Queue,
        pub create_program: unsafe extern "C" fn(
            Context,
            u32,
            *const *const u8,
            *const usize,
            *mut i32,
        ) -> Program,
        pub build_program: unsafe extern "C" fn(
            Program,
            u32,
            *const DeviceId,
            *const u8,
            *const c_void,
            *mut c_void,
        ) -> i32,
        pub get_program_build_info:
            unsafe extern "C" fn(Program, DeviceId, u32, usize, *mut c_void, *mut usize) -> i32,
        pub create_kernel: unsafe extern "C" fn(Program, *const u8, *mut i32) -> Kernel,
        pub create_buffer: unsafe extern "C" fn(Context, u64, usize, *mut c_void, *mut i32) -> Mem,
        pub write_buffer: unsafe extern "C" fn(
            Queue,
            Mem,
            u32,
            usize,
            usize,
            *const c_void,
            u32,
            *const Event,
            *mut Event,
        ) -> i32,
        pub read_buffer: unsafe extern "C" fn(
            Queue,
            Mem,
            u32,
            usize,
            usize,
            *mut c_void,
            u32,
            *const Event,
            *mut Event,
        ) -> i32,
        pub map_buffer: unsafe extern "C" fn(
            Queue,
            Mem,
            u32,
            u64,
            usize,
            usize,
            u32,
            *const Event,
            *mut Event,
            *mut i32,
        ) -> *mut c_void,
        pub unmap_mem: unsafe extern "C" fn(
            Queue,
            Mem,
            *mut c_void,
            u32,
            *const Event,
            *mut Event,
        ) -> i32,
        pub set_kernel_arg: unsafe extern "C" fn(Kernel, u32, usize, *const c_void) -> i32,
        pub copy_buffer: unsafe extern "C" fn(
            Queue,
            Mem,
            Mem,
            usize,
            usize,
            usize,
            u32,
            *const Event,
            *mut Event,
        ) -> i32,
        pub enqueue_kernel: unsafe extern "C" fn(
            Queue,
            Kernel,
            u32,
            *const usize,
            *const usize,
            *const usize,
            u32,
            *const Event,
            *mut Event,
        ) -> i32,
        pub finish: unsafe extern "C" fn(Queue) -> i32,
        pub get_event_info:
            unsafe extern "C" fn(Event, u32, usize, *mut c_void, *mut usize) -> i32,
        pub wait_for_events: unsafe extern "C" fn(u32, *const Event) -> i32,
        pub barrier_with_wait: unsafe extern "C" fn(Queue, u32, *const Event, *mut Event) -> i32,
        pub release_event: unsafe extern "C" fn(Event) -> i32,
        pub release_mem: unsafe extern "C" fn(Mem) -> i32,
        pub release_kernel: unsafe extern "C" fn(Kernel) -> i32,
        pub release_program: unsafe extern "C" fn(Program) -> i32,
        pub release_queue: unsafe extern "C" fn(Queue) -> i32,
        pub release_context: unsafe extern "C" fn(Context) -> i32,
    }

    // The loader handle is intentionally leaked: the probe runs once per
    // process and `clRelease*` on a still-referenced library is not worth the
    // bookkeeping (the module's `Drop` releases every CL object it created).
    #[cfg(windows)]
    fn open_library() -> Option<Handle> {
        extern "system" {
            fn LoadLibraryA(name: *const u8) -> Handle;
        }
        for name in [b"OpenCL.dll\0".as_slice(), b"opencl.dll\0"] {
            let h = unsafe { LoadLibraryA(name.as_ptr()) };
            if !h.is_null() {
                return Some(h);
            }
        }
        None
    }

    #[cfg(unix)]
    fn open_library() -> Option<Handle> {
        extern "C" {
            fn dlopen(filename: *const u8, flag: i32) -> Handle;
        }
        const RTLD_NOW: i32 = 2;
        for name in [b"libOpenCL.so.1\0".as_slice(), b"libOpenCL.so\0"] {
            let h = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if !h.is_null() {
                return Some(h);
            }
        }
        None
    }

    #[cfg(windows)]
    fn symbol(lib: Handle, name: &str) -> *mut c_void {
        extern "system" {
            fn GetProcAddress(h: Handle, name: *const u8) -> *mut c_void;
        }
        let mut c = Vec::with_capacity(name.len() + 1);
        c.extend_from_slice(name.as_bytes());
        c.push(0);
        unsafe { GetProcAddress(lib, c.as_ptr()) }
    }

    #[cfg(unix)]
    fn symbol(lib: Handle, name: &str) -> *mut c_void {
        extern "C" {
            fn dlsym(handle: Handle, symbol: *const u8) -> *mut c_void;
        }
        let mut c = Vec::with_capacity(name.len() + 1);
        c.extend_from_slice(name.as_bytes());
        c.push(0);
        unsafe { dlsym(lib, c.as_ptr()) }
    }

    macro_rules! load {
        ($lib:expr, $name:literal) => {{
            let p = symbol($lib, $name);
            if p.is_null() {
                return Err(format!("OpenCL loader has no symbol `{}`", $name));
            }
            // SAFETY: the ICD exports these with the C ABI and the signatures
            // above are the OpenCL 1.2 declarations, unchanged in 2.x/3.x.
            unsafe { std::mem::transmute(p) }
        }};
    }

    impl Api {
        pub fn load() -> Result<Self, String> {
            let lib = open_library().ok_or_else(|| "no OpenCL ICD loader found".to_string())?;
            Ok(Api {
                get_platform_ids: load!(lib, "clGetPlatformIDs"),
                get_device_ids: load!(lib, "clGetDeviceIDs"),
                get_device_info: load!(lib, "clGetDeviceInfo"),
                create_context: load!(lib, "clCreateContext"),
                create_queue: load!(lib, "clCreateCommandQueue"),
                create_program: load!(lib, "clCreateProgramWithSource"),
                build_program: load!(lib, "clBuildProgram"),
                get_program_build_info: load!(lib, "clGetProgramBuildInfo"),
                create_kernel: load!(lib, "clCreateKernel"),
                create_buffer: load!(lib, "clCreateBuffer"),
                copy_buffer: load!(lib, "clEnqueueCopyBuffer"),
                write_buffer: load!(lib, "clEnqueueWriteBuffer"),
                read_buffer: load!(lib, "clEnqueueReadBuffer"),
                map_buffer: load!(lib, "clEnqueueMapBuffer"),
                unmap_mem: load!(lib, "clEnqueueUnmapMemObject"),
                set_kernel_arg: load!(lib, "clSetKernelArg"),
                enqueue_kernel: load!(lib, "clEnqueueNDRangeKernel"),
                finish: load!(lib, "clFinish"),
                get_event_info: load!(lib, "clGetEventProfilingInfo"),
                wait_for_events: load!(lib, "clWaitForEvents"),
                barrier_with_wait: load!(lib, "clEnqueueBarrierWithWaitList"),
                release_event: load!(lib, "clReleaseEvent"),
                release_mem: load!(lib, "clReleaseMemObject"),
                release_kernel: load!(lib, "clReleaseKernel"),
                release_program: load!(lib, "clReleaseProgram"),
                release_queue: load!(lib, "clReleaseCommandQueue"),
                release_context: load!(lib, "clReleaseContext"),
            })
        }
    }

    pub fn check(what: &str, err: i32) -> Result<(), String> {
        if err == CL_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what} failed (cl error {err})"))
        }
    }
}

/// Batched Stockham autosort, one launch per stage.
///
/// Radix-2 is enough for the decoder: `blocklen` is 32768 = 2^15, and both the
/// forward r2c and every inverse are power-of-two transforms. All `batch`
/// transforms in a buffer run in one launch per stage (15 launches per pass),
/// which is how a real port would issue them: the stage is embarrassingly
/// parallel, so device occupancy comes from batching, not from launching early.
const KERNEL_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef double2 cplx;

__kernel void stockham(__global const cplx *x,
                       __global cplx *y,
                       __global const cplx *tw,
                       const uint n,
                       const uint s,
                       const uint total,
                       const uint batch,
                       const int inverse,
                       const int scale_last,
                       const double inv_n)
{
    const size_t gid = get_global_id(0);
    const size_t per = (size_t)total >> 1;      /* butterflies per transform */
    const size_t b = gid / per;
    if (b >= (size_t)batch) return;
    const size_t i = gid - b * per;
    const uint p = (uint)(i / (size_t)s);
    const uint q = (uint)(i - (size_t)p * (size_t)s);
    const uint m = n >> 1;
    if (p >= m) return;

    const size_t base = b * (size_t)total;
    const cplx a = x[base + (size_t)q + (size_t)s * (size_t)p];
    const cplx c = x[base + (size_t)q + (size_t)s * (size_t)(p + m)];

    cplx w = tw[(size_t)p * (size_t)s];
    if (inverse) w.y = -w.y;

    cplx t;
    t.x = a.x + c.x;
    t.y = a.y + c.y;
    cplx d;
    d.x = (a.x - c.x) * w.x - (a.y - c.y) * w.y;
    d.y = (a.x - c.x) * w.y + (a.y - c.y) * w.x;

    if (scale_last) {
        t.x *= inv_n; t.y *= inv_n;
        d.x *= inv_n; d.y *= inv_n;
    }
    y[base + (size_t)q + (size_t)s * (size_t)(2u * p)] = t;
    y[base + (size_t)q + (size_t)s * (size_t)(2u * p + 1u)] = d;
}
"#;

/// The sinc gather: one thread per output pixel, the reference's own arithmetic.
///
/// Per sample the reference does `coord = f32(locs[i])` clamped into the
/// interior, `coord_int = coord as usize`, `frac = coord - coord_int`,
/// `phase = min((frac * 2^16 + 0.5) as usize, 2^16-1)` (7.4.0's nearest-row
/// selection; `alpha` is 0), then 16 taps of `result += f64(buf[start+t] *
/// lut[row+t])` and a final `(adjust * result) as f32`. Every one of those
/// steps is reproduced here in the same order, which is why the probe can
/// assert bit equality rather than a tolerance: the f32 product and the f64
/// accumulate chain are per-sample scalar arithmetic, so a device lane and a
/// CPU lane compute the same value.
///
/// `adjust` is precomputed on the host (it needs the wow median and threshold,
/// a whole-field statistic) — the port would keep that pass on the CPU too.
const SINC_SRC: &str = r#"
__kernel void sinc_gather(__global const float *buf, const uint buflen,
                          __global const float *locs,
                          __global const double *adjust,
                          __global const float *lut,
                          __global float *out,
                          const uint n_out, const uint base)
{
    uint j = get_global_id(0);
    if (j >= n_out) return;
    uint i = base + j;
    const float half_m1 = 7.0f;                       /* 16/2 - 1 */
    float maxc = (float)buflen - 16.0f - half_m1;
    float coord = locs[i];
    coord = coord < half_m1 ? half_m1 : coord;
    coord = coord > maxc ? maxc : coord;
    uint ci = (uint)coord;
    float frac = coord - (float)ci;
    uint phase = (uint)(frac * 65536.0f + 0.5f);
    if (phase > 65535u) phase = 65535u;
    __global const float *row = lut + (size_t)phase * 16u;
    uint start = ci - 7u;
    double acc = 0.0;
    for (uint t = 0; t < 16u; t++) {
        acc += (double)(buf[start + t] * row[t]);
    }
    out[j] = (float)(adjust[i] * acc);
}
"#;

/// OpenCL handles are opaque pointers, so the automatic `Send`/`Sync` impls do
/// not apply. Both are safe here for the same reason: every call goes through the
/// same command queue, `LateGroup` is only reachable behind a `Mutex`, and the
/// OpenCL API itself is thread-safe. Without these the device handle cannot be
/// carried into the prefetch worker, which is where most of a field's blocks are
/// demodulated.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}
unsafe impl Send for DuccGpu<'_> {}
unsafe impl Sync for DuccGpu<'_> {}

/// A device, its queue, the two kernels and the two ping-pong buffers the FFT
/// runs between. Dropping it releases every CL object.
struct Gpu {
    api: ffi::Api,
    context: ffi::Context,
    device: ffi::DeviceId,
    queue: ffi::Queue,
    /// A second queue used only for host<->device transfers. Copies on the
    /// compute queue would be ordered behind the kernels that precede them; on
    /// their own queue they run on the DMA engine concurrently with compute,
    /// which is what makes the chunked pipeline overlap at all.
    tqueue: ffi::Queue,
    program: ffi::Program,
    fft_kernel: ffi::Kernel,
    sinc_kernel: ffi::Kernel,
    bufs: [ffi::Mem; 2],
    tw: ffi::Mem,
    n: usize,
    max_batch: usize,
    name: String,
    vendor: String,
    units: u32,
    gmem: u64,
}

/// A device buffer that releases itself; borrowing `Gpu` keeps it from
/// outliving the context that owns it.
struct Buffer<'a> {
    api: &'a ffi::Api,
    mem: ffi::Mem,
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.api.release_mem)(self.mem);
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            (self.api.finish)(self.queue);
            (self.api.release_kernel)(self.fft_kernel);
            (self.api.release_kernel)(self.sinc_kernel);
            (self.api.release_program)(self.program);
            for m in self.bufs {
                (self.api.release_mem)(m);
            }
            (self.api.release_mem)(self.tw);
            (self.api.release_queue)(self.queue);
            (self.api.release_queue)(self.tqueue);
            (self.api.release_context)(self.context);
        }
    }
}

impl Gpu {
    fn new(n: usize, max_batch: usize) -> Result<Self, String> {
        let api = ffi::Api::load()?;

        let mut nplat = 0u32;
        unsafe {
            ffi::check(
                "clGetPlatformIDs",
                (api.get_platform_ids)(0, std::ptr::null_mut(), &mut nplat),
            )?
        };
        if nplat == 0 {
            return Err("no OpenCL platform".into());
        }
        let mut plats = vec![std::ptr::null_mut(); nplat as usize];
        unsafe {
            ffi::check(
                "clGetPlatformIDs",
                (api.get_platform_ids)(nplat, plats.as_mut_ptr(), &mut nplat),
            )?
        };

        // Prefer a GPU; fall back to whatever the platform offers so the probe
        // still reports a number on a CPU OpenCL runtime.
        let mut found: Option<(ffi::PlatformId, ffi::DeviceId)> = None;
        for ty in [ffi::CL_DEVICE_TYPE_GPU, ffi::CL_DEVICE_TYPE_ALL] {
            for p in &plats {
                let mut ndev = 0u32;
                let rc = unsafe {
                    (api.get_device_ids)(*p, ty, 0, std::ptr::null_mut(), &mut ndev)
                };
                if rc != ffi::CL_SUCCESS || ndev == 0 {
                    continue;
                }
                let mut devs = vec![std::ptr::null_mut(); ndev as usize];
                let rc = unsafe {
                    (api.get_device_ids)(*p, ty, ndev, devs.as_mut_ptr(), &mut ndev)
                };
                if rc == ffi::CL_SUCCESS && ndev > 0 {
                    found = Some((*p, devs[0]));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        let (_platform, device) = found.ok_or_else(|| "no OpenCL device".to_string())?;

        fn device_str(api: &ffi::Api, device: ffi::DeviceId, param: u32) -> String {
            let mut buf = [0u8; 512];
            let mut got = 0usize;
            let rc = unsafe {
                (api.get_device_info)(
                    device,
                    param,
                    buf.len(),
                    buf.as_mut_ptr() as *mut c_void,
                    &mut got,
                )
            };
            if rc != ffi::CL_SUCCESS {
                return "?".into();
            }
            let end = buf.iter().position(|&b| b == 0).unwrap_or(got);
            String::from_utf8_lossy(&buf[..end]).into_owned()
        }
        let name = device_str(&api, device, ffi::CL_DEVICE_NAME);
        let vendor = device_str(&api, device, ffi::CL_DEVICE_VENDOR);
        let mut u32_out = 0u32;
        let rc = unsafe {
            (api.get_device_info)(
                device,
                ffi::CL_DEVICE_MAX_COMPUTE_UNITS,
                4,
                &mut u32_out as *mut u32 as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        let units = if rc == ffi::CL_SUCCESS { u32_out } else { 0 };
        let mut u64_out = 0u64;
        let rc = unsafe {
            (api.get_device_info)(
                device,
                ffi::CL_DEVICE_GLOBAL_MEM_SIZE,
                8,
                &mut u64_out as *mut u64 as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        let gmem = if rc == ffi::CL_SUCCESS { u64_out } else { 0 };

        // f64 is the whole point: the demod path is complex f64, and an f32
        // kernel would be a bigger arithmetic change than the FFT order alone.
        let mut dp = 0u64;
        unsafe {
            (api.get_device_info)(
                device,
                ffi::CL_DEVICE_DOUBLE_FP_CONFIG,
                8,
                &mut dp as *mut u64 as *mut c_void,
                std::ptr::null_mut(),
            );
        }
        if dp == 0 {
            return Err(format!(
                "device `{name}` reports no double-precision config (no cl_khr_fp64)"
            ));
        }

        let mut err = 0i32;
        let context = unsafe {
            (api.create_context)(
                std::ptr::null(),
                1,
                &device,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut err,
            )
        };
        ffi::check("clCreateContext", err)?;
        let queue =
            unsafe { (api.create_queue)(context, device, ffi::CL_QUEUE_PROFILING_ENABLE, &mut err) };
        ffi::check("clCreateCommandQueue", err)?;
        let tqueue =
            unsafe { (api.create_queue)(context, device, ffi::CL_QUEUE_PROFILING_ENABLE, &mut err) };
        ffi::check("clCreateCommandQueue (transfers)", err)?;

        let program_src = format!("{KERNEL_SRC}\n{SINC_SRC}");
        let src_ptr = program_src.as_ptr();
        let src_len = program_src.len();
        let program = unsafe {
            (api.create_program)(
                context,
                1,
                &src_ptr,
                &src_len as *const usize,
                &mut err,
            )
        };
        ffi::check("clCreateProgramWithSource", err)?;
        let rc = unsafe {
            (api.build_program)(
                program,
                1,
                &device,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if rc != ffi::CL_SUCCESS {
            let mut log = vec![0u8; 8192];
            let mut got = 0usize;
            unsafe {
                (api.get_program_build_info)(
                    program,
                    device,
                    ffi::CL_PROGRAM_BUILD_LOG,
                    log.len(),
                    log.as_mut_ptr() as *mut c_void,
                    &mut got,
                );
            }
            let end = log.iter().position(|&b| b == 0).unwrap_or(got);
            return Err(format!(
                "build failed (cl error {rc}): {}",
                String::from_utf8_lossy(&log[..end])
            ));
        }
        let fft_kernel = unsafe {
            (api.create_kernel)(program, b"stockham\0".as_ptr(), &mut err)
        };
        ffi::check("clCreateKernel(stockham)", err)?;
        let sinc_kernel = unsafe {
            (api.create_kernel)(program, b"sinc_gather\0".as_ptr(), &mut err)
        };
        ffi::check("clCreateKernel(sinc_gather)", err)?;

        let bytes = max_batch * n * 16;
        let mut alloc = |size: usize| -> Result<ffi::Mem, String> {
            let m = unsafe {
                (api.create_buffer)(
                    context,
                    ffi::CL_MEM_READ_WRITE,
                    size,
                    std::ptr::null_mut(),
                    &mut err,
                )
            };
            ffi::check("clCreateBuffer", err)?;
            Ok(m)
        };
        let bufs = [alloc(bytes)?, alloc(bytes)?];

        // tw[j] = exp(-2*pi*i*j/n), 0 <= j < n/2, uploaded once per process.
        let mut tw_host: Vec<Complex64> = Vec::with_capacity(n / 2);
        for j in 0..n / 2 {
            let a = -2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
            tw_host.push(Complex64::new(a.cos(), a.sin()));
        }
        let tw = unsafe {
            (api.create_buffer)(
                context,
                ffi::CL_MEM_READ_WRITE,
                n / 2 * 16,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        ffi::check("clCreateBuffer(tw)", err)?;

        let g = Gpu {
            api,
            context,
            device,
            queue,
            tqueue,
            program,
            fft_kernel,
            sinc_kernel,
            bufs,
            tw,
            n,
            max_batch,
            name,
            vendor,
            units,
            gmem,
        };
        g.write(g.tw, tw_host.as_ptr() as *const c_void, n / 2 * 16)?;
        Ok(g)
    }

    fn write(&self, mem: ffi::Mem, src: *const c_void, bytes: usize) -> Result<(), String> {
        let rc = unsafe {
            (self.api.write_buffer)(
                self.queue,
                mem,
                1,
                0,
                bytes,
                src,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        ffi::check("clEnqueueWriteBuffer", rc)
    }

    fn read(&self, mem: ffi::Mem, dst: *mut c_void, bytes: usize) -> Result<(), String> {
        let rc = unsafe {
            (self.api.read_buffer)(
                self.queue,
                mem,
                1,
                0,
                bytes,
                dst,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        ffi::check("clEnqueueReadBuffer", rc)
    }

    /// Upload `batch` transforms worth of complex f64 into buffer `slot`.
    fn upload(&self, slot: usize, data: &[Complex64]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(data);
        assert_eq!(bytes, self.n * 16 * (data.len() / self.n));
        self.write(self.bufs[slot], data.as_ptr() as *const c_void, bytes)
    }

    fn download(&self, slot: usize, batch: usize) -> Result<Vec<Complex64>, String> {
        let mut out = vec![Complex64::new(0.0, 0.0); batch * self.n];
        self.read(
            self.bufs[slot],
            out.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(&out[..]),
        )?;
        Ok(out)
    }

    /// One full batched transform: `log2(n)` Stockham launches, ping-pong.
    ///
    /// Returns `(device_ns, wall_ns, result_slot)` — the device figure is the
    /// sum of the per-launch event deltas, the wall figure includes launch
    /// overhead and the final `clFinish`.
    ///
    /// The event deltas are only valid once the commands have completed, so
    /// every event is kept alive until after `clFinish` and read then — reading
    /// them at enqueue time returns `CL_PROFILING_INFO_NOT_AVAILABLE` and a
    /// silent 0.
    fn run(&self, batch: usize, inverse: bool) -> Result<(u64, u64, usize), String> {
        assert!(batch <= self.max_batch);
        let stages = self.n.trailing_zeros() as usize;
        let global = batch * (self.n / 2);
        let mut slot = 0usize;
        let mut events: Vec<ffi::Event> = Vec::with_capacity(stages);
        let wall0 = Instant::now();
        for l in 0..stages {
            let n_l = (self.n >> l) as u32;
            let s_l = (1u32 << l) as u32;
            let batch_u = batch as u32;
            let inverse_i = i32::from(inverse);
            let scale_last = i32::from(inverse && l + 1 == stages);
            let inv_n = 1.0f64 / self.n as f64;
            let src = self.bufs[slot];
            let dst = self.bufs[1 - slot];
            unsafe {
                let k = self.fft_kernel;
                let set = |i: u32, size: usize, ptr: *const c_void| {
                    ffi::check(
                        "clSetKernelArg",
                        (self.api.set_kernel_arg)(k, i, size, ptr),
                    )
                };
                set(0, std::mem::size_of::<ffi::Mem>(), &src as *const _ as *const c_void)?;
                set(1, std::mem::size_of::<ffi::Mem>(), &dst as *const _ as *const c_void)?;
                set(2, std::mem::size_of::<ffi::Mem>(), &self.tw as *const _ as *const c_void)?;
                set(3, 4, &n_l as *const u32 as *const c_void)?;
                set(4, 4, &s_l as *const u32 as *const c_void)?;
                set(5, 4, &(self.n as u32) as *const u32 as *const c_void)?;
                set(6, 4, &batch_u as *const u32 as *const c_void)?;
                set(7, 4, &inverse_i as *const i32 as *const c_void)?;
                set(8, 4, &scale_last as *const i32 as *const c_void)?;
                set(9, 8, &inv_n as *const f64 as *const c_void)?;

                let mut ev: ffi::Event = std::ptr::null_mut();
                let gsz = [global];
                let rc = (self.api.enqueue_kernel)(
                    self.queue,
                    self.fft_kernel,
                    1,
                    std::ptr::null(),
                    gsz.as_ptr(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    &mut ev,
                );
                ffi::check("clEnqueueNDRangeKernel", rc)?;
                events.push(ev);
            }
            slot = 1 - slot;
        }
        unsafe { ffi::check("clFinish", (self.api.finish)(self.queue))? };
        let wall_ns = wall0.elapsed().as_nanos() as u64;
        let mut dev_ns = 0u64;
        for ev in events {
            let mut start = 0u64;
            let mut end = 0u64;
            unsafe {
                (self.api.get_event_info)(
                    ev,
                    ffi::CL_PROFILING_COMMAND_START,
                    8,
                    &mut start as *mut u64 as *mut c_void,
                    std::ptr::null_mut(),
                );
                (self.api.get_event_info)(
                    ev,
                    ffi::CL_PROFILING_COMMAND_END,
                    8,
                    &mut end as *mut u64 as *mut c_void,
                    std::ptr::null_mut(),
                );
                (self.api.release_event)(ev);
            }
            dev_ns += end.saturating_sub(start);
        }
        Ok((dev_ns, wall_ns, slot))
    }

    /// A device buffer that releases itself when it drops.
    fn alloc(&self, bytes: usize) -> Result<Buffer<'_>, String> {
        let mut err = 0i32;
        let mem = unsafe {
            (self.api.create_buffer)(
                self.context,
                ffi::CL_MEM_READ_WRITE,
                bytes,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        ffi::check("clCreateBuffer", err)?;
        Ok(Buffer {
            api: &self.api,
            mem,
        })
    }

    /// One sinc-gather launch: `n_out` output pixels starting at `base`.
    ///
    /// `mems` is `[buf, locs, adjust, lut, out]` in kernel-argument order.
    fn run_sinc(
        &self,
        mems: [ffi::Mem; 5],
        buflen: u32,
        n_out: u32,
        base: u32,
    ) -> Result<(u64, u64), String> {
        let [buf, locs, adjust, lut, out] = mems;
        let mut ev: ffi::Event = std::ptr::null_mut();
        let wall0 = Instant::now();
        unsafe {
            let k = self.sinc_kernel;
            let set = |i: u32, size: usize, ptr: *const c_void| {
                ffi::check("clSetKernelArg", (self.api.set_kernel_arg)(k, i, size, ptr))
            };
            let psz = std::mem::size_of::<ffi::Mem>();
            set(0, psz, &buf as *const _ as *const c_void)?;
            set(1, 4, &buflen as *const u32 as *const c_void)?;
            set(2, psz, &locs as *const _ as *const c_void)?;
            set(3, psz, &adjust as *const _ as *const c_void)?;
            set(4, psz, &lut as *const _ as *const c_void)?;
            set(5, psz, &out as *const _ as *const c_void)?;
            set(6, 4, &n_out as *const u32 as *const c_void)?;
            set(7, 4, &base as *const u32 as *const c_void)?;
            let gsz = [n_out as usize];
            let rc = (self.api.enqueue_kernel)(
                self.queue,
                self.sinc_kernel,
                1,
                std::ptr::null(),
                gsz.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                &mut ev,
            );
            ffi::check("clEnqueueNDRangeKernel(sinc)", rc)?;
        }
        unsafe { ffi::check("clFinish", (self.api.finish)(self.queue))? };
        let wall_ns = wall0.elapsed().as_nanos() as u64;
        let mut start = 0u64;
        let mut end = 0u64;
        unsafe {
            (self.api.get_event_info)(
                ev,
                ffi::CL_PROFILING_COMMAND_START,
                8,
                &mut start as *mut u64 as *mut c_void,
                std::ptr::null_mut(),
            );
            (self.api.get_event_info)(
                ev,
                ffi::CL_PROFILING_COMMAND_END,
                8,
                &mut end as *mut u64 as *mut c_void,
                std::ptr::null_mut(),
            );
            (self.api.release_event)(ev);
        }
        Ok((end.saturating_sub(start), wall_ns))
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// `median_f64` from `scale_field_sinc`, transcribed (`select_nth_unstable_by`
/// with `total_cmp`, mean of the two middle order statistics for an even
/// count). The *algorithm* matters here, not just the value: this runs inside
/// the per-field pre-pass, and a full sort would be ~10x the cost of the real
/// selection and would swamp the gather the probe is trying to price.
fn median_f64(values: &mut [f64]) -> f64 {
    assert!(!values.is_empty());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        let (left, hi, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
        let hi = *hi;
        let lo = *left.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        (lo + hi) / 2.0
    } else {
        let (_, median, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
        *median
    }
}

/// The reference's level-adjust rule with `wow_level_adjust_smoothing == 0`,
/// hoisted out of the gather so the device kernel can read it as an array.
///
/// Returns `(median, threshold, adjust)`; `adjust[i]` is the per-sample factor
/// the gather multiplies its accumulator by.
fn compute_adjust(wow: &[f64], threshold_k: f64) -> (f64, f64, Vec<f64>) {
    let mut copy = wow.to_vec();
    let median = median_f64(&mut copy);
    for w in copy.iter_mut() {
        *w = (*w - median).abs();
    }
    let mad = median_f64(&mut copy);
    let threshold = if mad > 0.0 { threshold_k * mad } else { 0.001 };
    let adjust = wow
        .iter()
        .map(|&w| {
            if (w - median).abs() > threshold {
                median
            } else {
                w
            }
        })
        .collect();
    (median, threshold, adjust)
}

/// Deterministic pseudo-random complex data (a plain LCG keeps the probe
/// dependency-free and reproducible).
fn lcg_complex(count: usize, seed: u64) -> Vec<Complex64> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    (0..count).map(|_| Complex64::new(next(), next())).collect()
}

/// Single-threaded CPU baselines: the whole demod kernel, and the FFT layer
/// alone with the kernel's own batching (2 forward r2c + one k=2 and one k=4
/// inverse batch — the shapes `demod_block_cpu` issues).
fn cpu_baselines(n: usize, blocks: usize) -> (f64, f64, usize) {
    let spec = DecoderSpec::new(&DecodeRequest::default()).expect("spec");
    let levels = CalibLevels::defaults(spec.sys);
    let dspec = DemodSpecRef::with_plans(
        spec.freq,
        spec.freq_half,
        spec.freq_hz,
        spec.blocklen,
        &spec.filters,
        &levels,
    );
    assert_eq!(spec.blocklen, n, "blocklen changed; probe assumes n = blocklen");

    // A synthetic but well-conditioned block: the demod cost does not depend on
    // the sample values beyond denormal behaviour, and this keeps the probe
    // free of capture files.
    let data: Vec<f32> = (0..spec.blocklen)
        .map(|i| {
            let a = (i as f32) * 0.0007;
            (12000.0 + 6000.0 * a.sin()) + ((i as f32) * 0.13).cos() * 400.0
        })
        .collect();
    let mtf_pow = compute_mtf_pow(&spec.filters.mtf, 1.0);

    // Warm the ducc plans and the thread-local scratch first.
    let _ = demod_block_cpu(&data, 1.0, &dspec, true, Some(&mtf_pow), 0, 0);

    let t = Instant::now();
    for b in 0..blocks {
        let out = demod_block_cpu(
            &data,
            1.0,
            &dspec,
            true,
            Some(&mtf_pow),
            0,
            b as u64,
        );
        std::hint::black_box(&out);
    }
    let kernel_ms = t.elapsed().as_secs_f64() * 1e3 / blocks as f64;

    // FFT layer only.
    let real: Vec<f64> = (0..n)
        .map(|i| ((i as f64) * 0.381_966_011_250_105_1).fract() - 0.5)
        .collect();
    let mut four = vec![Complex64::new(0.0, 0.0); 4 * n];
    let mut fwd = Vec::with_capacity(n);
    let mut fft_layer = |four: &mut Vec<Complex64>, fwd: &mut Vec<Complex64>| {
        ffi_ducc::fft_real_full_into(&real, fwd);
        ffi_ducc::ifft_batch_rows_inplace(2, n, &mut four[..2 * n]);
        ffi_ducc::fft_real_full_into(&real, fwd);
        ffi_ducc::ifft_batch_rows_inplace(4, n, &mut four[..4 * n]);
    };
    fft_layer(&mut four, &mut fwd);

    let t = Instant::now();
    for _ in 0..blocks {
        fft_layer(&mut four, &mut fwd);
    }
    let fft_ms = t.elapsed().as_secs_f64() * 1e3 / blocks as f64;
    (kernel_ms, fft_ms, blocks)
}

#[test]
#[ignore = "perf probe: needs an OpenCL f64 device; prints a table, asserts only correctness"]
fn gpu_fft_probe() {
    // blocklen is 32768 for NTSC at any input rate in the shipped range; the
    // power of two is what makes the radix-2 kernel valid.
    let n = 32768usize;
    let batch = BLOCKS_PER_FIELD * 8;
    let rounds: usize = std::env::var("LD_GPU_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);

    // Match the decoder's pool split (12 demod threads -> a 3-thread tail pool)
    // so the CPU sinc baseline is the production one. `build_global` is a no-op
    // if a pool already exists; the printed thread count says which ran.
    crate::set_worker_threads(12);

    println!("\n== OpenCL f64 FFT probe ==");
    println!(
        "geometry: {batch} transforms x n={n} per field \
         ({BLOCKS_PER_FIELD} blocks x [6 inverse + 2 forward])"
    );

    let gpu = match Gpu::new(n, batch) {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: no usable OpenCL f64 device: {e}");
            return;
        }
    };
    println!(
        "device:   {} {} (CUs={}, global mem {:.1} GiB)",
        gpu.vendor,
        gpu.name,
        gpu.units,
        gpu.gmem as f64 / (1u64 << 30) as f64
    );

    // --- 1. correctness -----------------------------------------------------
    let check_batch = 4;
    let input = lcg_complex(check_batch * n, 0x5eed_1234);
    gpu.upload(0, &input).expect("upload");
    let (_, _, slot) = gpu.run(check_batch, true).expect("inverse");
    let gpu_out = gpu.download(slot, check_batch).expect("download");

    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for b in 0..check_batch {
        let want = ffi_ducc::ifft(&input[b * n..(b + 1) * n]);
        for i in 0..n {
            let d = (gpu_out[b * n + i] - want[i]).norm();
            worst = worst.max(d);
            scale = scale.max(want[i].norm());
        }
    }
    let rel = worst / scale;
    println!(
        "\n[1] correctness: max |gpu-ducc| = {worst:.3e} (relative {rel:.3e}) \
         vs ducc ifft, {check_batch} transforms"
    );

    // Round trip: forward then inverse must return the input (this also pins
    // the forward transform, since the inverse is pinned above).
    gpu.upload(0, &input).expect("upload");
    let (_, _, s1) = gpu.run(check_batch, false).expect("forward");
    // Swap so the second pass starts from the forward result.
    let fwd_out = gpu.download(s1, check_batch).expect("download fwd");
    let src_slot = 1 - s1;
    gpu.upload(src_slot, &fwd_out).expect("upload fwd");
    let (_, _, s2) = gpu.run(check_batch, true).expect("inverse");
    let back = gpu.download(s2, check_batch).expect("download round trip");
    let mut worst_rt = 0.0f64;
    for i in 0..check_batch * n {
        worst_rt = worst_rt.max((back[i] - input[i]).norm());
    }
    println!(
        "[1] round trip:  max |fwd->inv - input| = {worst_rt:.3e} \
         (input max {scale:.3e})"
    );
    if rel > 1e-12 {
        println!("ABORT: device FFT is not accurate enough to time meaningfully ({rel:.3e})");
        return;
    }

    // --- 2. device timing ---------------------------------------------------
    gpu.run(batch, true).expect("warmup");
    let mut dev = Vec::new();
    let mut wall = Vec::new();
    for _ in 0..rounds {
        let (d, w, _) = gpu.run(batch, true).expect("timed run");
        dev.push(d as f64 / 1e6);
        wall.push(w as f64 / 1e6);
    }
    let dev_ms = median(&mut dev);
    let wall_ms = median(&mut wall);

    // Host->device bandwidth on a field-sized transfer, for the per-field
    // round-trip estimate.
    let transfer_bytes = 16 * 1024 * 1024;
    let chunk = vec![Complex64::new(0.0, 0.0); transfer_bytes / 16];
    let reps = 8;
    let t = Instant::now();
    for _ in 0..reps {
        gpu.write(gpu.bufs[0], chunk.as_ptr() as *const c_void, transfer_bytes)
            .expect("write");
    }
    let h2d_gbs = (transfer_bytes * reps) as f64 / t.elapsed().as_secs_f64() / 1e9;

    // A Stockham pass moves 2 x batch x n x 16 bytes (read + write) per stage.
    let pass_bytes = 2 * batch * n * 16 * (n.trailing_zeros() as usize);
    let eff_gbs = pass_bytes as f64 / (wall_ms / 1e3) / 1e9;

    println!(
        "\n[2] device:    {dev_ms:.2} ms/field device, {wall_ms:.2} ms/field wall \
         (15 launches x {batch} transforms; {:.2} ms per transform)",
        wall_ms / batch as f64
    );
    println!(
        "    traffic:   {:.0} MB/field over 15 ping-pong passes = {eff_gbs:.0} GB/s effective \
         -- a radix-2 multi-pass FFT is bound by video-memory bandwidth, not FLOPs",
        pass_bytes as f64 / 1e6
    );
    println!("    H2D:       {h2d_gbs:.1} GB/s (blocking 16 MiB writes)");

    // --- 3. CPU baselines ---------------------------------------------------
    let (cpu_kernel_block_ms, cpu_fft_block_ms, blocks) = cpu_baselines(n, 8);
    let cpu_kernel_ms = cpu_kernel_block_ms * BLOCKS_PER_FIELD as f64;
    let cpu_fft_ms = cpu_fft_block_ms * BLOCKS_PER_FIELD as f64;
    let fft_ratio = cpu_fft_ms / cpu_kernel_ms;
    println!(
        "\n[3] CPU:       whole kernel {cpu_kernel_ms:.1} ms/field CPU \
         ({cpu_kernel_block_ms:.2} ms/block, {blocks} timed) x {BLOCKS_PER_FIELD} blocks, 1 thread"
    );
    println!(
        "    CPU FFT layer alone {cpu_fft_ms:.1} ms/field CPU ({:.2} ms/block) = {:.0}% of the \
         kernel; the other {:.0}% is cmul/unwrap/clip/audio/cut",
        cpu_fft_block_ms,
        fft_ratio * 100.0,
        (1.0 - fft_ratio) * 100.0
    );
    println!(
        "    in situ:   dcpu is 128 ms/field ({:.2} ms/block) at 11.5 ms wall -- 12 threads on 8 \
         cores cost ~2.3x in contention, so single-thread cost is never divided by the thread count",
        DEMOD_WALL_MS * POOL_EFFICIENCY / BLOCKS_PER_FIELD as f64
    );

    // --- 4. device sinc gather ---------------------------------------------
    //
    // The largest serial step, and the only one besides the FFT with device-
    // shaped work. Everything below is built here: a field of demodulated video,
    // the per-output-sample coordinates and wow factors the line-location spline
    // produces, and the reference's level-adjust array. The CPU reference is the
    // *production function*, not a replica, so the device output is compared
    // against the real scaler.
    let spec2 = DecoderSpec::new(&DecodeRequest::default()).expect("spec");
    let outwidth = spec2.sys_outlinelen;
    let linesout = spec2.sys_field_lines[0];
    let n_out = linesout * outwidth;
    let lineoffset = 0usize; // 7.4.0 uses 0/1 per field; the work is identical
    let base = outwidth * (lineoffset + 1);
    let n_loc = base + n_out;
    let buf_len = BLOCKS_PER_FIELD * spec2.blocksize();
    assert_eq!(SINC_TAP_COUNT, 16, "the sinc kernel hardcodes 16 taps");
    assert_eq!(SINC_PHASE_COUNT, 1 << 16, "the sinc kernel hardcodes 2^16 phases");

    let buf: Vec<f32> = (0..buf_len)
        .map(|i| {
            let a = (i as f32) * 0.0007;
            12000.0 + 6000.0 * a.sin() + 400.0 * ((i as f32) * 0.13).cos()
        })
        .collect();
    let step = spec2.linelen as f64 / outwidth as f64;
    let locs: Vec<f64> = (0..n_loc)
        .map(|i| (i as f64) * step + 0.4 * ((i as f64) * 0.017).sin())
        .collect();
    // Mild wow plus periodic spikes so the level-adjust threshold branch is
    // exercised by the comparison instead of being dead code.
    let wow: Vec<f64> = (0..n_loc)
        .map(|i| {
            let w = 1.0 + 0.0008 * ((i as f64) * 0.0013).sin();
            if i % 40_000 == 7 {
                w * 1.05
            } else {
                w
            }
        })
        .collect();
    let lut = sinc_lut();
    let mut cpu_out = vec![0.0f32; n_out];
    let sinc_params = SincScaleParams {
        lineoffset,
        outwidth,
        wow_level_adjust_smoothing: 0.0,
        level_adjust_threshold: 15.0,
    };

    // The host pre-pass a port keeps off the device: wow median, MAD, threshold
    // and the per-sample adjust array.
    let (median_w, threshold, adjust) = {
        let mut pre_times = Vec::new();
        let mut last = None;
        for _ in 0..rounds {
            let t = Instant::now();
            last = Some(compute_adjust(&wow, 15.0));
            pre_times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let _ = median(&mut pre_times);
        last.unwrap()
    };
    let pre_ms = {
        let mut v = Vec::new();
        for _ in 0..rounds {
            let t = Instant::now();
            std::hint::black_box(compute_adjust(&wow, 15.0));
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        median(&mut v)
    };

    let mut cpu_times = Vec::new();
    for _ in 0..rounds {
        let t = Instant::now();
        scale_field_sinc(&buf, &mut cpu_out, &locs, &wow, lut, sinc_params);
        cpu_times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let sinc_cpu_ms = median(&mut cpu_times);
    let gather_cpu_ms = (sinc_cpu_ms - pre_ms).max(0.0);

    // Device side: upload the field once, then time the gather alone.
    let d_buf = gpu.alloc(buf.len() * 4).expect("alloc buf");
    let d_locs = gpu.alloc(n_loc * 4).expect("alloc locs");
    let d_adj = gpu.alloc(n_loc * 8).expect("alloc adjust");
    let d_lut = gpu.alloc(lut.len() * 4).expect("alloc lut");
    let d_out = gpu.alloc(n_out * 4).expect("alloc out");
    let locs_f32: Vec<f32> = locs.iter().map(|&v| v as f32).collect();
    gpu.write(d_buf.mem, buf.as_ptr() as *const c_void, buf.len() * 4)
        .expect("upload buf");
    gpu.write(d_locs.mem, locs_f32.as_ptr() as *const c_void, n_loc * 4)
        .expect("upload locs");
    gpu.write(d_adj.mem, adjust.as_ptr() as *const c_void, n_loc * 8)
        .expect("upload adjust");
    gpu.write(d_lut.mem, lut.as_ptr() as *const c_void, lut.len() * 4)
        .expect("upload lut");
    let mems = [d_buf.mem, d_locs.mem, d_adj.mem, d_lut.mem, d_out.mem];
    let sin = |g: &Gpu| {
        g.run_sinc(mems, buf_len as u32, n_out as u32, base as u32)
            .expect("sinc launch")
    };
    sin(&gpu);

    let mut gpu_out = vec![0.0f32; n_out];
    gpu.read(d_out.mem, gpu_out.as_mut_ptr() as *mut c_void, n_out * 4)
        .expect("read sinc out");
    let mut mism = 0usize;
    let mut worst = 0.0f64;
    for i in 0..n_out {
        if gpu_out[i].to_bits() != cpu_out[i].to_bits() {
            mism += 1;
            worst = worst.max((gpu_out[i] as f64 - cpu_out[i] as f64).abs());
        }
    }
    let level = cpu_out.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));

    let mut sdev = Vec::new();
    let mut swall = Vec::new();
    for _ in 0..rounds {
        let (d, w) = sin(&gpu);
        sdev.push(d as f64 / 1e6);
        swall.push(w as f64 / 1e6);
    }
    let sinc_dev_ms = median(&mut sdev);
    let sinc_wall_ms = median(&mut swall);

    println!(
        "\n[4] sinc:      {n_out} output pixels ({linesout}x{outwidth}) x {SINC_TAP_COUNT} taps \
         from a {buf_len}-sample field, {} LUT rows of {SINC_TAP_COUNT}",
        lut.len() / SINC_TAP_COUNT
    );
    println!(
        "    vs CPU:    {mism} of {n_out} samples differ bit-for-bit (worst |d| {worst:.3e} on a \
         {level:.1} level) -- the gather is scalar per-sample arithmetic, so the device \
         reproduces it exactly"
    );
    println!(
        "    CPU:       {sinc_cpu_ms:.2} ms/field = {pre_ms:.2} ms pre-pass (wow median/MAD/adjust, \
         median={median_w:.9}, thr={threshold:.6}) + {gather_cpu_ms:.2} ms gather"
    );
    println!(
        "               (on {} tail-pool threads; `down` as a whole is 5.67 ms in LD_TIMING -- the \
         scaler is only part of that bundle, the rest is wow/dropout/audio/encode)",
        crate::tail_pool_threads()
    );
    println!(
        "    device:    {sinc_dev_ms:.3} ms device, {sinc_wall_ms:.3} ms wall = {:.0}x the CPU gather",
        gather_cpu_ms / sinc_wall_ms
    );

    // --- 5. verdict ---------------------------------------------------------
    //
    // The comparison that matters is against the demod's **in-situ wall**, not
    // against single-thread cost divided by the thread count: the pool runs 12
    // threads on 8 physical cores over a bandwidth-bound kernel, so its total
    // CPU is ~2.3x the sum of single-threaded block times (128 ms vs 56 ms) and
    // only ~11x of that total becomes wall.
    let fft_wall_now = DEMOD_WALL_MS * fft_ratio;
    let nonfft_wall = DEMOD_WALL_MS * (1.0 - fft_ratio);

    // Per-field bytes a real port would move over PCIe: the block samples down
    // (f32) and the demodulated channels back (four f32 channels per block).
    let samples_per_field = (n * 21) as f64; // ~21 blocks of new samples/field
    let down_mb = samples_per_field * 4.0 / 1e6;
    let up_mb = samples_per_field * 4.0 * 4.0 / 1e6;
    let xfer_ms = (down_mb + up_mb) / 1e3 / h2d_gbs * 1e3;

    // Conservative: the non-FFT stages (cmul, unwrap, clip) stay on the pool at
    // their current in-situ cost, and the GPU gets no credit for them.
    let gpu_kernel_lb = wall_ms + nonfft_wall + xfer_ms;
    let iter_gpu = gpu_kernel_lb.max(TAIL_MS);
    let gain = (ITER_NOW_MS - iter_gpu) / ITER_NOW_MS;

    // Second scenario, now measured end to end: the sinc gather runs on the
    // device, so its CPU cost leaves the tail (the pre-pass stays on the host),
    // and the demod output no longer has to come back -- only the luma does.
    let luma_mb = (n_out * 4) as f64 / 1e6;
    let locs_mb = (n_loc * 4) as f64 / 1e6;
    let adj_mb = (n_loc * 8) as f64 / 1e6;
    let xfer_sinc_ms = (down_mb + locs_mb + adj_mb + luma_mb) / 1e3 / h2d_gbs * 1e3;
    // Only the gather leaves the CPU; the pre-pass (median/MAD/adjust) and the
    // rest of `down` stay, so the tail shrinks by the gather alone.
    let tail_after_sinc = TAIL_MS - gather_cpu_ms;
    let gpu_kernel_lb2 = wall_ms + nonfft_wall + sinc_wall_ms + xfer_sinc_ms;
    let iter_gpu_sinc = gpu_kernel_lb2.max(tail_after_sinc);
    let gain_sinc = (ITER_NOW_MS - iter_gpu_sinc) / ITER_NOW_MS;

    println!("\n[5] per-field budget (LD_TIMING medians, this box):");
    println!(
        "    now:            iter {ITER_NOW_MS:.2} ms = demod {DEMOD_WALL_MS:.2} ms wall \
         (hidden under the tail) + {PF_WAIT_MS:.2} ms exposed wait"
    );
    println!(
        "    FFT layer:      {fft_wall_now:.2} ms of that wall; non-FFT {nonfft_wall:.2} ms stays \
         on the pool"
    );
    println!(
        "    tail:           {TAIL_MS:.2} ms serial (down 5.67 incl. this {sinc_cpu_ms:.2} sinc, \
         asm 2.53, proc 2.26, vits 0.55, out 0.10) -- the GPU cannot touch proc/vits"
    );
    println!(
        "    PCIe estimate:  {down_mb:.1} MB/field (down) + {up_mb:.1} MB (up) = {xfer_ms:.2} ms; \
         with the sinc on device {xfer_sinc_ms:.2} ms (locs+adjust down, luma up)"
    );
    println!(
        "    device vs pool: {wall_ms:.2} ms vs {fft_wall_now:.2} ms for the FFT layer alone \
         ({:.2}x)",
        fft_wall_now / wall_ms
    );
    println!(
        "    GPU kernel lb:  {gpu_kernel_lb:.2} ms/field (device {wall_ms:.2} + non-FFT \
         {nonfft_wall:.2} + xfer {xfer_ms:.2})"
    );
    println!(
        "    -> GPU FFT only:   iter {iter_gpu:.2} ms, gain {:+.1}%",
        gain * 100.0
    );
    println!(
        "       (capped: even an infinitely fast device FFT cannot beat the {TAIL_MS:.2} ms tail, so \
         the ceiling for this scenario is {:+.1}%)",
        (ITER_NOW_MS - TAIL_MS) / ITER_NOW_MS * 100.0
    );
    println!(
        "    GPU kernel 2:   {gpu_kernel_lb2:.2} ms/field (device {wall_ms:.2} + non-FFT \
         {nonfft_wall:.2} + sinc {sinc_wall_ms:.3} + xfer {xfer_sinc_ms:.2})"
    );
    println!(
        "    -> GPU FFT + sinc: iter {iter_gpu_sinc:.2} ms, gain {:+.1}% (tail becomes \
         {tail_after_sinc:.2} ms; asm/proc/vits are then the wall)",
        gain_sinc * 100.0
    );
    println!(
        "\n    VERDICT: device FFT is {:.2}x the pool's FFT layer and {:.2}x the whole demod wall; \
         the sinc gather is {:.0}x the CPU gather.",
        fft_wall_now / wall_ms,
        DEMOD_WALL_MS / wall_ms,
        gather_cpu_ms / sinc_wall_ms
    );
    if gpu_kernel_lb >= DEMOD_WALL_MS {
        println!(
            "    The device kernel is not faster than the pool it would replace: no win available, \
             do not build it."
        );
    } else if gain_sinc < 0.15 {
        println!(
            "    FFT-only is capped at {:+.1}% by the serial tail, and even with the measured sinc \
             gather moved to the device the model reaches only {:+.1}% -- not worth it.",
            (ITER_NOW_MS - TAIL_MS) / ITER_NOW_MS * 100.0,
            gain_sinc * 100.0
        );
    } else {
        println!(
            "    FFT-only is capped at {:+.1}% by the serial tail; the measured sinc gather moves \
             too, and the modelled win clears 15%. Worth a spike only if a second output class is \
             acceptable -- the sinc gather is bit-exact on the device, but the FFT is not, so no \
             `.tbc` can be verified against the reference.",
            (ITER_NOW_MS - TAIL_MS) / ITER_NOW_MS * 100.0
        );
    }
    println!(
        "    (Non-parity by construction *for the Stockham kernel measured here*: its radix order \
         cannot reproduce ducc's rounding. See `gpu_bitexact_probe` for the ducc-replica \
         kernels, which are bit-exact but cost more launches.)\n"
    );
}

// ===========================================================================
// Bit-exact ducc replica on the device
// ===========================================================================
//
// The Stockham kernel above is mathematically right and numerically different
// from ducc, which makes any `--gpu` path a second output class. This section
// asks the other question: can the device reproduce ducc's *own* arithmetic?
//
// It can. ducc's 1-D transforms on the decode path are plain IEEE `+`, `-`, `*`
// in a fixed order -- no FMA (baseline SSE2 build) and no reassociation -- so a
// kernel that issues the same operations in the same order returns the same
// bits. `decode::ducc_plan` is the receiver-verified CPU model of that plan
// (gated against `ffi_ducc` at 8..32768); this runs the same plan on the device
// and compares against ducc directly.
//
// The one thing that can silently break it is contraction: OpenCL C *permits* a
// compiler to fuse `a*b + c` into an FMA, and an FMA is a different rounding.
// `#pragma OPENCL FP_CONTRACT OFF` is the portable way to forbid it (OpenCL C
// 1.2+); the probe also forbids it by construction where it can, by splitting
// every product into its own statement.

const DUCC_OCL_SRC: &str = r#"
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define HSQT2 0.707106781186547524400844362104849

inline double2 cadd(double2 a, double2 b) { return (double2)(a.x+b.x, a.y+b.y); }
inline double2 csub(double2 a, double2 b) { return (double2)(a.x-b.x, a.y-b.y); }
inline double2 cmx(double2 a, double2 b)
{ return (double2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }

inline void rot90(double2 *a, int fwd)
{ double t = a->x; if (fwd) { a->x = a->y; a->y = -t; } else { a->x = -a->y; a->y = t; } }

inline void rot45(double2 *a, int fwd)
{
  double t = a->x;
  if (fwd) { a->x = HSQT2*(a->x + a->y); a->y = HSQT2*(a->y - t); }
  else     { a->x = HSQT2*(a->x - a->y); a->y = HSQT2*(a->y + t); }
}

inline void rot135(double2 *a, int fwd)
{
  double t = a->x;
  if (fwd) { a->x = HSQT2*(a->y - a->x); a->y = HSQT2*(-t - a->y); }
  else     { a->x = HSQT2*(-a->x - a->y); a->y = HSQT2*(t - a->y); }
}

inline void pmi(double2 *a, double2 *b)
{ double2 t = *a; *a = cadd(t, *b); *b = csub(t, *b); }

inline double2 smul(double2 v, double2 w, int fwd)
{
  if (fwd) return (double2)(v.x*w.x + v.y*w.y, v.y*w.x - v.x*w.y);
  return (double2)(v.x*w.x - v.y*w.y, v.x*w.y + v.y*w.x);
}

inline double2 uroot(__global const double2 *v1, __global const double2 *v2,
                     int idx, int n, int mask, int shift)
{
  int lo = (idx <= (n >> 1));
  int i = lo ? idx : (n - idx);
  double2 m = cmx(v1[i & mask], v2[i >> shift]);
  return lo ? m : (double2)(m.x, -m.y);
}

// cfftp8::exec_, both halves. `wa` is the twiddle row for this `i`, or 0 for
// the untwiddled form.
inline void bf8(double2 *v, double2 *o, int fwd, __global const double2 *wa)
{
  double2 a0, a1, a2, a3, a4, a5, a6, a7;
  a1 = cadd(v[1], v[5]);
  a5 = csub(v[1], v[5]);
  a3 = cadd(v[3], v[7]);
  a7 = csub(v[3], v[7]);
  rot90(&a7, fwd);
  pmi(&a1, &a3);
  rot90(&a3, fwd);
  pmi(&a5, &a7);
  rot45(&a5, fwd);
  rot135(&a7, fwd);
  a0 = cadd(v[0], v[4]);
  a4 = csub(v[0], v[4]);
  a2 = cadd(v[2], v[6]);
  a6 = csub(v[2], v[6]);
  if (wa == 0)
  {
    o[0] = cadd(cadd(a0, a2), a1);
    o[4] = csub(cadd(a0, a2), a1);
    o[2] = cadd(csub(a0, a2), a3);
    o[6] = csub(csub(a0, a2), a3);
    rot90(&a6, fwd);
    o[1] = cadd(cadd(a4, a6), a5);
    o[5] = csub(cadd(a4, a6), a5);
    o[3] = cadd(csub(a4, a6), a7);
    o[7] = csub(csub(a4, a6), a7);
  }
  else
  {
    pmi(&a0, &a2);
    o[0] = cadd(a0, a1);
    o[4] = smul(csub(a0, a1), wa[3], fwd);
    o[2] = smul(cadd(a2, a3), wa[1], fwd);
    o[6] = smul(csub(a2, a3), wa[5], fwd);
    rot90(&a6, fwd);
    pmi(&a4, &a6);
    o[1] = smul(cadd(a4, a5), wa[0], fwd);
    o[5] = smul(csub(a4, a5), wa[4], fwd);
    o[3] = smul(cadd(a6, a7), wa[2], fwd);
    o[7] = smul(csub(a6, a7), wa[6], fwd);
  }
}

inline void bf4(double2 *v, double2 *o, int fwd, __global const double2 *wa)
{
  double2 t2 = cadd(v[0], v[2]);
  double2 t1 = csub(v[0], v[2]);
  double2 t3 = cadd(v[1], v[3]);
  double2 t4 = csub(v[1], v[3]);
  rot90(&t4, fwd);
  if (wa == 0)
  {
    o[0] = cadd(t2, t3);
    o[1] = cadd(t1, t4);
    o[2] = csub(t2, t3);
    o[3] = csub(t1, t4);
  }
  else
  {
    o[0] = cadd(t2, t3);
    o[1] = smul(cadd(t1, t4), wa[0], fwd);
    o[2] = smul(csub(t2, t3), wa[1], fwd);
    o[3] = smul(csub(t1, t4), wa[2], fwd);
  }
}

// One radix pass over `bt` independent blocks of `ip*1` points laid out
// contiguously. Item count is bt*l1*ido; the three index maps are
// cfftpN::exec_'s.
#define PASS_BODY(RADIX, BF)                                                  \
  int t = get_global_id(0);                                                   \
  int blk = t / per_block;                                                    \
  int u = t - blk * per_block;                                                \
  int base = blk * ip;                                                        \
  double2 v[RADIX];                                                           \
  double2 o[RADIX];                                                           \
  int b;                                                                      \
  if (l1 == 1) {                                                              \
    int i = u;                                                                \
    for (b = 0; b < RADIX; b++) v[b] = src[base + i + ido*b];                 \
    BF(v, o, fwd, i == 0 ? 0 : wa + (i-1)*(RADIX-1));                         \
    for (b = 0; b < RADIX; b++) dst[base + i + ido*b] = o[b];                 \
  } else if (ido == 1) {                                                      \
    int k = u;                                                                \
    for (b = 0; b < RADIX; b++) v[b] = src[base + b + RADIX*k];               \
    BF(v, o, fwd, 0);                                                         \
    for (b = 0; b < RADIX; b++) dst[base + k + l1*b] = o[b];                  \
  } else {                                                                    \
    int i = u % ido;                                                          \
    int k = u / ido;                                                          \
    for (b = 0; b < RADIX; b++) v[b] = src[base + i + ido*(b + RADIX*k)];     \
    BF(v, o, fwd, i == 0 ? 0 : wa + (i-1)*(RADIX-1));                         \
    for (b = 0; b < RADIX; b++) dst[base + i + ido*(k + l1*b)] = o[b];        \
  }

__kernel void fft_pass8(__global const double2 *src, __global double2 *dst,
                        __global const double2 *wa, int ip, int l1, int ido,
                        int fwd, int per_block)
{ PASS_BODY(8, bf8) }

__kernel void fft_pass4(__global const double2 *src, __global double2 *dst,
                        __global const double2 *wa, int ip, int l1, int ido,
                        int fwd, int per_block)
{ PASS_BODY(4, bf4) }

__kernel void fft_pass2(__global const double2 *src, __global double2 *dst,
                        __global const double2 *wa, int ip, int l1, int ido,
                        int fwd, int per_block)
{
  int t = get_global_id(0);
  int blk = t / per_block;
  int u = t - blk * per_block;
  int base = blk * ip;
  if (l1 == 1) {
    int i = u;
    double2 x0 = src[base + i];
    double2 x1 = src[base + i + ido];
    dst[base + i] = cadd(x0, x1);
    dst[base + i + ido] = (i == 0) ? csub(x0, x1) : smul(csub(x0, x1), wa[i-1], fwd);
  } else if (ido == 1) {
    int k = u;
    double2 x0 = src[base + 2*k];
    double2 x1 = src[base + 2*k + 1];
    dst[base + k] = cadd(x0, x1);
    dst[base + k + l1] = csub(x0, x1);
  } else {
    int i = u % ido;
    int k = u / ido;
    int at = base + i + ido*2*k;
    double2 x0 = src[at];
    double2 x1 = src[at + ido];
    int ot = base + i + ido*k;
    dst[ot] = cadd(x0, x1);
    dst[ot + ido*l1] = (i == 0) ? csub(x0, x1) : smul(csub(x0, x1), wa[i-1], fwd);
  }
}

// cfft_multipass, `l1 == 1 && ido > 1`: blocks gathered at stride `ido`.
// `stride` is the block's own length (the packet's `ip`), so `bt` blocks are
// handled by block-indexing the global id -- that is what lets a field's blocks
// share one launch.
__kernel void node_gather(__global const double2 *src, __global double2 *dst,
                          int ip, int ido, int stride)
{
  int per = ido * ip;
  int g = get_global_id(0);
  int blk = g / per;
  int t = g - blk * per;
  int i = t / ip;
  int m = t - i * ip;
  dst[blk*stride + i*ip + m] = src[blk*stride + i + ido*m];
}

// ... and scattered back with the outer twiddles.
__kernel void node_scatter(__global const double2 *src, __global double2 *dst,
                           __global const double2 *v1, __global const double2 *v2,
                           int nroots, int mask, int shift, int rfct,
                           int ip, int ido, int fwd, int stride)
{
  int per = ido * ip;
  int g = get_global_id(0);
  int blk = g / per;
  int t = g - blk * per;
  int i = t / ip;
  int m = t - i * ip;
  double2 val = src[blk*stride + i*ip + m];
  if (i != 0 && m != 0)
    val = smul(val, uroot(v1, v2, rfct*m*i, nroots, mask, shift), fwd);
  dst[blk*stride + i + ido*m] = val;
}

// cfft_multipass, `ido == 1`: contiguous blocks, transposed on the way out.
__kernel void node_transpose(__global const double2 *src, __global double2 *dst,
                             int ip, int l1, int stride)
{
  int per = l1 * ip;
  int g = get_global_id(0);
  int blk = g / per;
  int t = g - blk * per;
  int n = t / ip;
  int m = t - n * ip;
  dst[blk*stride + n + m*l1] = src[blk*stride + n*ip + m];
}

// rfftp_complexify, forward=false: half spectrum -> half-length complex input.
__kernel void c2r_pre(__global const double2 *sp, __global double2 *cch,
                      __global const double2 *v1, __global const double2 *v2,
                      int nroots, int mask, int shift, int n)
{
  int i = get_global_id(0);
  int nh = n >> 1;
  if (i == 0)
  {
    cch[0] = (double2)(sp[0].x + sp[nh].x, sp[0].x - sp[nh].x);
    return;
  }
  int xi = nh - i;
  if (i > xi) return;
  double2 t1 = (double2)(sp[i].x, sp[i].y);
  double2 t2 = (double2)(sp[xi].x, -sp[xi].y);
  double2 xe = cadd(t1, t2);
  double2 xo = cmx(csub(t1, t2), uroot(v1, v2, i, nroots, mask, shift));
  double2 lo = (double2)(xe.x - xo.y, xe.y + xo.x);
  double2 hi = (double2)(xe.x + xo.y, -xe.y + xo.x);
  cch[i] = lo;
  cch[xi] = hi;
}

// The tail of c2r: `in[i] = res[i]*fct` over all n doubles.
__kernel void c2r_scale(__global const double2 *src, __global double2 *dst,
                        int nh, double fct)
{
  int t = get_global_id(0);
  if (t >= nh) return;
  double2 v = src[t];
  dst[t] = (double2)(v.x*fct, v.y*fct);
}

// --- batched late group (`--gpu`) -----------------------------------------
// `w = n/2 + 1` bins per block, `nh = n/2`. All three take a block index out of
// the global id, so one launch covers `bt` independent blocks.
__kernel void r2c_post_b(__global const double2 *res, __global double2 *sp,
                         __global const double2 *v1, __global const double2 *v2,
                         int nroots, int mask, int shift, int n)
{
  int nh = n >> 1;
  int w = nh + 1;
  int g = get_global_id(0);
  int blk = g / w;
  int i = g - blk * w;
  int rb = blk * nh;
  int sb = blk * w;
  if (i == 0)
  {
    sp[sb] = (double2)(res[rb].x + res[rb].y, 0.0);
    return;
  }
  if (i == nh)
  {
    sp[sb + nh] = (double2)(res[rb].x - res[rb].y, 0.0);
    return;
  }
  int xi = nh - i;
  // Each mirrored pair must be written by *one* work item, the lower index.
  // Without this guard both work items compute both bins: the two forms are
  // equal in exact arithmetic (`get(nh-j) == -conj(get(j))`), but the root
  // table's entries are products of two rounded table values, so the two
  // computations disagree in the last bit at ~half the bins -- and which store
  // lands last is a race, i.e. the spectrum was 1 ulp off ducc's, differently
  // from run to run. `r2c_post` and `c2r_pre_r` already have this guard; this
  // is the batched form of the same loop (`gpu_late_group_f64_gate` pins it).
  if (i > xi) return;
  double2 ri = res[rb + i];
  double2 rxi = res[rb + xi];
  double2 xe = (double2)(ri.x + rxi.x, ri.y - rxi.y);
  double2 tw = uroot(v1, v2, i, nroots, mask, shift);
  double2 t = (double2)(ri.y + rxi.y, rxi.x - ri.x);
  double2 xo = (double2)(t.x*tw.x - t.y*(-tw.y), t.x*(-tw.y) + t.y*tw.x);
  sp[sb + i] = (double2)(0.5*(xe.x + xo.x), 0.5*(xe.y + xo.y));
  sp[sb + xi] = (double2)(0.5*(xe.x - xo.x), 0.5*(xo.y - xe.y));
}

__kernel void c2r_pre_b(__global const double2 *sp, __global double2 *cch,
                        __global const double2 *v1, __global const double2 *v2,
                        int nroots, int mask, int shift, int n, int ch)
{
  int nh = n >> 1;
  int w = nh + 1;
  int g = get_global_id(0);
  int blk = g / nh;
  int i = g - blk * nh;
  int sb = (blk * 3 + ch) * w;
  int cb = blk * nh;
  if (i == 0)
  {
    cch[cb] = (double2)(sp[sb].x + sp[sb + nh].x, sp[sb].x - sp[sb + nh].x);
    return;
  }
  int xi = nh - i;
  if (i > xi) return;
  double2 t1 = sp[sb + i];
  double2 t2 = (double2)(sp[sb + xi].x, -sp[sb + xi].y);
  double2 xe = cadd(t1, t2);
  double2 xo = cmx(csub(t1, t2), uroot(v1, v2, i, nroots, mask, shift));
  cch[cb + i] = (double2)(xe.x - xo.y, xe.y + xo.x);
  cch[cb + xi] = (double2)(xe.x + xo.y, -xe.y + xo.x);
}

// The same pre-pack for a row-major batch: row `r` is already the spectrum row
// (`prod` is `(blk*3 + ch)`, which is the row index when the three channels are
// stacked). One launch and one plan chain then cover all three channels, which
// is three times fewer launches for byte-identical arithmetic.
__kernel void c2r_pre_r(__global const double2 *sp, __global double2 *cch,
                        __global const double2 *v1, __global const double2 *v2,
                        int nroots, int mask, int shift, int n)
{
  int nh = n >> 1;
  int w = nh + 1;
  int g = get_global_id(0);
  int blk = g / nh;
  int i = g - blk * nh;
  int sb = blk * w;
  int cb = blk * nh;
  if (i == 0)
  {
    cch[cb] = (double2)(sp[sb].x + sp[sb + nh].x, sp[sb].x - sp[sb + nh].x);
    return;
  }
  int xi = nh - i;
  if (i > xi) return;
  double2 t1 = sp[sb + i];
  double2 t2 = (double2)(sp[sb + xi].x, -sp[sb + xi].y);
  double2 xe = cadd(t1, t2);
  double2 xo = cmx(csub(t1, t2), uroot(v1, v2, i, nroots, mask, shift));
  cch[cb + i] = (double2)(xe.x - xo.y, xe.y + xo.x);
  cch[cb + xi] = (double2)(xe.x + xo.y, -xe.y + xo.x);
}

// One filter product per (block, channel) bin. `np_cmul` is numpy's complex
// multiply, which is `fmaddsub` in hardware: both products are fused here or
// every bin moves.
__kernel void video_prod(__global const double2 *spec, __global const double2 *filt,
                         __global double2 *prod, int w)
{
  int g = get_global_id(0);
  int per = 3 * w;
  int blk = g / per;
  int t = g - blk * per;
  int ch = t / w;
  int i = t - ch * w;
  double2 a = spec[blk * w + i];
  double2 b = filt[ch * w + i];
  double2 v;
  v.x = fma(a.x, b.x, -(a.y * b.y));
  v.y = fma(a.x, b.y, a.y * b.x);
  prod[(blk * 3 + ch) * w + i] = v;
}

// `cut(ifft(x).re as f32)`: scale by `fct`, take `.re`, cast to f32 and gather
// the kept range -- the same three steps `irfft` + `rolled_f32_range` take.
__kernel void c2r_emit(__global const double2 *src, __global float *dst,
                       int nh, int cstart, int len, double fct, int base)
{
  int g = get_global_id(0);
  int blk = g / len;
  int i = g - blk * len;
  int j = cstart + i;
  double2 c = src[blk * nh + (j >> 1)];
  double v = (j & 1) ? c.y : c.x;
  dst[base + blk * len + i] = (float)(v * fct);
}

// rfftp_complexify, forward=true: half-length complex result -> spectrum.
// The `general_r2c` packing is folded in: bin k is `(out[2k-1], out[2k])`.
__kernel void r2c_post(__global const double2 *res, __global double2 *sp,
                       __global const double2 *v1, __global const double2 *v2,
                       int nroots, int mask, int shift, int n)
{
  int i = get_global_id(0);
  int nh = n >> 1;
  if (i == 0)
  {
    sp[0] = (double2)(res[0].x + res[0].y, 0.0);
    return;
  }
  if (i == nh)
  {
    sp[nh] = (double2)(res[0].x - res[0].y, 0.0);
    return;
  }
  int xi = nh - i;
  if (i > xi) return;
  double2 ri = res[i];
  double2 rxi = res[xi];
  double2 xe = (double2)(ri.x + rxi.x, ri.y - rxi.y);
  double2 tw = uroot(v1, v2, i, nroots, mask, shift);
  double2 t = (double2)(ri.y + rxi.y, rxi.x - ri.x);
  double2 xo = (double2)(t.x*tw.x - t.y*(-tw.y), t.x*(-tw.y) + t.y*tw.x);
  sp[i] = (double2)(0.5*(xe.x + xo.x), 0.5*(xe.y + xo.y));
  sp[xi] = (double2)(0.5*(xe.x - xo.x), 0.5*(xo.y - xe.y));
}
"#;

/// One kernel argument. The names are the OpenCL types, not Rust ones, so the
/// call sites read like the kernel signatures.
#[derive(Clone, Copy)]
enum Arg<'a> {
    Buf(ffi::Mem),
    Int(i32),
    Dbl(f64),
    _Phantom(std::marker::PhantomData<&'a ()>),
}

/// A pre-built node of a plan: the passes with their twiddle tables resident on
/// the device.
struct NodeDev {
    ip: usize,
    l1: usize,
    ido: usize,
    rfct: usize,
    passes: Vec<PassDev>,
    /// Buffer indices of the two packet children, when this node is a packet
    /// node (`l1 == ido == 1` only, for the shapes the decoder uses).
    children: Option<(usize, usize)>,
}

struct PassDev {
    radix: usize,
    l1: usize,
    ido: usize,
    wa: usize,
}

/// The bit-exact device harness: one program, the kernels, the plan's twiddle
/// tables and the roots table. Dropping it releases everything it created.
struct DuccGpu<'a> {
    gpu: &'a Gpu,
    program: ffi::Program,
    kernels: Vec<(&'static str, ffi::Kernel)>,
    mems: Vec<ffi::Mem>,
    /// When set, each launch carries an event so `profile_take` can report the
    /// kernels' own time apart from the CPU-side submit that precedes them.
    events: std::cell::RefCell<Vec<ffi::Event>>,
    profiling: std::cell::Cell<bool>,
}

impl Drop for DuccGpu<'_> {
    fn drop(&mut self) {
        // The `Gpu` that owns the context is dropped after this one (it is
        // declared first at the call site), so these objects are still valid.
        unsafe {
            for (_, k) in &self.kernels {
                (self.gpu.api.release_kernel)(*k);
            }
            (self.gpu.api.release_program)(self.program);
            for m in &self.mems {
                (self.gpu.api.release_mem)(*m);
            }
        }
    }
}

impl<'a> DuccGpu<'a> {
    fn new(gpu: &'a Gpu) -> Result<Self, String> {
        let mut err = 0i32;
        let src = DUCC_OCL_SRC;
        let sp = src.as_ptr();
        let len = src.len();
        let program = unsafe {
            (gpu.api.create_program)(gpu.context, 1, &sp, &len as *const usize, &mut err)
        };
        ffi::check("clCreateProgramWithSource (bit-exact)", err)?;
        let device = gpu.device;
        let rc = unsafe {
            (gpu.api.build_program)(
                program,
                1,
                &device,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if rc != ffi::CL_SUCCESS {
            let mut log = vec![0u8; 8192];
            let mut got = 0usize;
            unsafe {
                (gpu.api.get_program_build_info)(
                    program,
                    device,
                    ffi::CL_PROGRAM_BUILD_LOG,
                    log.len(),
                    log.as_mut_ptr() as *mut c_void,
                    &mut got,
                );
            }
            let end = log.iter().position(|&b| b == 0).unwrap_or(got.min(log.len()));
            return Err(format!(
                "bit-exact program failed to build (cl error {rc}): {}",
                String::from_utf8_lossy(&log[..end])
            ));
        }
        let mut kernels = Vec::new();
        for name in [
            "fft_pass8",
            "fft_pass4",
            "fft_pass2",
            "node_gather",
            "node_scatter",
            "node_transpose",
            "c2r_pre",
            "c2r_scale",
            "r2c_post",
            "r2c_post_b",
            "c2r_pre_b",
            "c2r_pre_r",
            "video_prod",
            "c2r_emit",
        ] {
            let mut c = Vec::with_capacity(name.len() + 1);
            c.extend_from_slice(name.as_bytes());
            c.push(0);
            let k = unsafe { (gpu.api.create_kernel)(program, c.as_ptr(), &mut err) };
            ffi::check(&format!("clCreateKernel({name})"), err)?;
            kernels.push((name, k));
        }
        Ok(Self {
            gpu,
            program,
            kernels,
            mems: Vec::new(),
            events: std::cell::RefCell::new(Vec::new()),
            profiling: std::cell::Cell::new(false),
        })
    }

    fn kernel(&self, name: &str) -> ffi::Kernel {
        self.kernels
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, k)| *k)
            .unwrap()
    }

    /// A read/write device buffer, kept alive by the harness.
    fn alloc(&mut self, bytes: usize) -> Result<usize, String> {
        let mut err = 0i32;
        let mem = unsafe {
            (self.gpu.api.create_buffer)(
                self.gpu.context,
                ffi::CL_MEM_READ_WRITE,
                bytes,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        ffi::check("clCreateBuffer (bit-exact)", err)?;
        self.mems.push(mem);
        Ok(self.mems.len() - 1)
    }

    /// A buffer whose host-side storage is page-locked (`CL_MEM_ALLOC_HOST_PTR`),
    /// so map/unmap hands back a stable pointer instead of copying through the
    /// driver's bounce buffer on every transfer. That copy is what the probe's
    /// `[2b]` table measures as the pageable rate (8.9 GB/s up, 7.7 down).
    fn alloc_pinned(&mut self, bytes: usize) -> Result<usize, String> {
        let mut err = 0i32;
        let mem = unsafe {
            (self.gpu.api.create_buffer)(
                self.gpu.context,
                ffi::CL_MEM_READ_WRITE | ffi::CL_MEM_ALLOC_HOST_PTR,
                bytes,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        ffi::check("clCreateBuffer (pinned)", err)?;
        self.mems.push(mem);
        Ok(self.mems.len() - 1)
    }

    /// Map a pinned buffer for the host, `bytes` long. The caller copies and
    /// then calls [`DuccGpu::unmap`]; blocking map keeps this free of event
    /// bookkeeping, and a page-locked map returns immediately once the queue
    /// has drained.
    fn map(&self, idx: usize, flags: u64, bytes: usize) -> Result<*mut u8, String> {
        let mut err = 0i32;
        let p = unsafe {
            (self.gpu.api.map_buffer)(
                self.gpu.queue,
                self.mems[idx],
                1, // CL_TRUE: block until mapped
                flags,
                0,
                bytes,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut err,
            )
        };
        ffi::check("clEnqueueMapBuffer", err)?;
        if p.is_null() {
            return Err("clEnqueueMapBuffer returned null".into());
        }
        Ok(p as *mut u8)
    }

    fn unmap(&self, idx: usize, ptr: *mut u8) -> Result<(), String> {
        let rc = unsafe {
            (self.gpu.api.unmap_mem)(
                self.gpu.queue,
                self.mems[idx],
                ptr as *mut c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        ffi::check("clEnqueueUnmapMemObject", rc)
    }

    fn write<T>(&self, idx: usize, data: &[T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(data);
        let rc = unsafe {
            (self.gpu.api.write_buffer)(
                self.gpu.queue,
                self.mems[idx],
                1,
                0,
                bytes,
                data.as_ptr() as *const c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        ffi::check("clEnqueueWriteBuffer (bit-exact)", rc)
    }

    fn read<T>(&self, idx: usize, out: &mut [T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(out);
        let rc = unsafe {
            (self.gpu.api.read_buffer)(
                self.gpu.queue,
                self.mems[idx],
                1,
                0,
                bytes,
                out.as_mut_ptr() as *mut c_void,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        ffi::check("clEnqueueReadBuffer (bit-exact)", rc)
    }

    fn copy(&self, dst: usize, src: usize, bytes: usize) -> Result<(), String> {
        let rc = unsafe {
            (self.gpu.api.copy_buffer)(
                self.gpu.queue,
                self.mems[src],
                self.mems[dst],
                0,
                0,
                bytes,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        ffi::check("clEnqueueCopyBuffer (bit-exact)", rc)
    }

    fn dispatch(&self, name: &str, global: usize, args: &[Arg<'_>]) -> Result<(), String> {
        let ev = self.dispatch_inner(name, global, args, false)?;
        if !self.profiling.get() && !ev.is_null() {
            unsafe { (self.gpu.api.release_event)(ev) };
        }
        Ok(())
    }

    /// [`DuccGpu::dispatch`] that always hands the launch's event back, so the
    /// caller can wait for exactly this chunk's work (the pipelined path
    /// downloads a chunk while the next one runs).

    fn dispatch_ev(&self, name: &str, global: usize, args: &[Arg<'_>]) -> Result<ffi::Event, String> {
        self.dispatch_inner(name, global, args, true)
    }

    /// `want` forces an event even when the profiler is off; the caller then
    /// owns it (`wait_chunk` releases it). In profiling mode the event is also
    /// registered for `profile_take`, which is why `wait_chunk` does not
    /// release it there.
    fn dispatch_inner(
        &self,
        name: &str,
        global: usize,
        args: &[Arg<'_>],
        want: bool,
    ) -> Result<ffi::Event, String> {
        let k = self.kernel(name);
        for (i, a) in args.iter().enumerate() {
            let (size, ptr): (usize, *const c_void) = match a {
                Arg::Buf(m) => (std::mem::size_of::<ffi::Mem>(), m as *const ffi::Mem as *const c_void),
                Arg::Int(v) => (std::mem::size_of::<i32>(), v as *const i32 as *const c_void),
                Arg::Dbl(v) => (std::mem::size_of::<f64>(), v as *const f64 as *const c_void),
                Arg::_Phantom(_) => unreachable!(),
            };
            let rc = unsafe { (self.gpu.api.set_kernel_arg)(k, i as u32, size, ptr) };
            if rc != ffi::CL_SUCCESS {
                return Err(format!("{name}: clSetKernelArg[{i}] failed ({rc})"));
            }
        }
        let gs = [global];
        let mut ev: ffi::Event = std::ptr::null_mut();
        let profiling = self.profiling.get();
        let rc = unsafe {
            (self.gpu.api.enqueue_kernel)(
                self.gpu.queue,
                k,
                1,
                std::ptr::null(),
                gs.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                if profiling || want {
                    &mut ev as *mut ffi::Event
                } else {
                    std::ptr::null_mut()
                },
            )
        };
        ffi::check(&format!("{name}: clEnqueueNDRangeKernel"), rc)?;
        if profiling {
            // Keep it for `profile_take`; the handle stays valid until then,
            // which is what lets `wait_chunk` use it in profiling mode too.
            self.events.borrow_mut().push(ev);
        }
        Ok(ev)
    }

    /// Release a chunk's last-launch event -- unless the profiler owns it.
    fn release_chunk(&self, ev: ffi::Event) {
        if !ev.is_null() && !self.profiling.get() {
            unsafe { (self.gpu.api.release_event)(ev) };
        }
    }

    /// Sum the profiled kernel times since the last take, in milliseconds, and
    /// drop the events.
    fn profile_take(&self) -> f64 {
        let evs = std::mem::take(&mut *self.events.borrow_mut());
        let mut total = 0u64;
        for ev in evs {
            let (mut start, mut end) = (0u64, 0u64);
            unsafe {
                (self.gpu.api.get_event_info)(
                    ev,
                    ffi::CL_PROFILING_COMMAND_START,
                    8,
                    &mut start as *mut u64 as *mut c_void,
                    std::ptr::null_mut(),
                );
                (self.gpu.api.get_event_info)(
                    ev,
                    ffi::CL_PROFILING_COMMAND_END,
                    8,
                    &mut end as *mut u64 as *mut c_void,
                    std::ptr::null_mut(),
                );
                (self.gpu.api.release_event)(ev);
            }
            total += end.saturating_sub(start);
        }
        total as f64 / 1e6
    }

    fn finish(&self) -> Result<(), String> {
        ffi::check("clFinish", unsafe { (self.gpu.api.finish)(self.gpu.queue) })
    }
}

/// A `UnityRoots` table resident on the device.
struct RootsDev {
    v1: usize,
    v2: usize,
    n: i32,
    mask: i32,
    shift: i32,
}

impl RootsDev {
    fn upload(dg: &mut DuccGpu<'_>, roots: &ducc_plan::UnityRoots) -> Result<Self, String> {
        let v1 = dg.alloc(roots.v1().len() * 16)?;
        dg.write(v1, roots.v1())?;
        let v2 = dg.alloc(roots.v2().len() * 16)?;
        dg.write(v2, roots.v2())?;
        Ok(Self {
            v1,
            v2,
            n: roots.len() as i32,
            mask: roots.mask() as i32,
            shift: roots.shift() as i32,
        })
    }
}

/// Build the device-side plan tree: every pass with its twiddle table resident.
fn build_dev(
    dg: &mut DuccGpu<'_>,
    nodes: &mut Vec<NodeDev>,
    node: &ducc_plan::Node,
) -> Result<usize, String> {
    let slot = nodes.len();
    nodes.push(NodeDev {
        ip: node.ip,
        l1: node.l1,
        ido: node.ido,
        rfct: node.rfct,
        passes: Vec::new(),
        children: None,
    });
    match &node.body {
        Body::Chain(passes) => {
            let mut pd = Vec::with_capacity(passes.len());
            for p in passes {
                let radix = match p.kind {
                    Kind::P8 => 8,
                    Kind::P4 => 4,
                    Kind::P2 => 2,
                };
                // An empty table still needs a valid buffer to bind.
                let wa = dg.alloc(p.wa.len().max(1) * 16)?;
                if !p.wa.is_empty() {
                    dg.write(wa, &p.wa)?;
                }
                pd.push(PassDev {
                    radix,
                    l1: p.l1,
                    ido: p.ido,
                    wa,
                });
            }
            nodes[slot].passes = pd;
        }
        Body::Packets(a, b) => {
            let ia = build_dev(dg, nodes, a)?;
            let ib = build_dev(dg, nodes, b)?;
            nodes[slot].children = Some((ia, ib));
        }
    }
    Ok(slot)
}

/// One pass chain over `bt` contiguous `ip`-blocks, ping-ponging the way
/// `chain_run` does. Returns the buffer index holding the result.
fn dev_chain(
    dg: &DuccGpu<'_>,
    nodes: &[NodeDev],
    slot: usize,
    bt: usize,
    fwd: bool,
    src: usize,
    spare: usize,
) -> Result<usize, String> {
    let node = &nodes[slot];
    let mut cur = src;
    let mut other = spare;
    for p in &node.passes {
        // `cfftp8` with `l1 == 1` works in place and does not swap; on the
        // device that is safe because each element belongs to exactly one work
        // item, which reads all eight of its inputs before writing any.
        let in_place = p.radix == 8 && p.l1 == 1;
        let out = if in_place { cur } else { other };
        let name = match p.radix {
            8 => "fft_pass8",
            4 => "fft_pass4",
            _ => "fft_pass2",
        };
        let global = bt * (node.ip / p.radix);
        dg.dispatch(
            name,
            global,
            &[
                Arg::Buf(dg.mems[cur]),
                Arg::Buf(dg.mems[out]),
                Arg::Buf(dg.mems[p.wa]),
                Arg::Int(node.ip as i32),
                Arg::Int(p.l1 as i32),
                Arg::Int(p.ido as i32),
                Arg::Int(if fwd { 1 } else { 0 }),
                Arg::Int((p.l1 * p.ido) as i32),
            ],
        )?;
        if !in_place {
            std::mem::swap(&mut cur, &mut other);
        }
    }
    Ok(cur)
}

/// Scratch buffers a bit-exact transform runs between. The roles are fixed so
/// no kernel ever reads and writes the same element in one launch: `inp` is the
/// uploaded input, `cch` the c2r pre-pass output, `g` the packet gather, `h`
/// the chain ping-pong partner, `i` the packet scatter, `o` the output and `x`
/// the scaling destination.
struct Scratch {
    inp: usize,
    cch: usize,
    g: usize,
    h: usize,
    i: usize,
    o: usize,
    x: usize,
}

/// Run a whole plan on the device, returning the buffer index of the result.
///
/// `bt` is the number of independent `ip`-blocks laid out contiguously in the
/// buffers; a pass chain handles any `bt` in one launch, which is what makes
/// batching a field's blocks worthwhile. Two-packet plans (ducc splits any
/// `n > 10000`) block-index the same launch, so both plan shapes batch.
fn dev_node(
    dg: &DuccGpu<'_>,
    nodes: &[NodeDev],
    roots: &RootsDev,
    slot: usize,
    bt: usize,
    fwd: bool,
    src: usize,
    sc: &Scratch,
) -> Result<usize, String> {
    let node = &nodes[slot];
    let Some((ia, ib)) = node.children else {
        return dev_chain(dg, nodes, slot, bt, fwd, src, sc.h);
    };
    let (a, b) = (&nodes[ia], &nodes[ib]);
    if a.l1 != 1 || b.ido != 1 {
        return Err("unexpected packet layout".into());
    }
    // a: gather the `ido` stride-`ido` blocks, transform them, twiddle back.
    // Every packet kernel block-indexes its global id by the block length, so
    // the same launch covers all `bt` transforms.
    dg.dispatch(
        "node_gather",
        bt * a.ido * a.ip,
        &[
            Arg::Buf(dg.mems[src]),
            Arg::Buf(dg.mems[sc.g]),
            Arg::Int(a.ip as i32),
            Arg::Int(a.ido as i32),
            Arg::Int(node.ip as i32),
        ],
    )?;
    let res_a = dev_chain(dg, nodes, ia, bt * a.ido, fwd, sc.g, sc.h)?;
    dg.dispatch(
        "node_scatter",
        bt * a.ido * a.ip,
        &[
            Arg::Buf(dg.mems[res_a]),
            Arg::Buf(dg.mems[sc.i]),
            Arg::Buf(dg.mems[roots.v1]),
            Arg::Buf(dg.mems[roots.v2]),
            Arg::Int(roots.n),
            Arg::Int(roots.mask),
            Arg::Int(roots.shift),
            Arg::Int(node.rfct as i32),
            Arg::Int(a.ip as i32),
            Arg::Int(a.ido as i32),
            Arg::Int(if fwd { 1 } else { 0 }),
            Arg::Int(node.ip as i32),
        ],
    )?;
    let res_b = dev_chain(dg, nodes, ib, bt * b.l1, fwd, sc.i, sc.h)?;
    // `b` writes transposed into the dedicated output buffer; `res_b` is only
    // read here, never written, so the transpose cannot alias its source.
    dg.dispatch(
        "node_transpose",
        bt * b.l1 * b.ip,
        &[
            Arg::Buf(dg.mems[res_b]),
            Arg::Buf(dg.mems[sc.o]),
            Arg::Int(b.ip as i32),
            Arg::Int(b.l1 as i32),
            Arg::Int(node.ip as i32),
        ],
    )?;
    Ok(sc.o)
}

/// A plan resident on the device: its node tree, its roots table and the slot
/// of the root node.
struct PlanDev {
    nodes: Vec<NodeDev>,
    roots: RootsDev,
    root: usize,
    n: usize,
}

impl PlanDev {
    fn build(dg: &mut DuccGpu<'_>, plan: &C2cPlan) -> Result<Self, String> {
        let roots = RootsDev::upload(dg, plan.roots())?;
        let mut nodes = Vec::new();
        let root = build_dev(dg, &mut nodes, &plan.root)?;
        Ok(Self {
            nodes,
            roots,
            root,
            n: plan.n,
        })
    }
}

/// Scratch and input buffers, one set per transform size.
struct Bufs {
    sc: Scratch,
}

fn alloc_bufs(dg: &mut DuccGpu<'_>, n_complex: usize) -> Result<Bufs, String> {
    let bytes = n_complex * 16;
    let inp = dg.alloc(bytes)?;
    Ok(Bufs {
        sc: Scratch {
            inp,
            cch: dg.alloc(bytes)?,
            g: dg.alloc(bytes)?,
            h: dg.alloc(bytes)?,
            i: dg.alloc(bytes)?,
            o: dg.alloc(bytes)?,
            x: dg.alloc(bytes)?,
        },
    })
}

fn dev_c2c(
    dg: &DuccGpu<'_>,
    pd: &PlanDev,
    sc: &Scratch,
    fwd: bool,
    input: &[C],
) -> Result<Vec<C>, String> {
    dg.write(sc.inp, input)?;
    let res = dev_node(dg, &pd.nodes, &pd.roots, pd.root, 1, fwd, sc.inp, sc)?;
    let mut out = res;
    if !fwd {
        // `pocketfft_c::exec` applies `fct = 1/n` to the result.
        dg.dispatch(
            "c2r_scale",
            pd.n,
            &[
                Arg::Buf(dg.mems[res]),
                Arg::Buf(dg.mems[sc.x]),
                Arg::Int(pd.n as i32),
                Arg::Dbl(1.0 / pd.n as f64),
            ],
        )?;
        out = sc.x;
    }
    dg.finish()?;
    let mut v = vec![(0.0, 0.0); pd.n];
    dg.read(out, &mut v)?;
    Ok(v)
}

/// Launch-only version, for timing: no upload, no readback, no finish.
fn dev_c2c_launch(
    dg: &DuccGpu<'_>,
    pd: &PlanDev,
    sc: &Scratch,
    fwd: bool,
) -> Result<(), String> {
    let res = dev_node(dg, &pd.nodes, &pd.roots, pd.root, 1, fwd, sc.inp, sc)?;
    if !fwd {
        dg.dispatch(
            "c2r_scale",
            pd.n,
            &[
                Arg::Buf(dg.mems[res]),
                Arg::Buf(dg.mems[sc.x]),
                Arg::Int(pd.n as i32),
                Arg::Dbl(1.0 / pd.n as f64),
            ],
        )?;
    }
    Ok(())
}

/// `r2c(32768)` through the complexify, on the device. `n/2 + 1` work items:
/// the last one is the Nyquist bin, which `cfft`'s packing writes separately.
fn dev_r2c(
    dg: &DuccGpu<'_>,
    inner: &PlanDev,
    outer_roots: &RootsDev,
    sc: &Scratch,
    n: usize,
    input: &[f64],
) -> Result<Vec<C>, String> {
    let as_c: Vec<C> = (0..n / 2).map(|i| (input[2 * i], input[2 * i + 1])).collect();
    dg.write(sc.inp, &as_c)?;
    let res = dev_node(dg, &inner.nodes, &inner.roots, inner.root, 1, true, sc.inp, sc)?;
    let out = sc.o;
    dg.dispatch(
        "r2c_post",
        n / 2 + 1,
        &[
            Arg::Buf(dg.mems[res]),
            Arg::Buf(dg.mems[out]),
            Arg::Buf(dg.mems[outer_roots.v1]),
            Arg::Buf(dg.mems[outer_roots.v2]),
            Arg::Int(outer_roots.n),
            Arg::Int(outer_roots.mask),
            Arg::Int(outer_roots.shift),
            Arg::Int(n as i32),
        ],
    )?;
    dg.finish()?;
    let mut sp = vec![(0.0, 0.0); n / 2 + 1];
    dg.read(out, &mut sp)?;
    Ok(sp)
}

fn dev_r2c_launch(
    dg: &DuccGpu<'_>,
    inner: &PlanDev,
    outer_roots: &RootsDev,
    sc: &Scratch,
    n: usize,
) -> Result<(), String> {
    let res = dev_node(dg, &inner.nodes, &inner.roots, inner.root, 1, true, sc.inp, sc)?;
    dg.dispatch(
        "r2c_post",
        n / 2 + 1,
        &[
            Arg::Buf(dg.mems[res]),
            Arg::Buf(dg.mems[sc.o]),
            Arg::Buf(dg.mems[outer_roots.v1]),
            Arg::Buf(dg.mems[outer_roots.v2]),
            Arg::Int(outer_roots.n),
            Arg::Int(outer_roots.mask),
            Arg::Int(outer_roots.shift),
            Arg::Int(n as i32),
        ],
    )?;
    Ok(())
}

fn dev_c2r(
    dg: &DuccGpu<'_>,
    inner: &PlanDev,
    outer_roots: &RootsDev,
    sc: &Scratch,
    n: usize,
    spectrum: &[C],
) -> Result<Vec<f64>, String> {
    dg.write(sc.inp, spectrum)?;
    let res = dev_c2r_launch(dg, inner, outer_roots, sc, n)?;
    dg.finish()?;
    let mut out = vec![0.0f64; n];
    dg.read(res, &mut out)?;
    Ok(out)
}

fn dev_c2r_launch(
    dg: &DuccGpu<'_>,
    inner: &PlanDev,
    outer_roots: &RootsDev,
    sc: &Scratch,
    n: usize,
) -> Result<usize, String> {
    // pre: spectrum -> half-length complex input
    let cch = sc.cch;
    dg.dispatch(
        "c2r_pre",
        n / 2,
        &[
            Arg::Buf(dg.mems[sc.inp]),
            Arg::Buf(dg.mems[cch]),
            Arg::Buf(dg.mems[outer_roots.v1]),
            Arg::Buf(dg.mems[outer_roots.v2]),
            Arg::Int(outer_roots.n),
            Arg::Int(outer_roots.mask),
            Arg::Int(outer_roots.shift),
            Arg::Int(n as i32),
        ],
    )?;
    let res = dev_node(dg, &inner.nodes, &inner.roots, inner.root, 1, false, cch, sc)?;
    dg.dispatch(
        "c2r_scale",
        n / 2,
        &[
            Arg::Buf(dg.mems[res]),
            Arg::Buf(dg.mems[sc.x]),
            Arg::Int((n / 2) as i32),
            Arg::Dbl(1.0 / n as f64),
        ],
    )?;
    Ok(sc.x)
}

/// Run one inverse c2c of length `n` on the device and report how many bins
/// differ from `ffi_ducc`.
fn bitexact_c2c(dg: &mut DuccGpu<'_>, n: usize, seed: u64) -> Result<(usize, usize), String> {
    let plan = C2cPlan::new(n);
    let pd = PlanDev::build(dg, &plan)?;
    let bufs = alloc_bufs(dg, n)?;
    let input = lcg_complex(n, seed);
    let want = ffi_ducc::ifft(&input);
    let got = dev_c2c(dg, &pd, &bufs.sc, false, &to_c(&input))?;
    Ok(diff_count(&to_c(&want), &got))
}

// ---------------------------------------------------------------------------
// `--gpu`: the demod late group, batched over a field's blocks
// ---------------------------------------------------------------------------

/// The block kernel's late group, on the device.
///
/// Per block that is one forward real transform of the clamped demod series,
/// three half-spectrum filter products, and three real inverses -- the shapes
/// `gpu_bitexact_probe` pins to `ffi_ducc` bit for bit. Everything upstream of
/// it (the input spectrum, hilbert, EFM, rfhpf, the audio slices, unwrap and the
/// clamp) stays on the CPU: moving it would cost more in PCIe than the CPU work
/// it removes, which the probe's `[2b]` table measures.
///
/// Batching is the whole point: `run` issues one launch per plan pass for *all*
/// blocks, so a field costs ~28 launches instead of ~1300, and the block-kernel
/// arithmetic is unchanged.
pub(crate) struct LateGroup {
    dg: DuccGpu<'static>,
    blocklen: usize,
    nh: usize,
    w: usize,
    fwd: PlanDev,
    inv: PlanDev,
    roots: RootsDev,
    max_bt: usize,
    // Buffers, all `max_bt` blocks wide. `g`/`h`/`i`/`o` are the packet path's
    // gather, chain partner, scatter and transpose destinations -- they must be
    // distinct, because `node_scatter` writes `i` while reading the chain
    // result that may itself be `g` or `h`.
    inp: usize,
    spec: usize,
    prod: usize,
    cch: usize,
    g: usize,
    h: usize,
    i: usize,
    o: usize,
    out: usize,
    filt: usize,
    /// `LD_GPU_STATS=1`: accumulated (ns, blocks, batches) so the in-situ cost
    /// per block can be compared with the probe's warm figure. This is what
    /// tells a clock-state penalty apart from a kernel that is simply slow.
    stats: std::cell::Cell<(u64, u64, u64)>,
    /// `LD_GPU_PROF=1`: where a batch's time goes, accumulated per batch as
    /// (upload, submit, finish, download, device kernels, blocks, batches,
    /// wall) in ns. `finish - kernels` is the transfer/launch time inside the
    /// queue; the host copies are the upload/download sections. A profiled run
    /// pays a few microseconds per launch for the events, so these figures are
    /// for attribution, not for an FPS claim.
    prof: bool,
    prof_acc: std::cell::Cell<[u64; 8]>,
    /// Always-on per-field counters, shared with the decoder so `LD_TIMING` can
    /// print the device share of the iteration next to the CPU stages. They are
    /// atomics *outside* the batch mutex on purpose: reading them through the
    /// mutex would block the main thread behind an in-flight batch, which both
    /// perturbs the run and hides the wait from the measurement it is supposed
    /// to explain.
    counters: Arc<GpuCounters>,
}

/// Device time and block count since the last drain, lock-free.
#[derive(Default)]
pub(crate) struct GpuCounters {
    ns: std::sync::atomic::AtomicU64,
    blocks: std::sync::atomic::AtomicU64,
}

impl GpuCounters {
    fn add(&self, ns: u64, blocks: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.ns.fetch_add(ns, Relaxed);
        self.blocks.fetch_add(blocks, Relaxed);
    }

    /// Milliseconds of device time and blocks since the last call.
    pub(crate) fn take(&self) -> (f64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.ns.swap(0, Relaxed) as f64 / 1e6,
            self.blocks.swap(0, Relaxed),
        )
    }
}

impl LateGroup {
    /// Open the first fp64 device and build the plans for `blocklen`.
    pub(crate) fn open(blocklen: usize, filters: &[Vec<Complex64>], max_bt: usize) -> Result<Self, String> {
        let filters: Vec<&Vec<Complex64>> = filters.iter().take(3).collect();
        if filters.len() != 3 {
            return Err(format!("expected 3 FVideo_rfft filters, got {}", filters.len()));
        }
        if blocklen % 8 != 0 {
            return Err(format!("blocklen {blocklen} is not a multiple of 8"));
        }
        let gpu: &'static Gpu = Box::leak(Box::new(Gpu::new(blocklen, 1)?));
        let mut dg = DuccGpu::new(gpu)?;
        let nh = blocklen / 2;
        let w = nh + 1;
        let max_bt = max_bt.max(1);
        let fwd = PlanDev::build(&mut dg, &C2cPlan::new(nh))?;
        let inv = PlanDev::build(&mut dg, &C2cPlan::new(nh))?;
        let roots = RootsDev::upload(&mut dg, C2cPlan::new(blocklen).roots())?;
        // The two per-batch transfers go through page-locked buffers: they are
        // the only host<->device traffic on the hot path.
        let inp = dg.alloc_pinned(max_bt * nh * 16)?;
        let spec = dg.alloc(max_bt * w * 16)?;
        let prod = dg.alloc(max_bt * 3 * w * 16)?;
        // The inverse chain is row-major over all three channels at once, so
        // its five scratch buffers are `3 * max_bt` rows wide (the forward
        // transform only uses the first `max_bt`, which is harmless).
        let rows3 = 3 * max_bt;
        let cch = dg.alloc(rows3 * nh * 16)?;
        let g = dg.alloc(rows3 * nh * 16)?;
        let h = dg.alloc(rows3 * nh * 16)?;
        let i = dg.alloc(rows3 * nh * 16)?;
        let o = dg.alloc(rows3 * nh * 16)?;
        // All three channels' kept ranges, one readback for the batch.
        let out = dg.alloc_pinned(max_bt * 3 * blocklen * 4)?;
        let filt = dg.alloc(3 * w * 16)?;
        let mut fhost: Vec<C> = vec![(0.0, 0.0); 3 * w];
        for (ch, f) in filters.iter().enumerate() {
            for i in 0..w {
                fhost[ch * w + i] = (f[i].re, f[i].im);
            }
        }
        dg.write(filt, &fhost)?;
        Ok(Self {
            dg,
            blocklen,
            nh,
            w,
            fwd,
            inv,
            roots,
            max_bt,
            inp,
            spec,
            prod,
            cch,
            g,
            h,
            i,
            o,
            out,
            filt,
            stats: std::cell::Cell::new((0, 0, 0)),
            prof: std::env::var_os("LD_GPU_PROF").is_some(),
            prof_acc: std::cell::Cell::new([0; 8]),
            counters: Arc::new(GpuCounters::default()),
        })
    }

    pub(crate) fn blocklen(&self) -> usize {
        self.blocklen
    }

    /// The lock-free per-field counters (`gpu=` / `gpub=` on the `LD_TIMING`
    /// line). The batches belong to a prefetch pass, which is why the figure is
    /// "in this field's window" rather than "for this field's own samples".
    pub(crate) fn counters(&self) -> Arc<GpuCounters> {
        Arc::clone(&self.counters)
    }

    /// "`gfx1201` (Advanced Micro Devices, Inc.)", for the start-up log line.
    pub(crate) fn describe(&self) -> String {
        format!("`{}` ({})", self.dg.gpu.name, self.dg.gpu.vendor)
    }

    /// Run the late group for `blocks` (each `blocklen` f64) and return, per
    /// block, the three channels cut to `cstart .. cstart + len` as f32.
    pub(crate) fn run(
        &self,
        blocks: &[&[f64]],
        cstart: usize,
        len: usize,
    ) -> Result<Vec<[Vec<f32>; 3]>, String> {
        let bt = blocks.len();
        if bt == 0 {
            return Ok(Vec::new());
        }
        let t0 = std::time::Instant::now();
        let mut r = self.run_inner(blocks, cstart, len);
        // `LD_GPU_X2=1`: a throwaway second pass over the same batch, so the
        // device stage can be lengthened by a known amount while everything else
        // is unchanged. That is how the *end-to-end* sensitivity to this stage is
        // measured (does a ms saved here show up as a ms of wall?), which is what
        // decided the pipelining question below. The output is the first pass's,
        // so a run with this set still produces the same files.
        if std::env::var_os("LD_GPU_X2").is_some() {
            r = self.run_inner(blocks, cstart, len);
        }
        self.counters.add(t0.elapsed().as_nanos() as u64, bt as u64);
        if self.prof {
            let a = self.prof_acc.get();
            if a[6] >= 20 {
                self.prof_acc.set([0; 8]);
                let per = |v: u64| v as f64 / 1e6 / a[5] as f64;
                let total = per(a[7]);
                crate::teeprintln!(
                    "GPUPROF batches={} blocks={} ms/block: upload={:.4} submit={:.4} finish={:.4} download={:.4} total={:.4} | kernels={:.4} ({:.0}% of total, {:.3} ms/transform)",
                    a[6],
                    a[5],
                    per(a[0]),
                    per(a[1]),
                    per(a[2]),
                    per(a[3]),
                    total,
                    per(a[4]),
                    if total > 0.0 { per(a[4]) / total * 100.0 } else { 0.0 },
                    per(a[4]) / 4.0,
                );
            }
        }
        if std::env::var_os("LD_GPU_STATS").is_some() {
            let (ns, blks, batches) = self.stats.get();
            let (ns, blks, batches) = (
                ns + t0.elapsed().as_nanos() as u64,
                blks + bt as u64,
                batches + 1,
            );
            self.stats.set((ns, blks, batches));
            if batches % 200 == 0 {
                crate::teeprintln!(
                    "GPUSTATS batches={batches} blocks={blks} {:.3} ms/block {:.2} ms/field",
                    ns as f64 / 1e6 / blks as f64,
                    ns as f64 / 1e6 / blks as f64 * BLOCKS_PER_FIELD as f64
                );
                self.stats.set((0, 0, 0));
            }
        }
        r
    }

    fn run_inner(
        &self,
        blocks: &[&[f64]],
        cstart: usize,
        len: usize,
    ) -> Result<Vec<[Vec<f32>; 3]>, String> {
        let bt = blocks.len();
        // A pass can hand over more blocks than the buffers hold (a first
        // window has 80+ missing blocks); the device cost is per launch, so
        // splitting into `max_bt`-sized chunks keeps the launch count low
        // without reallocating anything.
        if bt > self.max_bt {
            let mut out = Vec::with_capacity(bt);
            for chunk in blocks.chunks(self.max_bt) {
                out.extend(self.run_inner(chunk, cstart, len)?);
            }
            return Ok(out);
        }
        let (nh, w, n) = (self.nh, self.w, self.blocklen);
        let dg = &self.dg;
        dg.profiling.set(self.prof);
        let t_batch = Instant::now();
        let t_up0 = Instant::now();

        // Host -> device: a `double2` is two consecutive f64, so the block's
        // `n` doubles *are* its `n/2` complex values. Mapping the pinned buffer
        // and copying each block straight in drops both the old per-block
        // repack into a `Vec<Complex>` (n/2 tuple constructions, 256 KB of
        // memcpy) and the driver's bounce copy.
        {
            let bytes = bt * n * 8;
            let base = dg.map(self.inp, ffi::CL_MAP_WRITE, bytes)?;
            for (b, blk) in blocks.iter().enumerate() {
                debug_assert_eq!(blk.len(), n);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        blk.as_ptr() as *const u8,
                        base.add(b * n * 8),
                        n * 8,
                    );
                }
            }
            dg.unmap(self.inp, base)?;
        }
        let t_up = t_up0.elapsed().as_nanos() as u64;
        let t_sub0 = Instant::now();

        let ev = self.launch_batch(bt, nh, w, n, cstart, len, 1.0 / n as f64)?;
        dg.release_chunk(ev);
        let t_sub = t_sub0.elapsed().as_nanos() as u64;
        let t_fin0 = Instant::now();
        dg.finish()?;
        let t_fin = t_fin0.elapsed().as_nanos() as u64;
        let t_kern = if self.prof { dg.profile_take() } else { 0.0 };
        let t_dn0 = Instant::now();
        // Read the batch's three channel ranges straight out of the mapped
        // buffer, which also removes the old whole-batch staging `Vec<f32>`.
        let mut result: Vec<[Vec<f32>; 3]> = vec![Default::default(); bt];
        {
            // Rows are `prod`'s own order -- `block * 3 + channel` -- because
            // the merged chain emits in that order.
            let base = dg.map(self.out, ffi::CL_MAP_READ, 3 * bt * len * 4)?;
            for (b, r) in result.iter_mut().enumerate() {
                for (ch, slot) in r.iter_mut().enumerate() {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            base.add((b * 3 + ch) * len * 4) as *const f32,
                            len,
                        )
                    };
                    *slot = src.to_vec();
                }
            }
            dg.unmap(self.out, base)?;
        }
        let t_dn = t_dn0.elapsed().as_nanos() as u64;
        if self.prof {
            let mut a = self.prof_acc.get();
            a[0] += t_up;
            a[1] += t_sub;
            a[2] += t_fin;
            a[3] += t_dn;
            a[4] += (t_kern * 1e6) as u64;
            a[5] += bt as u64;
            a[6] += 1;
            a[7] += t_batch.elapsed().as_nanos() as u64;
            self.prof_acc.set(a);
        }
        Ok(result)
    }

    /// The whole batch's launches: the forward transform, the three filter
    /// products, then the row-major inverse chain and the cut/emit, all with
    /// `cnt` blocks per launch (the chain runs `3 * cnt` rows). The returned
    /// event is the last launch's.
    fn launch_batch(
        &self,
        cnt: usize,
        nh: usize,
        w: usize,
        n: usize,
        cstart: usize,
        len: usize,
        fct: f64,
    ) -> Result<ffi::Event, String> {
        let dg = &self.dg;
        let res = dev_node(
            dg,
            &self.fwd.nodes,
            &self.fwd.roots,
            self.fwd.root,
            cnt,
            true,
            self.inp,
            &self.scratch(),
        )?;
        dg.dispatch(
            "r2c_post_b",
            cnt * w,
            &[
                Arg::Buf(dg.mems[res]),
                Arg::Buf(dg.mems[self.spec]),
                Arg::Buf(dg.mems[self.roots.v1]),
                Arg::Buf(dg.mems[self.roots.v2]),
                Arg::Int(self.roots.n),
                Arg::Int(self.roots.mask),
                Arg::Int(self.roots.shift),
                Arg::Int(n as i32),
            ],
        )?;
        // Three filter products per block, in one launch.
        dg.dispatch(
            "video_prod",
            cnt * 3 * w,
            &[
                Arg::Buf(dg.mems[self.spec]),
                Arg::Buf(dg.mems[self.filt]),
                Arg::Buf(dg.mems[self.prod]),
                Arg::Int(w as i32),
            ],
        )?;

        // Three real inverses per block, as *one* row-major chain: `prod` is
        // already laid out `(blk*3 + ch)*w`, so stacking the three channels
        // makes row `r` the spectrum row directly, and a single pre-pack + plan
        // + emit covers all three.
        let rows = 3 * cnt;
        dg.dispatch(
            "c2r_pre_r",
            rows * nh,
            &[
                Arg::Buf(dg.mems[self.prod]),
                Arg::Buf(dg.mems[self.cch]),
                Arg::Buf(dg.mems[self.roots.v1]),
                Arg::Buf(dg.mems[self.roots.v2]),
                Arg::Int(self.roots.n),
                Arg::Int(self.roots.mask),
                Arg::Int(self.roots.shift),
                Arg::Int(n as i32),
            ],
        )?;
        let row = dev_node(
            dg,
            &self.inv.nodes,
            &self.inv.roots,
            self.inv.root,
            rows,
            false,
            self.cch,
            &self.scratch(),
        )?;
        dg.dispatch_ev(
            "c2r_emit",
            rows * len,
            &[
                Arg::Buf(dg.mems[row]),
                Arg::Buf(dg.mems[self.out]),
                Arg::Int(nh as i32),
                Arg::Int(cstart as i32),
                Arg::Int(len as i32),
                Arg::Dbl(fct),
                Arg::Int(0),
            ],
        )
    }

    /// The `Scratch` view the plan runner needs.
    fn scratch(&self) -> Scratch {
        Scratch {
            inp: self.inp,
            cch: self.cch,
            g: self.g,
            h: self.h,
            i: self.i,
            o: self.o,
            x: self.out,
        }
    }
}

fn bits(a: C) -> (u64, u64) {
    (a.0.to_bits(), a.1.to_bits())
}

fn to_c(v: &[Complex64]) -> Vec<C> {
    v.iter().map(|c| (c.re, c.im)).collect()
}

fn to_complex(v: &[C]) -> Vec<Complex64> {
    v.iter().map(|c| Complex64::new(c.0, c.1)).collect()
}

fn diff_count(want: &[C], got: &[C]) -> (usize, usize) {
    let mut bad = 0;
    let mut first = usize::MAX;
    for (i, (a, b)) in want.iter().zip(got).enumerate() {
        if bits(*a) != bits(*b) {
            bad += 1;
            if first == usize::MAX {
                first = i;
            }
        }
    }
    (bad, if first == usize::MAX { 0 } else { first })
}

/// **Can a GPU reproduce ducc bit for bit?**
///
/// The `gpu_fft_probe` above prices a Stockham kernel, which cannot: a radix-2
/// decimation rounds differently from ducc's radix-8 passes. This probe runs
/// ducc's *own* plan (from `decode::ducc_plan`) on the device instead -- same
/// passes, same twiddles, same operation order, with `FP_CONTRACT OFF` so no
/// product is fused -- and compares the bytes against `ffi_ducc`.
///
/// Usage:
///
/// ```text
/// cargo test -p ld-decode --release gpu_bitexact_probe -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs an OpenCL f64 device; prints a table, asserts only correctness"]
fn gpu_bitexact_probe() {
    let n = 32768usize;
    let gpu = match Gpu::new(n, 1) {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: {e}");
            return;
        }
    };
    println!(
        "\n[bit-exact ducc replica]  device `{}` ({}, {} CUs)",
        gpu.name, gpu.vendor, gpu.units
    );

    let mut dg = match DuccGpu::new(&gpu) {
        Ok(d) => d,
        Err(e) => {
            println!("SKIP: {e}");
            return;
        }
    };

    let plan32 = C2cPlan::new(n);
    let plan16 = C2cPlan::new(n / 2);
    let pd32 = PlanDev::build(&mut dg, &plan32).unwrap();
    let pd16 = PlanDev::build(&mut dg, &plan16).unwrap();
    let roots_outer = RootsDev::upload(&mut dg, plan32.roots()).unwrap();
    let bufs = alloc_bufs(&mut dg, n).unwrap();
    let sc = &bufs.sc;

    // --- the data: the dynamic range the demod kernel actually feeds it ---
    let mut r = lcg_complex(n, 12345);
    let mut input = vec![Complex64::new(0.0, 0.0); n];
    for (i, v) in input.iter_mut().enumerate() {
        let a = if i % 2 == 0 { 4096.0 } else { 64.0 };
        *v = Complex64::new(r[i % r.len()].re * a, r[i % r.len()].im * a);
    }
    r.clear();

    println!("\n[1] correctness against ffi_ducc (bit compare)");
    let mut failures = Vec::new();

    // Size ladder first: a break at n=8 is a butterfly bug, one that only shows
    // at 32768 is the packet path or the twiddle tables.
    for &m in &[8usize, 16, 32, 64, 128, 512, 4096] {
        match bitexact_c2c(&mut dg, m, 99) {
            Ok((bad, first)) => println!(
                "    c2c inverse n={m:<6}: {bad} of {m} differ{}",
                if bad > 0 { format!(", first at {first}") } else { String::new() }
            ),
            Err(e) => println!("    c2c inverse n={m:<6}: ERROR {e}"),
        }
    }

    let want = ffi_ducc::ifft(&input);
    let got_c = dev_c2c(&dg, &pd32, sc, false, &to_c(&input)).unwrap();
    let got: Vec<Complex64> = got_c.iter().map(|c| Complex64::new(c.0, c.1)).collect();
    let (bad, first) = diff_count(&to_c(&want), &got_c);
    println!("    c2c inverse n=32768 : {bad} of {n} differ{} (5 passes, 9 launches)", if bad > 0 { format!(", first at {first}") } else { String::new() });
    if bad > 0 {
        failures.push(format!("c2c inverse {bad}/{n}"));
        for k in first..(first + 3).min(n) {
            println!("        [{k}] ducc {:+.17e}{:+.17e}i  gpu {:+.17e}{:+.17e}i", want[k].re, want[k].im, got[k].re, got[k].im);
        }
    }

    // Where the difference lives: the replica against ducc, then the device
    // against the replica. One of the two lines must be all zeros.
    let real_in: Vec<f64> = (0..n).map(|i| input[i].re).collect();
    let rep_sp = ducc_plan::r2c(n, &real_in);
    let (bad, first) = diff_count(&to_c(&ffi_ducc::rfft(&real_in)), &rep_sp);
    println!("    replica r2c         : {bad} of {} differ{}", n / 2 + 1, if bad > 0 { format!(", first at {first}") } else { String::new() });
    let rep_r = ducc_plan::c2r(n, &rep_sp);
    let mut bad_rep = 0;
    for (a, b) in ffi_ducc::irfft(&to_complex(&rep_sp), n).iter().zip(&rep_r) {
        if a.to_bits() != b.to_bits() {
            bad_rep += 1;
        }
    }
    println!("    replica c2r         : {bad_rep} of {n} differ");

    let want_sp = ffi_ducc::rfft(&real_in);
    let got_sp = dev_r2c(&dg, &pd16, &roots_outer, sc, n, &real_in).unwrap();
    let (bad, first) = diff_count(&to_c(&want_sp), &got_sp);
    println!("    r2c forward n=32768 : {bad} of {} differ{}", n / 2 + 1, if bad > 0 { format!(", first at {first}") } else { String::new() });
    if bad > 0 {
        failures.push(format!("r2c {bad}/{}", n / 2 + 1));
        for k in first..(first + 3).min(n / 2 + 1) {
            println!("        [{k}] ducc {:+.17e}{:+.17e}i  gpu {:+.17e}{:+.17e}i", want_sp[k].re, want_sp[k].im, got_sp[k].0, got_sp[k].1);
        }
    }

    let want_r = ffi_ducc::irfft(&want_sp, n);
    let got_r = dev_c2r(&dg, &pd16, &roots_outer, sc, n, &to_c(&want_sp)).unwrap();
    let mut bad_r = 0;
    for (a, b) in want_r.iter().zip(&got_r) {
        if a.to_bits() != b.to_bits() {
            bad_r += 1;
        }
    }
    println!("    c2r inverse n=32768 : {bad_r} of {n} differ");
    if bad_r > 0 {
        failures.push(format!("c2r {bad_r}/{n}"));
    }

    // --- timing: launches only, the way a decode would use it ---
    // 200, not 20: a 20-round sample is ~0.6 ms of work, which measures the
    // card's idle clock state and reads up to 3x slow even after a warm-up
    // burst. At 200 the reading is stable to a few percent run to run.
    let rounds: usize = std::env::var("LD_GPU_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let time_dev = |label: &str, f: &dyn Fn() -> Result<(), String>| -> (f64, f64) {
        f().unwrap();
        dg.finish().unwrap();
        dg.profiling.set(true);
        let t = Instant::now();
        for _ in 0..rounds {
            f().unwrap();
        }
        dg.finish().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
        // Profiled kernel time: the part a batch would still have to pay. The
        // difference is CPU-side submit, which batching amortizes instead.
        let gpu_ms = dg.profile_take() / rounds as f64;
        dg.profiling.set(false);
        println!(
            "        {label}: {ms:.3} ms/transform wall, {gpu_ms:.3} ms in kernels, {:.3} ms submit",
            ms - gpu_ms
        );
        (ms, gpu_ms)
    };
    // Clock ramp: an idle RDNA card sits at its lowest clock state, and a
    // short sample is a fraction of a millisecond of work -- at 20 rounds the
    // same kernel reads *three times* slower than it does warm. Warm all three
    // shapes before timing anything.
    for _ in 0..2000 {
        dev_c2c_launch(&dg, &pd32, sc, false).unwrap();
        dev_r2c_launch(&dg, &pd16, &roots_outer, sc, n).unwrap();
        dev_c2r_launch(&dg, &pd16, &roots_outer, sc, n).unwrap();
    }
    dg.finish().unwrap();

    println!("\n[2] device cost per transform, {rounds} rounds (warm)");
    let (d_c2c, g_c2c) = time_dev("c2c inverse", &|| dev_c2c_launch(&dg, &pd32, sc, false));
    let (d_r2c, g_r2c) = time_dev("r2c forward", &|| dev_r2c_launch(&dg, &pd16, &roots_outer, sc, n));
    let (d_c2r, g_c2r) = time_dev("c2r inverse", &|| {
        dev_c2r_launch(&dg, &pd16, &roots_outer, sc, n).map(|_| ())
    });

    // --- [2b] the transfer bill, which is what a real `--gpu` pays and this
    // probe does not. One block: upload the raw samples (i16 -> f64, 256 KB),
    // download the block spectrum (512 KB) plus the five inverse results
    // (hilbert 512 KB, EFM 512 KB, rfhpf 256 KB, three video channels 256 KB
    // each) = 2.5 MB. Over 22 blocks that is ~62 MB per field.
    {
        let up_bytes = 256 * 1024usize;
        let down_bytes = 2560 * 1024usize;
        let up_buf = dg.alloc(up_bytes).unwrap();
        let down_buf = dg.alloc(down_bytes).unwrap();
        let src = vec![0u8; up_bytes];
        let mut dst = vec![0u8; down_bytes];
        let reps = 50;
        // warm
        for _ in 0..10 {
            dg.write(up_buf, &src).unwrap();
            dg.read(down_buf, &mut dst).unwrap();
        }
        dg.finish().unwrap();
        let t = Instant::now();
        for _ in 0..reps {
            dg.write(up_buf, &src).unwrap();
        }
        dg.finish().unwrap();
        let up_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            dg.read(down_buf, &mut dst).unwrap();
        }
        dg.finish().unwrap();
        let down_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let per_block = up_ms + down_ms;
        let per_field = per_block * BLOCKS_PER_FIELD as f64;
        println!("\n[2b] transfer bill (measured, blocking clEnqueueWrite/ReadBuffer)");
        println!(
            "        upload 256 KB: {up_ms:.3} ms ({:.1} GB/s)   download 2.5 MB: {down_ms:.3} ms ({:.1} GB/s)",
            up_bytes as f64 / up_ms / 1e6,
            down_bytes as f64 / down_ms / 1e6
        );
        println!(
            "        per block {per_block:.3} ms -> per field ({BLOCKS_PER_FIELD} blocks) {per_field:.2} ms,\n        \
             which is {} the 8.69 ms pool the GPU would replace -- before any kernel time.",
            if per_field < 8.69 { "under" } else { "OVER" }
        );
        let _ = (up_buf, down_buf);

        // The same bill with page-locked staging (`CL_MEM_ALLOC_HOST_PTR` +
        // map/unmap), which is what `LateGroup` uses. The pageable path above
        // copies through a driver bounce buffer on every transfer; this one
        // does not. It is the single change that turned `--gpu` from -6% into
        // +12% end to end, so it is worth being able to re-measure.
        let up = dg.alloc_pinned(up_bytes).unwrap();
        let down = dg.alloc_pinned(down_bytes).unwrap();
        for _ in 0..10 {
            let p = dg.map(up, ffi::CL_MAP_WRITE, up_bytes).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), p, up_bytes) };
            dg.unmap(up, p).unwrap();
        }
        dg.finish().unwrap();
        let t = Instant::now();
        for _ in 0..reps {
            let p = dg.map(up, ffi::CL_MAP_WRITE, up_bytes).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), p, up_bytes) };
            dg.unmap(up, p).unwrap();
        }
        dg.finish().unwrap();
        let pin_up_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            let p = dg.map(down, ffi::CL_MAP_READ, down_bytes).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(p, dst.as_mut_ptr(), down_bytes) };
            dg.unmap(down, p).unwrap();
        }
        dg.finish().unwrap();
        let pin_down_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let pin_per_block = pin_up_ms + pin_down_ms;
        println!(
            "        pinned:  up {pin_up_ms:.3} ms ({:.1} GB/s)  down {pin_down_ms:.3} ms ({:.1} GB/s)  = {:.3} ms/block, {:.2} ms/field ({:.0}% of the pageable bill)",
            up_bytes as f64 / pin_up_ms / 1e6,
            down_bytes as f64 / pin_down_ms / 1e6,
            pin_per_block,
            pin_per_block * BLOCKS_PER_FIELD as f64,
            pin_per_block / per_block * 100.0
        );
    }

    // --- the CPU side, same shapes, through the pool's own entry points ---
    let mut four = vec![Complex64::new(0.0, 0.0); 4 * n];
    let mut fwd: Vec<Complex64> = Vec::with_capacity(n);
    let mut cpu = |rounds: usize| -> (f64, f64, f64) {
        let t = Instant::now();
        for _ in 0..rounds {
            ffi_ducc::ifft(&input);
        }
        let c2c_ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
        let t = Instant::now();
        for _ in 0..rounds {
            ffi_ducc::rfft(&real_in);
        }
        let r2c_ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
        let t = Instant::now();
        for _ in 0..rounds {
            ffi_ducc::irfft(&want_sp, n);
        }
        let c2r_ms = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
        let _ = (&mut four, &mut fwd);
        (c2c_ms, r2c_ms, c2r_ms)
    };
    let (c_c2c, c_r2c, c_c2r) = cpu(rounds);
    println!("\n[3] CPU (ffi_ducc, one core) per transform");
    println!("        c2c inverse: {c_c2c:.3} ms   r2c: {c_r2c:.3} ms   c2r: {c_c2r:.3} ms");

    // --- the per-field model: 22 blocks x (3 c2c + 1 r2c + 3 c2r) ---
    // hilbert, rfhpf and the EFM row are full complex inverses; the block
    // spectrum is one forward r2c; demod_05/demod_burst/EFM-plus-video come
    // from the three real inverses.
    let per_field_dev = BLOCKS_PER_FIELD as f64 * (3.0 * d_c2c + d_r2c + 3.0 * d_c2r);
    let per_field_gpu = BLOCKS_PER_FIELD as f64 * (3.0 * g_c2c + g_r2c + 3.0 * g_c2r);
    let per_field_cpu = BLOCKS_PER_FIELD as f64 * (3.0 * c_c2c + c_r2c + 3.0 * c_c2r);
    let launches = BLOCKS_PER_FIELD * (3 * 9 + 7 + 3 * 8);
    println!("\n[4] per field (22 blocks x 7 transforms)");
    println!("        device: {per_field_dev:.2} ms over ~{launches} launches ({per_field_gpu:.2} ms of it kernels)");
    println!("        CPU   : {per_field_cpu:.2} ms (single core; the pool spreads it over {POOL_THREADS:.0} threads = {:.2} ms wall)", per_field_cpu / POOL_THREADS);
    println!(
        "        measured pool FFT layer: {:.2} ms/field -> device is {:.2}x it",
        DEMOD_WALL_MS * 0.756,
        per_field_dev / (DEMOD_WALL_MS * 0.756)
    );

    let verdict_exact = failures.is_empty();
    println!("\n[5] verdict");
    if verdict_exact {
        println!("    The device reproduces ducc bit for bit on all three shapes: the parity objection");
        println!("    to `--gpu` does not apply to this kernel. What is left is cost, not correctness.");
    } else {
        println!("    NOT bit-exact: {}", failures.join(", "));
    }
    println!(
        "    Whole-plan replication works because ducc's 1-D entry point is scalar: there is no\n    \
         cross-lane lane-width shortcut to reproduce, only its pass order, its twiddle table and\n    \
         FP_CONTRACT OFF (which the AMD compiler honours — without it every bin moves)."
    );
    println!(
        "    Device FFT layer is {:.2} ms/field against the pool's {:.2} ms; the exposed win is\n    \
         capped by the serial tail either way ({:.2} ms of a {:.2} ms iteration).",
        per_field_dev,
        DEMOD_WALL_MS * 0.756,
        TAIL_MS,
        ITER_NOW_MS
    );
    println!(
        "    {launches} launches/field cost {:.2} ms of submit and {per_field_gpu:.2} ms of kernels.\n    \
         A batch of `bt` blocks per launch divides the launches by `bt` and leaves the kernel\n    \
         time where it is, so the floor is the kernel figure -- the only number that decides\n    \
         whether `--gpu` can beat the {:.2} ms pool.\n",
        per_field_dev - per_field_gpu,
        DEMOD_WALL_MS * 0.756
    );

    assert!(
        verdict_exact,
        "device replica is not bit-exact: {}",
        failures.join(", ")
    );
}


/// **The `--gpu` gate: does the batched device late group equal the CPU's bytes?**
///
/// `gpu_bitexact_probe` pins the three transform shapes one at a time. This runs
/// the *whole* production late group -- several blocks per launch, the filter
/// products, the three-channel emit, one readback -- and compares it against the
/// CPU path the decoder falls back to, for every sample of every channel.
///
/// It compares the **f32** channels, which have a ~2^-29 blind spot per sample:
/// a 1-ulp f64 error only moves the f32 when the value lands within a hair of a
/// float midpoint, so this test is necessary but not sufficient. The f64 gate
/// (`gpu_late_group_f64_gate`) is what pins the pipeline's arithmetic, and its
/// absence is how the `r2c_post_b` mirror-pair race survived 0-of-285408 here.
///
/// ```text
/// cargo test -p ld-decode --release gpu_late_group_matches_cpu -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs an OpenCL f64 device; skips cleanly when there is none"]
fn gpu_late_group_matches_cpu() {
    use crate::decode::demodblock;

    let spec = DecoderSpec::new(&DecodeRequest::default()).expect("spec");
    let levels = CalibLevels::defaults(spec.sys);
    let dspec = demodblock::DemodSpecRef::with_plans(
        spec.freq,
        spec.freq_half,
        spec.freq_hz,
        spec.blocklen,
        &spec.filters,
        &levels,
    );
    let lg = match LateGroup::open(spec.blocklen, &spec.filters.fvideo_rfft, 8) {
        Ok(l) => l,
        Err(e) => {
            println!("SKIP: {e}");
            return;
        }
    };
    println!("\n[batch gate] device {} , {} blocks per launch", lg.describe(), 8);

    let mtf_pow = compute_mtf_pow(&spec.filters.mtf, 1.0);
    // Three blocks of loud, well-conditioned RF: the demod arithmetic does not
    // depend on the values, and a batch of 3 exercises the block indexing
    // without landing on a single-block edge case.
    let blocks: Vec<Vec<f32>> = (0..3u64)
        .map(|b| {
            (0..spec.blocklen)
                .map(|i| {
                    let a = (i as f32) * 0.0007 + b as f32 * 1.7;
                    (12000.0 + 6000.0 * a.sin()) + ((i as f32) * 0.13).cos() * 400.0
                })
                .collect()
        })
        .collect();

    // CPU: the production head/late split, one block at a time.
    let mut heads = Vec::new();
    let mut want = Vec::new();
    for (b, data) in blocks.iter().enumerate() {
        let (head, dump) =
            demodblock::demod_head_only(data, 1.0, &dspec, true, Some(&mtf_pow), 0, b as u64);
        want.push(demodblock::demod_late_cpu(&head, &dspec, true, dump));
        heads.push(head);
    }

    // Device: the same heads, the late group for all three in one go.
    let (start, end) = demodblock::kept_range(spec.blocklen, &dspec, true);
    let series: Vec<&[f64]> = heads.iter().map(|h| h.demod_clamped.as_slice()).collect();
    let got = lg
        .run(&series, start, end - start)
        .expect("device late group");

    assert_eq!(got.len(), want.len(), "block count changed");
    let mut bad = 0usize;
    let mut first: Option<(usize, usize, usize)> = None;
    for (b, (g, w)) in got.iter().zip(&want).enumerate() {
        for ch in 0..3 {
            assert_eq!(g[ch].len(), w[ch].len(), "block {b} channel {ch} length");
            for (i, (a, c)) in g[ch].iter().zip(&w[ch]).enumerate() {
                if a.to_bits() != c.to_bits() {
                    bad += 1;
                    if first.is_none() {
                        first = Some((b, ch, i));
                    }
                }
            }
        }
    }
    let total = want.iter().map(|v| v.iter().map(|c| c.len()).sum::<usize>()).sum::<usize>();
    println!("[batch gate] {bad} of {total} samples differ from the CPU late group");
    if let Some((b, ch, i)) = first {
        println!(
            "    first at block {b} channel {ch} sample {i}: cpu {:+.17e} gpu {:+.17e}",
            want[b][ch][i], got[b][ch][i]
        );
    }
    assert_eq!(bad, 0, "device late group is not bit-exact ({bad} of {total})");
}

/// **Is the batched late group bit-exact in f64?**
///
/// `gpu_late_group_matches_cpu` compares the three cut **f32** channels, and an
/// f32 compare cannot see a 1-ulp f64 error: the f64 value has to land within
/// 2^-29 of a float midpoint to move the f32 at all, so a systematic f64
/// last-bit difference passes that gate 0-of-285408 and then shows up in a
/// full decode as a handful of 1-LSB pixels. This gate compares the three steps
/// the f32 test is blind to, in f64:
///
/// 1. the *batched* half-spectrum pack (`r2c_post_b`, not the probe's
///    mirror-skipping `r2c_post`) against `ffi_ducc::rfft`;
/// 2. the filter product against `np_cmul`;
/// 3. the whole inverse path (`c2r_pre_r` + the plan chain + `1/n`) against
///    `ducc_plan::c2r`, which is itself gated to `ffi_ducc::irfft`.
///
/// ```text
/// cargo test -p ld-decode --release gpu_late_group_f64 -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs an OpenCL f64 device; skips cleanly when there is none"]
fn gpu_late_group_f64_gate() {
    let spec = DecoderSpec::new(&DecodeRequest::default()).expect("spec");
    let lg = match LateGroup::open(spec.blocklen, &spec.filters.fvideo_rfft, 8) {
        Ok(l) => l,
        Err(e) => {
            println!("SKIP: {e}");
            return;
        }
    };
    let (n, nh, w) = (lg.blocklen, lg.nh, lg.w);
    let bt = 6usize;
    println!("\n[f64 gate] device {}", lg.describe());

    // Demod-shaped data: the clamped demod series is a large positive offset
    // plus a few kHz of structure, which is what sets the dynamic range of the
    // spectrum and of the products' rounding.
    let blocks: Vec<Vec<f64>> = (0..bt)
        .map(|b| {
            (0..n)
                .map(|i| {
                    let a = i as f64 * 0.0007 + b as f64 * 1.7;
                    30000.0 + 8000.0 * a.sin() + ((i as f64) * 0.13).cos() * 400.0
                })
                .collect()
        })
        .collect();

    // Upload, exactly as `run_inner` does.
    {
        let bytes = bt * n * 8;
        let base = lg.dg.map(lg.inp, ffi::CL_MAP_WRITE, bytes).expect("map in");
        for (b, blk) in blocks.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    blk.as_ptr() as *const u8,
                    base.add(b * n * 8),
                    n * 8,
                );
            }
        }
        lg.dg.unmap(lg.inp, base).expect("unmap in");
    }

    // --- 1. batched forward r2c, f64 ---
    let dg = &lg.dg;
    let res = dev_node(
        dg,
        &lg.fwd.nodes,
        &lg.fwd.roots,
        lg.fwd.root,
        bt,
        true,
        lg.inp,
        &lg.scratch(),
    )
    .expect("fwd chain");
    dg.dispatch(
        "r2c_post_b",
        bt * w,
        &[
            Arg::Buf(dg.mems[res]),
            Arg::Buf(dg.mems[lg.spec]),
            Arg::Buf(dg.mems[lg.roots.v1]),
            Arg::Buf(dg.mems[lg.roots.v2]),
            Arg::Int(lg.roots.n),
            Arg::Int(lg.roots.mask),
            Arg::Int(lg.roots.shift),
            Arg::Int(n as i32),
        ],
    )
    .expect("r2c_post_b");
    dg.finish().expect("finish r2c");
    let mut sp = vec![(0.0f64, 0.0f64); bt * w];
    dg.read(lg.spec, &mut sp).expect("read spec");

    let mut bad_sp = 0usize;
    let mut first_sp = None;
    for b in 0..bt {
        let want = ffi_ducc::rfft(&blocks[b]);
        for i in 0..w {
            let got = sp[b * w + i];
            let wv = (want[i].re, want[i].im);
            if bits(got) != bits(wv) {
                bad_sp += 1;
                if first_sp.is_none() {
                    first_sp = Some((b, i, wv, got));
                }
            }
        }
    }
    println!(
        "[1] batched r2c: {bad_sp} of {} bins differ from ffi_ducc::rfft",
        bt * w
    );
    if let Some((b, i, wv, got)) = first_sp {
        println!("    first block {b} bin {i}: ducc {:+.17e}{:+.17e}i  gpu {:+.17e}{:+.17e}i", wv.0, wv.1, got.0, got.1);
    }

    // --- 2. filter product, f64 ---
    dg.dispatch(
        "video_prod",
        bt * 3 * w,
        &[
            Arg::Buf(dg.mems[lg.spec]),
            Arg::Buf(dg.mems[lg.filt]),
            Arg::Buf(dg.mems[lg.prod]),
            Arg::Int(w as i32),
        ],
    )
    .expect("video_prod");
    dg.finish().expect("finish prod");
    let mut prod = vec![(0.0f64, 0.0f64); bt * 3 * w];
    dg.read(lg.prod, &mut prod).expect("read prod");
    let mut bad_prod = 0usize;
    let mut first_prod = None;
    for b in 0..bt {
        for ch in 0..3 {
            for i in 0..w {
                let a = sp[b * w + i];
                let f = spec.filters.fvideo_rfft[ch][i];
                let want = crate::spec::np_cmul(
                    Complex64::new(a.0, a.1),
                    Complex64::new(f.re, f.im),
                );
                let got = prod[(b * 3 + ch) * w + i];
                if bits(got) != bits((want.re, want.im)) {
                    bad_prod += 1;
                    if first_prod.is_none() {
                        first_prod = Some((b, ch, i));
                    }
                }
            }
        }
    }
    println!("[2] product:     {bad_prod} of {} bins differ from np_cmul", bt * 3 * w);
    if let Some((b, ch, i)) = first_prod {
        println!("    first block {b} channel {ch} bin {i}");
    }

    // --- 3. the whole inverse path, f64 ---
    dg.dispatch(
        "c2r_pre_r",
        3 * bt * nh,
        &[
            Arg::Buf(dg.mems[lg.prod]),
            Arg::Buf(dg.mems[lg.cch]),
            Arg::Buf(dg.mems[lg.roots.v1]),
            Arg::Buf(dg.mems[lg.roots.v2]),
            Arg::Int(lg.roots.n),
            Arg::Int(lg.roots.mask),
            Arg::Int(lg.roots.shift),
            Arg::Int(n as i32),
        ],
    )
    .expect("c2r_pre_r");
    let row = dev_node(
        dg,
        &lg.inv.nodes,
        &lg.inv.roots,
        lg.inv.root,
        3 * bt,
        false,
        lg.cch,
        &lg.scratch(),
    )
    .expect("inv chain");
    dg.finish().expect("finish inv");
    let mut inv = vec![(0.0f64, 0.0f64); 3 * bt * nh];
    dg.read(row, &mut inv).expect("read inv");
    let fct = 1.0 / n as f64;
    let mut bad_inv = 0usize;
    let mut first_inv = None;
    for b in 0..bt {
        for ch in 0..3 {
            let spectrum: Vec<Complex64> = (0..w)
                .map(|i| {
                    let v = prod[(b * 3 + ch) * w + i];
                    Complex64::new(v.0, v.1)
                })
                .collect();
            let want = ducc_plan::c2r(n, &to_c(&spectrum));
            for j in 0..n {
                let c = inv[(b * 3 + ch) * nh + (j >> 1)];
                let got = if j & 1 == 1 { c.1 } else { c.0 } * fct;
                if got.to_bits() != want[j].to_bits() {
                    bad_inv += 1;
                    if first_inv.is_none() {
                        first_inv = Some((b, ch, j, want[j], got));
                    }
                }
            }
        }
    }
    println!("[3] inverse:     {bad_inv} of {} samples differ from ducc_plan::c2r", bt * 3 * n);
    if let Some((b, ch, j, wv, got)) = first_inv {
        println!("    first block {b} channel {ch} sample {j}: cpu {:+.17e}  gpu {:+.17e}", wv, got);
    }

    assert_eq!(
        (bad_sp, bad_prod, bad_inv),
        (0, 0, 0),
        "batched late group is not f64-bit-exact"
    );
}
