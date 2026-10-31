//! A thin OpenCL layer, shared by the GPU demodulator and the CLI's stack.
//!
//! The OpenCL library is loaded at run time (`OpenCL.dll`, shipped with the
//! GPU driver on Windows; `libOpenCL.so.1` elsewhere), so building needs no
//! SDK and a machine without it only loses the GPU paths. One device and
//! context ([`Gpu::get`]) serve every user in the process; each user owns its
//! queue, so several decoders can submit at once.

use std::ffi::{c_char, c_void, CString};
use std::ptr::{null, null_mut};
use std::sync::{Arc, OnceLock};

use anyhow::{bail, Context as _, Result};
use libloading::Library;

type ClInt = i32;
type ClUint = u32;
type Handle = *mut c_void;

const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_MEM_READ_WRITE: u64 = 1 << 0;
const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
const CL_MAP_READ: u64 = 1 << 0;
const CL_MAP_WRITE: u64 = 1 << 1;
const CL_TRUE: u32 = 1;
const CL_FALSE: u32 = 0;

/// The OpenCL 1.2 entry points used here.
struct Api {
    _lib: Library,
    get_platform_ids: unsafe extern "system" fn(ClUint, *mut Handle, *mut ClUint) -> ClInt,
    get_device_ids: unsafe extern "system" fn(Handle, u64, ClUint, *mut Handle, *mut ClUint) -> ClInt,
    get_device_info: unsafe extern "system" fn(Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_context: unsafe extern "system" fn(*const isize, ClUint, *const Handle, *const c_void, *mut c_void, *mut ClInt) -> Handle,
    create_command_queue: unsafe extern "system" fn(Handle, Handle, u64, *mut ClInt) -> Handle,
    create_program_with_source: unsafe extern "system" fn(Handle, ClUint, *const *const c_char, *const usize, *mut ClInt) -> Handle,
    build_program: unsafe extern "system" fn(Handle, ClUint, *const Handle, *const c_char, *const c_void, *mut c_void) -> ClInt,
    get_program_build_info: unsafe extern "system" fn(Handle, Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_kernel: unsafe extern "system" fn(Handle, *const c_char, *mut ClInt) -> Handle,
    create_buffer: unsafe extern "system" fn(Handle, u64, usize, *mut c_void, *mut ClInt) -> Handle,
    release_mem_object: unsafe extern "system" fn(Handle) -> ClInt,
    release_kernel: unsafe extern "system" fn(Handle) -> ClInt,
    release_command_queue: unsafe extern "system" fn(Handle) -> ClInt,
    set_kernel_arg: unsafe extern "system" fn(Handle, ClUint, usize, *const c_void) -> ClInt,
    enqueue_write_buffer: unsafe extern "system" fn(Handle, Handle, u32, usize, usize, *const c_void, ClUint, *const Handle, *mut Handle) -> ClInt,
    enqueue_read_buffer: unsafe extern "system" fn(Handle, Handle, u32, usize, usize, *mut c_void, ClUint, *const Handle, *mut Handle) -> ClInt,
    enqueue_nd_range_kernel: unsafe extern "system" fn(Handle, Handle, ClUint, *const usize, *const usize, *const usize, ClUint, *const Handle, *mut Handle) -> ClInt,
    finish: unsafe extern "system" fn(Handle) -> ClInt,
    enqueue_map_buffer:
        unsafe extern "system" fn(Handle, Handle, u32, u64, usize, usize, ClUint, *const Handle, *mut Handle, *mut ClInt) -> *mut c_void,
    enqueue_unmap_mem_object: unsafe extern "system" fn(Handle, Handle, *mut c_void, ClUint, *const Handle, *mut Handle) -> ClInt,
}

impl Api {
    fn load() -> Result<Self> {
        let names: &[&str] = if cfg!(windows) {
            &["OpenCL.dll"]
        } else if cfg!(target_os = "macos") {
            &["/System/Library/Frameworks/OpenCL.framework/OpenCL"]
        } else {
            &["libOpenCL.so.1", "libOpenCL.so"]
        };
        let lib = names
            .iter()
            .find_map(|n| unsafe { Library::new(n) }.ok())
            .context("no OpenCL runtime found (it comes with the GPU driver)")?;
        // Copy the entry points out before `lib` moves into the struct.
        macro_rules! sym {
            ($name:literal) => {
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                    .with_context(|| format!("OpenCL runtime lacks {}", $name))?
            };
        }
        let get_platform_ids = sym!("clGetPlatformIDs");
        let get_device_ids = sym!("clGetDeviceIDs");
        let get_device_info = sym!("clGetDeviceInfo");
        let create_context = sym!("clCreateContext");
        let create_command_queue = sym!("clCreateCommandQueue");
        let create_program_with_source = sym!("clCreateProgramWithSource");
        let build_program = sym!("clBuildProgram");
        let get_program_build_info = sym!("clGetProgramBuildInfo");
        let create_kernel = sym!("clCreateKernel");
        let create_buffer = sym!("clCreateBuffer");
        let release_mem_object = sym!("clReleaseMemObject");
        let release_kernel = sym!("clReleaseKernel");
        let release_command_queue = sym!("clReleaseCommandQueue");
        let set_kernel_arg = sym!("clSetKernelArg");
        let enqueue_write_buffer = sym!("clEnqueueWriteBuffer");
        let enqueue_read_buffer = sym!("clEnqueueReadBuffer");
        let enqueue_nd_range_kernel = sym!("clEnqueueNDRangeKernel");
        let finish = sym!("clFinish");
        let enqueue_map_buffer = sym!("clEnqueueMapBuffer");
        let enqueue_unmap_mem_object = sym!("clEnqueueUnmapMemObject");
        Ok(Self {
            _lib: lib,
            get_platform_ids,
            get_device_ids,
            get_device_info,
            create_context,
            create_command_queue,
            create_program_with_source,
            build_program,
            get_program_build_info,
            create_kernel,
            create_buffer,
            release_mem_object,
            release_kernel,
            release_command_queue,
            set_kernel_arg,
            enqueue_write_buffer,
            enqueue_read_buffer,
            enqueue_nd_range_kernel,
            finish,
            enqueue_map_buffer,
            enqueue_unmap_mem_object,
        })
    }
}

fn check(code: ClInt, what: &str) -> Result<()> {
    if code != 0 {
        bail!("OpenCL {what} failed (error {code})");
    }
    Ok(())
}

/// The process's GPU: device and context.
pub struct Gpu {
    api: Api,
    device: Handle,
    context: Handle,
    pub name: String,
}

// OpenCL devices and contexts are thread-safe; queues and kernels are kept
// per user (see `Queue`, `Kernel`).
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    /// The shared GPU (the first OpenCL GPU device, or `LD_OPENCL_DEVICE`).
    pub fn get() -> Result<Arc<Gpu>> {
        static GPU: OnceLock<std::result::Result<Arc<Gpu>, String>> = OnceLock::new();
        GPU.get_or_init(|| Gpu::open().map(Arc::new).map_err(|e| format!("{e:#}")))
            .clone()
            .map_err(anyhow::Error::msg)
    }

    fn open() -> Result<Self> {
        let api = Api::load()?;
        unsafe {
            let mut np = 0;
            check((api.get_platform_ids)(0, null_mut(), &mut np), "clGetPlatformIDs")?;
            let mut platforms = vec![null_mut(); np as usize];
            check((api.get_platform_ids)(np, platforms.as_mut_ptr(), null_mut()), "clGetPlatformIDs")?;
            let mut devices = Vec::new();
            for &p in &platforms {
                let mut nd = 0;
                if (api.get_device_ids)(p, CL_DEVICE_TYPE_GPU, 0, null_mut(), &mut nd) != 0 || nd == 0 {
                    continue;
                }
                let mut d = vec![null_mut(); nd as usize];
                check((api.get_device_ids)(p, CL_DEVICE_TYPE_GPU, nd, d.as_mut_ptr(), null_mut()), "clGetDeviceIDs")?;
                devices.extend(d);
            }
            let pick = std::env::var("LD_OPENCL_DEVICE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
            let Some(&device) = devices.get(pick) else {
                bail!("no OpenCL GPU device (found {})", devices.len());
            };
            let mut err = 0;
            let context = (api.create_context)(null(), 1, &device, null(), null_mut(), &mut err);
            check(err, "clCreateContext")?;
            let mut gpu = Gpu { api, device, context, name: String::new() };
            gpu.name = gpu.info(CL_DEVICE_NAME)?;
            if !gpu.info(CL_DEVICE_EXTENSIONS)?.contains("cl_khr_fp64") {
                bail!("OpenCL device {} has no double precision (cl_khr_fp64)", gpu.name);
            }
            tracing::info!("OpenCL device {}", gpu.name);
            Ok(gpu)
        }
    }

    fn info(&self, what: u32) -> Result<String> {
        unsafe {
            let mut size = 0;
            check((self.api.get_device_info)(self.device, what, 0, null_mut(), &mut size), "clGetDeviceInfo")?;
            let mut buf = vec![0u8; size];
            check(
                (self.api.get_device_info)(self.device, what, size, buf.as_mut_ptr().cast(), null_mut()),
                "clGetDeviceInfo",
            )?;
            Ok(String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string())
        }
    }

    /// Build `source` with `options` and create its kernels `names`.
    pub fn kernels(self: &Arc<Self>, source: &str, options: &str, names: &[&str]) -> Result<Vec<Kernel>> {
        unsafe {
            let src = CString::new(source)?;
            let ptr = src.as_ptr();
            let mut err = 0;
            let program = (self.api.create_program_with_source)(self.context, 1, &ptr, null(), &mut err);
            check(err, "clCreateProgramWithSource")?;
            let opts = CString::new(options)?;
            if (self.api.build_program)(program, 1, &self.device, opts.as_ptr(), null(), null_mut()) != 0 {
                let mut size = 0;
                (self.api.get_program_build_info)(program, self.device, CL_PROGRAM_BUILD_LOG, 0, null_mut(), &mut size);
                let mut log = vec![0u8; size];
                (self.api.get_program_build_info)(
                    program,
                    self.device,
                    CL_PROGRAM_BUILD_LOG,
                    size,
                    log.as_mut_ptr().cast(),
                    null_mut(),
                );
                bail!("OpenCL kernel build failed:\n{}", String::from_utf8_lossy(&log));
            }
            names
                .iter()
                .map(|n| {
                    let name = CString::new(*n)?;
                    let mut err = 0;
                    let k = (self.api.create_kernel)(program, name.as_ptr(), &mut err);
                    check(err, "clCreateKernel")?;
                    Ok(Kernel { gpu: Arc::clone(self), handle: k })
                })
                .collect()
        }
    }

    pub fn queue(self: &Arc<Self>) -> Result<Queue> {
        let mut err = 0;
        let q = unsafe { (self.api.create_command_queue)(self.context, self.device, 0, &mut err) };
        check(err, "clCreateCommandQueue")?;
        Ok(Queue { gpu: Arc::clone(self), handle: q })
    }

    /// Page-locked host memory for transfers (see [`Pinned`]).
    pub fn pinned(self: &Arc<Self>, bytes: usize) -> Result<Pinned> {
        let mut p = Pinned { q: self.queue()?, mem: null_mut(), ptr: null_mut(), cap: 0 };
        p.ensure(bytes)?;
        Ok(p)
    }

    /// A device buffer of at least `bytes` (grown on demand by [`Buffer::ensure`]).
    pub fn buffer(self: &Arc<Self>, bytes: usize) -> Result<Buffer> {
        let mut b = Buffer { gpu: Arc::clone(self), mem: null_mut(), cap: 0 };
        b.ensure(bytes)?;
        Ok(b)
    }
}

pub struct Kernel {
    gpu: Arc<Gpu>,
    handle: Handle,
}

unsafe impl Send for Kernel {}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { (self.gpu.api.release_kernel)(self.handle) };
    }
}

/// A kernel argument: a buffer or a plain value.
pub enum Arg<'a> {
    Buf(&'a Buffer),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl Kernel {
    pub fn args(&self, args: &[Arg]) -> Result<()> {
        for (i, a) in args.iter().enumerate() {
            let (size, ptr): (usize, *const c_void) = match a {
                Arg::Buf(b) => (std::mem::size_of::<Handle>(), (&b.mem as *const Handle).cast()),
                Arg::I32(v) => (4, (v as *const i32).cast()),
                Arg::I64(v) => (8, (v as *const i64).cast()),
                Arg::F32(v) => (4, (v as *const f32).cast()),
                Arg::F64(v) => (8, (v as *const f64).cast()),
            };
            check(unsafe { (self.gpu.api.set_kernel_arg)(self.handle, i as u32, size, ptr) }, "clSetKernelArg")?;
        }
        Ok(())
    }
}

pub struct Buffer {
    gpu: Arc<Gpu>,
    mem: Handle,
    cap: usize,
}

unsafe impl Send for Buffer {}

impl Buffer {
    /// Grow to at least `bytes` (contents are lost when it grows).
    pub fn ensure(&mut self, bytes: usize) -> Result<()> {
        if self.cap >= bytes.max(1) {
            return Ok(());
        }
        if !self.mem.is_null() {
            unsafe { (self.gpu.api.release_mem_object)(self.mem) };
        }
        let cap = bytes.max(1).next_power_of_two();
        let mut err = 0;
        self.mem = unsafe { (self.gpu.api.create_buffer)(self.gpu.context, CL_MEM_READ_WRITE, cap, null_mut(), &mut err) };
        check(err, "clCreateBuffer")?;
        self.cap = cap;
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !self.mem.is_null() {
            unsafe { (self.gpu.api.release_mem_object)(self.mem) };
        }
    }
}

/// Page-locked host memory (an `ALLOC_HOST_PTR` buffer kept mapped): the
/// driver moves it by DMA, without the extra staging copy a transfer from
/// ordinary memory pays. Used as the host side of [`Queue::upload`] and
/// [`Queue::download`].
pub struct Pinned {
    q: Queue,
    mem: Handle,
    ptr: *mut u8,
    cap: usize,
}

unsafe impl Send for Pinned {}

impl Pinned {
    /// Grow to at least `bytes` (contents are lost when it grows).
    pub fn ensure(&mut self, bytes: usize) -> Result<()> {
        if self.cap >= bytes.max(1) {
            return Ok(());
        }
        self.release();
        let cap = bytes.max(1).next_power_of_two();
        let api = &self.q.gpu.api;
        let mut err = 0;
        let mem = unsafe {
            (api.create_buffer)(self.q.gpu.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, cap, null_mut(), &mut err)
        };
        check(err, "clCreateBuffer (pinned)")?;
        let ptr = unsafe {
            (api.enqueue_map_buffer)(self.q.handle, mem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, cap, 0, null(), null_mut(), &mut err)
        };
        check(err, "clEnqueueMapBuffer")?;
        (self.mem, self.ptr, self.cap) = (mem, ptr.cast(), cap);
        Ok(())
    }

    /// The first `n` elements, as `T` (the mapping is page aligned).
    pub fn slice_mut<T: Copy>(&mut self, n: usize) -> &mut [T] {
        assert!(n * std::mem::size_of::<T>() <= self.cap);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), n) }
    }

    pub fn slice<T: Copy>(&self, n: usize) -> &[T] {
        assert!(n * std::mem::size_of::<T>() <= self.cap);
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), n) }
    }

    fn release(&mut self) {
        if !self.mem.is_null() {
            let api = &self.q.gpu.api;
            unsafe {
                (api.enqueue_unmap_mem_object)(self.q.handle, self.mem, self.ptr.cast(), 0, null(), null_mut());
                (api.finish)(self.q.handle);
                (api.release_mem_object)(self.mem);
            }
            self.mem = null_mut();
        }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        self.release();
    }
}

pub struct Queue {
    gpu: Arc<Gpu>,
    handle: Handle,
}

unsafe impl Send for Queue {}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe { (self.gpu.api.release_command_queue)(self.handle) };
    }
}

impl Queue {
    /// Blocking write of `data` at byte `offset` of `buf`.
    pub fn write<T: Copy>(&self, buf: &Buffer, offset: usize, data: &[T]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        check(
            unsafe {
                (self.gpu.api.enqueue_write_buffer)(
                    self.handle,
                    buf.mem,
                    CL_TRUE,
                    offset,
                    std::mem::size_of_val(data),
                    data.as_ptr().cast(),
                    0,
                    null(),
                    null_mut(),
                )
            },
            "clEnqueueWriteBuffer",
        )
    }

    /// Blocking read into `out` from byte `offset` of `buf`.
    pub fn read<T: Copy>(&self, buf: &Buffer, offset: usize, out: &mut [T]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        check(
            unsafe {
                (self.gpu.api.enqueue_read_buffer)(
                    self.handle,
                    buf.mem,
                    CL_TRUE,
                    offset,
                    std::mem::size_of_val(out),
                    out.as_mut_ptr().cast(),
                    0,
                    null(),
                    null_mut(),
                )
            },
            "clEnqueueReadBuffer",
        )
    }

    /// Queue a copy of `bytes` from `src` (at byte `src_off`) to `buf` (at
    /// `offset`). It is asynchronous: `src` must not change before a
    /// [`Queue::finish`] or a blocking call on this queue.
    pub fn upload(&self, buf: &Buffer, offset: usize, src: &Pinned, src_off: usize, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        assert!(src_off + bytes <= src.cap);
        check(
            unsafe {
                (self.gpu.api.enqueue_write_buffer)(
                    self.handle,
                    buf.mem,
                    CL_FALSE,
                    offset,
                    bytes,
                    src.ptr.add(src_off).cast(),
                    0,
                    null(),
                    null_mut(),
                )
            },
            "clEnqueueWriteBuffer",
        )
    }

    /// Queue a copy of `bytes` from `buf` (at `offset`) into `dst` (at byte
    /// `dst_off`); valid after [`Queue::finish`].
    pub fn download(&self, buf: &Buffer, offset: usize, dst: &mut Pinned, dst_off: usize, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        assert!(dst_off + bytes <= dst.cap);
        check(
            unsafe {
                (self.gpu.api.enqueue_read_buffer)(
                    self.handle,
                    buf.mem,
                    CL_FALSE,
                    offset,
                    bytes,
                    dst.ptr.add(dst_off).cast(),
                    0,
                    null(),
                    null_mut(),
                )
            },
            "clEnqueueReadBuffer",
        )
    }

    /// Enqueue `kernel` over `global` work items (1 or 2 dimensions, the first
    /// rounded up to a multiple of `local`).
    pub fn launch(&self, kernel: &Kernel, global: &[usize], local: usize) -> Result<()> {
        let mut g = global.to_vec();
        g[0] = g[0].div_ceil(local) * local;
        let mut l = vec![1usize; g.len()];
        l[0] = local;
        check(
            unsafe {
                (self.gpu.api.enqueue_nd_range_kernel)(
                    self.handle,
                    kernel.handle,
                    g.len() as u32,
                    null(),
                    g.as_ptr(),
                    l.as_ptr(),
                    0,
                    null(),
                    null_mut(),
                )
            },
            "clEnqueueNDRangeKernel",
        )
    }

    pub fn finish(&self) -> Result<()> {
        check(unsafe { (self.gpu.api.finish)(self.handle) }, "clFinish")
    }
}
