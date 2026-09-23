// C shim exposing ducc0's 1-D real/complex FFTs (matching scipy 1.18's
// _duccfft rounding) to the Rust decoder via FFI.
//
// Compiled with clang targeting MSVC ABI, single-threaded.
//
// The shim is compiled ONCE PER ENGINE (see crates/ld-decode/build.rs):
//   sse2    — the scipy-wheel replica (default, bit-exact reference build)
//   avx2fma — -mavx2 -mfma (fastest, but rounds differently by construction)
//   avx2    — -mavx2 -mfma -ffp-contract=off (FMA-contraction hypothesis)
// DUCCQ_PREFIX gives each copy's extern "C" entry points distinct symbols so
// all three can link into one binary and be selected at runtime.
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <complex>

#ifndef DUCCQ_PREFIX
#define DUCCQ_PREFIX duccq_
#endif

#define DUCCQ_CONCAT2(a, b) a##b
#define DUCCQ_CONCAT(a, b) DUCCQ_CONCAT2(a, b)
#define FN(name) DUCCQ_CONCAT(DUCCQ_PREFIX, name)

// Give this engine its OWN copy of ducc's entry layer. Without this the three
// engine objects share one engine's compiled code, silently.
//
// ducc's entry points are explicitly instantiated (fft_inst_inc.h) in the
// externally-linked namespace `ducc0::detail_fft`, so all three copies emitted
// identically-named strong symbols (`ducc0::detail_fft::c2c<double>` is `T` in
// every object, verified with llvm-nm) and the LINKER KEPT ONE. Measured
// 2026-09-22, the two consequences were both invisible and both wrong:
//
//  * `fft_simdlen<double>` is 4 in an AVX2 object and 2 in the SSE2 object, but
//    the executing code was the SSE2 copy, so no 256-bit kernel could ever run
//    no matter which engine was selected.
//  * The kernel bodies live in anonymous namespaces (per-TU, correctly
//    separate), but the shared entry copy also read the SSE2 object's
//    `thread_local` `force_simul_batch()` flag — which the AVX2 shim never sets
//    (it sets its own). So the AVX2 copies ran `n_simul=1`: one SCALAR transform
//    at a time, never the batched path. That is what the long-recorded "AVX2 is
//    ~22% slower in situ" measured — not a codegen or frequency penalty.
//
// Renaming the namespace per engine fixes both: the macro is what
// `namespace detail_fft`, every `ducc0::detail_fft::` qualified name and the
// `using` declarations all spell, so they stay consistent. Only `detail_fft`
// needs it — `detail_mav` and the FFT pass classes are header-inline or
// anonymous-namespace templates, already per-TU.
#define detail_fft DUCCQ_CONCAT(duccq_fft_, DUCCQ_PREFIX)

#include "ducc0/fft/fftnd_impl.h"

using namespace ducc0;
using namespace ducc0::detail_mav;
using namespace ducc0::detail_fft;

static void duccq_fail(const char *what, const char *fn) {
  std::fprintf(stderr, "ducc_ffi[%s] exception: %s\n", fn, what);
  std::fflush(stderr);
}

// Env-gated (DUCC_TRACE_SIMUL=1) probe: report what this translation unit was
// actually compiled as, so an engine name can be checked against the SIMD
// support the compiler gave it — the check that found the shared-symbol bug
// above. Pairs with the dispatch trace in vendor/ducc0/fft/fftnd_impl.h.
#ifdef __AVX__
#define DUCCQ_P_AVX 1
#else
#define DUCCQ_P_AVX 0
#endif
#ifdef __AVX2__
#define DUCCQ_P_AVX2 1
#else
#define DUCCQ_P_AVX2 0
#endif
#ifdef __FMA__
#define DUCCQ_P_FMA 1
#else
#define DUCCQ_P_FMA 0
#endif
#define DUCCQ_ST2(x) #x
#define DUCCQ_ST(x) DUCCQ_ST2(x)
static void duccq_engine_probe(const char *fn) {
  static const bool trace = getenv("DUCC_TRACE_SIMUL") != nullptr;
  if (!trace) return;
  static bool done = false;
  if (done) return;
  done = true;
  std::fprintf(stderr,
    "TUPROBE [%s] %s AVX=%d AVX2=%d FMA=%d native_simd<double>=%zu fft_simdlen<double>=%zu\n",
    DUCCQ_ST(DUCCQ_PREFIX), fn, DUCCQ_P_AVX, DUCCQ_P_AVX2, DUCCQ_P_FMA,
    (size_t)ducc0::native_simd<double>::size(),
    (size_t)ducc0::detail_fft::fft_simdlen<double>);
  std::fflush(stderr);
}

extern "C" {

// Forward complex FFT (no normalization). out[0..2n-1].
void FN(fft)(int n, const double *in, double *out) {
  try {
    shape_t axes{ 0 };
    cfmav<complex<double>> cin((complex<double>*)in, shape_t{(size_t)n});
    vfmav<complex<double>> cout((complex<double>*)out, shape_t{(size_t)n});
    c2c(cin, cout, axes, FORWARD, 1.0, 1);
  } catch (const std::exception &e) { duccq_fail(e.what(), "fft"); }
  catch (...) { duccq_fail("unknown", "fft"); }
}

// Inverse complex FFT (normalized by 1/n). out[0..2n-1].
void FN(ifft)(int n, const double *in, double *out) {
  try {
    shape_t axes{ 0 };
    cfmav<complex<double>> cin((complex<double>*)in, shape_t{(size_t)n});
    vfmav<complex<double>> cout((complex<double>*)out, shape_t{(size_t)n});
    c2c(cin, cout, axes, BACKWARD, 1.0/double(n), 1);
  } catch (const std::exception &e) { duccq_fail(e.what(), "ifft"); }
  catch (...) { duccq_fail("unknown", "ifft"); }
}

// Forward real FFT -> half spectrum (n/2+1 complex). out size = 2*(n/2+1).
void FN(rfft)(int n, const double *in, double *out) {
  try {
    cfmav<double> rin(in, shape_t{(size_t)n});
    vfmav<complex<double>> cout((complex<double>*)out, shape_t{(size_t)(n/2+1)});
    r2c(rin, cout, 0, FORWARD, 1.0, 1);
  } catch (const std::exception &e) { duccq_fail(e.what(), "rfft"); }
  catch (...) { duccq_fail("unknown", "rfft"); }
}

// Inverse real FFT: half spectrum (n/2+1 complex) in, n real out. fct=1/n.
void FN(irfft)(int n, const double *in, double *out) {
  try {
    cfmav<complex<double>> cin((complex<double>*)in, shape_t{(size_t)(n/2+1)});
    vfmav<double> cout(out, shape_t{(size_t)n});
    c2r(cin, cout, 0, BACKWARD, 1.0/double(n), 1);
  } catch (const std::exception &e) { duccq_fail(e.what(), "irfft"); }
  catch (...) { duccq_fail("unknown", "irfft"); }
}

// Batched inverse c2c (fct=1/n) over `k` contiguous rows of length `n` —
// several independent block-sized spectra transformed together.
//
// Why this is worth a special entry: ducc's 1-D contiguous path takes a fast
// path whose `exec_simple` hands the SCALAR type index to the pass, so a lone
// transform never reaches ducc's SIMD kernels (which vectorize *across*
// independent transforms, not within one). Its multi-transform machinery does,
// but the `n_simul` heuristic disables it for long transforms. Batching through
// here opts into it; measured ~1.83x per transform at k=2/4/6/8, n=32768, with
// results bit-identical to the scalar path.
void FN(ifft_batch_rows)(int k, int n, const double *in, double *out) {
  try {
    duccq_engine_probe("ifft_batch_rows");
    force_simul_batch() = true;
    static const bool trace = getenv("DUCC_TRACE_SIMUL") != nullptr;
    if (trace)
      std::fprintf(stderr, "CALL [%s] batch k=%d n=%d forced=%d\n",
        DUCCQ_ST(DUCCQ_PREFIX), k, n, (int)force_simul_batch());
    shape_t shp{ (size_t)k, (size_t)n };
    stride_t strd{ (ptrdiff_t)n, 1 };
    cfmav<complex<double>> cin((complex<double>*)in, shp, strd);
    vfmav<complex<double>> cout((complex<double>*)out, shp, strd);
    c2c(cin, cout, shape_t{1}, BACKWARD, 1.0/double(n), 1);
    force_simul_batch() = false;
  } catch (const std::exception &e) {
    force_simul_batch() = false;
    duccq_fail(e.what(), "ifft_batch_rows");
  }
  catch (...) {
    force_simul_batch() = false;
    duccq_fail("unknown", "ifft_batch_rows");
  }
}

// In-place form of the above: the same batched c2c over `k` contiguous rows,
// with `in == out`.
//
// ducc's c2c takes its "inplace" path whenever the transform axis is
// contiguous and `n_bunch == 1` -- exactly this call's shape -- and that path
// reads the input straight out of `out` (`if (in.data()!=out.data())
// copy_input(...)`), transforms it there through its own line buffer and
// writes back. Handing it the same pointer therefore only drops that copy: the
// plan, the pass structure and every arithmetic op are the ones the
// out-of-place call already used, so the results are bit-identical (gated by
// `ifft_batch_rows_inplace_matches_out_of_place`).
void FN(ifft_batch_rows_ip)(int k, int n, double *buf) {
  try {
    force_simul_batch() = true;
    shape_t shp{ (size_t)k, (size_t)n };
    stride_t strd{ (ptrdiff_t)n, 1 };
    cfmav<complex<double>> cin((complex<double>*)buf, shp, strd);
    vfmav<complex<double>> cout((complex<double>*)buf, shp, strd);
    c2c(cin, cout, shape_t{1}, BACKWARD, 1.0/double(n), 1);
    force_simul_batch() = false;
  } catch (const std::exception &e) {
    force_simul_batch() = false;
    duccq_fail(e.what(), "ifft_batch_rows_ip");
  }
  catch (...) {
    force_simul_batch() = false;
    duccq_fail("unknown", "ifft_batch_rows_ip");
  }
}

} // extern "C"

extern "C" {

// Compile-time probes: the SIMD lane count this engine's FFT kernels select
// for `double` (sse2 => 2 lanes, avx2* => 4) and the native SIMD width the
// build targeted. Exposed so a test can prove the engines are genuinely
// different machine code and not three copies of the same SSE2 build.
int FN(simdlen)() { return int(fft1d_simdlen<double>); }
int FN(native_simdlen)() { return int(native_simd<double>::size()); }

} // extern "C"
