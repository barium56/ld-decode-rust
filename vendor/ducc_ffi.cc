// C shim exposing ducc0's 1-D real/complex FFTs (matching scipy 1.18's
// _duccfft rounding) to the Rust decoder via FFI.
//
// Compiled with clang targeting MSVC ABI, SSE2 homegrown SIMD, single-threaded.
//
// The shim is compiled ONCE PER ENGINE (see crates/ld-decode/build.rs):
//   sse2    — the scipy-wheel replica (default, bit-exact reference build)
//   avx2fma — -mavx2 -mfma (fastest, historically rounds differently)
//   avx2    — -mavx2 -mfma -ffp-contract=off (FMA-contraction hypothesis)
// DUCCQ_PREFIX gives each copy's extern "C" entry points distinct symbols so
// all three can link into one binary and be selected at runtime. ducc
// internals (fft1d_impl.h / fftnd_impl.h) are anonymous-namespace templates
// instantiated inside this TU, so the per-engine copies cannot clash.
#include <cstddef>
#include <cstdlib>
#include <complex>
#include "ducc0/fft/fftnd_impl.h"

#ifndef DUCCQ_PREFIX
#define DUCCQ_PREFIX duccq_
#endif

#define DUCCQ_CONCAT2(a, b) a##b
#define DUCCQ_CONCAT(a, b) DUCCQ_CONCAT2(a, b)
#define FN(name) DUCCQ_CONCAT(DUCCQ_PREFIX, name)

using namespace ducc0;
using namespace ducc0::detail_mav;
using namespace ducc0::detail_fft;

static void duccq_fail(const char *what, const char *fn) {
  std::fprintf(stderr, "ducc_ffi[%s] exception: %s\n", fn, what);
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
    force_simul_batch() = true;
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

} // extern "C"

extern "C" {

// Compile-time probes: the SIMD lane count this engine's FFT kernels select
// for `double` (sse2 => 2 lanes, avx2* => 4) and the native SIMD width the
// build targeted. Exposed so a test can prove the engines are genuinely
// different machine code and not three copies of the same SSE2 build.
int FN(simdlen)() { return int(fft1d_simdlen<double>); }
int FN(native_simdlen)() { return int(native_simd<double>::size()); }

} // extern "C"
