// C shim exposing ducc0's 1-D real/complex FFTs (matching scipy 1.18's
// _duccfft rounding) to the Rust decoder via FFI.
//
// Compiled with clang targeting MSVC ABI, SSE2 homegrown SIMD, single-threaded.
#include <cstddef>
#include <complex>
#include "ducc0/fft/fftnd_impl.h"

using namespace ducc0;
using namespace ducc0::detail_mav;
using namespace ducc0::detail_fft;

static void duccq_fail(const char *what, const char *fn) {
  std::fprintf(stderr, "ducc_ffi[%s] exception: %s\n", fn, what);
  std::fflush(stderr);
}

extern "C" {

// Forward complex FFT (no normalization). out[0..2n-1].
void duccq_fft(int n, const double *in, double *out) {
  try {
    shape_t axes{ 0 };
    cfmav<complex<double>> cin((complex<double>*)in, shape_t{(size_t)n});
    vfmav<complex<double>> cout((complex<double>*)out, shape_t{(size_t)n});
    c2c(cin, cout, axes, FORWARD, 1.0, 1);
  } catch (const std::exception &e) { duccq_fail(e.what(), "fft"); }
  catch (...) { duccq_fail("unknown", "fft"); }
}

// Inverse complex FFT (normalized by 1/n). out[0..2n-1].
void duccq_ifft(int n, const double *in, double *out) {
  try {
    shape_t axes{ 0 };
    cfmav<complex<double>> cin((complex<double>*)in, shape_t{(size_t)n});
    vfmav<complex<double>> cout((complex<double>*)out, shape_t{(size_t)n});
    c2c(cin, cout, axes, BACKWARD, 1.0/double(n), 1);
  } catch (const std::exception &e) { duccq_fail(e.what(), "ifft"); }
  catch (...) { duccq_fail("unknown", "ifft"); }
}

// Forward real FFT -> half spectrum (n/2+1 complex). out size = 2*(n/2+1).
void duccq_rfft(int n, const double *in, double *out) {
  try {
    cfmav<double> rin(in, shape_t{(size_t)n});
    vfmav<complex<double>> cout((complex<double>*)out, shape_t{(size_t)(n/2+1)});
    r2c(rin, cout, 0, FORWARD, 1.0, 1);
  } catch (const std::exception &e) { duccq_fail(e.what(), "rfft"); }
  catch (...) { duccq_fail("unknown", "rfft"); }
}

// Inverse real FFT: half spectrum (n/2+1 complex) in, n real out. fct=1/n.
void duccq_irfft(int n, const double *in, double *out) {
  try {
    cfmav<complex<double>> cin((complex<double>*)in, shape_t{(size_t)(n/2+1)});
    vfmav<double> cout(out, shape_t{(size_t)n});
    c2r(cin, cout, 0, BACKWARD, 1.0/double(n), 1);
  } catch (const std::exception &e) { duccq_fail(e.what(), "irfft"); }
  catch (...) { duccq_fail("unknown", "irfft"); }
}

} // extern "C"