/* Copyright (C) 2019-2021 Max-Planck-Society
   Author: Martin Reinecke */

/* SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-or-later */

/*
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef DUCC0_UNITY_ROOTS_H
#define DUCC0_UNITY_ROOTS_H

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace ducc0 {

namespace detail_unity_roots {

using namespace std;

// Platform-independent twiddle evaluation.
//
// Upstream calls the platform libm's `cos`/`sin` here. Outside MSVC that is
// glibc, whose results differ from UCRT's by 1-2 ulp on a few percent of
// arguments -- and because twiddles are baked into every transform, that is the
// dominant source of Linux-vs-Windows FFT drift. When DUCC_UCRT_SHIM is defined
// (build.rs does that for every non-MSVC target) these helpers call bit-exact
// ports of the UCRT functions instead, so both platforms build identical
// twiddles from identical numbers. Windows keeps the platform libm, which *is*
// UCRT, so nothing changes there.
#if defined(DUCC_UCRT_SHIM)
// Implemented in crates/ld-decode/src/optimized/ucrt_math.rs; the C ABI keeps
// the vendored header free of any Rust dependency at compile time.
extern "C" double ld_ucrt_cos(double);
extern "C" double ld_ucrt_sin(double);
template<typename U> inline U twiddle_cos(U x) { return U(ld_ucrt_cos(double(x))); }
template<typename U> inline U twiddle_sin(U x) { return U(ld_ucrt_sin(double(x))); }
#else
template<typename U> inline U twiddle_cos(U x) { return cos(x); }
template<typename U> inline U twiddle_sin(U x) { return sin(x); }
#endif

template<typename T, typename Tc> class UnityRoots
  {
  private:
    using Thigh = typename conditional<(sizeof(T)>sizeof(double)), T, double>::type;
    struct cmplx_ { Thigh r, i; };
    size_t N, mask, shift;
    vector<cmplx_> v1, v2;

    static cmplx_ calc(size_t x, size_t n, Thigh ang)
      {
      x<<=3;
      if (x<4*n) // first half
        {
        if (x<2*n) // first quadrant
          {
          if (x<n) return {twiddle_cos(Thigh(x)*ang), twiddle_sin(Thigh(x)*ang)};
          return {twiddle_sin(Thigh(2*n-x)*ang), twiddle_cos(Thigh(2*n-x)*ang)};
          }
        else // second quadrant
          {
          x-=2*n;
          if (x<n) return {-twiddle_sin(Thigh(x)*ang), twiddle_cos(Thigh(x)*ang)};
          return {-twiddle_cos(Thigh(2*n-x)*ang), twiddle_sin(Thigh(2*n-x)*ang)};
          }
        }
      else
        {
        x=8*n-x;
        if (x<2*n) // third quadrant
          {
          if (x<n) return {twiddle_cos(Thigh(x)*ang), -twiddle_sin(Thigh(x)*ang)};
          return {twiddle_sin(Thigh(2*n-x)*ang), -twiddle_cos(Thigh(2*n-x)*ang)};
          }
        else // fourth quadrant
          {
          x-=2*n;
          if (x<n) return {-twiddle_sin(Thigh(x)*ang), -twiddle_cos(Thigh(x)*ang)};
          return {-twiddle_cos(Thigh(2*n-x)*ang), -twiddle_sin(Thigh(2*n-x)*ang)};
          }
        }
      }
#if 0  // alternative version, similar speed, but maybe a bit more accurate
    static cmplx_ calc2(size_t x, size_t n)
      {
      static constexpr Thigh pi = Thigh(3.141592653589793238462643383279502884197L);
      Thigh n4 = Thigh(n<<2);

      x<<=3;
      if (x<4*n) // first half
        {
        if (x<2*n) // first quadrant
          {
          if (x<n)
            {
            auto ang = (x/n4)*pi;
            return {cos(ang), sin(ang)};
            }
          auto ang = ((2*n-x)/n4)*pi;
          return {sin(ang), cos(ang)};
          }
        else // second quadrant
          {
          x-=2*n;
          if (x<n)
            {
            auto ang = (x/n4)*pi;
            return {-sin(ang), cos(ang)};
            }
          auto ang = ((2*n-x)/n4)*pi;
          return {-cos(ang), sin(ang)};
          }
        }
      else
        {
        x=8*n-x;
        if (x<2*n) // third quadrant
          {
          if (x<n)
            {
            auto ang = (x/n4)*pi;
            return {cos(ang), -sin(ang)};
            }
          auto ang = ((2*n-x)/n4)*pi;
          return {sin(ang), -cos(ang)};
          }
        else // fourth quadrant
          {
          x-=2*n;
          if (x<n)
            {
            auto ang = (x/n4)*pi;
            return {-sin(ang), -cos(ang)};
            }
          auto ang = ((2*n-x)/n4)*pi;
          return {-cos(ang), -sin(ang)};
          }
        }
      }
#endif

  public:
    UnityRoots(size_t n)
      : N(n)
      {
      // The twiddle argument is evaluated in `double`, not `long double`.
      //
      // Upstream writes this as `Thigh(0.25L*pi/n)` with a `long double`
      // literal, which makes the argument platform-dependent: MSVC defines
      // `long double` as 64-bit, but g++/clang on Linux use x87 80-bit
      // extended, so Linux rounds twice -- 80-bit intermediate, then to
      // `Thigh` -- and lands 1 ulp away from the Windows value whenever n is
      // not a power of two (measured: n = 3, 6, 12, 15 differ; powers of two
      // are exact scalings and agree). ducc builds a `UnityRoots` per
      // prime-factor stage, so radix-3/5/7 plans hit the odd n case, not a
      // corner case.
      //
      // On Windows this change is a no-op by construction (the `L` literal
      // already collapsed to `double` there, verified: sizeof(long double) ==
      // 8 and 34/34 sizes agreed before the change), so it only moves Linux
      // onto the Windows values.
      static constexpr double pi = 3.141592653589793238462643383279502884197;
      Thigh ang = Thigh(0.25*pi/n);
      size_t nval = (n+2)/2;
      shift = 1;
      while((size_t(1)<<shift)*(size_t(1)<<shift) < nval) ++shift;
      mask = (size_t(1)<<shift)-1;
      v1.resize(mask+1);
      v1[0]={Thigh(1), Thigh(0)};
      for (size_t i=1; i<v1.size(); ++i)
        v1[i]=calc(i,n,ang);
      v2.resize((nval+mask)/(mask+1));
      v2[0]={Thigh(1), Thigh(0)};
      for (size_t i=1; i<v2.size(); ++i)
        v2[i]=calc(i*(mask+1),n,ang);
      }

    size_t size() const { return N; }
    size_t footprint() const { return (v1.size()+v2.size())*sizeof(cmplx_); }

    Tc operator[](size_t idx) const
      {
      if (2*idx<=N)
        {
        auto x1=v1[idx&mask], x2=v2[idx>>shift];
        return Tc(T(x1.r*x2.r-x1.i*x2.i), T(x1.r*x2.i+x1.i*x2.r));
        }
      idx = N-idx;
      auto x1=v1[idx&mask], x2=v2[idx>>shift];
      return Tc(T(x1.r*x2.r-x1.i*x2.i), -T(x1.r*x2.i+x1.i*x2.r));
      }
  };

template<typename T, typename Tc> class MultiExp
  {
  private:
    using Thigh = typename conditional<(sizeof(T)>sizeof(double)), T, double>::type;
    struct cmplx_ { Thigh r, i; };
    size_t N, mask, shift;
    vector<cmplx_> v1, v2;

  public:
    MultiExp(T ang0, size_t n)
      : N(n)
      {
      Thigh ang = ang0;
      size_t nval = n+2;
      shift = 1;
      while((size_t(1)<<shift)*(size_t(1)<<shift) < nval) ++shift;
      mask = (size_t(1)<<shift)-1;
      v1.resize(mask+1);
      v1[0]={Thigh(1), Thigh(0)};
      for (size_t i=1; i<v1.size(); ++i)
        v1[i] = {twiddle_cos(i*ang), twiddle_sin(i*ang)};
      v2.resize((nval+mask)/(mask+1));
      v2[0]={Thigh(1), Thigh(0)};
      for (size_t i=1; i<v2.size(); ++i)
        v2[i] = {twiddle_cos((i*(mask+1))*ang), twiddle_sin((i*(mask+1))*ang)};
      }

    size_t size() const { return N; }

    Tc operator[](size_t idx) const
      {
      auto x1=v1[idx&mask], x2=v2[idx>>shift];
      return Tc(T(x1.r*x2.r-x1.i*x2.i), T(x1.r*x2.i+x1.i*x2.r));
      }
  };

}

using detail_unity_roots::UnityRoots;
using detail_unity_roots::MultiExp;

}

#endif
