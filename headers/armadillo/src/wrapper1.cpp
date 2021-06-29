// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


#include <climits>
#include <limits>
#include <complex>

#include "armadillo_bits/config.hpp"

#undef ARMA_USE_WRAPPER
#undef ARMA_USE_FORTRAN_HIDDEN_ARGS

#include "armadillo_bits/compiler_setup.hpp"
#include "armadillo_bits/typedef_elem.hpp"
#include "armadillo_bits/include_atlas.hpp"
#include "armadillo_bits/include_superlu.hpp"


#if defined(ARMA_USE_EXTERN_RNG)
  #include <random>
  
  namespace arma
    {
    // NOTE: arma_rng_cxx11_instance is kept only for compatibility with earlier versions of armadillo
    // TODO: remove arma_rng_cxx11_instance when the major version is bumped
    
    #include "armadillo_bits/arma_rng_cxx11.hpp"
    thread_local arma_rng_cxx11 arma_rng_cxx11_instance;
    
    thread_local std::mt19937_64 mt19937_64_instance;
    }
#endif

#if defined(ARMA_USE_HDF5_ALT)
  
  #undef  H5_USE_110_API
  #define H5_USE_110_API
  
  #include <hdf5.h>
  
  #if defined(H5_USE_16_API_DEFAULT) || defined(H5_USE_16_API)
    // #pragma message ("disabling use of HDF5 due to its incompatible configuration")
    #undef ARMA_USE_HDF5_ALT
  #endif

#endif

namespace arma
{

#include "armadillo_bits/def_blas.hpp"
#include "armadillo_bits/def_lapack.hpp"
#include "armadillo_bits/def_arpack.hpp"
#include "armadillo_bits/def_superlu.hpp"
// no need to include def_hdf5.hpp -- it only contains #defines for when ARMA_USE_HDF5_ALT is not defined.


#if defined(ARMA_USE_HDF5_ALT)
  // Wrapper functions: arma::H5open() and arma::H5check_version() to hijack calls to H5open() and H5check_version()
  herr_t H5open()
    {
    return ::H5open();
    }
  
  herr_t H5check_version(unsigned majnum, unsigned minnum, unsigned relnum)
    {
    return ::H5check_version(majnum, minnum, relnum);
    }
#endif



// at this stage we have prototypes for the real blas, lapack and atlas functions

// now we make the wrapper functions


extern "C"
  {
  #if defined(ARMA_USE_BLAS)
    
    float arma_fortran_with_prefix(arma_sasum)(const blas_int* n, const float* x, const blas_int* incx)
      {
      return arma_fortran_sans_prefix(arma_sasum)(n, x, incx);
      }
    
    double arma_fortran_with_prefix(arma_dasum)(const blas_int* n, const double* x, const blas_int* incx)
      {
      return arma_fortran_sans_prefix(arma_dasum)(n, x, incx);
      }
    
    
    
    float arma_fortran_with_prefix(arma_snrm2)(const blas_int* n, const float* x, const blas_int* incx)
      {
      return arma_fortran_sans_prefix(arma_snrm2)(n, x, incx);
      }
    
    double arma_fortran_with_prefix(arma_dnrm2)(const blas_int* n, const double* x, const blas_int* incx)
      {
      return arma_fortran_sans_prefix(arma_dnrm2)(n, x, incx);
      }
    
    
    
    float arma_fortran_with_prefix(arma_sdot)(const blas_int* n, const float*  x, const blas_int* incx, const float*  y, const blas_int* incy)
      {
      return arma_fortran_sans_prefix(arma_sdot)(n, x, incx, y, incy);
      }
    
    double arma_fortran_with_prefix(arma_ddot)(const blas_int* n, const double* x, const blas_int* incx, const double* y, const blas_int* incy)
      {
      return arma_fortran_sans_prefix(arma_ddot)(n, x, incx, y, incy);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgemv)(const char* transA, const blas_int* m, const blas_int* n, const float*  alpha, const float*  A, const blas_int* ldA, const float*  x, const blas_int* incx, const float*  beta, float*  y, const blas_int* incy)
      {
      arma_fortran_sans_prefix(arma_sgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy);
      }
    
    void arma_fortran_with_prefix(arma_dgemv)(const char* transA, const blas_int* m, const blas_int* n, const double* alpha, const double* A, const blas_int* ldA, const double* x, const blas_int* incx, const double* beta, double* y, const blas_int* incy)
      {
      arma_fortran_sans_prefix(arma_dgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy);
      }
    
    void arma_fortran_with_prefix(arma_cgemv)(const char* transA, const blas_int* m, const blas_int* n, const blas_cxf* alpha, const blas_cxf* A, const blas_int* ldA, const blas_cxf* x, const blas_int* incx, const blas_cxf* beta, blas_cxf* y, const blas_int* incy)
      {
      arma_fortran_sans_prefix(arma_cgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy);
      }
    
    void arma_fortran_with_prefix(arma_zgemv)(const char* transA, const blas_int* m, const blas_int* n, const blas_cxd* alpha, const blas_cxd* A, const blas_int* ldA, const blas_cxd* x, const blas_int* incx, const blas_cxd* beta, blas_cxd* y, const blas_int* incy)
      {
      arma_fortran_sans_prefix(arma_zgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const float*  alpha, const float*  A, const blas_int* ldA, const float*  B, const blas_int* ldB, const float*  beta, float*  C, const blas_int* ldC)
      {
      arma_fortran_sans_prefix(arma_sgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
      }
    
    void arma_fortran_with_prefix(arma_dgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* B, const blas_int* ldB, const double* beta, double* C, const blas_int* ldC)
      {
      arma_fortran_sans_prefix(arma_dgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
      }
    
    void arma_fortran_with_prefix(arma_cgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const blas_cxf* alpha, const blas_cxf* A, const blas_int* ldA, const blas_cxf* B, const blas_int* ldB, const blas_cxf* beta, blas_cxf* C, const blas_int* ldC)
      {
      arma_fortran_sans_prefix(arma_cgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
      }
    
    void arma_fortran_with_prefix(arma_zgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const blas_cxd* alpha, const blas_cxd* A, const blas_int* ldA, const blas_cxd* B, const blas_int* ldB, const blas_cxd* beta, blas_cxd* C, const blas_int* ldC)
      {
      arma_fortran_sans_prefix(arma_zgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
      }
    
    
    
    void arma_fortran_with_prefix(arma_ssyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const  float* alpha, const  float* A, const blas_int* ldA, const  float* beta,  float* C, const blas_int* ldC)
      {
      arma_fortran_sans_prefix(arma_ssyrk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC);
      }
    
    void arma_fortran_with_prefix(arma_dsyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* beta, double* C, const blas_int* ldC)
      {
      arma_fortran_sans_prefix(arma_dsyrk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cherk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const  float* alpha, const blas_cxf* A, const blas_int* ldA, const  float* beta, blas_cxf* C, const blas_int* ldC)
      {
      arma_fortran_sans_prefix(arma_cherk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC);
      }
    
    void arma_fortran_with_prefix(arma_zherk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const double* alpha, const blas_cxd* A, const blas_int* ldA, const double* beta, blas_cxd* C, const blas_int* ldC)
      {
      arma_fortran_sans_prefix(arma_zherk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC);
      }
    
  #endif
  
  
  
  #if defined(ARMA_USE_LAPACK)
    
    void arma_fortran_with_prefix(arma_sgetrf)(const blas_int* m, const blas_int* n,    float* a, const blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgetrf)(m, n, a, lda, ipiv, info);
      }
    
    void arma_fortran_with_prefix(arma_dgetrf)(const blas_int* m, const blas_int* n,   double* a, const blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgetrf)(m, n, a, lda, ipiv, info);
      }
    
    void arma_fortran_with_prefix(arma_cgetrf)(const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgetrf)(m, n, a, lda, ipiv, info);
      }
    
    void arma_fortran_with_prefix(arma_zgetrf)(const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgetrf)(m, n, a, lda, ipiv, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const    float* a, const blas_int* lda, const blas_int* ipiv,    float* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgetrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const   double* a, const blas_int* lda, const blas_int* ipiv,   double* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgetrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const blas_cxf* a, const blas_int* lda, const blas_int* ipiv, blas_cxf* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgetrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const blas_cxd* a, const blas_int* lda, const blas_int* ipiv, blas_cxd* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgetrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgetri)(const blas_int* n,    float* a, const blas_int* lda, const blas_int* ipiv,    float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgetri)(n, a, lda, ipiv, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgetri)(const blas_int* n,   double* a, const blas_int* lda, const blas_int* ipiv,   double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgetri)(n, a, lda, ipiv, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgetri)(const blas_int* n, blas_cxf* a, const blas_int* lda, const blas_int* ipiv, blas_cxf* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgetri)(n, a, lda, ipiv, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgetri)(const blas_int* n, blas_cxd* a, const blas_int* lda, const blas_int* ipiv, blas_cxd* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgetri)(n, a, lda, ipiv, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_strtri)(const char* uplo, const char* diag, const blas_int* n,    float* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_strtri)(uplo, diag, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_dtrtri)(const char* uplo, const char* diag, const blas_int* n,   double* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dtrtri)(uplo, diag, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_ctrtri)(const char* uplo, const char* diag, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ctrtri)(uplo, diag, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_ztrtri)(const char* uplo, const char* diag, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ztrtri)(uplo, diag, n, a, lda, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_ssyev)(const char* jobz, const char* uplo, const blas_int* n,  float* a, const blas_int* lda,  float* w,  float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ssyev)(jobz, uplo, n, a, lda, w, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dsyev)(const char* jobz, const char* uplo, const blas_int* n, double* a, const blas_int* lda, double* w, double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dsyev)(jobz, uplo, n, a, lda, w, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cheev)(const char* jobz, const char* uplo, const blas_int* n, blas_cxf* a, const blas_int* lda,  float* w, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cheev)(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zheev)(const char* jobz, const char* uplo, const blas_int* n, blas_cxd* a, const blas_int* lda, double* w, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zheev)(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_ssyevd)(const char* jobz, const char* uplo, const blas_int* n,  float* a, const blas_int* lda,  float* w,  float* work, const blas_int* lwork, blas_int* iwork, const blas_int* liwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ssyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dsyevd)(const char* jobz, const char* uplo, const blas_int* n, double* a, const blas_int* lda, double* w, double* work, const blas_int* lwork, blas_int* iwork, const blas_int* liwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dsyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cheevd)(const char* jobz, const char* uplo, const blas_int* n, blas_cxf* a, const blas_int* lda,  float* w, blas_cxf* work, const blas_int* lwork,  float* rwork, const blas_int* lrwork, blas_int* iwork, const blas_int* liwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cheevd)(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zheevd)(const char* jobz, const char* uplo, const blas_int* n, blas_cxd* a, const blas_int* lda, double* w, blas_cxd* work, const blas_int* lwork, double* rwork, const blas_int* lrwork, blas_int* iwork, const blas_int* liwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zheevd)(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgeev)(const char* jobvl, const char* jobvr, const blas_int* n,  float* a, const blas_int* lda,  float* wr,  float* wi,  float* vl, const blas_int* ldvl,  float* vr, const blas_int* ldvr,  float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgeev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgeev)(const char* jobvl, const char* jobvr, const blas_int* n, double* a, const blas_int* lda, double* wr, double* wi, double* vl, const blas_int* ldvl, double* vr, const blas_int* ldvr, double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgeev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgeev)(const char* jobvl, const char* jobvr, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_cxf* w, blas_cxf* vl, const blas_int* ldvl, blas_cxf* vr, const blas_int* ldvr, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgeev)(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgeev)(const char* jobvl, const char* jobvr, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_cxd* w, blas_cxd* vl, const blas_int* ldvl, blas_cxd* vr, const blas_int* ldvr, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgeev)(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgeevx)(const char* balanc, const char* jobvl, const char* jobvr, const char* sense, const blas_int* n,  float* a, const blas_int* lda,  float* wr,  float* wi,  float* vl, const blas_int* ldvl,  float* vr, const blas_int* ldvr, blas_int* ilo, blas_int* ihi,  float* scale,  float* abnrm,  float* rconde,  float* rcondv,  float* work, const blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgeevx)(const char* balanc, const char* jobvl, const char* jobvr, const char* sense, const blas_int* n, double* a, const blas_int* lda, double* wr, double* wi, double* vl, const blas_int* ldvl, double* vr, const blas_int* ldvr, blas_int* ilo, blas_int* ihi, double* scale, double* abnrm, double* rconde, double* rcondv, double* work, const blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgeevx)(const char* balanc, const char* jobvl, const char* jobvr, const char* sense, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_cxf* w, blas_cxf* vl, const blas_int* ldvl, blas_cxf* vr, const blas_int* ldvr, blas_int* ilo, blas_int* ihi,  float* scale,  float* abnrm,  float* rconde,  float* rcondv, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, w, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgeevx)(const char* balanc, const char* jobvl, const char* jobvr, const char* sense, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_cxd* w, blas_cxd* vl, const blas_int* ldvl, blas_cxd* vr, const blas_int* ldvr, blas_int* ilo, blas_int* ihi, double* scale, double* abnrm, double* rconde, double* rcondv, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, w, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work, lwork, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sggev)(const char* jobvl, const char* jobvr, const blas_int* n,  float* a, const blas_int* lda,  float* b, const blas_int* ldb,  float* alphar,  float* alphai,  float* beta,  float* vl, const blas_int* ldvl,  float* vr, const blas_int* ldvr,  float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sggev)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);
      }
      
    void arma_fortran_with_prefix(arma_dggev)(const char* jobvl, const char* jobvr, const blas_int* n, double* a, const blas_int* lda, double* b, const blas_int* ldb, double* alphar, double* alphai, double* beta, double* vl, const blas_int* ldvl, double* vr, const blas_int* ldvr, double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dggev)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cggev)(const char* jobvl, const char* jobvr, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_cxf* b, const blas_int* ldb, blas_cxf* alpha, blas_cxf* beta, blas_cxf* vl, const blas_int* ldvl, blas_cxf* vr, const blas_int* ldvr, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cggev)(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zggev)(const char* jobvl, const char* jobvr, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_cxd* b, const blas_int* ldb, blas_cxd* alpha, blas_cxd* beta, blas_cxd* vl, const blas_int* ldvl, blas_cxd* vr, const blas_int* ldvr, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zggev)(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spotrf)(const char* uplo, const blas_int* n,    float* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_spotrf)(uplo, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_dpotrf)(const char* uplo, const blas_int* n,   double* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dpotrf)(uplo, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_cpotrf)(const char* uplo, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cpotrf)(uplo, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_zpotrf)(const char* uplo, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zpotrf)(uplo, n, a, lda, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spotrs)(const char* uplo, const blas_int* n, const blas_int* nrhs, const    float* a, const blas_int* lda,    float* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_spotrs)(uplo, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dpotrs)(const char* uplo, const blas_int* n, const blas_int* nrhs, const   double* a, const blas_int* lda,   double* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dpotrs)(uplo, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cpotrs)(const char* uplo, const blas_int* n, const blas_int* nrhs, const blas_cxf* a, const blas_int* lda, blas_cxf* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cpotrs)(uplo, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zpotrs)(const char* uplo, const blas_int* n, const blas_int* nrhs, const blas_cxd* a, const blas_int* lda, blas_cxd* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zpotrs)(uplo, n, nrhs, a, lda, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spbtrf)(const char* uplo, const blas_int* n, const blas_int* kd,    float* ab, const blas_int* ldab, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_spbtrf)(uplo, n, kd, ab, ldab, info);
      }
    
    void arma_fortran_with_prefix(arma_dpbtrf)(const char* uplo, const blas_int* n, const blas_int* kd,   double* ab, const blas_int* ldab, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dpbtrf)(uplo, n, kd, ab, ldab, info);
      }
    
    void arma_fortran_with_prefix(arma_cpbtrf)(const char* uplo, const blas_int* n, const blas_int* kd, blas_cxf* ab, const blas_int* ldab, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cpbtrf)(uplo, n, kd, ab, ldab, info);
      }
    
    void arma_fortran_with_prefix(arma_zpbtrf)(const char* uplo, const blas_int* n, const blas_int* kd, blas_cxd* ab, const blas_int* ldab, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zpbtrf)(uplo, n, kd, ab, ldab, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spotri)(const char* uplo, const blas_int* n,    float* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_spotri)(uplo, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_dpotri)(const char* uplo, const blas_int* n,   double* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dpotri)(uplo, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_cpotri)(const char* uplo, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cpotri)(uplo, n, a, lda, info);
      }
    
    void arma_fortran_with_prefix(arma_zpotri)(const char* uplo, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zpotri)(uplo, n, a, lda, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgeqrf)(const blas_int* m, const blas_int* n,    float* a, const blas_int* lda,    float* tau,    float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgeqrf)(const blas_int* m, const blas_int* n,   double* a, const blas_int* lda,   double* tau,   double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgeqrf)(const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_cxf* tau, blas_cxf* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgeqrf)(const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_cxd* tau, blas_cxd* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgeqp3)(const blas_int* m, const blas_int* n,    float* a, const blas_int* lda, blas_int* jpvt,    float* tau,    float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgeqp3)(m, n, a, lda, jpvt, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgeqp3)(const blas_int* m, const blas_int* n,   double* a, const blas_int* lda, blas_int* jpvt,   double* tau,   double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgeqp3)(m, n, a, lda, jpvt, tau, work, lwork, info);
      }
    
    
    void arma_fortran_with_prefix(arma_cgeqp3)(const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* jpvt, blas_cxf* tau, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgeqp3)(m, n, a, lda, jpvt, tau, work, lwork, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgeqp3)(const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* jpvt, blas_cxd* tau, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgeqp3)(m, n, a, lda, jpvt, tau, work, lwork, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sorgqr)(const blas_int* m, const blas_int* n, const blas_int* k,  float* a, const blas_int* lda, const  float* tau,  float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sorgqr)(m, n, k, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dorgqr)(const blas_int* m, const blas_int* n, const blas_int* k, double* a, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dorgqr)(m, n, k, a, lda, tau, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_cungqr)(const blas_int* m, const blas_int* n, const blas_int* k, blas_cxf* a, const blas_int* lda,   const blas_cxf* tau, blas_cxf* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cungqr)(m, n, k, a, lda, tau, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zungqr)(const blas_int* m, const blas_int* n, const blas_int* k, blas_cxd* a, const blas_int* lda,   const blas_cxd* tau, blas_cxd* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zungqr)(m, n, k, a, lda, tau, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n,  float* a, const blas_int* lda,  float* s,  float* u, const blas_int* ldu,  float* vt, const blas_int* ldvt,  float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n, double* a, const blas_int* lda, double* s, double* u, const blas_int* ldu, double* vt, const blas_int* ldvt, double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda,  float* s, blas_cxf* u, const blas_int* ldu, blas_cxf* vt, const blas_int* ldvt, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, double* s, blas_cxd* u, const blas_int* ldu, blas_cxd* vt, const blas_int* ldvt, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgesdd)(const char* jobz, const blas_int* m, const blas_int* n,  float* a, const blas_int* lda,  float* s,  float* u, const blas_int* ldu,  float* vt, const blas_int* ldvt,  float* work, const blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgesdd)(const char* jobz, const blas_int* m, const blas_int* n, double* a, const blas_int* lda, double* s, double* u, const blas_int* ldu, double* vt, const blas_int* ldvt, double* work, const blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgesdd)(const char* jobz, const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda,  float* s, blas_cxf* u, const blas_int* ldu, blas_cxf* vt, const blas_int* ldvt, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgesdd)(const char* jobz, const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, double* s, blas_cxd* u, const blas_int* ldu, blas_cxd* vt, const blas_int* ldvt, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgesv)(const blas_int* n, const blas_int* nrhs,    float* a, const blas_int* lda, blas_int* ipiv,    float* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dgesv)(const blas_int* n, const blas_int* nrhs,   double* a, const blas_int* lda, blas_int* ipiv,   double* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cgesv)(const blas_int* n, const blas_int* nrhs, blas_cxf* a, const blas_int* lda, blas_int* ipiv, blas_cxf* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zgesv)(const blas_int* n, const blas_int* nrhs, blas_cxd* a, const blas_int* lda, blas_int* ipiv, blas_cxd* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgesvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* nrhs,  float* a, const blas_int* lda,  float* af, const blas_int* ldaf, blas_int* ipiv, char* equed,  float* r,  float* c,  float* b, const blas_int* ldb,  float* x, const blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgesvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* nrhs, double* a, const blas_int* lda, double* af, const blas_int* ldaf, blas_int* ipiv, char* equed, double* r, double* c, double* b, const blas_int* ldb, double* x, const blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgesvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* nrhs, blas_cxf* a, const blas_int* lda, blas_cxf* af, const blas_int* ldaf, blas_int* ipiv, char* equed,  float* r,  float* c, blas_cxf* b, const blas_int* ldb, blas_cxf* x, const blas_int* ldx,  float* rcond,  float* ferr,  float* berr, blas_cxf* work,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgesvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* nrhs, blas_cxd* a, const blas_int* lda, blas_cxd* af, const blas_int* ldaf, blas_int* ipiv, char* equed, double* r, double* c, blas_cxd* b, const blas_int* ldb, blas_cxd* x, const blas_int* ldx, double* rcond, double* ferr, double* berr, blas_cxd* work, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sposv)(const char* uplo, const blas_int* n, const blas_int* nrhs,    float* a, const blas_int* lda,    float* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sposv)(uplo, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dposv)(const char* uplo, const blas_int* n, const blas_int* nrhs,   double* a, const blas_int* lda,   double* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dposv)(uplo, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cposv)(const char* uplo, const blas_int* n, const blas_int* nrhs, blas_cxf* a, const blas_int* lda, blas_cxf* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cposv)(uplo, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zposv)(const char* uplo, const blas_int* n, const blas_int* nrhs, blas_cxd* a, const blas_int* lda, blas_cxd* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zposv)(uplo, n, nrhs, a, lda, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sposvx)(const char* fact, const char* uplo, const blas_int* n, const blas_int* nrhs,  float* a, const blas_int* lda,  float* af, const blas_int* ldaf, char* equed,  float* s,  float* b, const blas_int* ldb,  float* x, const blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sposvx)(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dposvx)(const char* fact, const char* uplo, const blas_int* n, const blas_int* nrhs, double* a, const blas_int* lda, double* af, const blas_int* ldaf, char* equed, double* s, double* b, const blas_int* ldb, double* x, const blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dposvx)(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cposvx)(const char* fact, const char* uplo, const blas_int* n, const blas_int* nrhs, blas_cxf* a, const blas_int* lda, blas_cxf* af, const blas_int* ldaf, char* equed,  float* s, blas_cxf* b, const blas_int* ldb, blas_cxf* x, const blas_int* ldx,  float* rcond,  float* ferr,  float* berr, blas_cxf* work,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cposvx)(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zposvx)(const char* fact, const char* uplo, const blas_int* n, const blas_int* nrhs, blas_cxd* a, const blas_int* lda, blas_cxd* af, const blas_int* ldaf, char* equed, double* s, blas_cxd* b, const blas_int* ldb, blas_cxd* x, const blas_int* ldx, double* rcond, double* ferr, double* berr, blas_cxd* work, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zposvx)(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgels)(const char* trans, const blas_int* m, const blas_int* n, const blas_int* nrhs,    float* a, const blas_int* lda,    float* b, const blas_int* ldb,    float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgels)(const char* trans, const blas_int* m, const blas_int* n, const blas_int* nrhs,   double* a, const blas_int* lda,   double* b, const blas_int* ldb,   double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgels)(const char* trans, const blas_int* m, const blas_int* n, const blas_int* nrhs, blas_cxf* a, const blas_int* lda, blas_cxf* b, const blas_int* ldb, blas_cxf* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgels)(const char* trans, const blas_int* m, const blas_int* n, const blas_int* nrhs, blas_cxd* a, const blas_int* lda, blas_cxd* b, const blas_int* ldb, blas_cxd* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgelsd)(const blas_int* m, const blas_int* n, const blas_int* nrhs,  float* a, const blas_int* lda,  float* b, const blas_int* ldb,  float* S, const  float* rcond, blas_int* rank,  float* work, const blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgelsd)(const blas_int* m, const blas_int* n, const blas_int* nrhs, double* a, const blas_int* lda, double* b, const blas_int* ldb, double* S, const double* rcond, blas_int* rank, double* work, const blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgelsd)(const blas_int* m, const blas_int* n, const blas_int* nrhs, blas_cxf* a, const blas_int* lda, blas_cxf* b, const blas_int* ldb,  float* S, const  float* rcond, blas_int* rank, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, rwork, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgelsd)(const blas_int* m, const blas_int* n, const blas_int* nrhs, blas_cxd* a, const blas_int* lda, blas_cxd* b, const blas_int* ldb, double* S, const double* rcond, blas_int* rank, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, rwork, iwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_strtrs)(const char* uplo, const char* trans, const char* diag, const blas_int* n, const blas_int* nrhs, const    float* a, const blas_int* lda,    float* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_strtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dtrtrs)(const char* uplo, const char* trans, const char* diag, const blas_int* n, const blas_int* nrhs, const   double* a, const blas_int* lda,   double* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dtrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_ctrtrs)(const char* uplo, const char* trans, const char* diag, const blas_int* n, const blas_int* nrhs, const blas_cxf* a, const blas_int* lda, blas_cxf* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ctrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_ztrtrs)(const char* uplo, const char* trans, const char* diag, const blas_int* n, const blas_int* nrhs, const blas_cxd* a, const blas_int* lda, blas_cxd* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ztrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgbtrf)(const blas_int* m, const blas_int* n, const blas_int* kl, const blas_int* ku,    float* ab, const blas_int* ldab, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgbtrf)(m, n, kl, ku, ab, ldab, ipiv, info);
      }
    
    void arma_fortran_with_prefix(arma_dgbtrf)(const blas_int* m, const blas_int* n, const blas_int* kl, const blas_int* ku,   double* ab, const blas_int* ldab, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgbtrf)(m, n, kl, ku, ab, ldab, ipiv, info);
      }
    
    void arma_fortran_with_prefix(arma_cgbtrf)(const blas_int* m, const blas_int* n, const blas_int* kl, const blas_int* ku, blas_cxf* ab, const blas_int* ldab, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgbtrf)(m, n, kl, ku, ab, ldab, ipiv, info);
      }
    
    void arma_fortran_with_prefix(arma_zgbtrf)(const blas_int* m, const blas_int* n, const blas_int* kl, const blas_int* ku, blas_cxd* ab, const blas_int* ldab, blas_int* ipiv, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgbtrf)(m, n, kl, ku, ab, ldab, ipiv, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgbtrs)(const char* trans, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, const    float* ab, const blas_int* ldab, const blas_int* ipiv,    float* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgbtrs)(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dgbtrs)(const char* trans, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, const   double* ab, const blas_int* ldab, const blas_int* ipiv,   double* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgbtrs)(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cgbtrs)(const char* trans, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, const blas_cxf* ab, const blas_int* ldab, const blas_int* ipiv, blas_cxf* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgbtrs)(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zgbtrs)(const char* trans, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, const blas_cxd* ab, const blas_int* ldab, const blas_int* ipiv, blas_cxd* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgbtrs)(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgbsv)(const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs,    float* ab, const blas_int* ldab, blas_int* ipiv,    float* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dgbsv)(const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs,   double* ab, const blas_int* ldab, blas_int* ipiv,   double* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cgbsv)(const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, blas_cxf* ab, const blas_int* ldab, blas_int* ipiv, blas_cxf* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zgbsv)(const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, blas_cxd* ab, const blas_int* ldab, blas_int* ipiv, blas_cxd* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgbsvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs,  float* ab, const blas_int* ldab,  float* afb, const blas_int* ldafb, blas_int* ipiv, char* equed,  float* r,  float* c,  float* b, const blas_int* ldb,  float* x, const blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgbsvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, double* ab, const blas_int* ldab, double* afb, const blas_int* ldafb, blas_int* ipiv, char* equed, double* r, double* c, double* b, const blas_int* ldb, double* x, const blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgbsvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, blas_cxf* ab, const blas_int* ldab, blas_cxf* afb, const blas_int* ldafb, blas_int* ipiv, char* equed,  float* r,  float* c, blas_cxf* b, const blas_int* ldb, blas_cxf* x, const blas_int* ldx,  float* rcond,  float* ferr,  float* berr, blas_cxf* work,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgbsvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_int* nrhs, blas_cxd* ab, const blas_int* ldab, blas_cxd* afb, const blas_int* ldafb, blas_int* ipiv, char* equed, double* r, double* c, blas_cxd* b, const blas_int* ldb, blas_cxd* x, const blas_int* ldx, double* rcond, double* ferr, double* berr, blas_cxd* work, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgtsv)(const blas_int* n, const blas_int* nrhs,    float* dl,    float* d,    float* du,    float* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgtsv)(n, nrhs, dl, d, du, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_dgtsv)(const blas_int* n, const blas_int* nrhs,   double* dl,   double* d,   double* du,   double* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgtsv)(n, nrhs, dl, d, du, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_cgtsv)(const blas_int* n, const blas_int* nrhs, blas_cxf* dl, blas_cxf* d, blas_cxf* du, blas_cxf* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgtsv)(n, nrhs, dl, d, du, b, ldb, info);
      }
    
    void arma_fortran_with_prefix(arma_zgtsv)(const blas_int* n, const blas_int* nrhs, blas_cxd* dl, blas_cxd* d, blas_cxd* du, blas_cxd* b, const blas_int* ldb, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgtsv)(n, nrhs, dl, d, du, b, ldb, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgtsvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* nrhs, const  float* dl, const  float* d, const  float* du,  float* dlf,  float* df,  float* duf,  float* du2, blas_int* ipiv, const  float* b, const blas_int* ldb,  float* x, const blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgtsvx)(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgtsvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* nrhs, const double* dl, const double* d, const double* du, double* dlf, double* df, double* duf, double* du2, blas_int* ipiv, const double* b, const blas_int* ldb, double* x, const blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgtsvx)(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgtsvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* nrhs, const blas_cxf* dl, const blas_cxf* d, const blas_cxf* du, blas_cxf* dlf, blas_cxf* df, blas_cxf* duf, blas_cxf* du2, blas_int* ipiv, const blas_cxf* b, const blas_int* ldb, blas_cxf* x, const blas_int* ldx,  float* rcond,  float* ferr,  float* berr, blas_cxf* work,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgtsvx)(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgtsvx)(const char* fact, const char* trans, const blas_int* n, const blas_int* nrhs, const blas_cxd* dl, const blas_cxd* d, const blas_cxd* du, blas_cxd* dlf, blas_cxd* df, blas_cxd* duf, blas_cxd* du2, blas_int* ipiv, const blas_cxd* b, const blas_int* ldb, blas_cxd* x, const blas_int* ldx, double* rcond, double* ferr, double* berr, blas_cxd* work, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgtsvx)(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgees)(const char* jobvs, const char* sort, fn_select_s2 select, const blas_int* n,  float* a, const blas_int* lda, blas_int* sdim,  float* wr,  float* wi,  float* vs, const blas_int* ldvs,  float* work, const blas_int* lwork, blas_int* bwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
      }
      
    void arma_fortran_with_prefix(arma_dgees)(const char* jobvs, const char* sort, fn_select_d2 select, const blas_int* n, double* a, const blas_int* lda, blas_int* sdim, double* wr, double* wi, double* vs, const blas_int* ldvs, double* work, const blas_int* lwork, blas_int* bwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgees)(const char* jobvs, const char* sort, fn_select_c1 select, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* sdim, blas_cxf* w, blas_cxf* vs, const blas_int* ldvs, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* bwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgees)(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgees)(const char* jobvs, const char* sort, fn_select_z1 select, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* sdim, blas_cxd* w, blas_cxd* vs, const blas_int* ldvs, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* bwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgees)(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_strsyl)(const char* transa, const char* transb, const blas_int* isgn, const blas_int* m, const blas_int* n, const    float* a, const blas_int* lda, const    float* b, const blas_int* ldb,    float* c, const blas_int* ldc,  float* scale, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_strsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info);
      }
    
    void arma_fortran_with_prefix(arma_dtrsyl)(const char* transa, const char* transb, const blas_int* isgn, const blas_int* m, const blas_int* n, const   double* a, const blas_int* lda, const   double* b, const blas_int* ldb,   double* c, const blas_int* ldc, double* scale, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dtrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info);
      }
    
    void arma_fortran_with_prefix(arma_ctrsyl)(const char* transa, const char* transb, const blas_int* isgn, const blas_int* m, const blas_int* n, const blas_cxf* a, const blas_int* lda, const blas_cxf* b, const blas_int* ldb, blas_cxf* c, const blas_int* ldc,  float* scale, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ctrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info);
      }
    
    void arma_fortran_with_prefix(arma_ztrsyl)(const char* transa, const char* transb, const blas_int* isgn, const blas_int* m, const blas_int* n, const blas_cxd* a, const blas_int* lda, const blas_cxd* b, const blas_int* ldb, blas_cxd* c, const blas_int* ldc, double* scale, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ztrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgges)(const char* jobvsl, const char* jobvsr, const char* sort, fn_select_s3 selctg, const blas_int* n,  float* a, const blas_int* lda,  float* b, const blas_int* ldb, blas_int* sdim,  float* alphar,  float* alphai,  float* beta,  float* vsl, const blas_int* ldvsl,  float* vsr, const blas_int* ldvsr,  float* work, const blas_int* lwork, blas_int* bwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgges)(const char* jobvsl, const char* jobvsr, const char* sort, fn_select_d3 selctg, const blas_int* n, double* a, const blas_int* lda, double* b, const blas_int* ldb, blas_int* sdim, double* alphar, double* alphai, double* beta, double* vsl, const blas_int* ldvsl, double* vsr, const blas_int* ldvsr, double* work, const blas_int* lwork, blas_int* bwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgges)(const char* jobvsl, const char* jobvsr, const char* sort, fn_select_c2 selctg, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_cxf* b, const blas_int* ldb, blas_int* sdim, blas_cxf* alpha, blas_cxf* beta, blas_cxf* vsl, const blas_int* ldvsl, blas_cxf* vsr, const blas_int* ldvsr, blas_cxf* work, const blas_int* lwork,  float* rwork, blas_int* bwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alpha, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, rwork, bwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgges)(const char* jobvsl, const char* jobvsr, const char* sort, fn_select_z2 selctg, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_cxd* b, const blas_int* ldb, blas_int* sdim, blas_cxd* alpha, blas_cxd* beta, blas_cxd* vsl, const blas_int* ldvsl, blas_cxd* vsr, const blas_int* ldvsr, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* bwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alpha, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, rwork, bwork, info);
      }
    
    
    
    float  arma_fortran_with_prefix(arma_slange)(const char* norm, const blas_int* m, const blas_int* n, const    float* a, const blas_int* lda,  float* work)
      {
      return arma_fortran_sans_prefix(arma_slange)(norm, m, n, a, lda, work);
      }
    
    double arma_fortran_with_prefix(arma_dlange)(const char* norm, const blas_int* m, const blas_int* n, const   double* a, const blas_int* lda, double* work)
      {
      return arma_fortran_sans_prefix(arma_dlange)(norm, m, n, a, lda, work);
      }
    
    float  arma_fortran_with_prefix(arma_clange)(const char* norm, const blas_int* m, const blas_int* n, const blas_cxf* a, const blas_int* lda,  float* work)
      {
      return arma_fortran_sans_prefix(arma_clange)(norm, m, n, a, lda, work);
      }
    
    double arma_fortran_with_prefix(arma_zlange)(const char* norm, const blas_int* m, const blas_int* n, const blas_cxd* a, const blas_int* lda, double* work)
      {
      return arma_fortran_sans_prefix(arma_zlange)(norm, m, n, a, lda, work);
      }
    
    
    
    float  arma_fortran_with_prefix(arma_slansy)(const char* norm, const char* uplo, const blas_int* n, const    float* a, const blas_int* lda,  float* work)
      {
      return arma_fortran_sans_prefix(arma_slansy)(norm, uplo, n, a, lda, work);
      }
    
    double arma_fortran_with_prefix(arma_dlansy)(const char* norm, const char* uplo, const blas_int* n, const   double* a, const blas_int* lda, double* work)
      {
      return arma_fortran_sans_prefix(arma_dlansy)(norm, uplo, n, a, lda, work);
      }
    
    float  arma_fortran_with_prefix(arma_clansy)(const char* norm, const char* uplo, const blas_int* n, const blas_cxf* a, const blas_int* lda,  float* work)
      {
      return arma_fortran_sans_prefix(arma_clansy)(norm, uplo, n, a, lda, work);
      }
    
    double arma_fortran_with_prefix(arma_zlansy)(const char* norm, const char* uplo, const blas_int* n, const blas_cxd* a, const blas_int* lda, double* work)
      {
      return arma_fortran_sans_prefix(arma_zlansy)(norm, uplo, n, a, lda, work);
      }
    
    
    
    float  arma_fortran_with_prefix(arma_clanhe)(const char* norm, const char* uplo, const blas_int* n, const blas_cxf* a, const blas_int* lda,  float* work)
      {
      return arma_fortran_sans_prefix(arma_clanhe)(norm, uplo, n, a, lda, work);
      }
    
    double arma_fortran_with_prefix(arma_zlanhe)(const char* norm, const char* uplo, const blas_int* n, const blas_cxd* a, const blas_int* lda, double* work)
      {
      return arma_fortran_sans_prefix(arma_zlanhe)(norm, uplo, n, a, lda, work);
      }
    
    
    
    float  arma_fortran_with_prefix(arma_slangb)(const char* norm, const blas_int* n, const blas_int* kl, const blas_int* ku, const    float* ab, const blas_int* ldab,  float* work)
      {
      return arma_fortran_sans_prefix(arma_slangb)(norm, n, kl, ku, ab, ldab, work);
      }
    
    double arma_fortran_with_prefix(arma_dlangb)(const char* norm, const blas_int* n, const blas_int* kl, const blas_int* ku, const   double* ab, const blas_int* ldab, double* work)
      {
      return arma_fortran_sans_prefix(arma_dlangb)(norm, n, kl, ku, ab, ldab, work);
      }
    
    float  arma_fortran_with_prefix(arma_clangb)(const char* norm, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_cxf* ab, const blas_int* ldab,  float* work)
      {
      return arma_fortran_sans_prefix(arma_clangb)(norm, n, kl, ku, ab, ldab, work);
      }
    
    double arma_fortran_with_prefix(arma_zlangb)(const char* norm, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_cxd* ab, const blas_int* ldab, double* work)
      {
      return arma_fortran_sans_prefix(arma_zlangb)(norm, n, kl, ku, ab, ldab, work);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgecon)(const char* norm, const blas_int* n, const  float* a, const blas_int* lda, const  float* anorm,  float* rcond,  float* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgecon)(const char* norm, const blas_int* n, const double* a, const blas_int* lda, const double* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgecon)(const char* norm, const blas_int* n, const blas_cxf* a, const blas_int* lda, const  float* anorm,  float* rcond, blas_cxf* work,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgecon)(norm, n, a, lda, anorm, rcond, work, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgecon)(const char* norm, const blas_int* n, const blas_cxd* a, const blas_int* lda, const double* anorm, double* rcond, blas_cxd* work, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgecon)(norm, n, a, lda, anorm, rcond, work, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_spocon)(const char* uplo, const blas_int* n, const  float* a, const blas_int* lda, const  float* anorm,  float* rcond,  float* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_spocon)(uplo, n, a, lda, anorm, rcond, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dpocon)(const char* uplo, const blas_int* n, const double* a, const blas_int* lda, const double* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dpocon)(uplo, n, a, lda, anorm, rcond, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cpocon)(const char* uplo, const blas_int* n, const blas_cxf* a, const blas_int* lda, const  float* anorm,  float* rcond, blas_cxf* work,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cpocon)(uplo, n, a, lda, anorm, rcond, work, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zpocon)(const char* uplo, const blas_int* n, const blas_cxd* a, const blas_int* lda, const double* anorm, double* rcond, blas_cxd* work, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zpocon)(uplo, n, a, lda, anorm, rcond, work, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_strcon)(const char* norm, const char* uplo, const char* diag, const blas_int* n, const  float* a, const blas_int* lda,  float* rcond,  float* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_strcon)(norm, uplo, diag, n, a, lda, rcond, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dtrcon)(const char* norm, const char* uplo, const char* diag, const blas_int* n, const double* a, const blas_int* lda, double* rcond, double* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dtrcon)(norm, uplo, diag, n, a, lda, rcond, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_ctrcon)(const char* norm, const char* uplo, const char* diag, const blas_int* n, const blas_cxf* a, const blas_int* lda,  float* rcond, blas_cxf* work,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ctrcon)(norm, uplo, diag, n, a, lda, rcond, work, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_ztrcon)(const char* norm, const char* uplo, const char* diag, const blas_int* n, const blas_cxd* a, const blas_int* lda, double* rcond, blas_cxd* work, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ztrcon)(norm, uplo, diag, n, a, lda, rcond, work, rwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgbcon)(const char* norm, const blas_int* n, const blas_int* kl, const blas_int* ku, const  float* ab, const blas_int* ldab, const blas_int* ipiv, const  float* anorm,  float* rcond,  float* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgbcon)(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgbcon)(const char* norm, const blas_int* n, const blas_int* kl, const blas_int* ku, const double* ab, const blas_int* ldab, const blas_int* ipiv, const double* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgbcon)(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, iwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgbcon)(const char* norm, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_cxf* ab, const blas_int* ldab, const blas_int* ipiv, const  float* anorm,  float* rcond, blas_cxf* work,  float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgbcon)(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, rwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgbcon)(const char* norm, const blas_int* n, const blas_int* kl, const blas_int* ku, const blas_cxd* ab, const blas_int* ldab, const blas_int* ipiv, const double* anorm, double* rcond, blas_cxd* work, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgbcon)(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, rwork, info);
      }
    
    
    
    blas_int arma_fortran_with_prefix(arma_ilaenv)(const blas_int* ispec, const char* name, const char* opts, const blas_int* n1, const blas_int* n2, const blas_int* n3, const blas_int* n4)
      {
      return arma_fortran_sans_prefix(arma_ilaenv)(ispec, name, opts, n1, n2, n3, n4);
      }
    
    
    
    void arma_fortran_with_prefix(arma_slahqr)(const blas_int* wantt, const blas_int* wantz, const blas_int* n, const blas_int* ilo, const blas_int* ihi,  float* h, const blas_int* ldh,  float* wr,  float* wi, const blas_int* iloz, const blas_int* ihiz,  float* z, const blas_int* ldz, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_slahqr)(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
      }
    
    void arma_fortran_with_prefix(arma_dlahqr)(const blas_int* wantt, const blas_int* wantz, const blas_int* n, const blas_int* ilo, const blas_int* ihi, double* h, const blas_int* ldh, double* wr, double* wi, const blas_int* iloz, const blas_int* ihiz, double* z, const blas_int* ldz, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dlahqr)(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sstedc)(const char* compz, const blas_int* n,  float* d,  float* e,  float* z, const blas_int* ldz,  float* work, const blas_int* lwork, blas_int* iwork, const blas_int* liwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sstedc)(compz, n, d, e, z, ldz, work, lwork, iwork, liwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dstedc)(const char* compz, const blas_int* n, double* d, double* e, double* z, const blas_int* ldz, double* work, const blas_int* lwork, blas_int* iwork, const blas_int* liwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dstedc)(compz, n, d, e, z, ldz, work, lwork, iwork, liwork, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_strevc)(const char* side, const char* howmny, blas_int* select, const blas_int* n, const  float* t, const blas_int* ldt,  float* vl, const blas_int* ldvl,  float* vr, const blas_int* ldvr, const blas_int* mm, blas_int* m,  float* work, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_strevc)(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info);
      }
    
    void arma_fortran_with_prefix(arma_dtrevc)(const char* side, const char* howmny, blas_int* select, const blas_int* n, const double* t, const blas_int* ldt, double* vl, const blas_int* ldvl, double* vr, const blas_int* ldvr, const blas_int* mm, blas_int* m, double* work, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dtrevc)(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info);
      }
    
    
    
    void arma_fortran_with_prefix(arma_slarnv)(const blas_int* idist, blas_int* iseed, const blas_int* n,  float* x)
      {
      arma_fortran_sans_prefix(arma_slarnv)(idist, iseed, n, x);
      }
    
    void arma_fortran_with_prefix(arma_dlarnv)(const blas_int* idist, blas_int* iseed, const blas_int* n, double* x)
      {
      arma_fortran_sans_prefix(arma_dlarnv)(idist, iseed, n, x);
      }
    
    
    
    void arma_fortran_with_prefix(arma_sgehrd)(const blas_int* n, const blas_int* ilo, const blas_int* ihi,    float* a, const blas_int* lda,    float* tao,    float* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sgehrd)(n, ilo, ihi, a, lda, tao, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_dgehrd)(const blas_int* n, const blas_int* ilo, const blas_int* ihi,   double* a, const blas_int* lda,   double* tao,   double* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dgehrd)(n, ilo, ihi, a, lda, tao, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_cgehrd)(const blas_int* n, const blas_int* ilo, const blas_int* ihi, blas_cxf* a, const blas_int* lda, blas_cxf* tao, blas_cxf* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cgehrd)(n, ilo, ihi, a, lda, tao, work, lwork, info);
      }
    
    void arma_fortran_with_prefix(arma_zgehrd)(const blas_int* n, const blas_int* ilo, const blas_int* ihi, blas_cxd* a, const blas_int* lda, blas_cxd* tao, blas_cxd* work, const blas_int* lwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zgehrd)(n, ilo, ihi, a, lda, tao, work, lwork, info);
      }
    
  #endif
  
  
  
  #if defined(ARMA_USE_ATLAS)
    
    float wrapper_cblas_sasum(const int N, const float  *X, const int incX)
      {
      return      cblas_sasum(N, X, incX);
      }
    
    double wrapper_cblas_dasum(const int N, const double *X, const int incX)
      {
      return       cblas_dasum(N, X, incX);
      }
    
    
    
    float wrapper_cblas_snrm2(const int N, const float  *X, const int incX)
      {
      return      cblas_snrm2(N, X, incX);
      }
    
    double wrapper_cblas_dnrm2(const int N, const double *X, const int incX)
      {
      return       cblas_dnrm2(N, X, incX);
      }
    
    
    
    float wrapper_cblas_sdot(const int N, const float  *X, const int incX, const float  *Y, const int incY)
      {
      return      cblas_sdot(N, X, incX, Y, incY);
      }
    
    double wrapper_cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY)
      {
      return       cblas_ddot(N, X, incX, Y, incY);
      }
    
    void wrapper_cblas_cdotu_sub(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotu)
      {
                 cblas_cdotu_sub(N, X, incX, Y, incY, dotu);
      }
    
    void wrapper_cblas_zdotu_sub(const int N, const void *X, const int incX, const void *Y, const int incY, void *dotu)
      {
                 cblas_zdotu_sub(N, X, incX, Y, incY, dotu);
      }
    
    
    
    void wrapper_cblas_sgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha,
                             const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY)
      {
                 cblas_sgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
      }
    
    void wrapper_cblas_dgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha,
                             const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY)
      {
                 cblas_dgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
      }
    
    void wrapper_cblas_cgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const void *alpha,
                             const void *A, const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY)
      {
                 cblas_cgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
      }
    
    void wrapper_cblas_zgemv(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const int M, const int N, const void *alpha,
                             const void *A, const int lda, const void *X, const int incX, const void *beta, void *Y, const int incY)
      {
                 cblas_zgemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
      }
    
    
    
    void wrapper_cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                             const int M, const int N, const int K, const float alpha,
                             const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
      {
                 cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      }
    
    void wrapper_cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                             const int M, const int N, const int K, const double alpha,
                             const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
      {
                 cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      }
    
    void wrapper_cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                             const int M, const int N, const int K, const void *alpha,
                             const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc)
      {
                 cblas_cgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      }
    
    void wrapper_cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                             const int M, const int N, const int K, const void *alpha,
                             const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc)
      {
                 cblas_zgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      }
    
    
    
    void wrapper_cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                             const int N, const int K, const float alpha,
                             const float *A, const int lda, const float beta, float *C, const int ldc)
      {
                 cblas_ssyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
      }
    
    void wrapper_cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                             const int N, const int K, const double alpha,
                             const double *A, const int lda, const double beta, double *C, const int ldc)
      {
                 cblas_dsyrk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
      }
    
    
    
    void wrapper_cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                             const int N, const int K, const float alpha,
                             const void *A, const int lda, const float beta, void *C, const int ldc)
      {
                 cblas_cherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
      }
    
    void wrapper_cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans,
                             const int N, const int K, const double alpha,
                             const void *A, const int lda, const double beta, void *C, const int ldc)
      {
                 cblas_zherk(Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
      }
    
    
    
    int wrapper_clapack_sgetrf(const enum CBLAS_ORDER Order, const int M, const int N, float  *A, const int lda, int *ipiv)
      {
      return    clapack_sgetrf(Order, M, N, A, lda, ipiv);
      }
    
    int wrapper_clapack_dgetrf(const enum CBLAS_ORDER Order, const int M, const int N, double *A, const int lda, int *ipiv)
      {
      return    clapack_dgetrf(Order, M, N, A, lda, ipiv);
      }
    
    int wrapper_clapack_cgetrf(const enum CBLAS_ORDER Order, const int M, const int N, void   *A, const int lda, int *ipiv)
      {
      return    clapack_cgetrf(Order, M, N, A, lda, ipiv);
      }
    
    int wrapper_clapack_zgetrf(const enum CBLAS_ORDER Order, const int M, const int N, void   *A, const int lda, int *ipiv)
      {
      return    clapack_zgetrf(Order, M, N, A, lda, ipiv);
      }
    
    
    
    int wrapper_clapack_sgetri(const enum CBLAS_ORDER Order, const int N, float  *A, const int lda, const int *ipiv)
      {
      return    clapack_sgetri(Order, N, A, lda, ipiv);
      }
    
    int wrapper_clapack_dgetri(const enum CBLAS_ORDER Order, const int N, double *A, const int lda, const int *ipiv)
      {
      return    clapack_dgetri(Order, N, A, lda, ipiv);
      }
    
    int wrapper_clapack_cgetri(const enum CBLAS_ORDER Order, const int N, void   *A, const int lda, const int *ipiv)
      {
      return    clapack_cgetri(Order, N, A, lda, ipiv);
      }
    
    int wrapper_clapack_zgetri(const enum CBLAS_ORDER Order, const int N, void   *A, const int lda, const int *ipiv)
      {
      return    clapack_zgetri(Order, N, A, lda, ipiv);
      }
    
    
    
    int wrapper_clapack_sgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS, float  *A, const int lda, int *ipiv, float  *B, const int ldb)
      {
      return    clapack_sgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
      }
    
    int wrapper_clapack_dgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS, double *A, const int lda, int *ipiv, double *B, const int ldb)
      {
      return    clapack_dgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
      }
    
    int wrapper_clapack_cgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS, void   *A, const int lda, int *ipiv, void   *B, const int ldb)
      {
      return    clapack_cgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
      }
    
    int wrapper_clapack_zgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS, void   *A, const int lda, int *ipiv, void   *B, const int ldb)
      {
      return    clapack_zgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
      }
    
    
    
    int wrapper_clapack_spotrf(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,  float *A, const int lda)
      {
      return    clapack_spotrf(Order, Uplo, N, A, lda);
      }
    
    int wrapper_clapack_dpotrf(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, double *A, const int lda)
      {
      return    clapack_dpotrf(Order, Uplo, N, A, lda);
      }
    
    int wrapper_clapack_cpotrf(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,   void *A, const int lda)
      {
      return    clapack_cpotrf(Order, Uplo, N, A, lda);
      }
    
    int wrapper_clapack_zpotrf(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,   void *A, const int lda)
      {
      return    clapack_zpotrf(Order, Uplo, N, A, lda);
      }
    
    
    
    int wrapper_clapack_spotri(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,  float *A, const int lda)
      {
      return    clapack_spotri(Order, Uplo, N, A, lda);
      }
    
    int wrapper_clapack_dpotri(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, double *A, const int lda)
      {
      return    clapack_dpotri(Order, Uplo, N, A, lda);
      }
    
    int wrapper_clapack_cpotri(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,   void *A, const int lda)
      {
      return    clapack_cpotri(Order, Uplo, N, A, lda);
      }
    
    int wrapper_clapack_zpotri(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N,   void *A, const int lda)
      {
      return    clapack_zpotri(Order, Uplo, N, A, lda);
      }
    
    
    
    int wrapper_clapack_sposv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const int NRHS,  float *A, const int lda,  float *B, const int ldb)
      {
      return    clapack_sposv(Order, Uplo, N, NRHS, A, lda, B, ldb);
      }
    
    int wrapper_clapack_dposv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const int NRHS, double *A, const int lda, double *B, const int ldb)
      {
      return    clapack_dposv(Order, Uplo, N, NRHS, A, lda, B, ldb);
      }
    
    int wrapper_clapack_cposv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const int NRHS,   void *A, const int lda,   void *B, const int ldb)
      {
      return    clapack_cposv(Order, Uplo, N, NRHS, A, lda, B, ldb);
      }
    
    int wrapper_clapack_zposv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const int N, const int NRHS,   void *A, const int lda,   void *B, const int ldb)
      {
      return    clapack_zposv(Order, Uplo, N, NRHS, A, lda, B, ldb);
      }
    
    
  #endif



  #if defined(ARMA_USE_ARPACK)

    void arma_fortran_with_prefix(arma_snaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, float* resid, blas_int* ncv, float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, float* workd, float* workl, blas_int* lworkl, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_snaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info);
      }

    void arma_fortran_with_prefix(arma_dnaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dnaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info);
      }

    void arma_fortran_with_prefix(arma_cnaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, void* resid, blas_int* ncv, void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, void* workd, void* workl, blas_int* lworkl, float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cnaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info);
      }

    void arma_fortran_with_prefix(arma_znaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, void* resid, blas_int* ncv, void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, void* workd, void* workl, blas_int* lworkl, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_znaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info);
      }


    void arma_fortran_with_prefix(arma_sneupd)(blas_int* rvec, char* howmny, blas_int* select, float* dr, float* di, float* z, blas_int* ldz, float* sigmar, float* sigmai, float* workev, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, float* resid, blas_int* ncv, float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, float* workd, float* workl, blas_int* lworkl, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sneupd)(rvec, howmny, select, dr, di, z, ldz, sigmar, sigmai, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info);
      }

    void arma_fortran_with_prefix(arma_dneupd)(blas_int* rvec, char* howmny, blas_int* select, double* dr, double* di, double* z, blas_int* ldz, double* sigmar, double* sigmai, double* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dneupd)(rvec, howmny, select, dr, di, z, ldz, sigmar, sigmai, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info);
      }

    void arma_fortran_with_prefix(arma_cneupd)(blas_int* rvec, char* howmny, blas_int* select, void* d, void* z, blas_int* ldz, void* sigma, void* workev, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, void* resid, blas_int* ncv, void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, void* workd, void* workl, blas_int* lworkl, float* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_cneupd)(rvec, howmny, select, d, z, ldz, sigma, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info);
      }

    void arma_fortran_with_prefix(arma_zneupd)(blas_int* rvec, char* howmny, blas_int* select, void* d, void* z, blas_int* ldz, void* sigma, void* workev, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, void* resid, blas_int* ncv, void* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, void* workd, void* workl, blas_int* lworkl, double* rwork, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_zneupd)(rvec, howmny, select, d, z, ldz, sigma, workev, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, info);
      }


    void arma_fortran_with_prefix(arma_ssaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, float* resid, blas_int* ncv, float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, float* workd, float* workl, blas_int* lworkl, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_ssaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info);
      }

    void arma_fortran_with_prefix(arma_dsaupd)(blas_int* ido, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dsaupd)(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info);
      }


    void arma_fortran_with_prefix(arma_sseupd)(blas_int* rvec, char* howmny, blas_int* select, float* d, float* z, blas_int* ldz, float* sigma, char* bmat, blas_int* n, char* which, blas_int* nev, float* tol, float* resid, blas_int* ncv, float* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, float* workd, float* workl, blas_int* lworkl, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_sseupd)(rvec, howmny, select, d, z, ldz, sigma, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info);
      }

    void arma_fortran_with_prefix(arma_dseupd)(blas_int* rvec, char* howmny, blas_int* select, double* d, double* z, blas_int* ldz, double* sigma, char* bmat, blas_int* n, char* which, blas_int* nev, double* tol, double* resid, blas_int* ncv, double* v, blas_int* ldv, blas_int* iparam, blas_int* ipntr, double* workd, double* workl, blas_int* lworkl, blas_int* info)
      {
      arma_fortran_sans_prefix(arma_dseupd)(rvec, howmny, select, d, z, ldz, sigma, bmat, n, which, nev, tol, resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, info);
      }

  #endif
  
  
  
  #if defined(ARMA_USE_SUPERLU)
    
    void wrapper_sgssv(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, superlu::SuperMatrix* e, superlu::SuperMatrix* f, superlu::SuperMatrix* g, superlu::SuperLUStat_t* h, int* i)
      {
      sgssv(a,b,c,d,e,f,g,h,i);
      }
    
    void wrapper_dgssv(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, superlu::SuperMatrix* e, superlu::SuperMatrix* f, superlu::SuperMatrix* g, superlu::SuperLUStat_t* h, int* i)
      {
      dgssv(a,b,c,d,e,f,g,h,i);
      }
    
    void wrapper_cgssv(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, superlu::SuperMatrix* e, superlu::SuperMatrix* f, superlu::SuperMatrix* g, superlu::SuperLUStat_t* h, int* i)
      {
      cgssv(a,b,c,d,e,f,g,h,i);
      }
    
    void wrapper_zgssv(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, superlu::SuperMatrix* e, superlu::SuperMatrix* f, superlu::SuperMatrix* g, superlu::SuperLUStat_t* h, int* i)
      {
      zgssv(a,b,c,d,e,f,g,h,i);
      }
    
    
    
    
    void wrapper_sgssvx(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, int* e, char* f,  float* g,  float* h, superlu::SuperMatrix* i, superlu::SuperMatrix* j, void* k, int l, superlu::SuperMatrix* m, superlu::SuperMatrix* n,  float* o,  float* p,  float* q,  float* r, superlu::GlobalLU_t* s, superlu::mem_usage_t* t, superlu::SuperLUStat_t* u, int* v)
      {
      sgssvx(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v);
      }
    
    void wrapper_dgssvx(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, int* e, char* f, double* g, double* h, superlu::SuperMatrix* i, superlu::SuperMatrix* j, void* k, int l, superlu::SuperMatrix* m, superlu::SuperMatrix* n, double* o, double* p, double* q, double* r, superlu::GlobalLU_t* s, superlu::mem_usage_t* t, superlu::SuperLUStat_t* u, int* v)
      {
      dgssvx(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v);
      }
    
    void wrapper_cgssvx(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, int* e, char* f,  float* g,  float* h, superlu::SuperMatrix* i, superlu::SuperMatrix* j, void* k, int l, superlu::SuperMatrix* m, superlu::SuperMatrix* n,  float* o,  float* p,  float* q,  float* r, superlu::GlobalLU_t* s, superlu::mem_usage_t* t, superlu::SuperLUStat_t* u, int* v)
      {
      cgssvx(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v);
      }
    
    void wrapper_zgssvx(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, int* e, char* f, double* g, double* h, superlu::SuperMatrix* i, superlu::SuperMatrix* j, void* k, int l, superlu::SuperMatrix* m, superlu::SuperMatrix* n, double* o, double* p, double* q, double* r, superlu::GlobalLU_t* s, superlu::mem_usage_t* t, superlu::SuperLUStat_t* u, int* v)
      {
      zgssvx(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v);
      }
    
    
    
    
    void wrapper_sgstrf(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int d, int e, int* f, void* g, int h, int* i, int* j, superlu::SuperMatrix* k, superlu::SuperMatrix* l, superlu::GlobalLU_t* m, superlu::SuperLUStat_t* n, int* o)
      {
      sgstrf(a, b, d, e, f, g, h, i, j, k, l, m, n, o);
      }
    
    void wrapper_dgstrf(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int d, int e, int* f, void* g, int h, int* i, int* j, superlu::SuperMatrix* k, superlu::SuperMatrix* l, superlu::GlobalLU_t* m, superlu::SuperLUStat_t* n, int* o)
      {
      dgstrf(a, b, d, e, f, g, h, i, j, k, l, m, n, o);
      }
    
    void wrapper_cgstrf(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int d, int e, int* f, void* g, int h, int* i, int* j, superlu::SuperMatrix* k, superlu::SuperMatrix* l, superlu::GlobalLU_t* m, superlu::SuperLUStat_t* n, int* o)
      {
      cgstrf(a, b, d, e, f, g, h, i, j, k, l, m, n, o);
      }
    
    void wrapper_zgstrf(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int d, int e, int* f, void* g, int h, int* i, int* j, superlu::SuperMatrix* k, superlu::SuperMatrix* l, superlu::GlobalLU_t* m, superlu::SuperLUStat_t* n, int* o)
      {
      zgstrf(a, b, d, e, f, g, h, i, j, k, l, m, n, o);
      }
    
    
    
    
    void wrapper_sgstrs(superlu::trans_t a, superlu::SuperMatrix* b, superlu::SuperMatrix* c, int* d, int* e, superlu::SuperMatrix* f, superlu::SuperLUStat_t* g, int* h)
      {
      sgstrs(a, b, c, d, e, f, g, h);
      }
    
    void wrapper_dgstrs(superlu::trans_t a, superlu::SuperMatrix* b, superlu::SuperMatrix* c, int* d, int* e, superlu::SuperMatrix* f, superlu::SuperLUStat_t* g, int* h)
      {
      dgstrs(a, b, c, d, e, f, g, h);
      }
    
    void wrapper_cgstrs(superlu::trans_t a, superlu::SuperMatrix* b, superlu::SuperMatrix* c, int* d, int* e, superlu::SuperMatrix* f, superlu::SuperLUStat_t* g, int* h)
      {
      cgstrs(a, b, c, d, e, f, g, h);
      }
    
    void wrapper_zgstrs(superlu::trans_t a, superlu::SuperMatrix* b, superlu::SuperMatrix* c, int* d, int* e, superlu::SuperMatrix* f, superlu::SuperLUStat_t* g, int* h)
      {
      zgstrs(a, b, c, d, e, f, g, h);
      }
    
    
    
    
    float  wrapper_slangs(char* norm, superlu::SuperMatrix* A)
      {
      return slangs(norm, A);
      }
    
    double wrapper_dlangs(char* norm, superlu::SuperMatrix* A)
      {
      return dlangs(norm, A);
      }
    
    float  wrapper_clangs(char* norm, superlu::SuperMatrix* A)
      {
      return clangs(norm, A);
      }
    
    double wrapper_zlangs(char* norm, superlu::SuperMatrix* A)
      {
      return zlangs(norm, A);
      }
    
    
    
    void wrapper_sgscon(char* norm, superlu::SuperMatrix* L, superlu::SuperMatrix* U,  float anorm,  float* rcond, superlu::SuperLUStat_t* stat, int* info)
      {
      sgscon(norm, L, U, anorm, rcond, stat, info);
      }
    
    void wrapper_dgscon(char* norm, superlu::SuperMatrix* L, superlu::SuperMatrix* U, double anorm, double* rcond, superlu::SuperLUStat_t* stat, int* info)
      {
      dgscon(norm, L, U, anorm, rcond, stat, info);
      }
    
    void wrapper_cgscon(char* norm, superlu::SuperMatrix* L, superlu::SuperMatrix* U,  float anorm,  float* rcond, superlu::SuperLUStat_t* stat, int* info)
      {
      cgscon(norm, L, U, anorm, rcond, stat, info);
      }
    
    void wrapper_zgscon(char* norm, superlu::SuperMatrix* L, superlu::SuperMatrix* U, double anorm, double* rcond, superlu::SuperLUStat_t* stat, int* info)
      {
      zgscon(norm, L, U, anorm, rcond, stat, info);
      }
    
    
    
    void wrapper_StatInit(superlu::SuperLUStat_t* a)
      {
      StatInit(a);
      }

    void wrapper_StatFree(superlu::SuperLUStat_t* a)
      {
      StatFree(a);
      }
      
    void wrapper_set_default_options(superlu::superlu_options_t* a)
      {
      set_default_options(a);
      }

    void wrapper_get_perm_c(int a, superlu::SuperMatrix* b, int* c)
      {
      get_perm_c(a, b, c);
      }
    
    int wrapper_sp_ienv(int a)
      {
      return sp_ienv(a);
      }
    
    void wrapper_sp_preorder(superlu::superlu_options_t* a, superlu::SuperMatrix* b, int* c, int* d, superlu::SuperMatrix* e)
      {
      sp_preorder(a, b, c, d, e);
      }
    
    void wrapper_Destroy_SuperNode_Matrix(superlu::SuperMatrix* a)
      {
      Destroy_SuperNode_Matrix(a);
      }

    void wrapper_Destroy_CompCol_Matrix(superlu::SuperMatrix* a)
      {
      Destroy_CompCol_Matrix(a);
      }

    void wrapper_Destroy_CompCol_Permuted(superlu::SuperMatrix* a)
      {
      Destroy_CompCol_Permuted(a);
      }

    void wrapper_Destroy_SuperMatrix_Store(superlu::SuperMatrix* a)
      {
      Destroy_SuperMatrix_Store(a);
      }
    
    void* wrapper_superlu_malloc(size_t a)
      {
      return superlu_malloc(a);
      }
    
    void wrapper_superlu_free(void* a)
      {
      superlu_free(a);
      }
    
  #endif
  
  
  
  #if defined(ARMA_USE_HDF5_ALT)
  
    hid_t arma_H5Tcopy(hid_t dtype_id)
      {
      return H5Tcopy(dtype_id);
      }
    
    hid_t arma_H5Tcreate(H5T_class_t cl, size_t size)
      {
      return H5Tcreate(cl, size);
      }
    
    herr_t arma_H5Tinsert(hid_t dtype_id, const char* name, size_t offset, hid_t field_id)
      {
      return H5Tinsert(dtype_id, name, offset, field_id);
      }
    
    htri_t arma_H5Tequal(hid_t dtype_id1, hid_t dtype_id2)
      {
      return H5Tequal(dtype_id1, dtype_id2);
      }
    
    herr_t arma_H5Tclose(hid_t dtype_id)
      {
      return H5Tclose(dtype_id);
      }
    
    hid_t arma_H5Dopen(hid_t loc_id, const char* name, hid_t dapl_id)
      {
      return H5Dopen(loc_id, name, dapl_id);
      }
    
    hid_t arma_H5Dget_type(hid_t dataset_id)
      {
      return H5Dget_type(dataset_id);
      }
    
    hid_t arma_H5Dcreate(hid_t loc_id, const char* name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id)
      {
      return H5Dcreate(loc_id, name, dtype_id, space_id, lcpl_id, dcpl_id, dapl_id);
      }
    
    herr_t arma_H5Dwrite(hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id, const void* buf)
      {
      return H5Dwrite(dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, buf);
      }
    
    herr_t arma_H5Dclose(hid_t dataset_id)
      {
      return H5Dclose(dataset_id);
      }
    
    hid_t arma_H5Dget_space(hid_t dataset_id)
      {
      return H5Dget_space(dataset_id);
      }
    
    herr_t arma_H5Dread(hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id, void* buf)
      {
      return H5Dread(dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, buf);
      }
    
    int arma_H5Sget_simple_extent_ndims(hid_t space_id)
      {
      return H5Sget_simple_extent_ndims(space_id);
      }
    
    int arma_H5Sget_simple_extent_dims(hid_t space_id, hsize_t* dims, hsize_t* maxdims)
      {
      return H5Sget_simple_extent_dims(space_id, dims, maxdims);
      }
    
    herr_t arma_H5Sclose(hid_t space_id)
      {
      return H5Sclose(space_id);
      }
    
    hid_t arma_H5Screate_simple(int rank, const hsize_t* current_dims, const hsize_t* maximum_dims)
      {
      return H5Screate_simple(rank, current_dims, maximum_dims);
      }
    
    herr_t arma_H5Ovisit(hid_t object_id, H5_index_t index_type, H5_iter_order_t order, H5O_iterate_t op, void* op_data)
      {
      return H5Ovisit(object_id, index_type, order, op, op_data);
      }
    
    herr_t arma_H5Eset_auto(hid_t estack_id, H5E_auto_t func, void* client_data)
      {
      return H5Eset_auto(estack_id, func, client_data);
      }
    
    herr_t arma_H5Eget_auto(hid_t estack_id, H5E_auto_t* func, void** client_data)
      {
      return H5Eget_auto(estack_id, func, client_data);
      }
    
    hid_t arma_H5Fopen(const char* name, unsigned flags, hid_t fapl_id)
      {
      return H5Fopen(name, flags, fapl_id);
      }
    
    hid_t arma_H5Fcreate(const char* name, unsigned flags, hid_t fcpl_id, hid_t fapl_id)
      {
      return H5Fcreate(name, flags, fcpl_id, fapl_id);
      }
    
    herr_t arma_H5Fclose(hid_t file_id)
      {
      return H5Fclose(file_id);
      }
    
    htri_t arma_H5Fis_hdf5(const char* name)
      {
      return H5Fis_hdf5(name);
      }
    
    hid_t arma_H5Gcreate(hid_t loc_id, const char* name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id)
      {
      return H5Gcreate(loc_id, name, lcpl_id, gcpl_id, gapl_id);
      }
    
    hid_t arma_H5Gopen(hid_t loc_id, const char* name, hid_t gapl_id)
      {
      return H5Gopen(loc_id, name, gapl_id);
      }
    
    herr_t arma_H5Gclose(hid_t group_id)
      {
      return H5Gclose(group_id);
      }
    
    htri_t arma_H5Lexists(hid_t loc_id, const char* name, hid_t lapl_id)
      {
      return H5Lexists(loc_id, name, lapl_id);
      }
    
    herr_t arma_H5Ldelete(hid_t loc_id, const char* name, hid_t lapl_id)
      {
      return H5Ldelete(loc_id, name, lapl_id);
      }
    
    
    // H5T_NATIVE_* types.  The rhs here expands to some macros.
    hid_t arma_H5T_NATIVE_UCHAR  = H5T_NATIVE_UCHAR;
    hid_t arma_H5T_NATIVE_CHAR   = H5T_NATIVE_CHAR;
    hid_t arma_H5T_NATIVE_SHORT  = H5T_NATIVE_SHORT;
    hid_t arma_H5T_NATIVE_USHORT = H5T_NATIVE_USHORT;
    hid_t arma_H5T_NATIVE_INT    = H5T_NATIVE_INT;
    hid_t arma_H5T_NATIVE_UINT   = H5T_NATIVE_UINT;
    hid_t arma_H5T_NATIVE_LONG   = H5T_NATIVE_LONG;
    hid_t arma_H5T_NATIVE_ULONG  = H5T_NATIVE_ULONG;
    hid_t arma_H5T_NATIVE_LLONG  = H5T_NATIVE_LLONG;
    hid_t arma_H5T_NATIVE_ULLONG = H5T_NATIVE_ULLONG;
    hid_t arma_H5T_NATIVE_FLOAT  = H5T_NATIVE_FLOAT;
    hid_t arma_H5T_NATIVE_DOUBLE = H5T_NATIVE_DOUBLE;

  #endif
  
  
  }  // end of extern "C"


}  // end of namespace arma
