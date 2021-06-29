// Copyright 2011-2017 Ryan Curtin (http://www.ratml.org/)
// Copyright 2017 National ICT Australia (NICTA)
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

#include <armadillo>

#include "catch.hpp"

using namespace arma;

TEST_CASE("fn_eigs_gen_odd_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    mat d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.1) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.1) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 4;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    mat d(m);

    // Eigendecompose, getting first 4 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_opts_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 4;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    mat d(m);

    // Eigendecompose, getting first 4 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "lm", opts);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_odd_sigma_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const double sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += (sigma+0.001)*speye(n_rows, n_rows);
    mat d(m);

    // Eigendecompose, getting first 5 eigenvectors around 1.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.1) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.1) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_sigma_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 4;
  const double sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += (sigma+0.001)*speye(n_rows, n_rows);
    mat d(m);

    // Eigendecompose, getting first 4 eigenvectors around 1.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_sigma_opts_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 4;
  const double sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += (sigma+0.001)*speye(n_rows, n_rows);
    mat d(m);

    // Eigendecompose, getting first 4 eigenvectors around 1.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma, opts);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_odd_sm_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += 0.001*speye(n_rows, n_rows);
    mat d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm");

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.1) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.1) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_sm_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 4;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += 0.001*speye(n_rows, n_rows);
    mat d(m);

    // Eigendecompose, getting first 4 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm");

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_sm_opts_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 4;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    sp_mat m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += 0.001*speye(n_rows, n_rows);
    mat d(m);

    // Eigendecompose, getting first 4 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm", opts);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-4) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-4) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_odd_float_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    Mat<float> d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.001) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_float_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    Mat<float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_float_opts_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    Mat<float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "lm", opts);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_odd_float_sigma_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const float sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    m += (sigma+0.001)*speye<SpMat<float>>(n_rows, n_rows);
    Mat<float> d(m);

    // Eigendecompose, getting first 5 eigenvectors around 1.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.001) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_float_sigma_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const float sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    m += (sigma+0.001)*speye<SpMat<float>>(n_rows, n_rows);
    Mat<float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma);

    // Do the same for the dense case around 1.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_float_sigma_opts_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const float sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    m += (sigma+0.001)*speye<SpMat<float>>(n_rows, n_rows);
    Mat<float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma, opts);

    // Do the same for the dense case around 1.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_odd_float_sm_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    m += 0.001*speye<SpMat<float>>(n_rows, n_rows);
    Mat<float> d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm");

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.001) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_float_sm_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    m += 0.001*speye<SpMat<float>>(n_rows, n_rows);
    Mat<float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm");

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_float_sm_opts_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    for(uword i = 0; i < n_rows; ++i)
      {
      m(i, i) += 5 * double(i) / double(n_rows);
      }
    m += 0.001*speye<SpMat<float>>(n_rows, n_rows);
    Mat<float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm", opts);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_odd_complex_float_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_float_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_float_opts_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "lm", opts);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_odd_complex_float_sigma_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const cx_float sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_fmat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += (sigma+cx_float(0.001,0))*speye< SpMat<cx_float> >(n_rows, n_rows);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 5 eigenvectors around 1.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_float_sigma_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const cx_float sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_fmat z(8, 8);
    z.sprandu(8, 8, 0.5);
    m.submat(2, 2, 9, 9) += 8 * z;
    m += (sigma+cx_float(0.001,0))*speye< SpMat<cx_float> >(n_rows, n_rows);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 8 eigenvectors around 1.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_float_sigma_opts_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const cx_float sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_fmat z(8, 8);
    z.sprandu(8, 8, 0.5);
    m.submat(2, 2, 9, 9) += 8 * z;
    m += (sigma+cx_float(0.001,0))*speye< SpMat<cx_float> >(n_rows, n_rows);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 8 eigenvectors around 1.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma, opts);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_odd_complex_float_sm_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_fmat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += cx_float(0.001,0)*speye< SpMat<cx_float> >(n_rows, n_rows);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm");

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_float_sm_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_fmat z(8, 8);
    z.sprandu(8, 8, 0.5);
    m.submat(2, 2, 9, 9) += 8 * z;
    m += cx_float(0.001,0)*speye< SpMat<cx_float> >(n_rows, n_rows);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm");

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_float_sm_opts_test")
  {
  const uword n_rows = 12;
  const uword n_eigval = 8;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_float> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_fmat z(8, 8);
    z.sprandu(8, 8, 0.5);
    m.submat(2, 2, 9, 9) += 8 * z;
    m += cx_float(0.001,0)*speye< SpMat<cx_float> >(n_rows, n_rows);
    Mat<cx_float> d(m);

    // Eigendecompose, getting first 8 eigenvectors.
    Col<cx_float> sp_eigval;
    Mat<cx_float> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm", opts);

    // Do the same for the dense case.
    Col<cx_float> eigval;
    Mat<cx_float> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_float(sp_eigval(i)).real() - eigval(k).real()) < 0.001) &&
            (std::abs(cx_float(sp_eigval(i)).imag() - eigval(k).imag()) < 0.001) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("eigs_gen_odd_complex_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(size_t j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_test")
  {
  const uword n_rows = 15;
  const uword n_eigval = 6;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 6 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_opts_test")
  {
  const uword n_rows = 15;
  const uword n_eigval = 6;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 6 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "lm", opts);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("eigs_gen_odd_complex_sigma_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const cx_double sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_mat z(5, 5);
    z.sprandu(5, 5, 0.5);
    m.submat(2, 2, 6, 6) += 5 * z;
    m += (sigma+cx_double(0.001,0))*speye< SpMat<cx_double> >(n_rows, n_rows);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 5 eigenvectors around 1.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(size_t j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_sigma_test")
  {
  const uword n_rows = 15;
  const uword n_eigval = 6;
  const cx_double sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_mat z(8, 8);
    z.sprandu(8, 8, 0.5);
    m.submat(2, 2, 9, 9) += 8 * z;
    m += (sigma+cx_double(0.001,0))*speye< SpMat<cx_double> >(n_rows, n_rows);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 6 eigenvectors around 1.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_sigma_opts_test")
  {
  const uword n_rows = 15;
  const uword n_eigval = 6;
  const cx_double sigma = 1.0;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    sp_cx_mat z(8, 8);
    z.sprandu(8, 8, 0.5);
    m.submat(2, 2, 9, 9) += 8 * z;
    m += (sigma+cx_double(0.001,0))*speye< SpMat<cx_double> >(n_rows, n_rows);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 6 eigenvectors around 1.0.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, sigma, opts);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("eigs_gen_odd_complex_sm_test")
  {
  const uword n_rows = 10;
  const uword n_eigval = 5;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    m += cx_double(0.001,0)*speye< SpMat<cx_double> >(n_rows, n_rows);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm");

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(size_t j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_sm_test")
  {
  const uword n_rows = 15;
  const uword n_eigval = 6;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    m += cx_double(0.001,0)*speye< SpMat<cx_double> >(n_rows, n_rows);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 6 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm");

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }



TEST_CASE("fn_eigs_gen_even_complex_sm_opts_test")
  {
  const uword n_rows = 15;
  const uword n_eigval = 6;
  const uword n_trials = 10;
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    SpMat<cx_double> m;
    m.sprandu(n_rows, n_rows, 0.3);
    m += cx_double(0.001,0)*speye< SpMat<cx_double> >(n_rows, n_rows);
    Mat<cx_double> d(m);

    // Eigendecompose, getting first 6 eigenvectors.
    Col<cx_double> sp_eigval;
    Mat<cx_double> sp_eigvec;
    eigs_opts opts{}; opts.maxiter = 10000; opts.tol = 1e-12;
    const bool status_sparse = eigs_gen(sp_eigval, sp_eigvec, m, n_eigval, "sm", opts);

    // Do the same for the dense case.
    Col<cx_double> eigval;
    Mat<cx_double> eigvec;
    const bool status_dense = eig_gen(eigval, eigvec, d);
    
    if( (status_sparse == false) || (status_dense == false) )  { continue; }  else  { ++count; }
    
    uvec used(n_rows, fill::zeros);

    for(uword i=0; i < n_eigval; ++i)
      {
      // Sorting these is difficult.
      // Find which one is the likely dense eigenvalue.
      uword dense_eval = n_rows + 1;
      for(uword k = 0; k < n_rows; ++k)
        {
        if ((std::abs(cx_double(sp_eigval(i)).real() - eigval(k).real()) < 1e-10) &&
            (std::abs(cx_double(sp_eigval(i)).imag() - eigval(k).imag()) < 1e-10) &&
            (used(k) == 0))
          {
          dense_eval = k;
          used(k) = 1;
          break;
          }
        }

      REQUIRE( dense_eval != n_rows + 1 );

      REQUIRE( std::abs(sp_eigval(i)) == Approx(std::abs(eigval(dense_eval))).epsilon(0.01) );
      for(uword j = 0; j < n_rows; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) == Approx(std::abs(eigvec(j, dense_eval))).epsilon(0.01) );
        }
      }
    }
  
  REQUIRE(count > 0);
  }
