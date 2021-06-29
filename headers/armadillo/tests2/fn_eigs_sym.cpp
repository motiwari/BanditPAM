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

TEST_CASE("fn_eigs_test")
  {
  for (size_t trial = 0; trial < 10; ++trial)
    {
    // Test ARPACK decomposition of sparse matrices.
    sp_mat m(1000, 1000);
    sp_vec dd;
    for (size_t i = 0; i < 10; ++i)
      {
      dd.sprandu(1000, 1, 0.15);
      double eig = 10.0 * randu<double>();
      m += eig * dd * dd.t();
      }
    mat d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    vec sp_eigval;
    mat sp_eigvec;
    eigs_sym(sp_eigval, sp_eigvec, m, 5);

    // Do the same for the dense case.
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, d);

    for (uword i = 0; i < 5; ++i)
      {
      // It may be pointed the wrong direction.
      REQUIRE( sp_eigval(i) == Approx(eigval(i + 995)).epsilon(0.01) );

      for (uword j = 0; j < 1000; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) ==
                 Approx(std::abs(eigvec(j, i + 995))).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_eigs_float_test")
  {
  for (size_t trial = 0; trial < 10; ++trial)
    {
    // Test ARPACK decomposition of sparse matrices.
    SpMat<float> m(100, 100);
    SpCol<float> dd;
    for (size_t i = 0; i < 10; ++i)
      {
      dd.sprandu(100, 1, 0.15);
      float eig = 10.0 * randu<float>();
      m += eig * dd * dd.t();
      }
    Mat<float> d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    Col<float> sp_eigval;
    Mat<float> sp_eigvec;
    eigs_sym(sp_eigval, sp_eigvec, m, 5);

    // Do the same for the dense case.
    Col<float> eigval;
    Mat<float> eigvec;
    eig_sym(eigval, eigvec, d);

    for (uword i = 0; i < 5; ++i)
      {
      // It may be pointed the wrong direction.
      REQUIRE( sp_eigval(i) == Approx(eigval(i + 95)).epsilon(0.01) );

      for (uword j = 0; j < 100; ++j)
        {
        REQUIRE(std::abs(sp_eigvec(j, i)) ==
                Approx(std::abs(eigvec(j, i + 95))).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_eigs_sm_test")
  {
  for (size_t trial = 0; trial < 10; ++trial)
    {
    // Test ARPACK decomposition of sparse matrices.
    sp_mat m(100, 100);
    for (uword i = 0; i < 100; ++i)
      {
      m(i, i) = i + 10;
      }
    mat d(m);

    // Eigendecompose, getting first 5 eigenvectors.
    vec sp_eigval;
    mat sp_eigvec;
    eigs_sym(sp_eigval, sp_eigvec, m, 5, "sm");

    // Do the same for the dense case.
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, d);

    for (size_t i = 0; i < 5; ++i)
      {
      // It may be pointed the wrong direction.
      REQUIRE( sp_eigval(i) == Approx(eigval(i)).epsilon(0.01) );

      for (size_t j = 0; j < 100; ++j)
        {
        REQUIRE( std::abs(sp_eigvec(j, i)) ==
                 Approx(std::abs(eigvec(j, i))).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_eigs_sigma_test")
  {
  const uword n_trials = 10;
  
  uword count = 0;
  
  for(uword trial=0; trial < n_trials; ++trial)
    {
    // Test ARPACK decomposition of sparse matrices.
    sp_mat m; m.sprandu(100, 100, 0.1);
    m = m.t() + m;
    for(uword i = 0; i < 100; ++i)  { m(i, i) = i + 10; }
    mat d(m);

    // Eigendecompose, getting first 5 eigenvectors around 12.1.
    vec sp_eigval;
    mat sp_eigvec;
    const bool status_sparse = eigs_sym(sp_eigval, sp_eigvec, m, 5, 12.1);
    
    // Do the same for the dense case.
    vec eigval;
    mat eigvec;
    const bool status_dense = eig_sym(eigval, eigvec, d);
    
    if(status_sparse && status_dense)
      {
      ++count;
      
      for(uword i = 0; i < 5; ++i)
        {
        // It may be pointed the wrong direction.
        REQUIRE( sp_eigval(i) == Approx(eigval(i)).epsilon(0.01) );

        for (size_t j = 0; j < 100; ++j)
          {
          REQUIRE( std::abs(sp_eigvec(j, i)) ==
                   Approx(std::abs(eigvec(j, i))).epsilon(0.01) );
          }
        }
      }
    }
  
  REQUIRE( count > 0 );
  }
