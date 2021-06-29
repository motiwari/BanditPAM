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

#if defined(ARMA_USE_SUPERLU)

TEST_CASE("fn_spsolve_sparse_test")
  {
  // We want to spsolve a system of equations, AX = B, where we want to recover
  // X and we have A and B, and A is sparse.
  for (size_t t = 0; t < 10; ++t)
    {
    const uword size = 5 * (t + 1);

    mat rX;
    rX.randu(size, size);

    sp_mat A;
    A.sprandu(size, size, 0.25);
    A.diag().randu();
    A.diag() += 1;

    mat B = A * rX;

    mat X;
    bool result = spsolve(X, A, B);
    REQUIRE( result );

    // Dense solver.
    mat dA(A);
    mat dX = solve(dA, B);

    REQUIRE( X.n_cols == dX.n_cols );
    REQUIRE( X.n_rows == dX.n_rows );

    for (uword i = 0; i < dX.n_cols; ++i)
      {
      for (uword j = 0; j < dX.n_rows; ++j)
        {
        REQUIRE( (double) X(j, i) == Approx((double) dX(j, i)).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_sparse_nonsymmetric_test")
  {
  for (size_t t = 0; t < 10; ++t)
    {
    const uword r_size = 5 * (t + 1);
    const uword c_size = 3 * (t + 4);

    mat rX;
    rX.randu(r_size, c_size);

    sp_mat A;
    A.sprandu(r_size, r_size, 0.25);
    A.diag().randu();
    A.diag() += 1;

    mat B = A * rX;

    mat X;
    bool result = spsolve(X, A, B);
    REQUIRE( result );

    // Dense solver.
    mat dA(A);
    mat dX = solve(dA, B);

    REQUIRE( X.n_cols == dX.n_cols );
    REQUIRE( X.n_rows == dX.n_rows );

    for (uword i = 0; i < dX.n_cols; ++i)
      {
      for (uword j = 0; j < dX.n_rows; ++j)
        {
        REQUIRE( (double) X(j, i) == Approx((double) dX(j, i)).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_sparse_float_test")
  {
  // We want to spsolve a system of equations, AX = B, where we want to recover
  // X and we have A and B, and A is sparse.
  for (size_t t = 0; t < 10; ++t)
    {
    const uword size = 5 * (t + 1);

    fmat rX;
    rX.randu(size, size);

    SpMat<float> A;
    A.sprandu(size, size, 0.25);
    A.diag().randu();
    A.diag() += 1;

    fmat B = A * rX;

    fmat X;
    bool result = spsolve(X, A, B);
    REQUIRE( result );

    // Dense solver.
    fmat dA(A);
    fmat dX = solve(dA, B);

    REQUIRE( X.n_cols == dX.n_cols );
    REQUIRE( X.n_rows == dX.n_rows );

    for (size_t i = 0; i < dX.n_cols; ++i)
      {
      for (size_t j = 0; j < dX.n_rows; ++j)
        {
        REQUIRE( (float) X(j, i) == Approx((float) dX(j, i)).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_sparse_nonsymmetric_float_test")
  {
  for (size_t t = 0; t < 10; ++t)
    {
    const uword r_size = 5 * (t + 1);
    const uword c_size = 3 * (t + 4);

    fmat rX;
    rX.randu(r_size, c_size);

    SpMat<float> A;
    A.sprandu(r_size, r_size, 0.25);
    A.diag().randu();
    A.diag() += 1;

    fmat B = A * rX;

    fmat X;
    bool result = spsolve(X, A, B);
    REQUIRE( result );

    // Dense solver.
    fmat dA(A);
    fmat dX = solve(dA, B);

    REQUIRE( X.n_cols == dX.n_cols );
    REQUIRE( X.n_rows == dX.n_rows );

    for (uword i = 0; i < dX.n_cols; ++i)
      {
      for (uword j = 0; j < dX.n_rows; ++j)
        {
        REQUIRE( (float) X(j, i) == Approx((float) dX(j, i)).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_sparse_complex_float_test")
  {
  // We want to spsolve a system of equations, AX = B, where we want to recover
  // X and we have A and B, and A is sparse.
  for (size_t t = 0; t < 10; ++t)
    {
    const uword size = 5 * (t + 1);

    Mat<cx_float> rX;
    rX.randu(size, size);

    SpMat<cx_float> A;
    A.sprandu(size, size, 0.25);
    A.diag().randu();
    A.diag() += 1;

    Mat<cx_float> B = A * rX;

    Mat<cx_float> X;
    bool result = spsolve(X, A, B);
    REQUIRE( result );

    // Dense solver.
    Mat<cx_float> dA(A);
    Mat<cx_float> dX = solve(dA, B);

    REQUIRE( X.n_cols == dX.n_cols );
    REQUIRE( X.n_rows == dX.n_rows );

    for (uword i = 0; i < dX.n_cols; ++i)
      {
      for (uword j = 0; j < dX.n_rows; ++j)
        {
        REQUIRE( (float) std::abs((cx_float) X(j, i)) ==
                 Approx((float) std::abs((cx_float) dX(j, i))).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_sparse_nonsymmetric_complex_float_test")
  {
  for (size_t t = 0; t < 10; ++t)
    {
    const uword r_size = 5 * (t + 1);
    const uword c_size = 3 * (t + 4);

    Mat<cx_float> rX;
    rX.randu(r_size, c_size);

    SpMat<cx_float> A;
    A.sprandu(r_size, r_size, 0.25);
    A.diag().randu();
    A.diag() += 1;

    Mat<cx_float> B = A * rX;

    Mat<cx_float> X;
    bool result = spsolve(X, A, B);
    REQUIRE( result );

    // Dense solver.
    Mat<cx_float> dA(A);
    Mat<cx_float> dX = solve(dA, B);

    REQUIRE( X.n_cols == dX.n_cols );
    REQUIRE( X.n_rows == dX.n_rows );

    for (uword i = 0; i < dX.n_cols; ++i)
      {
      for (uword j = 0; j < dX.n_rows; ++j)
        {
        REQUIRE( (float) std::abs((cx_float) X(j, i)) ==
                 Approx((float) std::abs((cx_float) dX(j, i))).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_sparse_complex_test")
  {
  // We want to spsolve a system of equations, AX = B, where we want to recover
  // X and we have A and B, and A is sparse.
  for (size_t t = 0; t < 10; ++t)
    {
    const uword size = 5 * (t + 1);

    Mat<cx_double> rX;
    rX.randu(size, size);

    SpMat<cx_double> A;
    A.sprandu(size, size, 0.25);
    A.diag().randu();
    A.diag() += 1;

    Mat<cx_double> B = A * rX;

    Mat<cx_double> X;
    bool result = spsolve(X, A, B);
    REQUIRE( result );

    // Dense solver.
    Mat<cx_double> dA(A);
    Mat<cx_double> dX = solve(dA, B);

    REQUIRE( X.n_cols == dX.n_cols );
    REQUIRE( X.n_rows == dX.n_rows );

    for (uword i = 0; i < dX.n_cols; ++i)
      {
      for (uword j = 0; j < dX.n_rows; ++j)
        {
        REQUIRE( (double) std::abs((cx_double) X(j, i)) ==
                 Approx((double) std::abs((cx_double) dX(j, i))).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_sparse_nonsymmetric_complex_test")
  {
  for (size_t t = 0; t < 10; ++t)
    {
    const uword r_size = 5 * (t + 1);
    const uword c_size = 3 * (t + 4);

    Mat<cx_double> rX;
    rX.randu(r_size, c_size);

    SpMat<cx_double> A;
    A.sprandu(r_size, r_size, 0.25);
    A.diag().randu();
    A.diag() += 1;

    Mat<cx_double> B = A * rX;

    Mat<cx_double> X;
    bool result = spsolve(X, A, B);
    REQUIRE( result );

    // Dense solver.
    Mat<cx_double> dA(A);
    Mat<cx_double> dX = solve(dA, B);

    REQUIRE( X.n_cols == dX.n_cols );
    REQUIRE( X.n_rows == dX.n_rows );

    for (uword i = 0; i < dX.n_cols; ++i)
      {
      for (uword j = 0; j < dX.n_rows; ++j)
        {
        REQUIRE( (double) std::abs((cx_double) X(j, i)) ==
                 Approx((double) std::abs((cx_double) dX(j, i))).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_delayed_sparse_test")
  {
  const uword size = 10;

  mat rX;
  rX.randu(size, size);

  sp_mat A;
  A.sprandu(size, size, 0.25);
  A.diag().randu();
  A.diag() += 1;

  mat B = A * rX;

  mat X;
  bool result = spsolve(X, A, B);
  REQUIRE( result );

  mat dX = spsolve(A, B);

  REQUIRE( X.n_cols == dX.n_cols );
  REQUIRE( X.n_rows == dX.n_rows );

  for (uword i = 0; i < dX.n_cols; ++i)
    {
    for (uword j = 0; j < dX.n_rows; ++j)
      {
      REQUIRE( (double) X(j, i) == Approx((double) dX(j, i)).epsilon(0.01) );
      }
    }
  }



TEST_CASE("fn_spsolve_superlu_solve_test")
  {
  // Solve this matrix, as in the examples:
  // [[19  0  21 21  0]
  //  [12 21   0  0  0]
  //  [ 0 12  16  0  0]
  //  [ 0  0   0  5 21]
  //  [12 12   0  0 18]]
  sp_mat b(5, 5);
  b(0, 0) = 19;
  b(0, 2) = 21;
  b(0, 3) = 21;
  b(1, 0) = 12;
  b(1, 1) = 21;
  b(2, 1) = 12;
  b(2, 2) = 16;
  b(3, 3) = 5;
  b(3, 4) = 21;
  b(4, 0) = 12;
  b(4, 1) = 12;
  b(4, 4) = 18;

  mat db(b);

  sp_mat a;
  a.eye(5, 5);
  mat da(a);

  mat x;
  spsolve(x, a, db);

  mat dx = solve(da, db);

  for (uword i = 0; i < x.n_cols; ++i)
    {
    for (uword j = 0; j < x.n_rows; ++j)
      {
      REQUIRE( (double) x(j, i) == Approx(dx(j, i)).epsilon(0.01) );
      }
    }
  }



TEST_CASE("fn_spsolve_random_superlu_solve_test")
  {
  // Try to solve some random systems.
  const size_t iterations = 10;
  for (size_t it = 0; it < iterations; ++it)
    {
    sp_mat a;
    a.sprandu(50, 50, 0.3);
    sp_mat trueX;
    trueX.sprandu(50, 50, 0.3);

    sp_mat b = a * trueX;

    // Get things into the right format.
    mat db(b);

    mat x;

    spsolve(x, a, db);

    for (uword i = 0; i < x.n_cols; ++i)
      {
      for (uword j = 0; j < x.n_rows; ++j)
        {
        REQUIRE( x(j, i) == Approx((double) trueX(j, i)).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_float_superlu_solve_test")
  {
  // Solve this matrix, as in the examples:
  // [[19  0  21 21  0]
  //  [12 21   0  0  0]
  //  [ 0 12  16  0  0]
  //  [ 0  0   0  5 21]
  //  [12 12   0  0 18]]
  sp_fmat b(5, 5);
  b(0, 0) = 19;
  b(0, 2) = 21;
  b(0, 3) = 21;
  b(1, 0) = 12;
  b(1, 1) = 21;
  b(2, 1) = 12;
  b(2, 2) = 16;
  b(3, 3) = 5;
  b(3, 4) = 21;
  b(4, 0) = 12;
  b(4, 1) = 12;
  b(4, 4) = 18;

  fmat db(b);

  sp_fmat a;
  a.eye(5, 5);
  fmat da(a);

  fmat x;
  spsolve(x, a, db);

  fmat dx = solve(da, db);

  for (uword i = 0; i < x.n_cols; ++i)
    {
    for (uword j = 0; j < x.n_rows; ++j)
      {
      REQUIRE( (float) x(j, i) == Approx(dx(j, i)).epsilon(0.01) );
      }
    }
  }



TEST_CASE("fn_spsolve_float_random_superlu_solve_test")
  {
  // Try to solve some random systems.
  const size_t iterations = 10;
  for (size_t it = 0; it < iterations; ++it)
    {
    sp_fmat a;
    a.sprandu(50, 50, 0.3);
    sp_fmat trueX;
    trueX.sprandu(50, 50, 0.3);

    sp_fmat b = a * trueX;

    // Get things into the right format.
    fmat db(b);

    fmat x;

    spsolve(x, a, db);

    for (uword i = 0; i < x.n_cols; ++i)
      {
      for (uword j = 0; j < x.n_rows; ++j)
        {
        if (std::abs(trueX(j, i)) < 0.001)
          REQUIRE( std::abs(x(j, i)) < 0.005 );
        else
          REQUIRE( trueX(j, i) == Approx((float) x(j, i)).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_cx_float_superlu_solve_test")
  {
  // Solve this matrix, as in the examples:
  // [[19  0  21 21  0]
  //  [12 21   0  0  0]
  //  [ 0 12  16  0  0]
  //  [ 0  0   0  5 21]
  //  [12 12   0  0 18]] (imaginary part is the same)
  SpMat<cx_float> b(5, 5);
  b(0, 0) = cx_float(19, 19);
  b(0, 2) = cx_float(21, 21);
  b(0, 3) = cx_float(21, 21);
  b(1, 0) = cx_float(12, 12);
  b(1, 1) = cx_float(21, 21);
  b(2, 1) = cx_float(12, 12);
  b(2, 2) = cx_float(16, 16);
  b(3, 3) = cx_float(5, 5);
  b(3, 4) = cx_float(21, 21);
  b(4, 0) = cx_float(12, 12);
  b(4, 1) = cx_float(12, 12);
  b(4, 4) = cx_float(18, 18);

  Mat<cx_float> db(b);

  SpMat<cx_float> a;
  a.eye(5, 5);
  Mat<cx_float> da(a);

  Mat<cx_float> x;
  spsolve(x, a, db);

  Mat<cx_float> dx = solve(da, db);

  for (uword i = 0; i < x.n_cols; ++i)
    {
    for (uword j = 0; j < x.n_rows; ++j)
      {
      if (std::abs(x(j, i)) < 0.001 )
        {
        REQUIRE( std::abs(dx(j, i)) < 0.005 );
        }
      else
        {
        REQUIRE( ((cx_float) x(j, i)).real() ==
                 Approx(dx(j, i).real()).epsilon(0.01) );
        REQUIRE( ((cx_float) x(j, i)).imag() ==
                 Approx(dx(j, i).imag()).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_cx_float_random_superlu_solve_test")
  {
  // Try to solve some random systems.
  const size_t iterations = 10;
  for (size_t it = 0; it < iterations; ++it)
    {
    SpMat<cx_float> a;
    a.sprandu(50, 50, 0.3);
    SpMat<cx_float> trueX;
    trueX.sprandu(50, 50, 0.3);

    SpMat<cx_float> b = a * trueX;

    // Get things into the right format.
    Mat<cx_float> db(b);

    Mat<cx_float> x;

    spsolve(x, a, db);

    for (uword i = 0; i < x.n_cols; ++i)
      {
      for (uword j = 0; j < x.n_rows; ++j)
        {
        if (std::abs((cx_float) trueX(j, i)) < 0.001 )
          {
          REQUIRE( std::abs(x(j, i)) < 0.001 );
          }
        else
          {
          REQUIRE( ((cx_float) trueX(j, i)).real() ==
                   Approx(x(j, i).real()).epsilon(0.01) );
          REQUIRE( ((cx_float) trueX(j, i)).imag() ==
                   Approx(x(j, i).imag()).epsilon(0.01) );
          }
        }
      }
    }
  }



TEST_CASE("fn_spsolve_cx_superlu_solve_test")
  {
  // Solve this matrix, as in the examples:
  // [[19  0  21 21  0]
  //  [12 21   0  0  0]
  //  [ 0 12  16  0  0]
  //  [ 0  0   0  5 21]
  //  [12 12   0  0 18]] (imaginary part is the same)
  SpMat<cx_double> b(5, 5);
  b(0, 0) = cx_double(19, 19);
  b(0, 2) = cx_double(21, 21);
  b(0, 3) = cx_double(21, 21);
  b(1, 0) = cx_double(12, 12);
  b(1, 1) = cx_double(21, 21);
  b(2, 1) = cx_double(12, 12);
  b(2, 2) = cx_double(16, 16);
  b(3, 3) = cx_double(5, 5);
  b(3, 4) = cx_double(21, 21);
  b(4, 0) = cx_double(12, 12);
  b(4, 1) = cx_double(12, 12);
  b(4, 4) = cx_double(18, 18);

  cx_mat db(b);

  sp_cx_mat a;
  a.eye(5, 5);
  cx_mat da(a);

  cx_mat x;
  spsolve(x, a, db);

  cx_mat dx = solve(da, db);

  for (uword i = 0; i < x.n_cols; ++i)
    {
    for (uword j = 0; j < x.n_rows; ++j)
      {
      if (std::abs(x(j, i)) < 0.001)
        {
        REQUIRE( std::abs(dx(j, i)) < 0.005 );
        }
      else
        {
        REQUIRE( ((cx_double) x(j, i)).real() ==
                 Approx(dx(j, i).real()).epsilon(0.01) );
        REQUIRE( ((cx_double) x(j, i)).imag() ==
                 Approx(dx(j, i).imag()).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_cx_random_superlu_solve_test")
  {
  // Try to solve some random systems.
  const size_t iterations = 10;
  for (size_t it = 0; it < iterations; ++it)
    {
    sp_cx_mat a;
    a.sprandu(50, 50, 0.3);
    sp_cx_mat trueX;
    trueX.sprandu(50, 50, 0.3);

    sp_cx_mat b = a * trueX;

    // Get things into the right format.
    cx_mat db(b);

    cx_mat x;

    spsolve(x, a, db);

    for (uword i = 0; i < x.n_cols; ++i)
      {
      for (uword j = 0; j < x.n_rows; ++j)
        {
        if (std::abs((cx_double) trueX(j, i)) < 0.001)
          {
          REQUIRE( std::abs(x(j, i)) < 0.005 );
          }
        else
          {
          REQUIRE( ((cx_double) trueX(j, i)).real() ==
                   Approx(x(j, i).real()).epsilon(0.01) );
          REQUIRE( ((cx_double) trueX(j, i)).imag() ==
                   Approx(x(j, i).imag()).epsilon(0.01) );
          }
        }
      }
    }
  }



TEST_CASE("fn_spsolve_function_test")
  {
  sp_mat a;
  a.sprandu(50, 50, 0.3);
  sp_mat trueX;
  trueX.sprandu(50, 50, 0.3);

  sp_mat b = a * trueX;

  // Get things into the right format.
  mat db(b);

  mat x;

  // Mostly these are compilation tests.
  spsolve(x, a, db);
  x = spsolve(a, db); // Test another overload.
  x = spsolve(a, db + 0.0);
  spsolve(x, a, db + 0.0);

  for (uword i = 0; i < x.n_cols; ++i)
    {
    for (uword j = 0; j < x.n_rows; ++j)
      {
      REQUIRE( (double) trueX(j, i) == Approx(x(j, i)).epsilon(0.01) );
      }
    }
  }



TEST_CASE("fn_spsolve_float_function_test")
  {
  sp_fmat a;
  a.sprandu(50, 50, 0.3);
  sp_fmat trueX;
  trueX.sprandu(50, 50, 0.3);

  sp_fmat b = a * trueX;

  // Get things into the right format.
  fmat db(b);

  fmat x;

  // Mostly these are compilation tests.
  spsolve(x, a, db);
  x = spsolve(a, db); // Test another overload.
  x = spsolve(a, db + 0.0);
  spsolve(x, a, db + 0.0);

  for (uword i = 0; i < x.n_cols; ++i)
    {
    for (uword j = 0; j < x.n_rows; ++j)
      {
      if (std::abs(trueX(j, i)) < 0.001)
        {
        REQUIRE( std::abs(x(j, i)) < 0.001 );
        }
      else
        {
        REQUIRE( (float) trueX(j, i) == Approx(x(j, i)).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_cx_function_test")
  {
  sp_cx_mat a;
  a.sprandu(50, 50, 0.3);
  sp_cx_mat trueX;
  trueX.sprandu(50, 50, 0.3);

  sp_cx_mat b = a * trueX;

  // Get things into the right format.
  cx_mat db(b);

  cx_mat x;

  // Mostly these are compilation tests.
  spsolve(x, a, db);
  x = spsolve(a, db); // Test another overload.
  x = spsolve(a, db + cx_double(0.0));
  spsolve(x, a, db + cx_double(0.0));

  for (uword i = 0; i < x.n_cols; ++i)
    {
    for (uword j = 0; j < x.n_rows; ++j)
      {
      if (std::abs((cx_double) trueX(j, i)) < 0.001)
        {
        REQUIRE( std::abs(x(j, i)) < 0.005 );
        }
      else
        {
        REQUIRE( ((cx_double) trueX(j, i)).real() ==
                 Approx(x(j, i).real()).epsilon(0.01) );
        REQUIRE( ((cx_double) trueX(j, i)).imag() ==
                 Approx(x(j, i).imag()).epsilon(0.01) );
        }
      }
    }
  }



TEST_CASE("fn_spsolve_cx_float_function_test")
  {
  sp_cx_fmat a;
  a.sprandu(50, 50, 0.3);
  sp_cx_fmat trueX;
  trueX.sprandu(50, 50, 0.3);

  sp_cx_fmat b = a * trueX;

  // Get things into the right format.
  cx_fmat db(b);

  cx_fmat x;

  // Mostly these are compilation tests.
  spsolve(x, a, db);
  x = spsolve(a, db); // Test another overload.
  x = spsolve(a, db + cx_float(0.0));
  spsolve(x, a, db + cx_float(0.0));

  for (uword i = 0; i < x.n_cols; ++i)
    {
    for (uword j = 0; j < x.n_rows; ++j)
      {
      if (std::abs((cx_float) trueX(j, i)) < 0.001 )
        {
        REQUIRE( std::abs(x(j, i)) < 0.005 );
        }
      else
        {
        REQUIRE( ((cx_float) trueX(j, i)).real() ==
                 Approx(x(j, i).real()).epsilon(0.01) );
        REQUIRE( ((cx_float) trueX(j, i)).imag() ==
                 Approx(x(j, i).imag()).epsilon(0.01) );
        }
      }
    }
  }

#endif
