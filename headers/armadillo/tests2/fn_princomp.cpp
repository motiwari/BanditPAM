#include <armadillo>
#include "catch.hpp"

using namespace arma;

namespace
  {
  void
  initMatrix(mat& m)
    {
    for(uword ii = 0; ii < m.n_rows; ++ii)
    for(uword jj = 0; jj < m.n_cols; ++jj)
      {
      const int i = int(ii);
      const int j = int(jj);
      
      m(ii, jj) = 5 * (i % 17) + (i + j) % 13 - 7 * ((j + 2) % 5) + double(i)/double(m.n_rows);
      }
    }

  void checkEigenvectors(const mat& coeff)
    {
    // sign of the eigenvectors can be flipped
    REQUIRE(std::abs(coeff(0,0)) == Approx(2.2366412109e-01));
    REQUIRE(std::abs(coeff(0,1)) == Approx(3.1197826828e-01));
    REQUIRE(std::abs(coeff(0,2)) == Approx(5.1847537613e-02));
    REQUIRE(std::abs(coeff(1,0)) == Approx(2.2419616512e-01));
    REQUIRE(std::abs(coeff(1,1)) == Approx(2.7564301912e-01));
    REQUIRE(std::abs(coeff(1,2)) == Approx(1.0953921221e-01));
    REQUIRE(std::abs(coeff(2,0)) == Approx(2.2427613980e-01));
    REQUIRE(std::abs(coeff(2,1)) == Approx(1.6088934501e-01));
    REQUIRE(std::abs(coeff(2,2)) == Approx(2.3660988967e-01));
    }

  void checkScore(const mat& score)
    {
    REQUIRE(score(0,0) == Approx(-1.8538115696e+02));
    REQUIRE(score(0,1) == Approx(4.6671842099e+00));
    REQUIRE(score(0,2) == Approx(1.1026881736e+01));
    REQUIRE(score(1,0) == Approx(-1.6144314244e+02));
    REQUIRE(score(1,1) == Approx(8.0636602200e+00));
    REQUIRE(score(1,2) == Approx(8.5129014856e+00));
    REQUIRE(score(2,0) == Approx(-1.3750123749e+02));
    REQUIRE(score(2,1) == Approx(1.0312494525e+01));
    REQUIRE(score(2,2) == Approx(4.5214633042e+00));
    }

  void checkEigenvalues(const vec& latent)
    {
    REQUIRE(latent(0) == Approx(1.1989436021e+04));
    REQUIRE(latent(1) == Approx(9.2136913098e+01));
    REQUIRE(latent(2) == Approx(7.8335565832e+01));
    REQUIRE(latent(3) == Approx(2.4204644513e+01));
    REQUIRE(latent(4) == Approx(2.1302619671e+01));
    REQUIRE(latent(5) == Approx(1.1615198930e+01));
    REQUIRE(latent(6) == Approx(1.1040034957e+01));
    REQUIRE(latent(7) == Approx(7.7918177707e+00));
    REQUIRE(latent(8) == Approx(7.2862524567e+00));
    REQUIRE(latent(9) == Approx(6.5039856845e+00));
    }

  void checkHotteling(const vec& tsquared)
    {
    REQUIRE(tsquared(0) == Approx(7.1983631370e+02));
    REQUIRE(tsquared(1) == Approx(6.5616053343e+02));
    REQUIRE(tsquared(2) == Approx(5.6308987454e+02));
    REQUIRE(tsquared(3) == Approx(3.6908398978e+02));
    REQUIRE(tsquared(4) == Approx(2.4632493795e+02));
    REQUIRE(tsquared(5) == Approx(1.3213013367e+02));
    REQUIRE(tsquared(6) == Approx(5.7414718234e+01));
    REQUIRE(tsquared(7) == Approx(1.5157746233e+01));
    REQUIRE(tsquared(8) == Approx(1.7316032365e+01));
    REQUIRE(tsquared(9) == Approx(2.9290529527e+01));
    REQUIRE(tsquared(20) == Approx(2.6159738840e+02));
    }
  }

TEST_CASE("fn_princomp_1")
  {
  mat m(1000, 20);
  initMatrix(m);
  mat coeff = princomp(m);
  checkEigenvectors(coeff);
  }

TEST_CASE("fn_princomp_2")
  {
  mat m(1000, 20);
  initMatrix(m);
  mat coeff;
  princomp(coeff, m);
  checkEigenvectors(coeff);
  }

TEST_CASE("fn_princomp_3")
  {
  mat m(1000, 20);
  initMatrix(m);
  mat coeff;
  mat score;
  princomp(coeff, score, m);
  checkScore(score);
  checkEigenvectors(coeff);
  }

TEST_CASE("fn_princomp_4")
  {
  mat m(1000, 20);
  initMatrix(m);
  mat coeff;
  mat score;
  vec latent;
  princomp(coeff, score, latent, m);
  checkEigenvectors(coeff);
  checkScore(score);
  checkEigenvalues(latent);
  }

TEST_CASE("fn_princomp_5")
  {
  mat m(1000, 20);
  initMatrix(m);
  mat coeff;
  mat score;
  vec latent;
  vec tsquared;
  princomp(coeff, score, latent, tsquared, m);
  checkEigenvectors(coeff);
  checkScore(score);
  checkEigenvalues(latent);
  // checkHotteling(tsquared);  // TODO
  }

TEST_CASE("fn_princomp_6")
  {
  mat m(5, 20);
  initMatrix(m);
  mat coeff = princomp(m);
  REQUIRE(std::abs(coeff(0,0)) == Approx(2.4288979933e-01));
  REQUIRE(std::abs(coeff(0,1)) == Approx(3.9409505019e-16));
  REQUIRE(std::abs(coeff(0,2)) == Approx(1.2516285718e-02));
  REQUIRE(std::abs(coeff(1,0)) == Approx(2.4288979933e-01));
  REQUIRE(std::abs(coeff(1,1)) == Approx(2.9190770799e-16));
  REQUIRE(std::abs(coeff(1,2)) == Approx(1.2516285718e-02));
  REQUIRE(std::abs(coeff(2,0)) == Approx(2.4288979933e-01));
  REQUIRE(std::abs(coeff(2,1)) == Approx(3.4719806003e-17));
  REQUIRE(std::abs(coeff(2,2)) == Approx(1.2516285718e-02));
  REQUIRE(std::abs(coeff(19,19)) == Approx(9.5528446175e-01).epsilon(0.01));
  }

