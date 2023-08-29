#ifndef _HEADERS_BANDITPAM_COMMON_
#define _HEADERS_BANDITPAM_COMMON_

#include <myomp.h>

#ifdef R_INTERFACE
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#ifdef USE_DOUBLE
typedef double banditpam_float;
typedef arma::mat arma_mat;
typedef arma::rowvec arma_rowvec;
typedef arma::vec arma_vec;
#else
typedef float banditpam_float;
typedef arma::fmat arma_mat;
typedef arma::frowvec arma_rowvec;
typedef arma::fvec arma_vec;
#endif

#endif
