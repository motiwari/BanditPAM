// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;


/**
 * @file kmedoids.cpp
 *
 * Exposes KMedoids C++ class and methods to R
 */

#include <unistd.h>
#include <fstream>
#include <exception>
#include <filesystem>

#include "kmedoids_algorithm.hpp"

//' Create a new KMedoids object.
//'
//' @return an external ptr (Rcpp::XPtr) to a KMedoids object instance.
// [[Rcpp::export(.KMedoids__new)]]
SEXP KMedoids__new(IntegerVector k, IntegerVector max_iter, IntegerVector build_confidence, IntegerVector swap_confidence) {
  
  // create a pointer to an KMedoids object and wrap it
  // as an external pointer
  XPtr<km::KMedoids> ptr( new km::KMedoids((size_t) k[0], "BanditPAM", (size_t) max_iter[0], (size_t) build_confidence[0], (size_t) swap_confidence[0]), true );
  // return the external pointer to the R side
  return ptr;
}

//' Fit the KMedoids algorthm given the data and loss
//'
//' @param xp the km::KMedoids Object XPtr
//' @param data the data matrix
//' @param loss the loss indicator
//' @param distMat the optional distance matrix
// [[Rcpp::export(.KMedoids__fit)]]
void KMedoids__fit(SEXP xp, arma::mat data, std::vector< std::string > loss, SEXP distMat = R_NilValue) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);

  if (distMat != R_NilValue) {
    arma_mat mat = Rcpp::as<arma_mat>(distMat);
    std::reference_wrapper<const arma::mat> matRef(mat);
    ptr->fit(data, loss[0], matRef);
  } else {
    ptr->fit(data, loss[0], std::nullopt);
  }

}

//' Return the final medoids
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__get_medoids_final)]]
SEXP KMedoids__get_medoids_final(SEXP xp) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  return wrap(ptr->getMedoidsFinal());
}

//' Return the number of medoids property k
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__get_k)]]
SEXP KMedoids__get_k(SEXP xp) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  return wrap(ptr->getNMedoids());
}

//' Set the number of medoids property k
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__set_k)]]
void KMedoids__set_k(SEXP xp, IntegerVector k) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  ptr->setNMedoids(k[0]);
}

//' Return the max_iter property
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__get_max_iter)]]
SEXP KMedoids__get_max_iter(SEXP xp) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  return wrap(ptr->getMaxIter());
}

//' Set the max_iter property
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__set_iter)]]
void KMedoids__set_max_iter(SEXP xp, IntegerVector m) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  ptr->setMaxIter(m[0]);
}

//' Return the build_conf property
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__get_build_conf)]]
SEXP KMedoids__get_build_conf(SEXP xp) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  return wrap(ptr->getBuildConfidence());
}

//' Set the build_conf property
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__set_iter)]]
void KMedoids__set_build_conf(SEXP xp, IntegerVector bc) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  ptr->setBuildConfidence(bc[0]);
}

//' Return the swap_conf property
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__get_swap_conf)]]
SEXP KMedoids__get_swap_conf(SEXP xp) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  return wrap(ptr->getSwapConfidence());
}

//' Set the swap_conf property
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__set_iter)]]
void KMedoids__set_swap_conf(SEXP xp, IntegerVector bc) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  ptr->setSwapConfidence(bc[0]);
}

//' Return the loss_fn property
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__get_loss_fn)]]
SEXP KMedoids__get_loss_fn(SEXP xp) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  return wrap(ptr->getLossFn());
}

//' Set the loss_fn property
//'
//' @param xp the km::KMedoids Object XPtr
// [[Rcpp::export(.KMedoids__set_loss_fn)]]
void KMedoids__set_loss_fn(SEXP xp, std::vector< std::string > loss_fn ) {
  // grab the object as a XPtr (smart pointer)
  XPtr<km::KMedoids> ptr(xp);
  ptr->setLossFn(loss_fn[0]);
}


