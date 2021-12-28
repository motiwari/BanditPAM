/**
 * @file kmedoids_algorithm.cpp
 * @date 2020-06-10
 *
 * This file contains the primary C++ implementation of the BanditPAM code.
 *
 */

#include <omp.h>
#include <armadillo>
#include <unordered_map>
#include <regex>

#include "kmedoids_algorithm.hpp"
#include "fastpam1.hpp"
#include "pam.hpp"
#include "banditpam.hpp"

namespace km {
KMedoids::KMedoids(
  size_t n_medoids,
  const std::string& algorithm,
  size_t max_iter,
  size_t buildConfidence,
  size_t swapConfidence):
    n_medoids(n_medoids),
    algorithm(algorithm),
    max_iter(max_iter),
    buildConfidence(buildConfidence),
    swapConfidence(swapConfidence) {
  KMedoids::checkAlgorithm(algorithm);
}

KMedoids::~KMedoids() {}

void KMedoids::fit(const arma::mat& input_data, const std::string& loss) {
  batchSize = fmin(input_data.n_rows, batchSize);

  if (input_data.n_rows == 0) {
    throw std::invalid_argument("Dataset is empty");
  }

  KMedoids::setLossFn(loss);
  if (algorithm == "PAM") {
    static_cast<PAM*>(this)->fit_pam(input_data);
  } else if (algorithm == "BanditPAM") {
    static_cast<BanditPAM*>(this)->fit_bpam(input_data);
  } else if (algorithm == "FastPAM1") {
    static_cast<FastPAM1*>(this)->fit_fastpam1(input_data);
  }
}

arma::urowvec KMedoids::getMedoidsBuild() const {
  return medoid_indices_build;
}

arma::urowvec KMedoids::getMedoidsFinal() const {
  return medoid_indices_final;
}

arma::urowvec KMedoids::getLabels() const {
  return labels;
}

size_t KMedoids::getSteps() const {
  return steps;
}

size_t KMedoids::getNMedoids() const {
  return n_medoids;
}

void KMedoids::setNMedoids(size_t new_num) {
  n_medoids = new_num;
}

std::string KMedoids::getAlgorithm() const {
  return algorithm;
}

void KMedoids::setAlgorithm(const std::string& new_alg) {
  algorithm = new_alg;
  KMedoids::checkAlgorithm(algorithm);
}

size_t KMedoids::getMaxIter() const {
  return max_iter;
}

void KMedoids::setMaxIter(size_t new_max) {
  max_iter = new_max;
}


size_t KMedoids::getbuildConfidence() const {
  return buildConfidence;
}

void KMedoids::setbuildConfidence(size_t new_buildConfidence) {
  if (algorithm != "BanditPAM") {
    // TODO(@motiwari): Better error type
    throw "Cannot set buildConfidence when not using BanditPAM";
  }
  buildConfidence = new_buildConfidence;
}

size_t KMedoids::getswapConfidence() const {
  return swapConfidence;
}

void KMedoids::setswapConfidence(size_t new_swapConfidence) {
  if (algorithm != "BanditPAM") {
    // TODO(@motiwari): Better error type
    throw "Cannot set buildConfidence when not using BanditPAM";
  }
  swapConfidence = new_swapConfidence;
}

void KMedoids::setLossFn(std::string loss) {
  if (std::regex_match(loss, std::regex("L\\d*"))) {
    loss = loss.substr(1);
  }
  try {
    if (loss == "manhattan") {
      lossFn = &KMedoids::manhattan;
    } else if (loss == "cos") {
      lossFn = &KMedoids::cos;
    } else if (loss == "inf") {
      lossFn = &KMedoids::LINF;
    } else if (std::isdigit(loss.at(0))) {
      lossFn = &KMedoids::LP;
      lp     = atoi(loss.c_str());
    } else {
      throw std::invalid_argument("error: unrecognized loss function");
    }
  } catch (std::invalid_argument& e) {
    std::cout << e.what() << std::endl;
  }
}

void KMedoids::calc_best_distances_swap(
  const arma::mat& data,
  const arma::urowvec* medoid_indices,
  arma::rowvec* best_distances,
  arma::rowvec* second_distances,
  arma::urowvec* assignments) {
  #pragma omp parallel for
  for (size_t i = 0; i < data.n_cols; i++) {
    double best = std::numeric_limits<double>::infinity();
    double second = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < medoid_indices->n_cols; k++) {
      double cost = KMedoids::cachedLoss(data, i, (*medoid_indices)(k));
      if (cost < best) {
        (*assignments)(i) = k;
        second = best;
        best = cost;
      } else if (cost < second) {
        second = cost;
      }
    }
    (*best_distances)(i) = best;
    (*second_distances)(i) = second;
  }
}

double KMedoids::calc_loss(
  const arma::mat& data,
  const arma::urowvec* medoid_indices) {
  double total = 0;
  // TODO(@motiwari): is this parallel loop accumulating properly?
  #pragma omp parallel for
  for (size_t i = 0; i < data.n_cols; i++) {
    double cost = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < n_medoids; k++) {
      double currCost = KMedoids::cachedLoss(data, i, (*medoid_indices)(k));
      if (currCost < cost) {
        cost = currCost;
      }
    }
    total += cost;
  }
  return total;
}

double KMedoids::cachedLoss(
  const arma::mat& data,
  const size_t i,
  const size_t j,
  const bool use_cache) {
  if (!use_cache) {
    return (this->*lossFn)(data, i, j);
  }

  size_t n = data.n_cols;
  size_t m = fmin(n, ceil(log10(data.n_cols) * cache_multiplier));

  // test this is one of the early points in the permutation
  if (reindex.find(j) != reindex.end()) {
    // TODO(@motiwari): Potential race condition with shearing?
    // T1 begins to write to cache and then T2 access in the middle of write?
    if (cache[(m*i) + reindex[j]] == -1) {
      cache[(m*i) + reindex[j]] = (this->*lossFn)(data, i, j);
    }
    return cache[m*i + reindex[j]];
  }
  return (this->*lossFn)(data, i, j);
}

void KMedoids::checkAlgorithm(const std::string& algorithm) const {
  if ((algorithm != "BanditPAM") &&
      (algorithm != "PAM") &&
      (algorithm != "FastPAM1")) {
    // TODO(@motiwari): Better error type
    throw "unrecognized algorithm";
  }
}

double KMedoids::LP(const arma::mat& data,
  const size_t i,
  const size_t j) const {
  return arma::norm(data.col(i) - data.col(j), lp);
}

double KMedoids::LINF(
  const arma::mat& data,
  const size_t i,
  const size_t j) const {
  return arma::max(arma::abs(data.col(i) - data.col(j)));
}

double KMedoids::cos(const arma::mat& data,
  const size_t i,
  const size_t j) const {
  return arma::dot(
    data.col(i),
    data.col(j)) / (arma::norm(data.col(i))* arma::norm(data.col(j)));
}

double KMedoids::manhattan(const arma::mat& data,
  const size_t i,
  const size_t j) const {
  return arma::accu(arma::abs(data.col(i) - data.col(j)));
}
}  // namespace km
