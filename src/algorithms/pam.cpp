/**
 * @file pam.cpp
 * @date 2021-07-25
 *
 * This file contains a C++ implementation of the PAM algorithm.
 *
 */

#include "pam.hpp"

#include <armadillo>
#include <unordered_map>

namespace km {
void PAM::fitPAM(const arma::mat& inputData) {
  data = inputData;
  data = arma::trans(data);
  arma::urowvec medoid_indices(nMedoids);
  PAM::buildPAM(data, &medoid_indices);
  steps = 0;
  medoidIndicesBuild = medoid_indices;
  arma::urowvec assignments(data.n_cols);
  size_t i = 0;
  bool medoidChange = true;
  while (i < maxIter && medoidChange) {
    auto previous(medoid_indices);
    PAM::swapPAM(data, &medoid_indices, &assignments);
    medoidChange = arma::any(medoid_indices != previous);
    i++;
  }
  medoidIndicesFinal = medoid_indices;
  this->labels = assignments;
  this->steps = i;
}

void PAM::buildPAM(
  const arma::mat& data,
  arma::urowvec* medoid_indices) {
  size_t N = data.n_cols;
  arma::rowvec estimates(N, arma::fill::zeros);
  arma::rowvec bestDistances(N);
  bestDistances.fill(std::numeric_limits<double>::infinity());
  for (size_t k = 0; k < nMedoids; k++) {
    double minDistance = std::numeric_limits<double>::infinity();
    size_t best = 0;
    for (size_t i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        double cost = KMedoids::cachedLoss(data, i, j);
        // compares this with the cached best distance
        if (bestDistances(j) < cost) {
          cost = bestDistances(j);
        }
        total += cost;
      }
      if (total < minDistance) {
        minDistance = total;
        best = i;
      }
    }
    // update the medoid index for that of lowest cost
    (*medoid_indices)(k) = best;

    // update the medoid assignment and best_distance for this datapoint
    for (size_t l = 0; l < N; l++) {
      double cost = KMedoids::cachedLoss(data, l, (*medoid_indices)(k));
      if (cost < bestDistances(l)) {
        bestDistances(l) = cost;
      }
    }
  }
}

void PAM::swapPAM(
  const arma::mat& data,
  arma::urowvec* medoid_indices,
  arma::urowvec* assignments) {
  double minDistance = std::numeric_limits<double>::infinity();
  size_t best = 0;
  size_t medoid_to_swap = 0;
  size_t N = data.n_cols;
  arma::rowvec bestDistances(N);
  arma::rowvec secondBestDistances(N);

  KMedoids::calcBestDistancesSwap(
    data,
    medoid_indices,
    &bestDistances,
    &secondBestDistances,
    assignments);

  for (size_t k = 0; k < nMedoids; k++) {
    for (size_t i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // compute distance between base point and every other datapoint
        double cost = KMedoids::cachedLoss(data, i, j);
        // if x_j is NOT assigned to k: compares this with
        //   the cached best distance
        // if x_j is assigned to k: compares this with
        //   the cached second best distance
        if ((*assignments)(j) != k) {
          if (bestDistances(j) < cost) {
            cost = bestDistances(j);
          }
        } else {
          if (secondBestDistances(j) < cost) {
            cost = secondBestDistances(j);
          }
        }
        total += cost;
      }
      // if total distance for new base point is better than
      // that of the medoid, update the best index identified so far
      if (total < minDistance) {
        minDistance = total;
        best = i;
        medoid_to_swap = k;
      }
    }
  }
  (*medoid_indices)(medoid_to_swap) = best;
}
}  // namespace km
