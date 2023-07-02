/**
 * @file pam.cpp
 * @date 2021-07-25
 *
 * Contains a C++ implementation of the PAM algorithm.
 * The original PAM papers are:
 * 1) Leonard Kaufman and Peter J. Rousseeuw: Clustering by means of medoids.
 * 2) Leonard Kaufman and Peter J. Rousseeuw: Partitioning around medoids (program pam).
 */

#include "pam.hpp"

#include <unordered_map>

namespace km {
void PAM::fitPAM(
  const arma_mat& inputData,
  std::optional<std::reference_wrapper<const arma_mat>> distMat) {
  data = arma::trans(inputData);
  arma::urowvec medoidIndices(nMedoids);
  PAM::buildPAM(data, distMat, &medoidIndices);
  steps = 0;
  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);

  size_t i = 0;
  bool medoidChange = true;
  while (i < maxIter && medoidChange) {
    auto previous(medoidIndices);
    PAM::swapPAM(data, distMat, &medoidIndices, &assignments);
    medoidChange = arma::any(medoidIndices != previous);
    i++;
  }
  medoidIndicesFinal = medoidIndices;
  labels = assignments;
  steps = i;
}

void PAM::buildPAM(
  const arma_mat& data,
  std::optional<std::reference_wrapper<const arma_mat>> distMat,
  arma::urowvec* medoidIndices) {
  size_t N = data.n_cols;
  arma_rowvec estimates(N, arma::fill::zeros);
  arma_rowvec bestDistances(N);
  bestDistances.fill(std::numeric_limits<banditpam_float>::infinity());
  for (size_t k = 0; k < nMedoids; k++) {
    banditpam_float minDistance = std::numeric_limits<banditpam_float>::infinity();
    size_t best = 0;
    for (size_t i = 0; i < data.n_cols; i++) {
      banditpam_float total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        banditpam_float cost = (this->*lossFn)(data, i, j);
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
    (*medoidIndices)(k) = best;

    // update the medoid assignment and best_distance for this datapoint
    for (size_t l = 0; l < N; l++) {
      banditpam_float cost = (this->*lossFn)(data, l, (*medoidIndices)(k));
      if (cost < bestDistances(l)) {
        bestDistances(l) = cost;
      }
    }
  }
}

void PAM::swapPAM(
  const arma_mat& data,
  std::optional<std::reference_wrapper<const arma_mat>> distMat,
  arma::urowvec* medoidIndices,
  arma::urowvec* assignments) {
  banditpam_float minDistance = std::numeric_limits<banditpam_float>::infinity();
  size_t best = 0;
  size_t medoidToSwap = 0;
  size_t N = data.n_cols;
  arma_rowvec bestDistances(N);
  arma_rowvec secondBestDistances(N);

  KMedoids::calcBestDistancesSwap(
    data,
    distMat,
    medoidIndices,
    &bestDistances,
    &secondBestDistances,
    assignments);

  for (size_t k = 0; k < nMedoids; k++) {
    for (size_t i = 0; i < data.n_cols; i++) {
      banditpam_float total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // compute distance between base point and every other datapoint
        banditpam_float cost = (this->*lossFn)(data, i, j);
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
        medoidToSwap = k;
      }
    }
  }
  (*medoidIndices)(medoidToSwap) = best;
}
}  // namespace km
