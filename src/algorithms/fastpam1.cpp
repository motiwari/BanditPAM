/**
 * @file fastpam1.cpp
 * @date 2021-08-03
 *
 * This file contains the primary C++ implementation of the FastPAM1 code follows
 * from the paper: Erich Schubert and Peter J. Rousseeuw: Faster k-Medoids Clustering:
 * Improving the PAM, CLARA, and CLARANS Algorithms. (https://arxiv.org/pdf/1810.05691.pdf).
 * The original PAM papers are:
 * 1) Leonard Kaufman and Peter J. Rousseeuw: Clustering by means of medoids.
 * 2) Leonard Kaufman and Peter J. Rousseeuw: Partitioning around medoids (program pam).
 *
 */

#include "fastpam1.hpp"

#include <armadillo>
#include <unordered_map>

namespace km {
void FastPAM1::fitFastPAM1(const arma::fmat& inputData) {
  data = inputData;
  data = arma::trans(data);
  arma::urowvec medoidIndices(nMedoids);
  FastPAM1::buildFastPAM1(data, &medoidIndices);
  steps = 0;
  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);
  size_t iter = 0;
  bool medoidChange = true;
  while (iter < maxIter && medoidChange) {
    auto previous{medoidIndices};
    FastPAM1::swapFastPAM1(data, &medoidIndices, &assignments);
    medoidChange = arma::any(medoidIndices != previous);
    iter++;
  }
  medoidIndicesFinal = medoidIndices;
  labels = assignments;
  steps = iter;
}

void FastPAM1::buildFastPAM1(
  const arma::fmat& data,
  arma::urowvec* medoidIndices
) {
  size_t N = data.n_cols;
  arma::frowvec estimates(N, arma::fill::zeros);
  arma::frowvec bestDistances(N);
  bestDistances.fill(std::numeric_limits<float>::infinity());
  arma::frowvec sigma(N);
  for (size_t k = 0; k < nMedoids; k++) {
    float minDistance = std::numeric_limits<float>::infinity();
    int best = 0;
    // fixes a base datapoint
    for (size_t i = 0; i < data.n_cols; i++) {
      float total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // computes distance between base and all other points
        float cost = (this->*lossFn)(data, i, j);
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
      float cost = (this->*lossFn)(data, l, (*medoidIndices)(k));
      if (cost < bestDistances(l)) {
        bestDistances(l) = cost;
      }
    }
  }
}

void FastPAM1::swapFastPAM1(
  const arma::fmat& data,
  arma::urowvec* medoidIndices,
  arma::urowvec* assignments
) {
  float bestChange = 0;
  float minDistance = std::numeric_limits<float>::infinity();
  size_t best = 0;
  size_t medoid_to_swap = 0;
  size_t N = data.n_cols;
  arma::fmat sigma(nMedoids, N, arma::fill::zeros);
  arma::frowvec bestDistances(N);
  arma::frowvec secondBestDistances(N);
  arma::frowvec delta_td(nMedoids, arma::fill::zeros);

  // calculate quantities needed for swap, bestDistances and sigma
  KMedoids::calcBestDistancesSwap(
    data,
    medoidIndices,
    &bestDistances,
    &secondBestDistances,
    assignments);

  for (size_t i = 0; i < data.n_cols; i++) {
    float di = bestDistances(i);
    // compute loss change for making i a medoid
    delta_td.fill(-di);
    for (size_t j = 0; j < data.n_cols; j++) {
      if (j != i) {
        float dij = (this->*lossFn)(data, i, j);
        // update loss change for the current
        if (dij < secondBestDistances(j)) {
          delta_td.at((*assignments)(j)) += (dij - bestDistances(j));
        } else {
          delta_td.at((*assignments)(j)) +=
            (secondBestDistances(j) - bestDistances(j));
        }
        // reassignment check
        if (dij < bestDistances(j)) {
          // update loss change for others
          delta_td += (dij -  bestDistances(j));
          // remove the update for the current
          delta_td.at((*assignments)(j)) -= (dij -  bestDistances(j));
        }
      }
    }
    // choose the best medoid-to-swap
    arma::uword min_medoid = delta_td.index_min();
    // if the loss change is better than the best loss change,
    // update the best index identified so far
    if (delta_td.min() < bestChange) {
      bestChange = delta_td.min();
      best = i;
      medoid_to_swap = min_medoid;
    }
  }
  // update the loss and medoid if the loss is improved
  if (bestChange < 0) {
    minDistance = arma::sum(bestDistances) + bestChange;
    (*medoidIndices)(medoid_to_swap) = best;
  } else {
    minDistance = arma::sum(bestDistances);
  }
}
}  // namespace km
