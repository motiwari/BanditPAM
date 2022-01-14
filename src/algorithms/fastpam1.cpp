/**
 * @file fastpam1.cpp
 * @date 2021-08-03
 *
 * Contains the primary C++ implementation of the FastPAM1 code
 * from the paper: Erich Schubert and Peter J. Rousseeuw: Faster k-Medoids Clustering:
 * Improving the PAM, CLARA, and CLARANS Algorithms. (https://arxiv.org/pdf/1810.05691.pdf).
 * The original PAM papers are:
 * 1) Leonard Kaufman and Peter J. Rousseeuw: Clustering by means of medoids.
 * 2) Leonard Kaufman and Peter J. Rousseeuw: Partitioning around medoids (program pam).
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
  float minDistance = std::numeric_limits<float>::infinity();
  int best = 0;
  float total = 0;
  float cost = 0;
  // TODO(@motiwari): pragma omp parallel for?
  for (size_t k = 0; k < nMedoids; k++) {
    minDistance = std::numeric_limits<float>::infinity();
    best = 0;
    // fixes a base datapoint
    // TODO(@motiwari): pragma omp parallel for?
    for (size_t i = 0; i < data.n_cols; i++) {
      total = 0;
      // TODO(@motiwari): pragma omp parallel for?
      for (size_t j = 0; j < data.n_cols; j++) {
        // computes distance between base and all other points
        cost = KMedoids::cachedLoss(data, i, j);
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
    // TODO(@motiwari): pragma omp parallel for?
    for (size_t l = 0; l < N; l++) {
      cost = KMedoids::cachedLoss(data, l, (*medoidIndices)(k));
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
  size_t medoidToSwap = 0;
  size_t N = data.n_cols;
  arma::fmat sigma(nMedoids, N, arma::fill::zeros);
  arma::frowvec bestDistances(N);
  arma::frowvec secondBestDistances(N);
  arma::frowvec deltaTD(nMedoids, arma::fill::zeros);

  // calculate quantities needed for swap, bestDistances and sigma
  KMedoids::calcBestDistancesSwap(
    data,
    medoidIndices,
    &bestDistances,
    &secondBestDistances,
    assignments);

  float di = 0;
  float dij = 0;

  // TODO(@motiwari): pragma omp parallel for?
  for (size_t i = 0; i < data.n_cols; i++) {
    di = bestDistances(i);
    // compute loss change for making i a medoid
    deltaTD.fill(-di);
    // TODO(@motiwari): pragma omp parallel for?
    for (size_t j = 0; j < data.n_cols; j++) {
      if (j != i) {
        dij = KMedoids::cachedLoss(data, i, j);
        // update loss change for the current
        if (dij < secondBestDistances(j)) {
          deltaTD.at((*assignments)(j)) += (dij - bestDistances(j));
        } else {
          deltaTD.at((*assignments)(j)) +=
            (secondBestDistances(j) - bestDistances(j));
        }
        // reassignment check
        if (dij < bestDistances(j)) {
          // update loss change for others
          deltaTD += (dij -  bestDistances(j));
          // remove the update for the current
          deltaTD.at((*assignments)(j)) -= (dij -  bestDistances(j));
        }
      }
    }
    // choose the best medoid-to-swap
    arma::uword minMedoid = deltaTD.index_min();
    // if the loss change is better than the best loss change,
    // update the best index identified so far
    if (deltaTD.min() < bestChange) {
      bestChange = deltaTD.min();
      best = i;
      medoidToSwap = minMedoid;
    }
  }
  // update the loss and medoid if the loss is improved
  if (bestChange < 0) {
    minDistance = arma::sum(bestDistances) + bestChange;
    (*medoidIndices)(medoidToSwap) = best;
  } else {
    minDistance = arma::sum(bestDistances);
  }
}
}  // namespace km
