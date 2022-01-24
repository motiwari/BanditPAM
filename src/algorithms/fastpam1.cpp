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
        cost = (this->*lossFn)(data, i, j);
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
      cost = (this->*lossFn)(data, l, (*medoidIndices)(k));
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
  size_t swapIn = 0;
  size_t medoidToSwap = 0;
  size_t N = data.n_cols;
  size_t iter = 0;
  bool swapPerformed = true;
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
    assignments,
    swapPerformed);

  float di = 0;
  float dij = 0;

  while (swapPerformed && iter < maxIter) {
    iter++;
    // TODO(@motiwari): pragma omp parallel for?
    for (size_t i = 0; i < data.n_cols; i++) {
      di = bestDistances(i);

      // Consider making point i a medoid.
      // The total loss then contains at least one term, -di,
      // because the loss contribution for point i is reduced to 0
      deltaTD.fill(-di);
      // TODO(@motiwari): pragma omp parallel for?
      for (size_t j = 0; j < data.n_cols; j++) {
        if (j != i) {
          dij = (this->*lossFn)(data, i, j);
          if (dij < bestDistances(j)) {
            // Case 1: point i becomes the closest medoid for point j,
            // regardless of which medoid j was previously assigned to. deltaTD
            // will be negative across ALL possible medoid indices m
            deltaTD += (dij -  bestDistances(j));
          } else if (dij < secondBestDistances(j)) {
            // Case 2: i. If point i is closer than the second best
            // medoid but further than the best medoid (enforced by failing
            // the condition for the above if condition), point i will
            // become the closest medoid only when we remove its associated
            // medoid and add point i
            deltaTD.at((*assignments)(j)) += (dij - bestDistances(j));
          }  else {
            // Case 3: dij > secondBestDistances(j). Then the loss for point j
            // will not change for any medoid swapped out except for its
            // assignment, in which case it moves to its second nearest medoid
            deltaTD.at((*assignments)(j)) +=
              (secondBestDistances(j) - bestDistances(j));
          }
        }
      }

      // Determine the best medoid to swap out
      arma::uword swapOut = deltaTD.index_min();
      // If the loss change is better than the best loss change so far,
      // Update our running best statistics
      if (deltaTD.min() < bestChange) {
        bestChange = deltaTD.min();
        swapIn = i;
        medoidToSwap = swapOut;
      }
    }

    // Update the loss and perform the swap if the loss would be improved
    if (bestChange < 0) {
      minDistance = arma::sum(bestDistances) + bestChange;
      (*medoidIndices)(medoidToSwap) = swapIn;
      calcBestDistancesSwap(
        data,
        medoidIndices,
        &bestDistances,
        &secondBestDistances,
        assignments,
        swapPerformed);
    } else {
      minDistance = arma::sum(bestDistances);
      swapPerformed = false;
    }
  }
}
}  // namespace km
