/**
 * @file BanditFasterPAM.cpp
 * @date 2021-07-25
 *
 * Contains a C++ implementation of the BanditFasterPAM algorithm.
 * The original BanditFasterPAM papers are:
 * 1) Erich Schubert and Peter J. Rousseeuw: Fast and Eager k-Medoids Clustering:
 *  O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
 * 2) Erich Schubert and Peter J. Rousseeuw: Faster k-Medoids Clustering:
 *  Improving the PAM, CLARA, and CLARANS Algorithms
 */

#include "banditfasterpam.hpp"

#include <armadillo>
#include <unordered_map>

namespace km {
void BanditFasterPAM::fitBanditFasterPAM(
  const arma::fmat& inputData,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat) {
  data = arma::trans(inputData);
  arma::urowvec medoidIndices(nMedoids);

  // Note: even if we are using a distance matrix, we compute the permutation
  // in the block below because it is used elsewhere in the call stack
  // TODO(@motiwari): Remove need for data or permutation through when using
  //  a distance matrix
  // TODO(@motiwari): Break this duplicated code out
  if (this->useCache) {
    size_t n = data.n_cols;
    size_t m = fmin(n, cacheWidth);
    cache = new float[n * m];

    #pragma omp parallel for if (this->parallelize)
    for (size_t idx = 0; idx < m*n; idx++) {
      cache[idx] = -1;  // TODO(@motiwari): need better value here
    }

    permutation = arma::randperm(n);
    permutationIdx = 0;
    reindex = {};  // TODO(@motiwari): Can this intialization be removed?
    // TODO(@motiwari): Can we parallelize this?
    for (size_t counter = 0; counter < m; counter++) {
      reindex[permutation[counter]] = counter;
    }
  }


  BanditFasterPAM::buildBanditFasterPAM(data, distMat, &medoidIndices);
  steps = 0;
  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);
  BanditFasterPAM::swapBanditFasterPAM(data, distMat, &medoidIndices, &assignments);

  medoidIndicesFinal = medoidIndices;
  labels = assignments;
}

// TODO(@motiwari): consolidate this function which is identical to several other BUILD fns
void BanditFasterPAM::buildBanditFasterPAM(
  const arma::fmat& data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  arma::urowvec* medoidIndices) {
  size_t N = data.n_cols;
  arma::frowvec estimates(N, arma::fill::zeros);
  arma::frowvec bestDistances(N);
  bestDistances.fill(std::numeric_limits<float>::infinity());
  for (size_t k = 0; k < nMedoids; k++) {
    float minDistance = std::numeric_limits<float>::infinity();
    size_t best = 0;
    for (size_t i = 0; i < data.n_cols; i++) {
      float total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
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

arma::frowvec BanditFasterPAM::calcDeltaTDMs(
  arma::urowvec* assignments,
  arma::frowvec* bestDistances,
  arma::frowvec* secondBestDistances) {

  arma::frowvec Delta_TD_ms(nMedoids, arma::fill::zeros);
  size_t N = (*assignments).n_cols;
  for (size_t i = 0; i < data.n_cols; i++) {
    Delta_TD_ms((*assignments)(i)) += -(*bestDistances)(i) + (*secondBestDistances)(i);
  }
  return Delta_TD_ms / N;
}

float BanditFasterPAM::swapSigma(
  const size_t candidate,
  const arma::fmat& data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const arma::frowvec* bestDistances,
  const arma::frowvec* secondBestDistances,
  const arma::urowvec* assignments) {
  const size_t N = data.n_cols;
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly
  //  as last batch_size elements are dropped
  if (usePerm) {
    if ((permutationIdx + batchSize - 1) >= N) {
      permutationIdx = 0;
    }
    // inclusive of both indices
    referencePoints = permutation.subvec(
        permutationIdx,
        permutationIdx + batchSize - 1);
    permutationIdx += batchSize;
  } else {
    referencePoints = arma::randperm(N, batchSize);
  }

  arma::fvec sample(batchSize, arma::fill::zeros);
  // calculate change in loss for some subset of the data
  #pragma omp parallel for if (this->parallelize)
  for (size_t ref = 0; ref < batchSize; ref++) {
    // 0 for MISC when estimating sigma
    float cost =
        KMedoids::cachedLoss(data, distMat, candidate, referencePoints(ref), 0);

    if (cost < (*bestDistances)(ref)) {
      sample(ref) = cost - (*bestDistances)(ref);
    }
  }
  return arma::stddev(sample);
}

std::tuple<float, arma::frowvec> BanditFasterPAM::swapTarget(
  const arma::fmat& data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const size_t candidate,
  const arma::frowvec* bestDistances,
  const arma::frowvec* secondBestDistances,
  const arma::urowvec* assignments,
  const size_t exact) {
  const size_t N = data.n_cols;
  size_t tmpBatchSize = batchSize;
  if (exact > 0) {
    tmpBatchSize = N;
  }

  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly
  //  as last batch_size elements are dropped
  // TODO(@motiwari): Break this duplicated code into a function
  if (usePerm) {
    if ((permutationIdx + tmpBatchSize - 1) >= N) {
      permutationIdx = 0;
    }
    // inclusive of both indices
    referencePoints = permutation.subvec(
        permutationIdx,
        permutationIdx + tmpBatchSize - 1);
    permutationIdx += tmpBatchSize;
  } else {
    referencePoints = arma::randperm(N, tmpBatchSize);
  }

  float result = 0;
  arma::frowvec Delta_TD_ms_given_candidate_result(nMedoids, arma::fill::zeros);

  // TODO(@motiwari): parallelization here?
  for (size_t ref = 0; ref < tmpBatchSize; ref++) {
    float cost =
        KMedoids::cachedLoss(
            data,
            distMat,
            referencePoints(ref),
            candidate,
            2);  // 2 for SWAP
    size_t nearest = (*assignments)(referencePoints(ref));
    if (cost < (*bestDistances)(referencePoints(ref))) {
      // When nearest(o) is removed, the loss of point reference is second(o). The two lines below, when summed together,
      // properly do the bookkeeping so that the loss of point reference will now become d_cr. This is why we add d_cr and
      // subtract off second(o).
      result += cost - (*bestDistances)(referencePoints(ref));
      Delta_TD_ms_given_candidate_result(nearest) += (*bestDistances)(referencePoints(ref)) - (*secondBestDistances)(referencePoints(ref));
    } else if (cost < (*secondBestDistances)(referencePoints(ref))) {
      // Every point has been assigned to its second closest medoid. If we remove the nearest medoid and add
      // point candidate in here, then the updated change in loss for removing the nearest medoid will be
      // d_cr - second(o), since the reference point reference will be assigned to candidate when candidate is added and nearest(o)
      // is removed. In the initial Delta_TMs, we added a +second(o) to the loss for assigning point reference to its
      // second closest medoid when nearest(o) is removed.
      Delta_TD_ms_given_candidate_result(nearest) += cost - (*secondBestDistances)(referencePoints(ref));
    }
  }

  // TODO(@motiwari): we can probably avoid this division
  //  if we look at total loss, not average loss
  result /= tmpBatchSize;
  Delta_TD_ms_given_candidate_result /= tmpBatchSize;
  return std::make_tuple(result, Delta_TD_ms_given_candidate_result);
}

void BanditFasterPAM::swapBanditFasterPAM(
  const arma::fmat& data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  arma::urowvec* medoidIndices,
  arma::urowvec* assignments) {

  size_t N = data.n_cols;
  size_t K = nMedoids;
  arma::frowvec bestDistances(N);
  arma::frowvec secondBestDistances(N);

  // TODO(@motiwari): This is O(kn). Can remove by carrying through assignments from the BUILD step, but that will be
  //  O(kn) too. Since we only do this O(kn) once, we can amortize it over all eager SWAP steps.
  //  This modifies bestDistances, secondBestDistanaces, and assignments in-place.
  KMedoids::calcBestDistancesSwap(
    data,
    distMat,
    medoidIndices,
    &bestDistances,
    &secondBestDistances,
    assignments);

  bool converged{false};
  size_t x_last{N};

  // Calculate initial removal loss for each medoid
  arma::frowvec Delta_TD_ms_initial = BanditFasterPAM::calcDeltaTDMs(
    assignments,
    &bestDistances,
    &secondBestDistances);


  size_t iter = 0;
  while (iter < maxIter && !converged) {
    for (size_t candidate = 0; candidate < data.n_cols; candidate++) {
      if (candidate == x_last) {
        converged = true;
        break;
      }

      float candidate_estimate{0};
      float candidate_lcb{0};
      float candidate_ucb{0};
      float candidate_sigma{0};
      size_t numSamples{0};

      arma::frowvec Delta_TD_ms_given_candidate(nMedoids, arma::fill::zeros);

      // Calculate sigma for constructing CIs
      candidate_sigma = swapSigma(
        candidate,
        data,
        distMat,
        &bestDistances,
        &secondBestDistances,
        assignments);

      bool continue_sampling{true};

      // While still possible to get < -0.01, keep sampling
      while (continue_sampling) {
        if (numSamples + batchSize >= N) {
          std::tuple<float, arma::frowvec> result = swapTarget(
              data,
              distMat,
              candidate,
              &bestDistances,
              &secondBestDistances,
              assignments,
              1); // exact

          float candidate_result = std::get<0>(result);
          candidate_estimate = candidate_result;
          candidate_ucb = candidate_result;
          candidate_lcb = candidate_result;
          Delta_TD_ms_given_candidate = std::get<1>(result);
          continue_sampling = false; // Unnecessary because
          numSamples += N;
        } else {
          std::tuple<float, arma::frowvec> result = swapTarget(
            data,
            distMat,
            candidate,
            &bestDistances,
            &secondBestDistances,
            assignments,
            0); // not exact

          float candidate_result = std::get<0>(result);
          Delta_TD_ms_given_candidate = ((numSamples * Delta_TD_ms_given_candidate) + (std::get<1>(result) * batchSize)) / (numSamples + batchSize);
          candidate_estimate = ((numSamples * candidate_estimate) + (candidate_result * batchSize)) / (numSamples + batchSize);
          numSamples += batchSize;

          float adjust = swapConfidence + std::log(N) + std::log(nMedoids);

          // TODO(@motiwari): 5 is a fudge factor. Really, we should consider each of Delta_TD_ms_given_candidate as an
          //  r.v. and create a new confidence bound on the difference.

          float confBoundDelta = 5 * candidate_sigma * std::sqrt(adjust / numSamples);
          candidate_ucb = candidate_estimate + confBoundDelta;
          candidate_lcb = candidate_estimate - confBoundDelta;

//          std::cout << "Bounds are " << candidate_lcb << " to " << candidate_ucb << "\n";
        }
        arma::frowvec total_Delta_TD_ms(nMedoids);
        total_Delta_TD_ms = Delta_TD_ms_initial + Delta_TD_ms_given_candidate;
//        std::cout << "\n\n";
//        std::cout << "Initial Delta_TD_ms: " << Delta_TD_ms_initial;
//        std::cout << "Delta_TD_ms_given_candidate" << Delta_TD_ms_given_candidate;
//        std::cout << "total Delta TD ms" << total_Delta_TD_ms;
//        std::cout << "Bounds are " << candidate_lcb << " to " << candidate_ucb;
//        std::cout << "\n\n";
        arma::uword best_m_idx = total_Delta_TD_ms.index_min();
        continue_sampling = total_Delta_TD_ms(best_m_idx) + candidate_lcb < - 0.01;
        // -0.01 to avoid precision errors
        // TODO(@motiwari): Move 0.01 to a constants file
        // TODO(@motiwari): This is not a statistically correct comparison. The Delta_TD_ms are also a r.v., with their
        //  source of randomness the subsampled data used to update it. Right now, we just compare to sigma and fiddle
        //  with sigma to get a reasonable CI
        // TODO(@motiwari): This -0.1 / N should be moved to an approximate comparison
//        std::cout << "Computed  " << numSamples << " for candidate " << candidate << "\n";
        if (total_Delta_TD_ms(best_m_idx) + candidate_ucb < -0.0001  && (*medoidIndices)(best_m_idx) != candidate) {
          // Perform Swap
          std::cout << "Swapped medoid index " << best_m_idx << " (medoid " << (*medoidIndices)(best_m_idx) << ") with "
                    << candidate << "\n";
          iter++;
          (*medoidIndices)(best_m_idx) = candidate;

          // TODO(@motiwari): This is O(kn) per swap. Instead, we could do a single pass over all kn pairs and keep a
          //  heap of all k distances to medoids per datapoint. That way we will have all the proper j-th nearest medoids
          //  which are necessary for this eager swapping out of medoids. This would incur O(kn) space though.
          //  We should also see how Schubert does this... I believe his algorithm is also O(kn).
          KMedoids::calcBestDistancesSwap(
            data,
            distMat,
            medoidIndices,
            &bestDistances,
            &secondBestDistances,
            assignments);

          // Update \Delta_TD_m's. This function modifies Delat_TD_ms in place
          Delta_TD_ms_initial = BanditFasterPAM::calcDeltaTDMs(
            assignments,
            &bestDistances,
            &secondBestDistances);

          x_last = candidate;
          break;  // Don't sample this arm any more, sample next datapoint
        }
      }
    }
  }
  steps = iter;

  // Call it one last time to update the loss
  // This is O(kn) but amortized over all eager SWAP steps
  // TODO(@motiwari): make this a call to calcLoss instead
  KMedoids::calcBestDistancesSwap(
    data,
    distMat,
    medoidIndices,
    &bestDistances,
    &secondBestDistances,
    assignments,
    false); // no swap performed, update loss
}
}  // namespace km
