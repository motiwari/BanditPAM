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

  steps = 0;
  // BanditFasterPAM uses uniform random sampling instead of BUILD since
  // SWAP is so fast that it is not worth it to use BUILD
  arma::urowvec medoidIndices = randomInitialization(data.n_cols);
  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);
  // we should swap even if k = 1 because we used uniform random sampling
  BanditFasterPAM::swapBanditFasterPAM(
      data,
      distMat,
      &medoidIndices,
      &assignments);

  medoidIndicesFinal = medoidIndices;
  labels = assignments;
}

arma::urowvec BanditFasterPAM::randomInitialization(
    size_t n) {
  // from https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
  const size_t rangeFrom = 0;
  const size_t rangeTo = n-1;
  // create a random device
  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_int_distribution<size_t> distr(rangeFrom, rangeTo);
  // generate k random numbers
  arma::urowvec res(nMedoids);
  for (size_t i = 0; i < nMedoids; i++) {
    res[i] = distr(generator);
  }

  return res;
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

std::tuple<float, arma::fvec> BanditFasterPAM::swapSigma(
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

  arma::fvec candidateSamples(batchSize, arma::fill::zeros);
  arma::fmat medoidSamples(nMedoids, batchSize, arma::fill::zeros);
  // since each row of medoidSamples will have a different number of elements
  // due to differing numbers of points that are assigned to each medoid,
  // we need to keep track of the next index to update for each medoid
  arma::fvec medoidSamplesIndex(nMedoids, arma::fill::zeros);
  // calculate change in loss for some subset of the data
  #pragma omp parallel for if (this->parallelize)
  for (size_t ref = 0; ref < batchSize; ref++) {
    // 0 for MISC when estimating sigma
    float cost =
        KMedoids::cachedLoss(data, distMat, candidate, referencePoints(ref), 0);

    if (cost < (*bestDistances)(ref)) {
      candidateSamples(ref) = cost - (*bestDistances)(ref);
    } else if (cost < (*secondBestDistances)(ref)) {
      size_t k = (*assignments)(ref);
      medoidSamples(k, medoidSamplesIndex(k)) = cost - (*bestDistances)(ref);
      medoidSamplesIndex(k)++;
    }
  }

  return {arma::stddev(candidateSamples), arma::stddev(medoidSamples, 0, 1)};
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
  arma::urowvec candidates(nMedoids, arma::fill::ones);
  arma::urowvec exactMask(nMedoids, arma::fill::zeros);
  arma::frowvec estimates(nMedoids, arma::fill::zeros);
  arma::frowvec Delta_TD_ms_given_candidate_result;
  arma::frowvec lcbs(nMedoids);
  arma::frowvec ucbs(nMedoids);
  arma::urowvec numSamples(nMedoids, arma::fill::zeros);

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

  size_t x_last{N};

  // Calculate initial removal loss for each medoid
  arma::frowvec Delta_TD_ms_initial = BanditFasterPAM::calcDeltaTDMs(
    assignments,
    &bestDistances,
    &secondBestDistances);


  size_t iter = 0;
  size_t nSwaps = 0;
  while (iter < maxIter) {
    iter++;
    permutationIdx = 0;
    size_t swapsBefore = nSwaps;

    for (size_t candidate = 0; candidate < N; candidate++) {
      if (candidate == x_last) {  // no improvements found
        break;
      }

      arma::frowvec Delta_TD_ms = Delta_TD_ms_initial;

      // skip this iteration since candidate is already a medoid
      if (candidate == (*medoidIndices)((*assignments)(candidate))) {
        continue;
      }

      // Calculate sigma for constructing CIs
      std::tuple<float, arma::fvec> sigmas = swapSigma(
        candidate,
        data,
        distMat,
        &bestDistances,
        &secondBestDistances,
        assignments);

      float candidate_sigma = std::get<0>(sigmas);
      arma::fvec medoid_sigmas = std::get<1>(sigmas);

      // Reset variables when starting a new swap
      candidates.fill(1);
      exactMask.fill(0);
      estimates.fill(0);
      numSamples.fill(0);

      // while there is at least one candidate (float comparison issues)
      while (arma::accu(candidates) > 1.5) {
        // compute exactly if it's been samples more than N times and
        // hasn't been computed exactly already
        arma::urowvec compute_exactly =
            ((numSamples + batchSize) >= N) != (exactMask);

        // TODO: update all comments
        // Get unique candidate medoids from the candidates (second index)
        // Store all k x T in estimates
        // TODO(@motiwari): Move this declaration outside loop
        // Need unique values over second index
        // Sum the different columns
        // if any index appears in at least one, compute it exactly
        // TODO(@motiwari): make sure we're only computing exactly
        // for the relevant candidates
        arma::uvec compute_exactly_targets =
            arma::find(arma::sum(compute_exactly, 0) >= 1);

        if (compute_exactly_targets.size() > 0) {
          std::tuple<float, arma::frowvec> result = swapTarget(
              data,
              distMat,
              candidate,
              &bestDistances,
              &secondBestDistances,
              assignments,
              1); // exact

          float candidate_result = std::get<0>(result);
          Delta_TD_ms_given_candidate_result = std::get<1>(result);
          Delta_TD_ms_given_candidate_result += candidate_result; // add shared accumulator to each arm

          // result will be k x T
          // Now update the correct indices
          estimates.cols(compute_exactly_targets) = Delta_TD_ms_given_candidate_result;
          ucbs.cols(compute_exactly_targets) = Delta_TD_ms_given_candidate_result;
          lcbs.cols(compute_exactly_targets) = Delta_TD_ms_given_candidate_result;
          exactMask.cols(compute_exactly_targets).fill(1);
          numSamples.cols(compute_exactly_targets) += N;
          candidates = (lcbs < ucbs.min()) && (exactMask == 0);
        }
        if (arma::accu(candidates) < precision) {
          break;
        }

        // candidate_targets should be of size k
        // Sum the different columns
        // if any index appears in at least one column, sample it
        arma::uvec candidate_targets = arma::find(
            arma::sum(candidates, 0) >= 1);

        // TODO: do we need to pass in candidate_targets?
        std::tuple<float, arma::frowvec> result = swapTarget(
          data,
          distMat,
          candidate,
          &bestDistances,
          &secondBestDistances,
          assignments,
          0); // not exact

        float candidate_result = std::get<0>(result);
        Delta_TD_ms_given_candidate_result = std::get<1>(result);
        Delta_TD_ms_given_candidate_result += candidate_result; // add shared accumulator to each arm

        // TODO: ensure that correct columns are accessed
        estimates.cols(candidate_targets) =
            ((numSamples.cols(candidate_targets)
              % estimates.cols(candidate_targets))
             + (Delta_TD_ms_given_candidate_result.cols(candidate_targets) * batchSize)) / (batchSize +
             numSamples.cols(
                 candidate_targets));

        // numSamples should be k x N
        // select the T of N columns that are candidates
        numSamples.cols(candidate_targets) += batchSize;

        // use the sum of variances formula to get the variance of each arm
        float candidate_variance = pow(candidate_sigma, 2);
        arma::fvec medoid_variances = arma::square(medoid_sigmas);
        medoid_variances += candidate_variance;
        // construct CIs for each arm using the sum of variances
        double criticalValue = 1.96;  // assume samples follow normal distribution
        arma::fmat confBoundDelta = criticalValue * arma::sqrt(medoid_variances / batchSize);

        ucbs.cols(candidate_targets) = estimates.cols(candidate_targets)
                               + confBoundDelta;
        lcbs.cols(candidate_targets) = estimates.cols(candidate_targets)
                                       - confBoundDelta;

        candidates = (lcbs < ucbs.min()) && (exactMask == 0);
      }

      arma::uword best_m_idx = lcbs.index_min();
      // TODO(@motiwari): This is not a statistically correct comparison. The Delta_TD_ms are also a r.v., with their
      //  source of randomness the subsampled data used to update it. Right now, we just compare to sigma and fiddle
      //  with sigma to get a reasonable CI
      // TODO(@motiwari): This -0.1 / N should be moved to an approximate comparison
      //        std::cout << "Computed  " << numSamples << " for candidate " << candidate << "\n";
      if (Delta_TD_ms_given_candidate_result(best_m_idx) + ucbs(best_m_idx) < -0.0001  && (*medoidIndices)(best_m_idx) != candidate) {
        // Perform Swap
        //          std::cout << "Swapped medoid index " << best_m_idx << " (medoid " << (*medoidIndices)(best_m_idx) << ") with "
        //                    << candidate << "\n";
        nSwaps++;
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

        // Update \Delta_TD_m's. This function modifies Delta_TD_ms in place
        Delta_TD_ms_initial = BanditFasterPAM::calcDeltaTDMs(
            assignments,
            &bestDistances,
            &secondBestDistances);

        x_last = candidate;
        break;  // Don't sample this arm any more, sample next datapoint
      }
    }
    // If no swaps were performed, we are done
    // this conditional ensures we do not converge prematurely
    if (nSwaps == swapsBefore) {
      break;
    }
  }
  steps = nSwaps;

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
