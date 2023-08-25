/**
 * @file banditpam.cpp
 * @date 2021-07-25
 *
 * Contains the primary C++ implementation of the BanditPAM code.
 */

#include "banditpam.hpp"

#include <armadillo>
#include <unordered_map>
#include <cmath>
#include <vector>

namespace km {
void BanditPAM::fitBanditPAM(
  const arma::fmat &inputData,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat) {
  data = arma::trans(inputData);

  // Note: even if we are using a distance matrix, we compute the permutation
  // in the block below because it is used elsewhere in the call stack
  // TODO(@motiwari): Remove need for data or permutation through when using
  //  a distance matrix
  if (this->useCache) {
    size_t n = data.n_cols;
    size_t m = fmin(n, cacheWidth);
    cache = new float[n * m];

#pragma omp parallel for if (this->parallelize)
    for (size_t idx = 0; idx < m * n; idx++) {
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

  arma::fmat medoidMatrix(data.n_rows, nMedoids);
  arma::urowvec medoidIndices(nMedoids);
  steps = 0;
  BanditPAM::build(data, distMat, &medoidIndices, &medoidMatrix);

  buildLoss = KMedoids::calcLoss(data, distMat, &medoidIndices);

  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);
  if (nMedoids > 1) {
    BanditPAM::swap(data, distMat, &medoidIndices, &medoidMatrix, &assignments);
  }

  medoidIndicesFinal = medoidIndices;
  labels = assignments;
}

arma::frowvec BanditPAM::buildSigma(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const arma::frowvec &bestDistances, const bool useAbsolute) {
  size_t N = data.n_cols;
  // Temporarily increase the batch size
  // for precise estimation of the standard deviation
  int originalBatchSize = batchSize;
  batchSize = std::min(1000, static_cast<int>(N / 4));
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly as
  //  last batch_size elements are dropped
  if (usePerm) {
    if ((permutationIdx + batchSize - 1) >= N) {
      permutationIdx = 0;
    }
    // inclusive of both indices
    referencePoints =
      permutation.subvec(permutationIdx, permutationIdx + batchSize - 1);
    permutationIdx += batchSize;
  } else {
    referencePoints = arma::randperm(N, batchSize);
  }

  arma::fvec sample(batchSize);
  arma::frowvec updated_sigma(N);
#pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < batchSize; j++) {
      // 0 for MISC
      float cost =
        KMedoids::cachedLoss(data, distMat, i, referencePoints(j), 0);
      if (useAbsolute) {
        sample(j) = cost;
      } else {
        sample(j) = cost < bestDistances(referencePoints(j))
                      ? cost
                      : bestDistances(referencePoints(j));
        sample(j) -= bestDistances(referencePoints(j));
      }
    }
    updated_sigma(i) = arma::stddev(sample);
  }
  // reset batchSize to the original batch size as it's a global variable
  // used by other functions (e.g. buildTarget, swapTarget)
  batchSize = originalBatchSize;
  return updated_sigma;
}

arma::frowvec BanditPAM::buildTarget(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const arma::uvec *target, const arma::frowvec *bestDistances,
  const bool useAbsolute, const bool exact = false) {
  size_t N = data.n_cols;
  size_t tmpBatchSize = batchSize;
  if (exact) {
    tmpBatchSize = N;
  }
  arma::frowvec results(target->n_rows, arma::fill::zeros);
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly
  //  as last batch_size elements are dropped
  if (usePerm) {
    if ((permutationIdx + tmpBatchSize - 1) >= N) {
      permutationIdx = 0;
    }
    // inclusive of both indices
    referencePoints =
      permutation.subvec(permutationIdx, permutationIdx + tmpBatchSize - 1);
    permutationIdx += tmpBatchSize;
  } else {
    referencePoints = arma::randperm(N, tmpBatchSize);
  }

#pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < target->n_rows; i++) {
    float total = 0;
    for (size_t j = 0; j < referencePoints.n_rows; j++) {
      float cost =
        KMedoids::cachedLoss(data, distMat, (*target)(i), referencePoints(j),
                             1);  // 1 for BUILD
      if (useAbsolute) {
        total += cost;
      } else {
        total += cost < (*bestDistances)(referencePoints(j))
                   ? cost
                   : (*bestDistances)(referencePoints(j));
        total -= (*bestDistances)(referencePoints(j));
      }
    }
    results(i) = total / tmpBatchSize;
  }
  return results;
}

void BanditPAM::build(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  arma::urowvec *medoidIndices, arma::fmat *medoids) {
  size_t N = data.n_cols;
  arma::frowvec N_mat(N);
  N_mat.fill(N);
  size_t p = N;
  bool useAbsolute = true;
  arma::frowvec estimates(N, arma::fill::zeros);
  arma::frowvec bestDistances(N);
  bestDistances.fill(std::numeric_limits<float>::infinity());
  arma::frowvec sigma(N);
  arma::urowvec candidates(N, arma::fill::ones);
  arma::frowvec lcbs(N);
  arma::frowvec ucbs(N);
  arma::frowvec numSamples(N, arma::fill::zeros);
  arma::frowvec exactMask(N, arma::fill::zeros);
  float minSigma = 0.001;

  // TODO(@motiwari): #pragma omp parallel for if (this->parallelize)?
  for (size_t k = 0; k < nMedoids; k++) {
    // instantiate medoids one-by-one
    permutationIdx = 0;
    candidates.fill(1);
    numSamples.fill(0);
    exactMask.fill(0);
    estimates.fill(0);
    // compute std dev amongst batch of reference points
    sigma = buildSigma(data, distMat, bestDistances, useAbsolute);

    // Replace the zero entries in sigma with the minimum non-zero
    // sigma value. Otherwise, some candidates could have
    // a 0 confidence interval.
    // This step prevents the overestimated candidates from
    // discarding underestimated candidates,
    // which could lead to suboptimal results.
    minSigma = arma::min(arma::nonzeros(sigma));
    sigma.elem(arma::find(sigma == 0.0)).fill(minSigma);

    while (arma::sum(candidates) > precision) {
      // TODO(@motiwari): Do not need a matrix for this comparison,
      //  use broadcasting
      arma::umat compute_exactly =
        ((numSamples + batchSize) >= N_mat) != exactMask;
      if (arma::accu(compute_exactly) > 0) {
        arma::uvec targets = find(compute_exactly);
        arma::frowvec result =
          buildTarget(data, distMat, &targets, &bestDistances, useAbsolute,
                      (true ? N > 0 : false));
        estimates.cols(targets) = result;
        ucbs.cols(targets) = result;
        lcbs.cols(targets) = result;
        exactMask.cols(targets).fill(1);
        numSamples.cols(targets) += N;
        candidates.cols(targets).fill(0);
      }
      if (arma::sum(candidates) < precision) {
        break;
      }
      arma::uvec targets = arma::find(candidates);
      arma::frowvec result = buildTarget(data, distMat, &targets,
                                         &bestDistances, useAbsolute, false);
      // update the running average
      estimates.cols(targets) =
        ((numSamples.cols(targets) % estimates.cols(targets)) +
         (result * batchSize)) /
        (batchSize + numSamples.cols(targets));
      numSamples.cols(targets) += batchSize;
      arma::frowvec adjust(targets.n_rows);
      adjust.fill(p);
      // Assume buildConfidence is given in logspace
      adjust = buildConfidence + arma::log(adjust);
      arma::frowvec confBoundDelta =
        sigma.cols(targets) % arma::sqrt(adjust / numSamples.cols(targets));
      ucbs.cols(targets) = estimates.cols(targets) + confBoundDelta;
      lcbs.cols(targets) = estimates.cols(targets) - confBoundDelta;
      candidates = (lcbs <= ucbs.min()) && (exactMask == 0);
    }

    medoidIndices->at(k) = lcbs.index_min();
    medoids->unsafe_col(k) = data.unsafe_col((*medoidIndices)(k));

// don't need to do this on final iteration
#pragma omp parallel for if (this->parallelize)
    for (size_t i = 0; i < N; i++) {
      float cost = KMedoids::cachedLoss(data, distMat, i, (*medoidIndices)(k),
                                        0);  // 0 for MISC
      if (cost < bestDistances(i)) {
        bestDistances(i) = cost;
      }
    }
    // use difference of loss for sigma and sampling, not absolute
    useAbsolute = false;
  }
}

arma::fmat BanditPAM::swapSigma(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const arma::frowvec *bestDistances, const arma::frowvec *secondBestDistances,
  const arma::urowvec *assignments) {
  size_t N = data.n_cols;
  size_t K = nMedoids;
  arma::fmat updated_sigma(K, N, arma::fill::zeros);
  // temporarily increase the batch size for precise estimation
  // of the std dev
  int originalBatchSize = batchSize;
  batchSize = std::min(1000, static_cast<int>(N / 4));
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly
  //  as last batch_size elements are dropped
  if (usePerm) {
    if ((permutationIdx + batchSize - 1) >= N) {
      permutationIdx = 0;
    }
    // inclusive of both indices
    referencePoints =
      permutation.subvec(permutationIdx, permutationIdx + batchSize - 1);
    permutationIdx += batchSize;
  } else {
    referencePoints = arma::randperm(N, batchSize);
  }

  arma::fvec sample(batchSize);
// for each considered swap
#pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < K * N; i++) {
    // extract data point of swap
    size_t n = i / K;
    size_t k = i % K;

    // calculate change in loss for some subset of the data
    for (size_t j = 0; j < batchSize; j++) {
      // 0 for MISC when estimating sigma
      float cost =
        KMedoids::cachedLoss(data, distMat, n, referencePoints(j), 0);

      if (k == (*assignments)(referencePoints(j))) {
        if (cost < (*secondBestDistances)(referencePoints(j))) {
          sample(j) = cost;
        } else {
          sample(j) = (*secondBestDistances)(referencePoints(j));
        }
      } else {
        if (cost < (*bestDistances)(referencePoints(j))) {
          sample(j) = cost;
        } else {
          sample(j) = (*bestDistances)(referencePoints(j));
        }
      }
      sample(j) -= (*bestDistances)(referencePoints(j));
    }
    updated_sigma(k, n) = arma::stddev(sample);
  }
  // reset batchSize to the original batch size as it's a global variable
  // used by other functions (e.g. buildTarget, swapTarget)
  batchSize = originalBatchSize;
  return updated_sigma;
}

arma::fmat BanditPAM::swapTarget(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  const arma::urowvec *medoidIndices, const arma::uvec *targets,
  const arma::frowvec *bestDistances, const arma::frowvec *secondBestDistances,
  const arma::urowvec *assignments, const bool exact = false) {
  const size_t N = data.n_cols;
  const size_t T = targets->n_rows;
  arma::fmat results(nMedoids, T, arma::fill::zeros);

  // Targets should be a list of indices for target CANDIDATE points
  // Then update all corresponding EXISTING MEDOID indices targets.
  // If targets is a T-length vector, then the return value should be
  // a matrix of size K x T. We should perform the appropriate update then
  // in the swap() function.
  //
  // An alternate method to do this would be to pass only the (m, c)
  // Points under consideration. Then we wouldn't need to update all
  // k virtual arms for each candidate, just the ones that are passed
  // However, this would incur a .find() call to find all pairs
  // (m, c) where c == c', the arm under consideration. I believe this
  // would be an O(kn) cost. Instead, may need to use another data
  // structure to avoid this .find() call, like a tree where the top-level
  // nodes are the candidates and the bottom-level nodes are the corresponding
  // virtual arms.
  // A jagged array might also do the trick.

  size_t tmpBatchSize = batchSize;
  if (exact) {
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
    referencePoints =
      permutation.subvec(permutationIdx, permutationIdx + tmpBatchSize - 1);
    permutationIdx += tmpBatchSize;
  } else {
    referencePoints = arma::randperm(N, tmpBatchSize);
  }

// TODO(@motiwari): Declare variables outside of loops
#pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < T; i++) {
    // TODO(@motiwari): pragma omp parallel for?
    for (size_t j = 0; j < tmpBatchSize; j++) {
      float cost =
        KMedoids::cachedLoss(data, distMat, (*targets)(i), referencePoints(j),
                             2);  // 2 for SWAP
      size_t k = (*assignments)(referencePoints(j));
      if (cost < (*bestDistances)(referencePoints(j))) {
        // We might be able to change this to
        // .eachrow(every column but k)
        // since arma does this in-place and it should not introduce
        // complexity
        results.col(i) += cost - (*bestDistances)(referencePoints(j));
      }

      // If cost < bd, this second term will subtract off the "new cost"
      // added by the all-column call above inside the if
      results(k, i) +=
        std::fmin(cost, (*secondBestDistances)(referencePoints(j))) -
        std::fmin(cost, (*bestDistances)(referencePoints(j)));
    }
  }
  // TODO(@motiwari): we can probably avoid this division
  //  if we look at total loss, not average loss
  results /= tmpBatchSize;
  return results;
}

void BanditPAM::swap(
  const arma::fmat &data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  arma::urowvec *medoidIndices, arma::fmat *medoids,
  arma::urowvec *assignments) {
  size_t N = data.n_cols;
  size_t p = N;

  arma::fmat sigma(nMedoids, N, arma::fill::zeros);

  arma::frowvec bestDistances(N);
  arma::frowvec secondBestDistances(N);
  bool swapPerformed = true;
  arma::umat candidates(nMedoids, N, arma::fill::ones);
  arma::umat exactMask(nMedoids, N, arma::fill::zeros);
  arma::fmat estimates(nMedoids, N, arma::fill::zeros);
  arma::fmat lcbs(nMedoids, N);
  arma::fmat ucbs(nMedoids, N);
  arma::umat numSamples(nMedoids, N, arma::fill::zeros);
  float minSigma = 0.001;

  // calculate quantities needed for swap, bestDistances and sigma
  calcBestDistancesSwap(data, distMat, medoidIndices, &bestDistances,
                        &secondBestDistances, assignments, swapPerformed);

  // continue making swaps while loss is decreasing
  while (swapPerformed && steps < maxIter) {
    steps++;
    permutationIdx = 0;

    sigma = swapSigma(data, distMat, &bestDistances, &secondBestDistances,
                      assignments);

    // Fill in the zero sigma entries with the non-zero minimum sigma value
    minSigma = arma::min(arma::nonzeros(sigma));
    sigma.elem(arma::find(sigma == 0.0)).fill(minSigma);

    // Reset variables when starting a new swap
    candidates.fill(1);
    exactMask.fill(0);
    estimates.fill(0);
    numSamples.fill(0);

    // while there is at least one candidate (float comparison issues)
    while (arma::accu(candidates) > 1.5) {
      // compute exactly if it's been samples more than N times and
      // hasn't been computed exactly already
      arma::umat compute_exactly =
        ((numSamples + batchSize) >= N) != (exactMask);

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
        arma::fmat result =
          swapTarget(data, distMat, medoidIndices, &compute_exactly_targets,
                     &bestDistances, &secondBestDistances, assignments,
                     (true ? N > 0 : false));

        // result will be k x T
        // Now update the correct indices
        estimates.cols(compute_exactly_targets) = result;
        ucbs.cols(compute_exactly_targets) = result;
        lcbs.cols(compute_exactly_targets) = result;
        exactMask.cols(compute_exactly_targets).fill(1);
        numSamples.cols(compute_exactly_targets) += N;
        candidates = (lcbs <= ucbs.min()) && (exactMask == 0);
      }
      if (arma::accu(candidates) < precision) {
        break;
      }

      // candidate_targets should be of size T
      // Sum the different columns
      // if any index appears in at least one column, sample it
      arma::uvec candidate_targets = arma::find(arma::sum(candidates, 0) >= 1);

      // result will be k x T
      arma::fmat result =
        swapTarget(data, distMat, medoidIndices, &candidate_targets,
                   &bestDistances, &secondBestDistances, assignments, false);

      // candidate_targets should be of size T, 1
      estimates.cols(candidate_targets) =
        ((numSamples.cols(candidate_targets) %
          estimates.cols(candidate_targets)) +
         (result * batchSize)) /
        (batchSize + numSamples.cols(candidate_targets));

      // numSamples should be k x N
      // select the T of N columns that are candidates
      numSamples.cols(candidate_targets) += batchSize;

      arma::fmat adjust(nMedoids, candidate_targets.size());
      // TODO(@motiwari): Move this ::fill to the previous line
      adjust.fill(p);
      // Assume swapConfidence is given in logspace
      adjust = swapConfidence + arma::log(adjust);
      arma::fmat confBoundDelta =
        sigma.cols(candidate_targets) %
        arma::sqrt(adjust / numSamples.cols(candidate_targets));
      ucbs.cols(candidate_targets) =
        estimates.cols(candidate_targets) + confBoundDelta;
      lcbs.cols(candidate_targets) =
        estimates.cols(candidate_targets) - confBoundDelta;

      candidates = (lcbs <= ucbs.min()) && (exactMask == 0);
    }

    // Perform the medoid switch
    arma::uword newMedoid = lcbs.index_min();
    size_t k = newMedoid % nMedoids;
    size_t n = newMedoid / nMedoids;
    swapPerformed = (*medoidIndices)(k) != n;

    if (swapPerformed) {
      (*medoidIndices)(k) = n;
      medoids->col(k) = data.col((*medoidIndices)(k));
    }

    calcBestDistancesSwap(data, distMat, medoidIndices, &bestDistances,
                          &secondBestDistances, assignments, swapPerformed);
  }
}
}  // namespace km
