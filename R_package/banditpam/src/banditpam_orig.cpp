/**
 * @file banditpam_orig.cpp
 * @date 2021-07-25
 *
 * Contains the primary C++ implementation of the BanditPAM_orig code.
 */

#include "banditpam_orig.hpp"

#include <unordered_map>
#include <cmath>

namespace km {
void BanditPAM_orig::fitBanditPAM_orig(
  const arma_mat& inputData,
  std::optional<std::reference_wrapper<const arma_mat>> distMat) {
  data = arma::trans(inputData);

  // Note: even if we are using a distance matrix, we compute the permutation
  // in the block below because it is used elsewhere in the call stack
  // TODO(@motiwari): Remove need for data or permutation through when using
  //  a distance matrix
  if (this->useCache) {
    size_t n = data.n_cols;
    size_t m = fmin(n, cacheWidth);
    cache = new banditpam_float[n * m];

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

  arma_mat medoidMatrix(data.n_rows, nMedoids);
  arma::urowvec medoidIndices(nMedoids);
  steps = 0;
  BanditPAM_orig::build(data, distMat, &medoidIndices, &medoidMatrix);

  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);

  BanditPAM_orig::swap(
      data,
      distMat,
      &medoidIndices,
      &medoidMatrix,
      &assignments);

  medoidIndicesFinal = medoidIndices;
  labels = assignments;
}

arma_rowvec BanditPAM_orig::buildSigma(
  const arma_mat& data,
  std::optional<std::reference_wrapper<const arma_mat>> distMat,
  const arma_rowvec& bestDistances,
  const bool useAbsolute) {
  size_t N = data.n_cols;
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly as
  //  last batch_size elements are dropped
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

  arma_vec sample(batchSize);
  arma_rowvec updated_sigma(N);
  #pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < batchSize; j++) {
      // 0 for MISC
      banditpam_float cost =
          KMedoids::cachedLoss(data, distMat, i, referencePoints(j), 0);
      if (useAbsolute) {
        sample(j) = cost;
      } else {
        sample(j) = cost < bestDistances(referencePoints(j))
                          ? cost : bestDistances(referencePoints(j));
        sample(j) -= bestDistances(referencePoints(j));
      }
    }
    updated_sigma(i) = arma::stddev(sample);
  }
  return updated_sigma;
}

arma_rowvec BanditPAM_orig::buildTarget(
  const arma_mat& data,
  std::optional<std::reference_wrapper<const arma_mat>> distMat,
  const arma::uvec* target,
  const arma_rowvec* bestDistances,
  const bool useAbsolute,
  const size_t exact = 0) {
  size_t N = data.n_cols;
  size_t tmpBatchSize = batchSize;
  if (exact > 0) {
    tmpBatchSize = N;
  }
  arma_rowvec estimates(target->n_rows, arma::fill::zeros);
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly
  //  as last batch_size elements are dropped
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

  #pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < target->n_rows; i++) {
    banditpam_float total = 0;
    for (size_t j = 0; j < referencePoints.n_rows; j++) {
      banditpam_float cost =
          KMedoids::cachedLoss(
              data,
              distMat,
              (*target)(i),
              referencePoints(j),
              1);  // 1 for BUILD
      if (useAbsolute) {
        total += cost;
      } else {
        total += cost < (*bestDistances)(referencePoints(j))
                      ? cost : (*bestDistances)(referencePoints(j));
        total -= (*bestDistances)(referencePoints(j));
      }
    }
    estimates(i) = total / tmpBatchSize;
  }
  return estimates;
}

void BanditPAM_orig::build(
  const arma_mat& data,
  std::optional<std::reference_wrapper<const arma_mat>> distMat,
  arma::urowvec* medoidIndices,
  arma_mat* medoids) {
  size_t N = data.n_cols;
  arma_rowvec N_mat(N);
  N_mat.fill(N);
  size_t p = N;
  bool useAbsolute = true;
  arma_rowvec estimates(N, arma::fill::zeros);
  arma_rowvec bestDistances(N);
  bestDistances.fill(std::numeric_limits<banditpam_float>::infinity());
  arma_rowvec sigma(N);
  arma::urowvec candidates(N, arma::fill::ones);
  arma_rowvec lcbs(N);
  arma_rowvec ucbs(N);
  arma_rowvec numSamples(N, arma::fill::zeros);
  arma_rowvec exactMask(N, arma::fill::zeros);

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

    while (arma::sum(candidates) > precision) {
      // TODO(@motiwari): Do not need a matrix for this comparison,
      //  use broadcasting
      arma::umat compute_exactly =
          ((numSamples + batchSize) >= N_mat) != exactMask;
      if (arma::accu(compute_exactly) > 0) {
        arma::uvec targets = find(compute_exactly);
        arma_rowvec result = buildTarget(
          data,
          distMat,
          &targets,
          &bestDistances,
          useAbsolute,
          N);
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
      arma_rowvec result = buildTarget(
        data,
        distMat,
        &targets,
        &bestDistances,
        useAbsolute,
        0);
      // update the running average
      estimates.cols(targets) =
        ((numSamples.cols(targets) % estimates.cols(targets)) +
         (result * batchSize)) /
        (batchSize + numSamples.cols(targets));
      numSamples.cols(targets) += batchSize;
      arma_rowvec adjust(targets.n_rows);
      adjust.fill(p);
      // Assume buildConfidence is given in logspace
      adjust = buildConfidence + arma::log(adjust);
      arma_rowvec confBoundDelta =
        sigma.cols(targets) %
        arma::sqrt(adjust / numSamples.cols(targets));
      ucbs.cols(targets) = estimates.cols(targets) + confBoundDelta;
      lcbs.cols(targets) = estimates.cols(targets) - confBoundDelta;
      candidates = (lcbs < ucbs.min()) && (exactMask == 0);
    }

    medoidIndices->at(k) = lcbs.index_min();
    medoids->unsafe_col(k) = data.unsafe_col((*medoidIndices)(k));

    // don't need to do this on final iteration
    #pragma omp parallel for if (this->parallelize)
    for (size_t i = 0; i < N; i++) {
      banditpam_float cost = KMedoids::cachedLoss(
        data,
        distMat,
        i,
        (*medoidIndices)(k),
        0);  // 0 for MISC
      if (cost < bestDistances(i)) {
        bestDistances(i) = cost;
      }
    }
    // use difference of loss for sigma and sampling, not absolute
    useAbsolute = false;
  }
}

arma_mat BanditPAM_orig::swapSigma(
  const arma_mat& data,
  std::optional<std::reference_wrapper<const arma_mat>> distMat,
  const arma_rowvec* bestDistances,
  const arma_rowvec* secondBestDistances,
  const arma::urowvec* assignments) {
  size_t N = data.n_cols;
  size_t K = nMedoids;
  arma_mat updated_sigma(K, N, arma::fill::zeros);
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

  arma_vec sample(batchSize);
  // for each considered swap
  #pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < K * N; i++) {
    // extract data point of swap
    size_t n = i / K;
    size_t k = i % K;

    // calculate change in loss for some subset of the data
    for (size_t j = 0; j < batchSize; j++) {
      // 0 for MISC when estimating sigma
      banditpam_float cost =
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
  return updated_sigma;
}

arma_vec BanditPAM_orig::swapTarget(
  const arma_mat& data,
  std::optional<std::reference_wrapper<const arma_mat>> distMat,
  const arma::urowvec* medoidIndices,
  const arma::uvec* targets,
  const arma_rowvec* bestDistances,
  const arma_rowvec* secondBestDistances,
  const arma::urowvec* assignments,
  const size_t exact = 0) {
  size_t N = data.n_cols;
  arma_vec estimates(targets->n_rows, arma::fill::zeros);

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

  // TODO(@motiwari): Declare variables outside of loops
  #pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < targets->n_rows; i++) {
    banditpam_float total = 0;
    // extract data point of swap
    size_t n = (*targets)(i) / medoidIndices->n_cols;
    size_t k = (*targets)(i) % medoidIndices->n_cols;
    // calculate total loss for some subset of the data
    for (size_t j = 0; j < tmpBatchSize; j++) {
      // 2 for SWAP
      banditpam_float cost =
          KMedoids::cachedLoss(data, distMat, n, referencePoints(j), 2);
      if (k == (*assignments)(referencePoints(j))) {
        if (cost < (*secondBestDistances)(referencePoints(j))) {
          total += cost;
        } else {
          total += (*secondBestDistances)(referencePoints(j));
        }
      } else {
        if (cost < (*bestDistances)(referencePoints(j))) {
          total += cost;
        } else {
          total += (*bestDistances)(referencePoints(j));
        }
      }
      total -= (*bestDistances)(referencePoints(j));
    }
    // TODO(@motiwari): we can probably avoid this division
    //  if we look at total loss, not average loss
    estimates(i) = total / referencePoints.n_rows;
  }
  return estimates;
}

void BanditPAM_orig::swap(
  const arma_mat& data,
  std::optional<std::reference_wrapper<const arma_mat>> distMat,
  arma::urowvec* medoidIndices,
  arma_mat* medoids,
  arma::urowvec* assignments) {
  size_t N = data.n_cols;
  size_t p = N;

  arma_mat sigma(nMedoids, N, arma::fill::zeros);

  arma_rowvec bestDistances(N);
  arma_rowvec secondBestDistances(N);
  bool swapPerformed = true;
  arma::umat candidates(nMedoids, N, arma::fill::ones);
  arma::umat exactMask(nMedoids, N, arma::fill::zeros);
  arma_mat estimates(nMedoids, N, arma::fill::zeros);
  arma_mat lcbs(nMedoids, N);
  arma_mat ucbs(nMedoids, N);
  arma::umat numSamples(nMedoids, N, arma::fill::zeros);

  // calculate quantities needed for swap, bestDistances and sigma
  calcBestDistancesSwap(
    data,
    distMat,
    medoidIndices,
    &bestDistances,
    &secondBestDistances,
    assignments,
    swapPerformed);

  // continue making swaps while loss is decreasing
  while (swapPerformed && steps < maxIter) {
    steps++;
    permutationIdx = 0;

    sigma = swapSigma(
      data,
      distMat,
      &bestDistances,
      &secondBestDistances,
      assignments);

    // Reset variables when starting a new swap
    candidates.fill(1);
    exactMask.fill(0);
    estimates.fill(0);
    numSamples.fill(0);

    // while there is at least one candidate (banditpam_float comparison issues)
    while (arma::accu(candidates) > 1.5) {
      // compute exactly if it's been samples more than N times and
      // hasn't been computed exactly already
      arma::umat compute_exactly =
          ((numSamples + batchSize) >= N) != (exactMask);
      arma::uvec targets = arma::find(compute_exactly);

      if (targets.size() > 0) {
        arma_vec result = swapTarget(
          data,
          distMat,
          medoidIndices,
          &targets,
          &bestDistances,
          &secondBestDistances,
          assignments,
          N);
        estimates.elem(targets) = result;
        ucbs.elem(targets) = result;
        lcbs.elem(targets) = result;
        exactMask.elem(targets).fill(1);
        numSamples.elem(targets) += N;
        candidates = (lcbs < ucbs.min()) && (exactMask == 0);
      }
      if (arma::accu(candidates) < precision) {
        break;
      }
      targets = arma::find(candidates);
      arma_vec result = swapTarget(
          data,
          distMat,
          medoidIndices,
          &targets,
          &bestDistances,
          &secondBestDistances,
          assignments,
          0);
      estimates.elem(targets) =
          ((numSamples.elem(targets) % estimates.elem(targets)) +
           (result * batchSize)) /
          (batchSize + numSamples.elem(targets));
      numSamples.elem(targets) += batchSize;
      arma_vec adjust(targets.n_rows);
      adjust.fill(p);
      // Assume swapConfidence is given in logspace
      adjust = swapConfidence + arma::log(adjust);
      arma_vec confBoundDelta = sigma.elem(targets) %
                    arma::sqrt(adjust / numSamples.elem(targets));
      ucbs.elem(targets) = estimates.elem(targets) + confBoundDelta;
      lcbs.elem(targets) = estimates.elem(targets) - confBoundDelta;
      candidates = (lcbs < ucbs.min()) && (exactMask == 0);
    }

    // Perform the medoid switch
    arma::uword newMedoid = lcbs.index_min();
    // extract old and new medoids of swap
    size_t k = newMedoid % medoids->n_cols;
    size_t n = newMedoid / medoids->n_cols;
    swapPerformed = (*medoidIndices)(k) != n;
    steps++;

    if (swapPerformed) {
      (*medoidIndices)(k) = n;
      medoids->col(k) = data.col((*medoidIndices)(k));
    }
    calcBestDistancesSwap(
      data,
      distMat,
      medoidIndices,
      &bestDistances,
      &secondBestDistances,
      assignments,
      swapPerformed);
  }
}
}  // namespace km
