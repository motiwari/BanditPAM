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

namespace km {
void BanditPAM::fitBanditPAM(const arma::fmat& inputData) {
  data = arma::trans(inputData);

  if (this->useCacheP) {
    size_t n = data.n_cols;
    size_t m = fmin(n, ceil(log10(data.n_cols) * cacheMultiplier));
    cache = new float[n * m];

    #pragma omp parallel for
    for (size_t idx = 0; idx < m*n; idx++) {
      cache[idx] = -1;  // TODO(@motiwari): need better value here
    }

    permutation = arma::randperm(n);
    permutationIdx = 0;
    reindex = {};
    // TODO(@motiwari): Can we parallelize this?
    for (size_t counter = 0; counter < m; counter++) {
      reindex[permutation[counter]] = counter;
    }
  }

  arma::fmat medoidMatrix(data.n_rows, nMedoids);
  arma::urowvec medoidIndices(nMedoids);
  BanditPAM::build(data, &medoidIndices, &medoidMatrix);
  steps = 0;

  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);
  BanditPAM::swap(data, &medoidIndices, &medoidMatrix, &assignments);
  medoidIndicesFinal = medoidIndices;
  labels = assignments;
}

arma::frowvec BanditPAM::buildSigma(
  const arma::fmat& data,
  const arma::frowvec& bestDistances,
  const bool useAbsolute) {
  size_t N = data.n_cols;
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly as
  // last batch_size elements are dropped
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

  arma::fvec sample(batchSize);
  arma::frowvec updated_sigma(N);
  #pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < batchSize; j++) {
      float cost = KMedoids::cachedLoss(data, i, referencePoints(j));
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

arma::frowvec BanditPAM::buildTarget(
  const arma::fmat& data,
  const arma::uvec* target,
  const arma::frowvec* bestDistances,
  const bool useAbsolute,
  const size_t exact = 0) {
  size_t N = data.n_cols;
  size_t tmpBatchSize = batchSize;
  if (exact > 0) {
    tmpBatchSize = N;
  }
  arma::frowvec estimates(target->n_rows, arma::fill::zeros);
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly
  // as last batch_size elements are dropped
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

  #pragma omp parallel for
  for (size_t i = 0; i < target->n_rows; i++) {
    float total = 0;
    for (size_t j = 0; j < referencePoints.n_rows; j++) {
      float cost =
        KMedoids::cachedLoss(data, (*target)(i), referencePoints(j));
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

void BanditPAM::build(
  const arma::fmat& data,
  arma::urowvec* medoidIndices,
  arma::fmat* medoids) {
  size_t N = data.n_cols;
  arma::frowvec N_mat(N);
  N_mat.fill(N);
  size_t p = (buildConfidence * N);
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

  // TODO(@motiwari): #pragma omp parallel for?
  for (size_t k = 0; k < nMedoids; k++) {
    // instantiate medoids one-by-one
    permutationIdx = 0;
    size_t step_count = 0;
    candidates.fill(1);
    numSamples.fill(0);
    exactMask.fill(0);
    estimates.fill(0);
    // compute std dev amongst batch of reference points
    sigma = buildSigma(data, bestDistances, useAbsolute);

    while (arma::sum(candidates) > precision) {
      // TODO(@motiwari): Do not need a matrix for this comparison,
      // use broadcasting
      arma::umat compute_exactly =
        ((numSamples + batchSize) >= N_mat) != exactMask;
      if (arma::accu(compute_exactly) > 0) {
        arma::uvec targets = find(compute_exactly);
        arma::frowvec result = buildTarget(
          data,
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
      arma::frowvec result = buildTarget(
        data,
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
      arma::frowvec adjust(targets.n_rows);
      adjust.fill(p);
      adjust = arma::log(adjust);
      arma::frowvec confBoundDelta =
        sigma.cols(targets) %
        arma::sqrt(adjust / numSamples.cols(targets));
      ucbs.cols(targets) = estimates.cols(targets) + confBoundDelta;
      lcbs.cols(targets) = estimates.cols(targets) - confBoundDelta;
      candidates = (lcbs < ucbs.min()) && (exactMask == 0);
      step_count++;
    }

    medoidIndices->at(k) = lcbs.index_min();
    medoids->unsafe_col(k) = data.unsafe_col((*medoidIndices)(k));

    // don't need to do this on final iteration
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        float cost = KMedoids::cachedLoss(data, i, (*medoidIndices)(k));
        if (cost < bestDistances(i)) {
            bestDistances(i) = cost;
        }
    }
    // use difference of loss for sigma and sampling, not absolute
    useAbsolute = false;
  }
}

arma::fmat BanditPAM::swapSigma(
  const arma::fmat& data,
  const arma::frowvec* bestDistances,
  const arma::frowvec* secondBestDistances,
  const arma::urowvec* assignments) {
  size_t N = data.n_cols;
  size_t K = nMedoids;
  arma::fmat updated_sigma(K, N, arma::fill::zeros);
  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly
  // as last batch_size elements are dropped
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

  arma::fvec sample(batchSize);
  // for each considered swap
  #pragma omp parallel for
  for (size_t i = 0; i < K * N; i++) {
    // extract data point of swap
    size_t n = i / K;
    size_t k = i % K;

    // calculate change in loss for some subset of the data
    for (size_t j = 0; j < batchSize; j++) {
      float cost = KMedoids::cachedLoss(data, n, referencePoints(j));

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

arma::fvec BanditPAM::swapTarget(
  const arma::fmat& data,
  const arma::urowvec* medoidIndices,
  const arma::uvec* targets,
  const arma::frowvec* bestDistances,
  const arma::frowvec* secondBestDistances,
  const arma::urowvec* assignments,
  const size_t exact = 0) {
  size_t N = data.n_cols;
  arma::fvec estimates(targets->n_rows, arma::fill::zeros);

  size_t tmpBatchSize = batchSize;
  if (exact > 0) {
    tmpBatchSize = N;
  }

  arma::uvec referencePoints;
  // TODO(@motiwari): Make this wraparound properly
  // as last batch_size elements are dropped
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

  // for each considered swap
  #pragma omp parallel for
  for (size_t i = 0; i < targets->n_rows; i++) {
    float total = 0;
    // extract data point of swap
    size_t n = (*targets)(i) / medoidIndices->n_cols;
    size_t k = (*targets)(i) % medoidIndices->n_cols;
    // calculate total loss for some subset of the data
    for (size_t j = 0; j < tmpBatchSize; j++) {
      float cost = KMedoids::cachedLoss(data, n, referencePoints(j));
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
    estimates(i) = total / referencePoints.n_rows;
  }
  return estimates;
}

void BanditPAM::swap(
  const arma::fmat& data,
  arma::urowvec* medoidIndices,
  arma::fmat* medoids,
  arma::urowvec* assignments) {
  size_t N = data.n_cols;
  size_t p = (N * nMedoids * swapConfidence);

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

  // calculate quantities needed for swap, bestDistances and sigma
  calcBestDistancesSwap(
    data,
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
      &bestDistances,
      &secondBestDistances,
      assignments);

    candidates.fill(1);
    exactMask.fill(0);
    estimates.fill(0);
    numSamples.fill(0);

    // while there is at least one candidate (float comparison issues)
    while (arma::accu(candidates) > 0.5) {
      // compute exactly if it's been samples more than N times and
      // hasn't been computed exactly already
      arma::umat compute_exactly =
        ((numSamples + batchSize) >= N) != (exactMask);
      arma::uvec targets = arma::find(compute_exactly);

      if (targets.size() > 0) {
          arma::fvec result = swapTarget(
            data,
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
      arma::fvec result = swapTarget(
        data,
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
      arma::fvec adjust(targets.n_rows);
      adjust.fill(p);
      adjust = arma::log(adjust);
      arma::fvec confBoundDelta = sigma.elem(targets) %
                          arma::sqrt(adjust / numSamples.elem(targets));

      ucbs.elem(targets) = estimates.elem(targets) + confBoundDelta;
      lcbs.elem(targets) = estimates.elem(targets) - confBoundDelta;
      candidates = (lcbs < ucbs.min()) && (exactMask == 0);
      targets = arma::find(candidates);
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
        medoidIndices,
        &bestDistances,
        &secondBestDistances,
        assignments,
        swapPerformed);
  }
}
}  // namespace km
