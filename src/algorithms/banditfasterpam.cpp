/**
 * @file BanditFasterPAM.cpp
 * @date 2021-07-25
 *
 * Contains a C++ implementation of the BanditFasterPAM algorithm.
 * The original FasterPAM paper is:
 * 1) Erich Schubert and Peter J. Rousseeuw: Fast and Eager k-Medoids Clustering:
 *  O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
 */

#include "banditfasterpam.hpp"

#include <armadillo>
#include <unordered_map>
#include <string>

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

  arma::fmat medoidMatrix(data.n_rows, nMedoids);
  arma::urowvec medoidIndices(nMedoids);
  steps = 0;
  BanditFasterPAM::randomInitialization(data.n_cols, data, &medoidIndices, &medoidMatrix);

  // TODO(@Adarsh321123): uncomment later
//  buildLoss = KMedoids::calcLoss(data, distMat, &medoidIndices);

  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);
  assignments.fill(std::numeric_limits<size_t>::max());
  arma::urowvec secondAssignments(data.n_cols);
  secondAssignments.fill(std::numeric_limits<size_t>::max());
  // we should swap even if k = 1 since we used uniform random sampling to
  // initialize the one medoid
  BanditFasterPAM::swap(
      data,
      distMat,
      &medoidIndices,
      &medoidMatrix,
      &assignments,
      &secondAssignments);

  medoidIndicesFinal = medoidIndices;
  labels = assignments;
}

void BanditFasterPAM::randomInitialization(
    size_t n,
    const arma::fmat &data,
    arma::urowvec *medoidIndices,
    arma::fmat *medoids) {
  // from https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
  const size_t rangeFrom = 0;
  const size_t rangeTo = n-1;
  // create a random device
  std::random_device randDev;
  // Use the provided seed to initialize the random number generator
  std::mt19937 generator(0); // Use for comparison with FasterPAM
  //  std::mt19937 generator(randDev());
  std::uniform_int_distribution<size_t> distr(rangeFrom, rangeTo);
  // generate k random numbers
  arma::urowvec res(nMedoids);
  for (size_t k = 0; k < nMedoids; k++) {
    medoidIndices->at(k) = distr(generator);
    medoids->unsafe_col(k) = data.unsafe_col((*medoidIndices)(k));
  }
}

arma::fmat BanditFasterPAM::swapSigma(
    const arma::fmat &data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    const arma::frowvec *bestDistances,
    const arma::frowvec *secondBestDistances,
    const arma::urowvec *assignments,
    const size_t activeColumn) {
  size_t N = data.n_cols;
  size_t K = nMedoids;
  arma::fmat updated_sigma(K, N, arma::fill::zeros);
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

  arma::fvec sample(batchSize);
// for each considered swap
#pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < K * N; i++) {
    // extract data point of swap
    size_t n = i / K;

    // no need to calculate the sigma for non-active columns
    if (n != activeColumn) {
      continue;
    }

    size_t k = i % K;

    // calculate change in loss for some subset of the data
    for (size_t j = 0; j < batchSize; j++) {
      // 0 for MISC when estimating sigma
      float cost =
          KMedoids::cachedLoss(data, distMat, n,
                               referencePoints(j), 0);

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

arma::fmat BanditFasterPAM::swapTarget(
    const arma::fmat &data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    const arma::urowvec *medoidIndices,
    const arma::uvec *targets,
    const arma::frowvec *bestDistances,
    const arma::frowvec *secondBestDistances,
    const arma::urowvec *assignments,
    const bool exact = false) {
  const size_t N = data.n_cols;
  const size_t T = targets->n_rows;
  arma::fmat results(nMedoids, T, arma::fill::zeros);

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
    referencePoints = permutation.subvec(
        permutationIdx,
        permutationIdx + tmpBatchSize - 1);
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
          KMedoids::cachedLoss(
              data,
              distMat,
              (*targets)(i),
              referencePoints(j),
              2);  // 2 for SWAP
      size_t k = (*assignments)(referencePoints(j));
      if (cost < (*bestDistances)(referencePoints(j))) {
        // We might be able to change this to
        // .eachrow(every column but k)
        // since arma does this in-place and it should not introduce
        // complexity
        results.col(i) +=
            cost - (*bestDistances)(referencePoints(j));
      }

      // If cost < bd, this second term will subtract off the "new cost"
      // added by the all-column call above inside the if
      results(k, i) +=
          std::fmin(cost,
                    (*secondBestDistances)(referencePoints(j))) -
          std::fmin(cost, (*bestDistances)(referencePoints(j)));
    }
  }

  // TODO(@motiwari): we can probably avoid this division
  //  if we look at total loss, not average loss
  results /= tmpBatchSize;
  return results;
}

void BanditFasterPAM::calcBestDistancesSwapInitial(
    const arma::fmat &data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    const arma::urowvec *medoidIndices,
    arma::frowvec *bestDistances,
    arma::frowvec *secondBestDistances,
    arma::urowvec *assignments,
    arma::urowvec *secondAssignments,
    const bool swapPerformed) {
  // this function is similar to calcBestDistancesSwap, but it has slightly
  // different conditional logic and also updates secondAssignments (useful
  // later on)
#pragma omp parallel for if (this->parallelize)
  for (size_t i = 0; i < data.n_cols; i++) {
    float best = std::numeric_limits<float>::infinity();
    float second = std::numeric_limits<float>::infinity();
    for (size_t k = 0; k < medoidIndices->n_cols; k++) {
      // 0 for MISC
      float cost =
          KMedoids::cachedLoss(data, distMat, i,
                               (*medoidIndices)(k), 0);
      if (cost < best || i == (*medoidIndices)(k)) {
        (*secondAssignments)(i) = (*assignments)(i);
        (*assignments)(i) = k;
        second = best;
        best = cost;
      } else if ((*secondAssignments)(i) == std::numeric_limits<size_t>::max() || cost < second) {
        (*secondAssignments)(i) = k;
        second = cost;
      }
    }
    (*bestDistances)(i) = best;
    (*secondBestDistances)(i) = second;
  }

  if (!swapPerformed) {
    averageLoss = arma::accu(*bestDistances) / data.n_cols;
  }
}

std::tuple<size_t, float> BanditFasterPAM::updateSecondNearest(
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    const arma::urowvec *medoidIndices,
    size_t n,
    size_t k,
    size_t o,
    float djo) {
  // after making a swap, we need to update the second nearest medoid and
  // second nearest distance accordingly
  size_t secondMedoid = k;
  float secondDistance = djo;
  for (size_t i = 0; i < (*medoidIndices).size(); i++) {
    size_t mi = (*medoidIndices)(i);
    if (i == n || i == k) {
      continue;
    }

    float cost = KMedoids::cachedLoss(data, distMat, o,
                                      mi, 0);
    if (cost < secondDistance) {
      secondMedoid = i;
      secondDistance = cost;
    }
  }

  return {secondMedoid, secondDistance};
}

void BanditFasterPAM::calcBestDistancesSwapWithFPOptimizations(
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    const arma::urowvec *medoidIndices,
    arma::frowvec *bestDistances,
    arma::frowvec *secondBestDistances,
    arma::urowvec *assignments,
    arma::urowvec *secondAssignments,
    size_t k,
    size_t n) {
  // this function is similar to calcBestDistancesSwap, but it takes cues from
  // FasterPAM to be highly optimized.
  // this is because it doesn't compute all kn distances; it only computes
  // n distances and then calls updateSecondNearest if needed

  // update the distances from doing the swap
  for (size_t o = 0; o < (*bestDistances).size(); o++) {
    // special case where the index is the non-medoid we just swapped to become
    // the medoid
    if (o == n) {
      if ((*assignments)(o) != k) {
        (*secondAssignments)(o) = (*assignments)(o);
        (*secondBestDistances)(o) = (*bestDistances)(o);
      }

      (*assignments)(o) = k;
      (*bestDistances)(o) = 0;
      continue;
    }

    float djo = KMedoids::cachedLoss(data, distMat, o,
                                     n, 0);
    if ((*assignments)(o) == k) {
      if (djo < (*secondBestDistances)(o)) {
        (*assignments)(o) = k;
        (*bestDistances)(o) = djo;
      } else {
        (*assignments)(o) = (*secondAssignments)(o);
        (*bestDistances)(o) = (*secondBestDistances)(o);
        std::tuple<size_t, float> paramsSecond = updateSecondNearest(distMat, medoidIndices, (*assignments)(o), k, o, djo);
        (*secondAssignments)(o) = std::get<0>(paramsSecond);
        (*secondBestDistances)(o) = std::get<1>(paramsSecond);
      }
    } else {
      if (djo < (*bestDistances)(o)) {
        (*secondAssignments)(o) = (*assignments)(o);
        (*secondBestDistances)(o) = (*bestDistances)(o);
        (*assignments)(o) = k;
        (*bestDistances)(o) = djo;
      } else if (djo < (*secondBestDistances)(o)) {
        (*secondAssignments)(o) = k;
        (*secondBestDistances)(o) = djo;
      } else if ((*secondAssignments)(o) == k) {
        std::tuple<size_t, float> paramsSecond = updateSecondNearest(distMat, medoidIndices, (*assignments)(o), k, o, djo);
        (*secondAssignments)(o) = std::get<0>(paramsSecond);
        (*secondBestDistances)(o) = std::get<1>(paramsSecond);
      }
    }
  }
}

void BanditFasterPAM::swap(
    const arma::fmat &data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec *medoidIndices,
    arma::fmat *medoids,
    arma::urowvec *assignments,
    arma::urowvec *secondAssignments) {
  size_t N = data.n_cols;
  size_t p = N;
  bool continueSampling = true;

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

  // calculate quantities needed for swap
  calcBestDistancesSwapInitial(
      data,
      distMat,
      medoidIndices,
      &bestDistances,
      &secondBestDistances,
      assignments,
      secondAssignments,
      swapPerformed);

  size_t iter = 0;
  // no need to go back to previous data points for k = 1
  // so cap maxIter at N for k = 1
  if (nMedoids == 1) {
    maxIter = std::min(N, maxIter);
  }
  // continue making swaps while loss is decreasing
  while (iter < maxIter) {
    // TODO(@Adarsh321123): optimize this later with a single column vector
    size_t activeColumn = iter % N;

    iter++;
    permutationIdx = 0;

    sigma = swapSigma(
        data,
        distMat,
        &bestDistances,
        &secondBestDistances,
        assignments,
        activeColumn);

    // Reset variables when starting a new swap
    candidates.fill(0);

    for (size_t col = 0; col < candidates.n_cols; col++) {
      if (col == activeColumn) {
        // only the active column can have valid candidates
        candidates.col(col).fill(1);
      }
    }

    exactMask.fill(0);
    estimates.fill(0);
    numSamples.fill(0);
    continueSampling = true;
    ucbs.fill(std::numeric_limits<float>::infinity());
    lcbs.fill(std::numeric_limits<float>::infinity());

    while (continueSampling) {
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
      //  for the relevant candidates
      arma::uvec compute_exactly_targets =
          arma::find(arma::sum(compute_exactly, 0) >= 1);

      if (compute_exactly_targets.size() > 0) {
        arma::fmat result = swapTarget(
            data,
            distMat,
            medoidIndices,
            &compute_exactly_targets,
            &bestDistances,
            &secondBestDistances,
            assignments,
            (true ? N > 0 : false));

        // Now update the correct indices
        estimates.cols(compute_exactly_targets) = result;
        ucbs.cols(compute_exactly_targets) = result;
        lcbs.cols(compute_exactly_targets) = result;
        exactMask.cols(compute_exactly_targets).fill(1);

        numSamples.cols(compute_exactly_targets) += N;

        // TODO(@Adarsh321123): extract this into a function
        // keep sampling an arm if one of the following are true:
        // (a) the arm is overlapping with 0
        // (b) the arm is below 0 but overlapping with the best arm so far
        // this means that if an arm is above 0, we stop sampling it
        candidates = ((ucbs > 0) && (lcbs < 0) && (exactMask == 0)) ||
                     ((ucbs < 0) && (lcbs < ucbs.min()) && (exactMask == 0));

        // stop sampling if no candidates remain
        if (arma::accu(candidates) == 0) {
          continueSampling = false;
        }
        // if k = 1, then we need to stop sampling if the arm is below 0
        else if (nMedoids == 1 && ucbs(0, activeColumn) < 0) {
          continueSampling = false;
        }
        // if an arm is below 0 and disjoint with all other arms,
        // then we break out of the while loop to perform a swap with that arm
        else if (nMedoids != 1) {
          bool didBreak = false;
          for (size_t i = 0; i < nMedoids; i++) {  // for each arm
            didBreak = false;
            for (size_t j = 0; j < nMedoids; j++) {  // for all other arms
              if (i != j) {
                if (ucbs(i, activeColumn) < 0 && ucbs(i, activeColumn) < lcbs(j, activeColumn)) {
                  continue;
                } else {
                  didBreak = true;
                  break;  // this arm is not disjoint so we should try the next one
                }
              }
            }
            if (!didBreak) { // this arm is disjoint with all other arms
              continueSampling = false;
              break;
            }
          }
        }
      }
      // no need to continue sampling if doing the exact computation changed
      // the continueSampling flag
      if (!continueSampling) {
        break;
      }

      // candidate_targets should be of size T
      // Sum the different columns
      // if any index appears in at least one column, sample it
      arma::uvec candidate_targets = arma::find(
          arma::sum(candidates, 0) >= 1);

      arma::fmat result = swapTarget(
          data,
          distMat,
          medoidIndices,
          &candidate_targets,
          &bestDistances,
          &secondBestDistances,
          assignments,
          false);

      estimates.cols(candidate_targets) =
          ((numSamples.cols(candidate_targets)
            % estimates.cols(candidate_targets))
           + (result * batchSize)) / (batchSize +
           numSamples.cols(
               candidate_targets));

      numSamples.cols(candidate_targets) += batchSize;

      arma::fmat adjust(nMedoids, candidate_targets.size());
      // TODO(@motiwari): Move this ::fill to the previous line
      adjust.fill(p);
      // Assume swapConfidence is given in logspace
      // 0.1 empirically returns the best results for the settings we tested
      // in the paper
      adjust = 0.1 + arma::log(adjust);

      arma::fmat confBoundDelta = sigma.cols(candidate_targets) %
                                  arma::sqrt(adjust / numSamples.cols(
                                                          candidate_targets));
      ucbs.cols(candidate_targets) = estimates.cols(candidate_targets)
                                     + confBoundDelta;
      lcbs.cols(candidate_targets) = estimates.cols(candidate_targets)
                                     - confBoundDelta;

      // keep sampling an arm if one of the following are true:
      // (a) the arm is overlapping with 0
      // (b) the arm is below 0 but overlapping with the best arm so far
      // this means that if an arm is above 0, we stop sampling it
      candidates = ((ucbs > 0) && (lcbs < 0) && (exactMask == 0)) ||
                   ((ucbs < 0) && (lcbs < ucbs.min()) && (exactMask == 0));

      // stop sampling if no candidates remain
      if (arma::accu(candidates) == 0) {
        continueSampling = false;
      }
      // if k = 1, then we need to stop sampling if the arm is below 0
      else if (nMedoids == 1 && ucbs(0, activeColumn) < 0) {
        continueSampling = false;
      }
      // if an arm is below 0 and disjoint with all other arms,
      // then we break out of the while loop to perform a swap with that arm
      else if (nMedoids != 1) {
        bool didBreak = false;
        for (size_t i = 0; i < nMedoids; i++) {  // for each arm
          didBreak = false;
          for (size_t j = 0; j < nMedoids; j++) {  // for all other arms
            if (i != j) {
              if (ucbs(i, activeColumn) < 0 && ucbs(i, activeColumn) < lcbs(j, activeColumn)) {
                continue;
              } else {
                didBreak = true;
                break;  // this arm is not disjoint so we should try the next one
              }
            }
          }
          if (!didBreak) { // this arm is disjoint with all other arms
            continueSampling = false;
            break;
          }
        }
      }
    }

    // Perform the medoid switch
    arma::uword newMedoid = lcbs.index_min();
    size_t k = newMedoid % nMedoids;
    size_t n = newMedoid / nMedoids;
    // we must ensure that the change in loss is negative
    swapPerformed = (*medoidIndices)(k) != n && ucbs(newMedoid) < 0;

    if (swapPerformed) {
      steps++;
      // Perform Swap
      (*medoidIndices)(k) = n;
      medoids->col(k) = data.col((*medoidIndices)(k));

      calcBestDistancesSwapWithFPOptimizations(
        distMat,
        medoidIndices,
        &bestDistances,
        &secondBestDistances,
        assignments,
        secondAssignments,
        k,
        n);
    }

    averageLoss = arma::accu(bestDistances) / data.n_cols;
  }
}
}  // namespace km
