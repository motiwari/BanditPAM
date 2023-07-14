/**
 * @file fasterpam.cpp
 * @date 2021-07-25
 *
 * Contains a C++ implementation of the FasterPAM algorithm.
 * The original FasterPAM papers are:
 * 1) Erich Schubert and Peter J. Rousseeuw: Fast and Eager k-Medoids Clustering:
 *  O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms
 * 2) Erich Schubert and Peter J. Rousseeuw: Faster k-Medoids Clustering:
 *  Improving the PAM, CLARA, and CLARANS Algorithms
 */

#include "fasterpam.hpp"

#include <tuple>
#include <armadillo>
#include <vector>
#include <cassert>
#include <random>

namespace km {
  void FasterPAM::fitFasterPAM(
    const arma::fmat& inputData,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat) {
    useCache = false; // this allows cachedLoss to work appropriately
    data = arma::trans(inputData);
    // FasterPAM uses uniform random sampling instead of BUILD since
    // SWAP is so fast that it is not worth it to use BUILD
    arma::urowvec medoidIndices = randomInitialization(data.n_cols);
    steps = 0;
    medoidIndicesBuild = medoidIndices;
    size_t n = data.n_cols;
    arma::urowvec assignments(n);
    arma::urowvec secondAssignments(n);
    secondAssignments.fill(std::numeric_limits<size_t>::max());
    std::tuple<arma::urowvec, size_t> paramsFasterPAM = FasterPAM::swapFasterPAM(data, distMat, medoidIndices, assignments, secondAssignments);
    assignments = std::get<0>(paramsFasterPAM);
    size_t swaps = std::get<1>(paramsFasterPAM);
    medoidIndicesFinal = medoidIndices;
    labels = assignments;
    steps = swaps;
  }

  arma::urowvec FasterPAM::randomInitialization(
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

  float FasterPAM::initialAssignment(
    const arma::fmat &data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec medoidIndices,
    arma::frowvec *bestDistances,
    arma::frowvec *secondBestDistances,
    arma::urowvec *assignments,
    arma::urowvec *secondAssignments) {
    size_t n = data.n_cols;
    size_t firstCenter = medoidIndices[0];
    float loss = 0.0;
    for (size_t i = 0; i < n; i++) {
      float distNear = KMedoids::cachedLoss(data, distMat, i,
                                        firstCenter, 2);
      (*assignments)(i) = 0;
      (*bestDistances)(i) = distNear;
      (*secondAssignments)(i) = std::numeric_limits<size_t>::max();
      (*secondBestDistances)(i) = 0.0;
      for (size_t m = 1; m < medoidIndices.size(); m++) {
        size_t me = medoidIndices[m];
        float d = KMedoids::cachedLoss(data, distMat, i,
                                       me, 2);
        // determine how to fill the second nearest distance
        if (d < (*bestDistances)(i) || i == me) {
          (*secondAssignments)(i) = (*assignments)(i);
          (*secondBestDistances)(i) = (*bestDistances)(i);
          (*assignments)(i) = m;
          (*bestDistances)(i) = d;
        } else if ((*secondAssignments)(i) == std::numeric_limits<size_t>::max() || d < (*secondBestDistances)(i)) {
          (*secondAssignments)(i) = m;
          (*secondBestDistances)(i) = d;
        }
      }

      loss += (*bestDistances)(i);
    }

    return loss;
  }

  std::tuple<bool, float> FasterPAM::chooseMedoidWithinPartition(
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec assignments,
    arma::urowvec& medoidIndices,
    size_t m) {
    size_t first = medoidIndices[m];
    size_t best = first;
    float sumb = 0.0;

    // Calculate the sum of distances to the current medoid within the partition
    for (size_t i = 0; i < assignments.size(); i++) {
      size_t a = assignments[i];
      if (first != i && a == m) {
        sumb += KMedoids::cachedLoss(data, distMat, i,
                                     first, 2);
      }
    }

    // Find the best medoid within the partition
    for (size_t j = 0; j < assignments.size(); j++) {
      size_t aj = assignments[j];
      if (j != first && aj == m) {
        float sumj = 0.0;
        for (size_t i = 0; i < assignments.size(); i++) {
          size_t ai = assignments[i];
          if (i != j && ai == m) {
            sumj += KMedoids::cachedLoss(data, distMat, i,
                                         j, 2);
          }
        }

        if (sumj < sumb) {
          best = j;
          sumb = sumj;
        }
      }
    }

    medoidIndices[m] = best;
    return {best != first, sumb};
  }

  void FasterPAM::updateRemovalLoss(
    arma::frowvec *bestDistances,
    arma::frowvec *secondBestDistances,
    arma::frowvec& loss,
    arma::urowvec *assignments) {
    loss.fill(0.0);
    for (size_t i = 0; i < (*bestDistances).n_elem; i++) {
      loss[(*assignments)(i)] += (*secondBestDistances)(i) - (*bestDistances)(i);
    }
  }

  std::tuple<float, size_t> FasterPAM::findBestSwap(
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::frowvec& removalLoss,
    arma::frowvec *bestDistances,
    arma::frowvec *secondBestDistances,
    size_t j,
    arma::urowvec *assignments) {
    arma::frowvec ploss = removalLoss;
    float acc = 0.0;
    for (size_t o = 0; o < (*bestDistances).n_elem; o++) {
      float djo = KMedoids::cachedLoss(data, distMat, o,
                                       j, 2);
      if (djo < (*bestDistances)(o)) {
        acc += djo - (*bestDistances)(o);
        ploss[(*assignments)(o)] += (*bestDistances)(o) - (*secondBestDistances)(o);
      } else if (djo < (*secondBestDistances)(o)) {
        ploss[(*assignments)(o)] += djo - (*secondBestDistances)(o);
      }
    }

    // Find the medoid with the minimum change in loss
    auto it = std::min_element(std::begin(ploss), std::end(ploss));
    size_t b = std::distance(std::begin(ploss), it);
    float bloss = *it;
    return {bloss + acc, b};
  }

  std::tuple<size_t, float> FasterPAM::updateSecondNearest(
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec medoidIndices,
    size_t n,
    size_t b,
    size_t o,
    float djo) {
    size_t secondMedoid = b;
    float secondDistance = djo;
    for (size_t i = 0; i < medoidIndices.size(); i++) {
      size_t mi = medoidIndices[i];
      if (i == n || i == b) {
        continue;
      }

      float d = KMedoids::cachedLoss(data, distMat, o,
                                     mi, 2);
      if (d < secondDistance) {
        secondMedoid = i;
        secondDistance = d;
      }
    }

    return {secondMedoid, secondDistance};
  }

  float FasterPAM::doSwap(
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec& medoidIndices,
    arma::frowvec *bestDistances,
    arma::frowvec *secondBestDistances,
    arma::urowvec *assignments,
    arma::urowvec *secondAssignments,
    size_t b,
    size_t j) {
    size_t n = (*bestDistances).size();
    assert(("invalid medoid number",
            b < medoidIndices.size()));
    assert(("invalid object number",
            j < n));
    medoidIndices[b] = j;
    float loss = 0.0;
    // update the distances and loss from doing the swap
    for (size_t o = 0; o < n; o++) {
      if (o == j) {
        if ((*assignments)(o) != b) {
          (*secondAssignments)(o) = (*assignments)(o);
          (*secondBestDistances)(o) = (*bestDistances)(o);
        }

        (*assignments)(o) = b;
        (*bestDistances)(o) = 0;
        continue;
      }

      float djo = KMedoids::cachedLoss(data, distMat, o,
                                       j, 2);
      if ((*assignments)(o) == b) {
        if (djo < (*secondBestDistances)(o)) {
          (*assignments)(o) = b;
          (*bestDistances)(o) = djo;
        } else {
          (*assignments)(o) = (*secondAssignments)(o);
          (*bestDistances)(o) = (*secondBestDistances)(o);
          std::tuple<size_t, float> paramsSecond = updateSecondNearest(distMat, medoidIndices, (*assignments)(o), b, o, djo);
          (*secondAssignments)(o) = std::get<0>(paramsSecond);
          (*secondBestDistances)(o) = std::get<1>(paramsSecond);
        }
      } else {
        if (djo < (*bestDistances)(o)) {
          (*secondAssignments)(o) = (*assignments)(o);
          (*secondBestDistances)(o) = (*bestDistances)(o);
          (*assignments)(o) = b;
          (*bestDistances)(o) = djo;
        } else if (djo < (*secondBestDistances)(o)) {
          (*secondAssignments)(o) = b;
          (*secondBestDistances)(o) = djo;
        } else if ((*secondAssignments)(o) == b) {
          std::tuple<size_t, float> paramsSecond = updateSecondNearest(distMat, medoidIndices, (*assignments)(o), b, o, djo);
          (*secondAssignments)(o) = std::get<0>(paramsSecond);
          (*secondBestDistances)(o) = std::get<1>(paramsSecond);
        }
      }

      loss += (*bestDistances)(o);
    }

    return loss;
  }

  std::tuple<arma::urowvec, size_t> FasterPAM::swapFasterPAM(
    const arma::fmat &data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec& medoidIndices,
    arma::urowvec assignments,
    arma::urowvec secondAssignments) {
    size_t n = assignments.n_elem;
    // run a simplified algorithm if k = 1
    if (nMedoids == 1) {
      assignments.fill(0);
      std::tuple<bool, float> paramsMedoid = chooseMedoidWithinPartition(distMat, assignments, medoidIndices, 0);
      bool swapped = std::get<0>(paramsMedoid);
      float loss = std::get<1>(paramsMedoid);
      averageLoss = loss / data.n_cols;
      return {assignments, (swapped) ? 1 : 0};
    }

    assignments.fill(std::numeric_limits<size_t>::max());
    arma::frowvec bestDistances(n, arma::fill::zeros);
    arma::frowvec secondBestDistances(n, arma::fill::zeros);
    float loss = initialAssignment(data, distMat, medoidIndices, &bestDistances, &secondBestDistances, &assignments, &secondAssignments);
    arma::frowvec removalLoss(nMedoids, arma::fill::zeros);
    updateRemovalLoss(&bestDistances, &secondBestDistances, removalLoss, &assignments);
    size_t lastSwap = n;
    size_t nSwaps = 0;
    size_t iter = 0;

    // run the main SWAP algorithm until convergence
    while (iter < maxIter) {
      iter++;
      size_t swapsBefore = nSwaps;
      for (size_t j = 0; j < n; j++) {
        if (j == lastSwap) {
          break;
        }

        // skip this iteration since candidate is already a medoid
        if (j == medoidIndices[assignments(j)]) {
          continue;
        }

        std::tuple<float, size_t> paramsSwap = findBestSwap(distMat, removalLoss, &bestDistances, &secondBestDistances, j, &assignments);
        float change = std::get<0>(paramsSwap);
        size_t b = std::get<1>(paramsSwap);
        if (change >= 0) {
          continue;
        }

        nSwaps++;
        lastSwap = j;
        float newLoss = doSwap(distMat, medoidIndices, &bestDistances, &secondBestDistances, &assignments, &secondAssignments, b, j);
        if (newLoss >= loss) {
          break;
        }

        loss = newLoss;
        updateRemovalLoss(&bestDistances, &secondBestDistances, removalLoss, &assignments);
      }

      if (nSwaps == swapsBefore) {
        break;
      }
    }

    averageLoss = loss / data.n_cols;
    return { assignments, nSwaps };
  }
}  // namespace km
