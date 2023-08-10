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
#include <string>
#include <time.h>
#include <sys/time.h>

double get_wall_time_4(){
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    //  Handle error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

namespace km {
  void FasterPAM::fitFasterPAM(
    const arma::fmat& inputData,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat) {
    double wall0 = get_wall_time_4();
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

    double wall1 = get_wall_time_4();
    std::cout << "Wall Clock Time: " << wall1 - wall0 << "\n";
  }

  arma::urowvec FasterPAM::randomInitialization(
    size_t n) {
    // from https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
    const size_t rangeFrom = 0;
    const size_t rangeTo = n-1;
    // create a random device
    std::random_device randDev;
    //  std::mt19937 generator(randDev());  // TODO: uncomment
    // Use the provided seed to initialize the random number generator
    std::mt19937 generator(0); // TODO: remove
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
                                        firstCenter, 0);
      (*assignments)(i) = 0;
      (*bestDistances)(i) = distNear;
      (*secondAssignments)(i) = std::numeric_limits<size_t>::max();
      (*secondBestDistances)(i) = 0.0;
      for (size_t m = 1; m < medoidIndices.size(); m++) {
        size_t me = medoidIndices[m];
        float cost = KMedoids::cachedLoss(data, distMat, i,
                                       me, 0);
        // determine how to fill the second nearest distance
        if (cost < (*bestDistances)(i) || i == me) {
          (*secondAssignments)(i) = (*assignments)(i);
          (*secondBestDistances)(i) = (*bestDistances)(i);
          (*assignments)(i) = m;
          (*bestDistances)(i) = cost;
        } else if ((*secondAssignments)(i) == std::numeric_limits<size_t>::max() || cost < (*secondBestDistances)(i)) {
          (*secondAssignments)(i) = m;
          (*secondBestDistances)(i) = cost;
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

    std::cout << "Swapped medoid index " << first << " with "
              << best << "\n";

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
    size_t candidate,
    arma::urowvec *assignments) {
    arma::frowvec ploss = removalLoss;
    float acc = 0.0;
    for (size_t o = 0; o < (*bestDistances).n_elem; o++) {
      float djo = KMedoids::cachedLoss(data, distMat, o,
                                       candidate, 2);
      if (djo < (*bestDistances)(o)) {
        acc += djo - (*bestDistances)(o);
        ploss[(*assignments)(o)] += (*bestDistances)(o) - (*secondBestDistances)(o);
      } else if (djo < (*secondBestDistances)(o)) {
        ploss[(*assignments)(o)] += djo - (*secondBestDistances)(o);
      }
    }

    // Find the medoid with the minimum change in loss
    auto it = std::min_element(std::begin(ploss), std::end(ploss));
    size_t best_m_idx = std::distance(std::begin(ploss), it);
    float bloss = *it;
    return {bloss + acc, best_m_idx};
  }

  std::tuple<size_t, float> FasterPAM::updateSecondNearest(
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec medoidIndices,
    size_t n,
    size_t best_m_idx,
    size_t o,
    float djo) {
    size_t secondMedoid = best_m_idx;
    float secondDistance = djo;
    for (size_t i = 0; i < medoidIndices.size(); i++) {
      size_t mi = medoidIndices[i];
      if (i == n || i == best_m_idx) {
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

  float FasterPAM::doSwap(
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec& medoidIndices,
    arma::frowvec *bestDistances,
    arma::frowvec *secondBestDistances,
    arma::urowvec *assignments,
    arma::urowvec *secondAssignments,
    size_t best_m_idx,
    size_t candidate) {
    size_t n = (*bestDistances).size();
    assert(("invalid medoid number",
            best_m_idx < medoidIndices.size()));
    assert(("invalid object number",
            candidate < n));
    // Perform Swap
    std::cout << "Swapped medoid index " << best_m_idx << " (medoid " << (medoidIndices)(best_m_idx) << ") with "
              << candidate << "\n";
    medoidIndices[best_m_idx] = candidate;
    float loss = 0.0;
    // update the distances and loss from doing the swap
    for (size_t o = 0; o < n; o++) {
      if (o == candidate) {
        if ((*assignments)(o) != best_m_idx) {
          (*secondAssignments)(o) = (*assignments)(o);
          (*secondBestDistances)(o) = (*bestDistances)(o);
        }

        (*assignments)(o) = best_m_idx;
        (*bestDistances)(o) = 0;
        continue;
      }

      float djo = KMedoids::cachedLoss(data, distMat, o,
                                       candidate, 0);
      if ((*assignments)(o) == best_m_idx) {
        if (djo < (*secondBestDistances)(o)) {
          (*assignments)(o) = best_m_idx;
          (*bestDistances)(o) = djo;
        } else {
          (*assignments)(o) = (*secondAssignments)(o);
          (*bestDistances)(o) = (*secondBestDistances)(o);
          std::tuple<size_t, float> paramsSecond = updateSecondNearest(distMat, medoidIndices, (*assignments)(o), best_m_idx, o, djo);
          (*secondAssignments)(o) = std::get<0>(paramsSecond);
          (*secondBestDistances)(o) = std::get<1>(paramsSecond);
        }
      } else {
        if (djo < (*bestDistances)(o)) {
          (*secondAssignments)(o) = (*assignments)(o);
          (*secondBestDistances)(o) = (*bestDistances)(o);
          (*assignments)(o) = best_m_idx;
          (*bestDistances)(o) = djo;
        } else if (djo < (*secondBestDistances)(o)) {
          (*secondAssignments)(o) = best_m_idx;
          (*secondBestDistances)(o) = djo;
        } else if ((*secondAssignments)(o) == best_m_idx) {
          std::tuple<size_t, float> paramsSecond = updateSecondNearest(distMat, medoidIndices, (*assignments)(o), best_m_idx, o, djo);
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
      if (!swapped) {
        std::cout << "No swap performed" << std::endl;
      }
      float loss = std::get<1>(paramsMedoid);
      averageLoss = loss / data.n_cols;
      std::cout << "Sample complexity (swaps) = " << numSwapDistanceComputations << std::endl;
      std::cout << "Sample complexity (miscs) = " << numMiscDistanceComputations << std::endl;
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
//      std::cout << "Starting while loop at " << iter << " iterations" << std::endl;
//      std::cout << "maxIter: " << maxIter << std::endl;
      size_t swapsBefore = nSwaps;
      for (size_t candidate = 0; candidate < n; candidate++) {
        double wall0 = get_wall_time_4();
//        double iter_start = get_wall_time_4();
        if (candidate == lastSwap) {
          break;
        }

        // skip this iteration since candidate is already a medoid
        if (candidate == medoidIndices[assignments(candidate)]) {
          continue;
        }

        std::tuple<float, size_t> paramsSwap = findBestSwap(distMat, removalLoss, &bestDistances, &secondBestDistances, candidate, &assignments);
        float change = std::get<0>(paramsSwap);
        size_t best_m_idx = std::get<1>(paramsSwap);
        if (change >= 0) {
          iter++;
//          std::cout << "iter: " << iter << std::endl;
//          double iter_end = get_wall_time_4();
//          std::cout << "Time for iteration: " << iter_end - iter_start << std::endl;
          std::cout << "No swap performed" << std::endl;
          std::cout << "Sample complexity (swaps) = " << numSwapDistanceComputations << std::endl;
          std::cout << "Sample complexity (miscs) = " << numMiscDistanceComputations << std::endl;
          double wall1 = get_wall_time_4();
          std::cout << "Time for iteration: " << wall1 - wall0 << std::endl;
          if (iter >= maxIter) {
            break;
          }
          continue;
        }

        nSwaps++;
        lastSwap = candidate;
//        std::cout << "iter: " << iter << std::endl;
        float newLoss = doSwap(distMat, medoidIndices, &bestDistances, &secondBestDistances, &assignments, &secondAssignments, best_m_idx, candidate);
        if (newLoss >= loss) {
          break;
        }

        loss = newLoss;
        updateRemovalLoss(&bestDistances, &secondBestDistances, removalLoss, &assignments);
        iter++;
        std::cout << "Sample complexity (swaps) = " << numSwapDistanceComputations << std::endl;
        std::cout << "Sample complexity (miscs) = " << numMiscDistanceComputations << std::endl;
        double wall1 = get_wall_time_4();
        std::cout << "Time for iteration: " << wall1 - wall0 << std::endl;
//        double iter_end = get_wall_time_4();
//        std::cout << "Time for iteration: " << iter_end - iter_start << std::endl;
        if (iter >= maxIter) {
          break;
        }
      }

      if (nSwaps == swapsBefore) {
        break;
      }
    }

    averageLoss = loss / data.n_cols;
    return { assignments, nSwaps };
  }
}  // namespace km
