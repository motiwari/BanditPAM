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

#include <armadillo>
#include <unordered_map>

namespace km {
void FasterPAM::fitFasterPAM(
  const arma::fmat& inputData,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat) {
  data = arma::trans(inputData);
  arma::urowvec medoidIndices(nMedoids);
  FasterPAM::buildFasterPAM(data, distMat, &medoidIndices);
  steps = 0;
  medoidIndicesBuild = medoidIndices;
  arma::urowvec assignments(data.n_cols);
  FasterPAM::swapFasterPAM(data, distMat, &medoidIndices, &assignments);

  medoidIndicesFinal = medoidIndices;
  labels = assignments;
}

void FasterPAM::buildFasterPAM(
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

arma::frowvec FasterPAM::calcDeltaTDMs(
  arma::urowvec* assignments,
  arma::frowvec* bestDistances,
  arma::frowvec* secondBestDistances) {
  arma::frowvec Delta_TD_ms(nMedoids, arma::fill::zeros);
  for (size_t i = 0; i < data.n_cols; i++) {
    // Find which medoid point i is assigned to
    size_t m = (*assignments)(i);

    // Update \Delta_TD(ms) with -best(i) + secondBestDistances(i)
    Delta_TD_ms(m) += -(*bestDistances)(i) + (*secondBestDistances)(i);
  }
  return Delta_TD_ms;
}

void FasterPAM::swapFasterPAM(
  const arma::fmat& data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  arma::urowvec* medoidIndices,
  arma::urowvec* assignments) {
  size_t N = data.n_cols;
  arma::frowvec bestDistances(N);
  arma::frowvec secondBestDistances(N);

  // TODO(@motiwari): This is O(kn). Can remove by carrying through assignments from the BUILD step, but that will be
  //  O(kn) too. Since we only do this O(kn) once, we can amortize it over all eager SWAP steps.
  KMedoids::calcBestDistancesSwap(
    data,
    distMat,
    medoidIndices,
    &bestDistances,
    &secondBestDistances,
    assignments);

  bool converged{false};
  size_t x_last{data.n_cols};

  // Calculate initial removal loss for each medoid. This function modifies Delat_TD_ms in place
  arma::frowvec Delta_TD_ms_initial = FasterPAM::calcDeltaTDMs(
    assignments,
    &bestDistances,
    &secondBestDistances);

  arma::frowvec Delta_TD_ms;
  size_t iter = 0;
  while (iter < maxIter && !converged) {
    for (size_t candidate = 0; candidate < data.n_cols; candidate++) {
      if (candidate == x_last) {
        converged = true;
        break;
      }
      float Delta_TD_candidate = 0;
      Delta_TD_ms = Delta_TD_ms_initial; // TODO(@motiwari): Ensure this is copy assignment

      // NOTE: Can probably sample this loop
      for (size_t reference = 0; reference < data.n_cols; reference++) {
        float d_cr = KMedoids::cachedLoss(
            data,
            distMat,
            reference,
            candidate,
            0);  // 0 for MISC

        size_t nearest = (*assignments)(reference);
        if (d_cr < bestDistances(reference)) {
          // When nearest(o) is removed, the loss of point reference is second(o). The two lines below, when summed together,
          // properly do the bookkeeping so that the loss of point reference will now become d_cr. This is why we add d_cr and
          // subtract off second(o).
          Delta_TD_candidate += d_cr - bestDistances(reference);
          Delta_TD_ms(nearest) += bestDistances(reference) - secondBestDistances(reference);
        } else if (d_cr < secondBestDistances(reference)) {
          // Every point has been assigned to its second closest medoid. If we remove the nearest medoid and add
          // point candidate in here, then the updated change in loss for removing the nearest medoid will be
          // d_cr - second(o), since the reference point reference will be assigned to candidate when candidate is added and nearest(o)
          // is removed. In the initial Delta_TMs, we added a +second(o) to the loss for assigning point reference to its
          // second closest medoid when nearest(o) is removed.
          Delta_TD_ms(nearest) += d_cr - secondBestDistances(reference);
        }
      }

      arma::uword best_m_idx = Delta_TD_ms.index_min();
      // -0.01 to avoid precision errors
      // TODO(@motiwari): Move 0.01 to a constants file
//      std::cout << "\n\n";
//      std::cout << "Delta_TD_ms: " << Delta_TD_ms;
//      std::cout << "Delta_TD_candidate: " << Delta_TD_candidate << "\n";

      // TODO(@motiwari): This -0.1 / N should be moved to an approximate comparison
      // TODO(@motiwari): Right now explicitly prevent from swapping with itself, but should never happpen...
      if (Delta_TD_ms(best_m_idx) + Delta_TD_candidate < -0.001 && (*medoidIndices)(best_m_idx) != candidate) {
        // Perform Swap


        std::cout << "Swapped medoid index " << best_m_idx << " (medoid " << (*medoidIndices)(best_m_idx) << ") with " << candidate << "\n";
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
        Delta_TD_ms_initial = FasterPAM::calcDeltaTDMs(
          assignments,
          &bestDistances,
          &secondBestDistances);

        x_last = candidate;
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
