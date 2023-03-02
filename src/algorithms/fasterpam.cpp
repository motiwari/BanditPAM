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

  size_t i = 0;
  bool medoidChange = true;
  while (i < maxIter && medoidChange) {
    auto previous(medoidIndices);
    FasterPAM::swapFasterPAM(data, distMat, &medoidIndices, &assignments);
    medoidChange = arma::any(medoidIndices != previous);
    i++;
  }
  medoidIndicesFinal = medoidIndices;
  labels = assignments;
  steps = i;
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

void FasterPAM::calcDeltaTDMs(
  arma::urowvec* assignments,
  arma::frowvec* bestDistances,
  arma::frowvec* secondBestDistances,
  arma::frowvec* Delta_TD_ms) {
  for (size_t i = 0; i < data.n_cols; i++) {
    // Find which medoid point i is assigned to
    size_t m = (*assignments)(i);

    // Update \Delta_TD(ms) with -best(i) + secondBestDistances(i)
    (*Delta_TD_ms)(m) += -(*bestDistances)(i) + (*secondBestDistances)(i);
  }
}

void FasterPAM::swapFasterPAM(
  const arma::fmat& data,
  std::optional<std::reference_wrapper<const arma::fmat>> distMat,
  arma::urowvec* medoidIndices,
  arma::urowvec* assignments) {
  size_t N = data.n_cols;
  arma::frowvec bestDistances(N);
  arma::frowvec secondBestDistances(N);

  KMedoids::calcBestDistancesSwap(
    data,
    distMat,
    medoidIndices,
    &bestDistances,
    &secondBestDistances,
    assignments);


  arma::frowvec Delta_TD_ms(nMedoids, arma::fill::zeros);
  bool converged{false};
  size_t x_last{data.n_cols};

  // Calculate initial removal loss for each medoid. This function modifies Delat_TD_ms in place
  FasterPAM::calcDeltaTDMs(
    assignments,
    &bestDistances,
    &secondBestDistances,
    &Delta_TD_ms
    );


  while (!converged) {
    for (size_t x_c = 0; x_c < data.n_cols; x_c++) {
      if (x_c == x_last) {
        converged = true;
        break;
      }
      float Delta_TD_x_c = 0;


      // NOTE: Can probably sample this loop
      for (size_t x_o = 0; x_o < data.n_cols; x_o++) {
        float d_oj = KMedoids::cachedLoss(
            data,
            distMat,
            x_o,
            x_c,
            0);  // 0 for MISC

        size_t nearest = (*assignments)(x_o);
        if (d_oj < (bestDistances)(x_o)) {
          // When nearest(o) is removed, the loss of point x_o is second(o). The two lines below, when summed together,
          // properly do the bookkeeping so that the loss of point x_o will now become d_oj. This is why we add d_oj and
          // subtract off second(o).
          Delta_TD_x_c += d_oj - (bestDistances)(x_o);
          Delta_TD_ms(nearest) += (bestDistances)(x_o) - (secondBestDistances)(x_o);
        } else if (d_oj < (secondBestDistances)(x_o)) {
          // Every point has been assigned to its second closest medoid. If we remove the nearest medoid and add
          // point x_c in here, then the updated change in loss for removing the nearest medoid will be
          // d_oj - second(o), since the reference point x_o will be assigned to x_c when x_c is added and nearest(o)
          // is removed. In the initial Delta_TMs, we added a +second(o) to the loss for assigning point x_o to its
          // second closest medoid when nearest(o) is removed.
          Delta_TD_ms(nearest) += d_oj - (secondBestDistances)(x_o);
        }
      }

      arma::uword best_m_idx = Delta_TD_ms.index_min();
      Delta_TD_ms(best_m_idx) += Delta_TD_x_c; // Paired with line below
      if (Delta_TD_ms(best_m_idx) < 0) {
        // Perform Swap
        std::cout << "Swapped medoid index " << best_m_idx << " (medoid " << (*medoidIndices)(best_m_idx) << ") with " << x_c;
        (*medoidIndices)(best_m_idx) = x_c;

        // Update TD and assignments
        KMedoids::calcBestDistancesSwap(
            data,
            distMat,
            medoidIndices,
            &bestDistances,
            &secondBestDistances,
            assignments);

        // Update \Delta_TD_m's. This function modifies Delat_TD_ms in place
        FasterPAM::calcDeltaTDMs(
          assignments,
          &bestDistances,
          &secondBestDistances,
          &Delta_TD_ms
        );

        // Update x_last
        x_last = x_c;
      } else {
        Delta_TD_ms(best_m_idx) -= Delta_TD_x_c; // This allows us to avoid the .copy in Line 7 of the original algorithm
                                                // when no swap is performed
      }
    }
  }
}
}  // namespace km
