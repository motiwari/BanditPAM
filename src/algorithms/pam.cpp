/**
 * @file pam.cpp
 * @date 2021-07-25
 *
 * This file contains a C++ implementation of the PAM algorithm.
 *
 */

#include "pam.hpp"

#include <armadillo>
#include <unordered_map>

namespace km {
void PAM::fit_pam(const arma::mat& input_data) {
  data = input_data;
  data = arma::trans(data);
  arma::urowvec medoid_indices(n_medoids);
  PAM::build_pam(data, &medoid_indices);
  steps = 0;
  medoid_indices_build = medoid_indices;
  arma::urowvec assignments(data.n_cols);
  size_t i = 0;
  bool medoidChange = true;
  while (i < max_iter && medoidChange) {
    auto previous(medoid_indices);
    PAM::swap_pam(data, &medoid_indices, &assignments);
    medoidChange = arma::any(medoid_indices != previous);
    i++;
  }
  medoid_indices_final = medoid_indices;
  this->labels = assignments;
  this->steps = i;
}

void PAM::build_pam(
  const arma::mat& data,
  arma::urowvec* medoid_indices) {
  size_t N = data.n_cols;
  arma::rowvec estimates(N, arma::fill::zeros);
  arma::rowvec best_distances(N);
  best_distances.fill(std::numeric_limits<double>::infinity());
  for (size_t k = 0; k < n_medoids; k++) {
    double minDistance = std::numeric_limits<double>::infinity();
    size_t best = 0;
    for (size_t i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        double cost = KMedoids::cachedLoss(data, i, j);
        // compares this with the cached best distance
        if (best_distances(j) < cost) {
          cost = best_distances(j);
        }
        total += cost;
      }
      if (total < minDistance) {
        minDistance = total;
        best = i;
      }
    }
    // update the medoid index for that of lowest cost
    (*medoid_indices)(k) = best;

    // update the medoid assignment and best_distance for this datapoint
    for (size_t l = 0; l < N; l++) {
      double cost = KMedoids::cachedLoss(data, l, (*medoid_indices)(k));
      if (cost < best_distances(l)) {
        best_distances(l) = cost;
      }
    }
  }
}

void PAM::swap_pam(
  const arma::mat& data,
  arma::urowvec* medoid_indices,
  arma::urowvec* assignments) {
  double minDistance = std::numeric_limits<double>::infinity();
  size_t best = 0;
  size_t medoid_to_swap = 0;
  size_t N = data.n_cols;
  arma::rowvec best_distances(N);
  arma::rowvec second_distances(N);

  KMedoids::calc_best_distances_swap(
    data,
    medoid_indices,
    &best_distances,
    &second_distances,
    assignments);

  for (size_t k = 0; k < n_medoids; k++) {
    for (size_t i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // compute distance between base point and every other datapoint
        double cost = KMedoids::cachedLoss(data, i, j);
        // if x_j is NOT assigned to k: compares this with
        //   the cached best distance
        // if x_j is assigned to k: compares this with
        //   the cached second best distance
        if ((*assignments)(j) != k) {
          if (best_distances(j) < cost) {
            cost = best_distances(j);
          }
        } else {
          if (second_distances(j) < cost) {
            cost = second_distances(j);
          }
        }
        total += cost;
      }
      // if total distance for new base point is better than
      // that of the medoid, update the best index identified so far
      if (total < minDistance) {
        minDistance = total;
        best = i;
        medoid_to_swap = k;
      }
    }
  }
  (*medoid_indices)(medoid_to_swap) = best;
}
}  // namespace km
