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
#include <regex>

/**
 * \brief Runs naive PAM algorithm.
 *
 * Run the PAM algorithm to identify a dataset's medoids.
 *
 * @param input_data Input data to find the medoids of
 */
void PAM::fit_naive(const arma::mat& input_data) {
  data = input_data;
  data = arma::trans(data);
  arma::rowvec medoid_indices(n_medoids);
  // runs build step
  PAM::build_naive(data, medoid_indices);
  steps = 0;
  medoid_indices_build = medoid_indices;
  arma::rowvec assignments(data.n_cols);
  size_t i = 0;
  bool medoidChange = true;
  while (i < max_iter && medoidChange) {
    auto previous(medoid_indices);
    // runs swap step as necessary
    PAM::swap_naive(data, medoid_indices, assignments);
    medoidChange = arma::any(medoid_indices != previous);
    i++;
  }
  medoid_indices_final = medoid_indices;
  this->labels = assignments;
  this->steps = i;
}

/**
 * \brief Build step for the PAM algorithm
 *
 * Runs build step for the PAM algorithm. Loops over all datapoint and
 * checks its distance from every other datapoint in the dataset, then checks if
 * the total cost is less than that of the medoid (if a medoid exists yet).
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Uninitialized array of medoids that is modified in place
 * as medoids are identified
 */
void PAM::build_naive(
  const arma::mat& data,
  arma::rowvec& medoid_indices) {
  size_t N = data.n_cols;
  bool use_absolute = true;
  arma::rowvec estimates(N, arma::fill::zeros);
  arma::rowvec best_distances(N);
  best_distances.fill(std::numeric_limits<double>::infinity());
  arma::rowvec sigma(N); // standard deviation of induced losses on reference points
  for (size_t k = 0; k < n_medoids; k++) {
    double minDistance = std::numeric_limits<double>::infinity();
    size_t best = 0;
    sigma = km::KMedoids::build_sigma(
            data, best_distances, batchSize, use_absolute); // computes std dev amongst batch of reference points
    // fixes a base datapoint
    for (size_t i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // computes distance between base and all other points
        double cost = km::KMedoids::cachedLoss(data, i, j);
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
    medoid_indices(k) = best;

    // update the medoid assignment and best_distance for this datapoint
    for (size_t l = 0; l < N; l++) {
        double cost = km::KMedoids::cachedLoss(data, l, medoid_indices(k));
        if (cost < best_distances(l)) {
            best_distances(l) = cost;
        }
    }
    use_absolute = false; // use difference of loss for sigma and sampling,
                          // not absolute
  }
}

/**
 * \brief Swap step for the PAM algorithm
 *
 * Runs build step for the PAM algorithm. Loops over all datapoint and
 * checks its distance from every other datapoint in the dataset, then checks if
 * the total cost is less than that of the medoid.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Array of medoid indices created from the build step
 * that is modified in place as better medoids are identified
 * @param assignments Uninitialized array of indices corresponding to each
 * datapoint assigned the index of the medoid it is closest to
 */
void PAM::swap_naive(
  const arma::mat& data,
  arma::rowvec& medoid_indices,
  arma::rowvec& assignments) {
  double minDistance = std::numeric_limits<double>::infinity();
  size_t best = 0;
  size_t medoid_to_swap = 0;
  size_t N = data.n_cols;
  arma::mat sigma(n_medoids, N, arma::fill::zeros);
  arma::rowvec best_distances(N);
  arma::rowvec second_distances(N);

  // calculate quantities needed for swap, best_distances and sigma
  km::KMedoids::calc_best_distances_swap(
    data, medoid_indices, best_distances, second_distances, assignments);

  sigma = km::KMedoids::swap_sigma(data,
                                   batchSize,
                                   best_distances,
                                   second_distances,
                                   assignments);
  
  
  // iterate across the current medoids
  for (size_t k = 0; k < n_medoids; k++) {
    // for every point in our dataset, let it serve as a "base" point
    for (size_t i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // compute distance between base point and every other datapoint
        double cost = km::KMedoids::cachedLoss(data, i, j);
        // (i) if x_j is not assigned to k: compares this with the cached best distance 
        // (ii) if x_j is assigned to k: compares this with the cached second best distance
        if (assignments(j) != k) {
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
      // if total distance for new base point is better than that of the medoid,
      // update the best index identified so far
      if (total < minDistance) {
        minDistance = total;
        best = i;
        medoid_to_swap = k;
      }
    }
  }
  medoid_indices(medoid_to_swap) = best;
}
