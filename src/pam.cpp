/**
 * @file pam.cpp
 * @date 2021-07-25
 *
 * This file contains the primary C++ implementation of the PAM code.
 *
 */
#include "kmedoids_algorithm.hpp"

#include <carma.h>
#include <armadillo>
#include <unordered_map>
#include <regex>

/**
 * \brief Runs naive PAM algorithm.
 *
 * Run the naive PAM algorithm to identify a dataset's medoids.
 *
 * @param input_data Input data to find the medoids of
 */
void KMedoids::fit_naive(arma::mat input_data) {
  data = input_data;
  data = arma::trans(data);
  arma::rowvec medoid_indices(n_medoids);
  // runs build step
  KMedoids::build_naive(data, medoid_indices);
  steps = 0;

  medoid_indices_build = medoid_indices;
  arma::rowvec assignments(data.n_cols);
  size_t i = 0;
  bool medoidChange = true;
  while (i < max_iter && medoidChange) {
    auto previous(medoid_indices);
    // runs swa step as necessary
    KMedoids::swap_naive(data, medoid_indices, assignments);
    medoidChange = arma::any(medoid_indices != previous);
    i++;
  }
  medoid_indices_final = medoid_indices;
  labels = assignments;
  steps = i;
}

/**
 * \brief Build step for the naive algorithm
 *
 * Runs build step for the naive PAM algorithm. Loops over all datapoint and
 * checks its distance from every other datapoint in the dataset, then checks if
 * the total cost is less than that of the medoid (if a medoid exists yet).
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Uninitialized array of medoids that is modified in place
 * as medoids are identified
 */
void KMedoids::build_naive(
  arma::mat& data, 
  arma::rowvec& medoid_indices)
{ 
  size_t N = data.n_cols;
  int p = (buildConfidence * N); // reciprocal
  bool use_absolute = true;
  arma::rowvec estimates(N, arma::fill::zeros);
  arma::rowvec best_distances(N);
  best_distances.fill(std::numeric_limits<double>::infinity());
  arma::rowvec sigma(N); // standard deviation of induced losses on reference points
  for (size_t k = 0; k < n_medoids; k++) {
    double minDistance = std::numeric_limits<double>::infinity();
    int best = 0;
    KMedoids::build_sigma(
           data, best_distances, sigma, batchSize, use_absolute); // computes std dev amongst batch of reference points
    // fixes a base datapoint
    for (int i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // computes distance between base and all other points
        double cost = (this->*lossFn)(data, i, j);
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
    // updates the medoid index for that of lowest cost.
    medoid_indices(k) = best;

    // updates the best distance with this medoid 
    for (size_t l = 0; l < N; l++) {
        double cost = (this->*lossFn)(data, l, medoid_indices(k));
        if (cost < best_distances(l)) {
            best_distances(l) = cost;
        }
    }
    use_absolute = false; // use difference of loss for sigma and sampling,
                          // not absolute
    logHelper.loss_build.push_back(minDistance/N);
    logHelper.p_build.push_back((float)1/(float)p);
    logHelper.comp_exact_build.push_back(N);
  }
}

/**
 * \brief Swap step for the naive algorithm
 *
 * Runs build step for the naive PAM algorithm. Loops over all datapoint and
 * checks its distance from every other datapoint in the dataset, then checks if
 * the total cost is less than that of the medoid.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Array of medoid indices created from the build step
 * that is modified in place as better medoids are identified
 * @param assignments Uninitialized array of indices corresponding to each
 * datapoint assigned the index of the medoid it is closest to
 */
void KMedoids::swap_naive(
  arma::mat& data, 
  arma::rowvec& medoid_indices,
  arma::rowvec& assignments)
{
  double minDistance = std::numeric_limits<double>::infinity();
  size_t best = 0;
  size_t medoid_to_swap = 0;
  size_t N = data.n_cols;
  int p = (N * n_medoids * swapConfidence); // reciprocal
  arma::mat sigma(n_medoids, N, arma::fill::zeros);
  arma::rowvec best_distances(N);
  arma::rowvec second_distances(N);

  // calculate quantities needed for swap, best_distances and sigma
  calc_best_distances_swap(
    data, medoid_indices, best_distances, second_distances, assignments);

  swap_sigma(data,
              sigma,
              batchSize,
              best_distances,
              second_distances,
              assignments);
  
  // write the sigma distribution to logfile
  sigma_log(sigma);

  // iterate across the current medoids
  for (size_t k = 0; k < n_medoids; k++) {
    // for every point in our dataset, let it serve as a "base" point
    for (size_t i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // compute distance between base point and every other datapoint
        double cost = (this->*lossFn)(data, i, j);
        // (i) if x_j is not assigned to k: compares this with the cached best distance 
        // (ii) if x_j is assigned to k: compares this with the cached second best distance
        if (assignments(j) != k) {
          if (best_distances(j) < cost) {
            cost = best_distances(j);
          }
        } else if (second_distances(j) < cost) {
          cost = second_distances(j);
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
  logHelper.loss_swap.push_back(minDistance/N);
  logHelper.p_swap.push_back((float)1/(float)p);
  logHelper.comp_exact_swap.push_back(N*n_medoids);
}
