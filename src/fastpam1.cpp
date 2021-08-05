/**
 * @file fastpam1.cpp
 * @date 2021-08-03
 *
 * This file contains the primary C++ implementation of the FastPAM1 code.
 *
 */
#include "kmedoids_algorithm.hpp"
#include "log_helper.hpp"

#include <carma.h>
#include <armadillo>
#include <unordered_map>
#include <regex>

/**
 * \brief Runs naive FastPAM1 algorithm.
 *
 * Run the naive FastPAM1 algorithm to identify a dataset's medoids.
 *
 * @param input_data Input data to find the medoids of
 */
void km::KMedoids::fit_fastpam1(const arma::mat& input_data) {
  data = input_data;
  data = arma::trans(data);
  arma::rowvec medoid_indices(n_medoids);
  // runs build step
  km::KMedoids::build_fastpam1(data, medoid_indices);
  steps = 0;
  medoid_indices_build = medoid_indices;
  arma::rowvec assignments(data.n_cols);
  size_t iter = 0;
  bool medoidChange = true;
  while (iter < max_iter && medoidChange) {
    auto previous(medoid_indices);
    // runs swap step as necessary
    km::KMedoids::swap_fastpam1(data, medoid_indices, assignments);
    medoidChange = arma::any(medoid_indices != previous);
    iter++;
  }
  medoid_indices_final = medoid_indices;
  labels = assignments;
  steps = iter;
}

/**
 * \brief Build step for the FastPAM1 algorithm
 *
 * Runs build step for the FastPAM1 algorithm. Loops over all datapoint and
 * checks its distance from every other datapoint in the dataset, then checks if
 * the total cost is less than that of the medoid (if a medoid exists yet).
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Uninitialized array of medoids that is modified in place
 * as medoids are identified
 */
void km::KMedoids::build_fastpam1(
  const arma::mat& data, 
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
    km::KMedoids::build_sigma(
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
    // update the medoid index for that of lowest cost
    medoid_indices(k) = best;

    // update the medoid assignment and best_distance for this datapoint
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
 * \brief Swap step for the FastPAM1 algorithm
 *
 * Runs swap step for the FastPAM1 algorithm. Loops over all datapoint and
 * checks its distance from every other datapoint in the dataset, then checks if
 * the total cost is less than that of the medoid.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Array of medoid indices created from the build step
 * that is modified in place as better medoids are identified
 * @param assignments Uninitialized array of indices corresponding to each
 * datapoint assigned the index of the medoid it is closest to
 */
void km::KMedoids::swap_fastpam1(
  const arma::mat& data, 
  arma::rowvec& medoid_indices,
  arma::rowvec& assignments)
{
  double bestChange = 0; // best loss change 
  double minDistance = std::numeric_limits<double>::infinity();
  size_t best = 0;
  size_t medoid_to_swap = 0;
  size_t N = data.n_cols;
  int p = (N * n_medoids * swapConfidence); // reciprocal
  arma::mat sigma(n_medoids, N, arma::fill::zeros);
  arma::rowvec best_distances(N);
  arma::rowvec second_distances(N);
  arma::rowvec delta_td(n_medoids, arma::fill::zeros);

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
  // for every point in our dataset, let it serve as a new medoid 
  for (size_t i = 0; i < data.n_cols; i++) {
      double di = best_distances(i);
      // loss change for making i a medoid
      delta_td.fill(-di);
      for (size_t j = 0; j < data.n_cols; j++) {
          if (j != i) {
              // compute distance to new medoid
              double dij = (this->*lossFn)(data, i, j);
              // update loss change for the current 
              if (dij < second_distances(j)) {
                  delta_td.at(assignments(j)) += (dij - best_distances(j));
              } else {
                  delta_td.at(assignments(j)) += (second_distances(j) - best_distances(j));
              }
              // reassignment check 
              if (dij < best_distances(j)) {
                  // update loss change for others 
                  delta_td += (dij -  best_distances(j));
                  // remove the update for the current
                  delta_td.at(assignments(j)) -= (dij -  best_distances(j)); 
              }  
          }
      }
      // choose the best medoid-to-swap 
      arma::uword min_medoid = delta_td.index_min();
      // if the loss change is better than the best loss change
      // update the best index identified so far
      if (delta_td.min() < bestChange) {
          bestChange = delta_td.min();
          best = i;
          medoid_to_swap = min_medoid;
      }
  }
  // update the loss and medoid if the loss is improved
  if (bestChange < 0) {
      minDistance = arma::sum(best_distances) + bestChange;
      medoid_indices(medoid_to_swap) = best;
  } else {
      minDistance = arma::sum(best_distances);
  }
  logHelper.loss_swap.push_back(minDistance/N);
  logHelper.p_swap.push_back((float)1/(float)p);
  logHelper.comp_exact_swap.push_back(N*n_medoids);
}
