#ifndef HEADERS_ALGORITHMS_KMEDOIDS_ALGORITHM_HPP_
#define HEADERS_ALGORITHMS_KMEDOIDS_ALGORITHM_HPP_

#include <omp.h>
#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>
#include <tuple>
#include <functional>
#include <unordered_map>
#include <string>


namespace km {
/**
 *  \brief Class implementation for running KMedoids methods.
 *
 *  KMedoids class. Creates a KMedoids object that can be used to find the medoids
 *  for a particular set of input data.
 *
 *  @param n_medoids Number of medoids/clusters to create
 *  @param algorithm Algorithm used to find medoids; options are "BanditPAM",
 *    "PAM", or "FastPAM1"
 *  @param max_iter The maximum number of iterations the algorithm runs for
 *  @param buildConfidence Constant that affects the sensitivity of build confidence bounds
 *  @param swapConfidence Constant that affects the sensitiviy of swap confidence bounds
 */
class KMedoids {
 public:
  KMedoids(
    size_t n_medoids = 5,
    const std::string& algorithm = "BanditPAM",
    size_t max_iter = 1000,
    size_t buildConfidence = 1000,
    size_t swapConfidence = 10000);

  ~KMedoids();

  void fit(const arma::mat& inputData, const std::string& loss);

  // cache-related variables

  size_t cache_multiplier = 1000;

  float* cache;

  arma::uvec permutation;

  size_t permutation_idx;

  std::unordered_map<size_t, size_t> reindex;

  // set to false for debugging only, to measure speedup
  bool use_perm = true;

  // set to false for debugging only, to measure speedup
  bool use_cache_p = true;

  // The functions below are getters for read-only attributes

  arma::urowvec getMedoidsFinal();

  arma::urowvec getMedoidsBuild();

  arma::urowvec getLabels();

  size_t getSteps();

  // The functions below are get/set functions for attributes

  size_t getNMedoids();

  void setNMedoids(size_t new_num);

  std::string getAlgorithm();

  void setAlgorithm(const std::string& new_alg);

  size_t getMaxIter();

  void setMaxIter(size_t new_max);

  size_t getbuildConfidence();

  void setbuildConfidence(size_t new_buildConfidence);

  size_t getswapConfidence();

  void setswapConfidence(size_t new_swapConfidence);

  void setLossFn(std::string loss);

 protected:
  void calc_best_distances_swap(
    const arma::mat& data,
    arma::urowvec* medoidIndices,
    arma::rowvec* best_distances,
    arma::rowvec* second_distances,
    arma::urowvec* assignments);

  double calc_loss(
    const arma::mat& data,
    arma::urowvec* medoidIndices);

  // if you change use_cache, also change use_cache_p
  double cachedLoss(
    const arma::mat& data,
    size_t i,
    size_t j,
    bool use_cache = true);

  // Loss functions
  size_t lp;

  double LP(const arma::mat& data, size_t i, size_t j) const;

  double LINF(const arma::mat& data, size_t i, size_t j) const;

  double cos(const arma::mat& data, size_t i, size_t j) const;

  double manhattan(const arma::mat& data, size_t i, size_t j) const;

  void checkAlgorithm(const std::string& algorithm);

  // Constructor params
  size_t n_medoids;

  std::string algorithm;

  size_t max_iter;

  // Properties of the KMedoids instance
  arma::mat data;

  arma::urowvec labels;

  arma::urowvec medoid_indices_build;

  arma::urowvec medoid_indices_final;

  double (KMedoids::*lossFn)(
    const arma::mat& data,
    size_t i,
    size_t j)
    const;

  size_t steps;

  // Hyperparameters
  size_t buildConfidence;

  size_t swapConfidence;

  const double precision = 0.001;

  size_t batchSize = 100;
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_KMEDOIDS_ALGORITHM_HPP_
