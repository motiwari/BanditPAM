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

  // The functions below are getters for read-only attributes

  arma::urowvec getMedoidsFinal() const;

  arma::urowvec getMedoidsBuild() const;

  arma::urowvec getLabels() const;

  size_t getSteps() const;

  // The functions below are get/set functions for attributes

  size_t getNMedoids() const;

  void setNMedoids(size_t new_num);

  std::string getAlgorithm() const;

  void setAlgorithm(const std::string& new_alg);

  size_t getMaxIter() const;

  void setMaxIter(size_t new_max);

  size_t getbuildConfidence() const;

  void setbuildConfidence(size_t new_buildConfidence);

  size_t getswapConfidence() const;

  void setswapConfidence(size_t new_swapConfidence);

  void setLossFn(std::string loss);

  /// The cache will be of size cache_multiplier*nlogn
  size_t cache_multiplier = 1000;

  /// The cache which stores pairwise distance computations
  float* cache;

  /// The permutation in which to sample the reference points
  arma::uvec permutation;

  /// The index of our current position in the permutated points
  size_t permutation_idx;

  /// A map from permutation index of each point to its original index
  std::unordered_map<size_t, size_t> reindex;

  /// Used for debugging only to toggle a fixed permutation of points
  bool use_perm = true;

  /// Used for debugging only to toggle use of the cache
  bool use_cache_p = true;

 protected:
  void calc_best_distances_swap(
    const arma::mat& data,
    const arma::urowvec* medoidIndices,
    arma::rowvec* best_distances,
    arma::rowvec* second_distances,
    arma::urowvec* assignments);

  double calc_loss(
    const arma::mat& data,
    const arma::urowvec* medoidIndices);

  // NOTE: if you change use_cache, also change use_cache_p
  double cachedLoss(
    const arma::mat& data,
    const size_t i,
    const size_t j,
    const bool use_cache = true);

  // Loss functions
  /// If using an L_p loss, the value of p
  size_t lp;

  double LP(
    const arma::mat& data,
    const size_t i,
    const size_t j) const;

  double LINF(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  double cos(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  double manhattan(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  void checkAlgorithm(const std::string& algorithm) const;

  /// Number of medoids to use -- the "k" in k-medoids
  size_t n_medoids;

  /// k-medoids algorithm to use
  std::string algorithm;

  /// Maximum number of SWAP steps to perform
  size_t max_iter;

  /// Data to be clustered
  arma::mat data;

  /// Cluster assignments of each point
  arma::urowvec labels;

  /// Indices of the medoids after BUILD step
  arma::urowvec medoid_indices_build;

  /// Indices of the medoids after clustering
  arma::urowvec medoid_indices_final;

  /// Function pointer to the loss function to use
  double (KMedoids::*lossFn)(
    const arma::mat& data,
    const size_t i,
    const size_t j)
    const;

  /// Number of SWAP steps performed
  size_t steps;

  /// Governs the error rate of each BUILD step in BanditPAM
  size_t buildConfidence;

  /// Governs the error rate of each SWAP step in BanditPAM
  size_t swapConfidence;

  /// Used for floatcomparisons, primarily number of "arms" remaining
  const double precision = 0.001;

  /// Number of points to sample per reference batch
  size_t batchSize = 100;
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_KMEDOIDS_ALGORITHM_HPP_
