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
 * @brief KMedoids class. Creates a KMedoids object that can be used to find the medoids
 * for a particular set of input data.
 *
 * @param n_medoids Number of medoids to use and clusters to create
 * @param algorithm Algorithm used to find medoids: "BanditPAM", "PAM", or "FastPAM1"
 * @param max_iter The maximum number of SWAP steps the algorithm runs
 * @param buildConfidence Parameter that affects the width of BUILD confidence intervals
 * @param swapConfidence Parameter that affects the width of SWAP confidence intervals
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

  /**
   * @brief Finds medoids for the input data, given loss function.
   *
   * @param inputData Input data to cluster
   * @param loss The loss function used during medoid computation
   * 
   * @throws if the input data is empty.
   */
  void fit(const arma::mat& inputData, const std::string& loss);

  /**
   * @brief Returns the medoids at the end of the BUILD step.
   * 
   * @returns Medoids at the end of the BUILD step
   */
  arma::urowvec getMedoidsBuild() const;

  /**
   * @brief Returns the medoids after all SWAP steps have been run.
   * 
   * @returns Medoids at the end of all SWAP steps
   */
  arma::urowvec getMedoidsFinal() const;

  /**
   * @brief Returns the medoid assignments for each datapoint.
   * 
   * @returns Cluster assignment for each point
   */
  arma::urowvec getLabels() const;

  /**
   * @brief Returns the number of SWAP steps performed.
   * 
   * @returns Number of swap steps performed
   */
  size_t getSteps() const;

  /**
   * @brief Returns the number of medoids, k.
   * 
   * @returns Current value of k, the number of medoids/clusters
   */
  size_t getNMedoids() const;

  /**
   * @brief Sets the number of medoids, k.
   * 
   * @param new_num The new number, k, of medoids to use
   */
  void setNMedoids(size_t new_num);

  /**
   * @brief Returns the algorithm being used for k-medoids clustering.
   * 
   * @returns The name of the algorithm being used
   */
  std::string getAlgorithm() const;

  /**
   * @brief Sets the algorithm being used for k-medoids clustering.
   *
   * @param new_alg The new algorithm to use: "BanditPAM", "PAM", or "FastPAM1"
   */
  void setAlgorithm(const std::string& new_alg);

  /**
   * @brief Returns the maximum number of SWAP steps during clustering.
   * 
   * @returns The maximum number of SWAP steps that can be
   * performed during clustering
   */
  size_t getMaxIter() const;

  /**
   * @brief Sets the maximum number of SWAP steps during clustering.
   * 
   * @param new_max New maximum number of iterations to use
   */
  void setMaxIter(size_t new_max);

  /**
   * @brief Returns the buildConfidence, a parameter that affects the width
   * of the confidence intervals during the BUILD step.
   * 
   * @returns The current value of buildConfidence
   */
  size_t getbuildConfidence() const;

  /**
   * @brief Sets the buildConfidence, a parameter that affects the width
   * of the confidence intervals during the BUILD step.
   *
   * @param new_buildConfidence The new buildConfidence to use
   * 
   * @throws If attempting to set buildConfidence when not using BanditPAM
   */
  void setbuildConfidence(size_t new_buildConfidence);

  /**
   * @brief Returns the swapConfidence, a parameter that affects the width
   * of the confidence intervals during the BUILD step.
   * 
   * @returns The current value of swapConfidence
   */
  size_t getswapConfidence() const;

  /**
   * @brief Sets the swapConfidence, a parameter that affects the width
   * of the confidence intervals during the BUILD step.
   *
   * @param new_swapConfidence The new swapConfidence to use
   * 
   * @throws If attempting to set buildConfidence when not using BanditPAM
   */
  void setswapConfidence(size_t new_swapConfidence);

  /**
   * @brief Sets the loss function to use during clustering.
   *
   * @param loss Loss function to be used e.g. "L2"
   * 
   * @throws If the loss function is unrecognized
   */
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
  /**
   * @brief Calculates the best and second best distances for each datapoint to
   * the medoids in the current set of medoids.
   *
   * @param data Transposed data to cluster
   * @param medoidIndices Array of medoid indices corresponding to dataset entries
   * @param best_distances Array of best distances from each point to previous set
   * of medoids
   * @param second_best_distances Array of second smallest distances from each
   * point to previous set of medoids
   * @param assignments Assignments of datapoints to their closest medoid
   */
  void calc_best_distances_swap(
    const arma::mat& data,
    const arma::urowvec* medoidIndices,
    arma::rowvec* best_distances,
    arma::rowvec* second_distances,
    arma::urowvec* assignments);

  /**
   * @brief Calculate the overall loss for the given choice of medoids. 
   * 
   * @param data Transposed data to cluster
   * @param medoidIndices Indices of the medoids in the dataset
   * 
   * @returns The total (not average) loss
   */
  double calc_loss(
    const arma::mat& data,
    const arma::urowvec* medoidIndices);

  
  /**
   * @brief A wrapper around the given loss function that caches distances
   * between the given points.
   * 
   * NOTE: if you change use_cache, also change use_cache_p
   * 
   * @param data Transposed data to cluster
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   * @param use_cache Indices of the medoids in the dataset
   * 
   * @returns The distance between points i and j
   */
  double cachedLoss(
    const arma::mat& data,
    const size_t i,
    const size_t j,
    const bool use_cache = true); 


  /// If using an L_p loss, the value of p
  size_t lp;

  /**
   * @brief Computes the Lp distance between the 
   * datapoints of indices i and j in the dataset
   *
   * @param data Transposed data to cluster
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   * 
   * @returns The Lp distance between points i and j
   */
  double LP(
    const arma::mat& data,
    const size_t i,
    const size_t j) const;

  /**
   * @brief Computes the L-infinity distance between the 
   * datapoints of indices i and j in the dataset
   *
   * @param data Transposed data to cluster
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   * 
   * @returns The L-infinity distance between points i and j
   */
  double LINF(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  /**
   * @brief Computes the cosine distance between the 
   * datapoints of indices i and j in the dataset
   *
   * @param data Transposed data to cluster
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   * 
   * @returns The cosine distance between points i and j
   */
  double cos(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  /**
   * @brief Computes the Manhattan distance between the 
   * datapoints of indices i and j in the dataset
   *
   * @param data Transposed data to cluster
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   * 
   * @returns The Manhattan distance between points i and j
   */
  double manhattan(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  /**
   * @brief Checks whether algorithm choice is valid. The given 
   * algorithm must be either "BanditPAM", "PAM", or "FastPAM1". 
   *
   * @param algorithm Name of the k-medoids algorithm to use
   * 
   * @throws If the algorithm is invalid.
   */
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
