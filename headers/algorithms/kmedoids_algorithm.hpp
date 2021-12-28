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
   * @brief Finds medoids for the input data, given loss function
   *
   * @param inputData Input data to cluster
   * @param loss The loss function used during medoid computation
   */
  void fit(const arma::mat& inputData, const std::string& loss);

  /**
   * @brief Returns the medoids at the end of the BUILD step
   */
  arma::urowvec getMedoidsBuild() const;

  /**
   * @brief Returns the medoids after all SWAP steps have been run
   */
  arma::urowvec getMedoidsFinal() const;

  /**
   * @brief Returns the medoid assignments for each datapoint
   *
   * Returns the medoid each input datapoint is assigned to after KMedoids::fit
   * has been called and the final medoids have been identified
   */
  arma::urowvec getLabels() const;

  /**
   * @brief Returns the number of swap steps
   *
   * Returns the number of SWAP steps completed during the last call to
   * KMedoids::fit
   */
  size_t getSteps() const;

  /**
   * @brief Returns the number of medoids
   *
   * Returns the number of medoids to be identified during KMedoids::fit
   */
  size_t getNMedoids() const;

  /**
   * @brief Sets the number of medoids
   *
   * Sets the number of medoids to be identified during KMedoids::fit
   */
  void setNMedoids(size_t new_num);

  /**
   * @brief Returns the algorithm for KMedoids
   *
   * Returns the algorithm used for identifying the medoids during KMedoids::fit
   */
  std::string getAlgorithm() const;

  /**
   * @brief Sets the algorithm for KMedoids
   *
   * Sets the algorithm used for identifying the medoids during KMedoids::fit
   *
   * @param new_alg New algorithm to use
   */
  void setAlgorithm(const std::string& new_alg);

  /**
   * @brief Returns the maximum number of iterations for KMedoids
   *
   * Returns the maximum number of iterations that can be run during
   * KMedoids::fit
   */
  size_t getMaxIter() const;

  /**
   * @brief Sets the maximum number of SWAP steps
   * @param new_max New maximum number of iterations to use
   */
  void setMaxIter(size_t new_max);

  /**
   * @brief Returns the constant buildConfidence
   *
   * Returns the constant that affects the sensitivity of build confidence bounds
   * that can be run during KMedoids::fit
   */
  size_t getbuildConfidence() const;

  /**
   * @brief Sets the constant buildConfidence
   *
   * Sets the constant that affects the sensitivity of build confidence bounds
   * that can be run during KMedoids::fit
   *
   *  @param new_buildConfidence New buildConfidence
   */
  void setbuildConfidence(size_t new_buildConfidence);

  /**
   * @brief Returns the constant swapConfidence
   *
   * Returns the constant that affects the sensitivity of swap confidence bounds
   * that can be run during KMedoids::fit
   */
  size_t getswapConfidence() const;

  /**
   * @brief Sets the constant swapConfidence
   *
   * Sets the constant that affects the sensitivity of swap confidence bounds
   * that can be run during KMedoids::fit
   *
   * @param new_swapConfidence New swapConfidence
   */
  void setswapConfidence(size_t new_swapConfidence);

  /**
   * @brief Sets the loss function
   *
   * Sets the loss function used during KMedoids::fit
   *
   * @param loss Loss function to be used e.g. L2
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
   * @brief Calculates distances in swap step
   *
   * Calculates the best and second best distances for each datapoint to one of
   * the medoids in the current medoid set.
   *
   * @param data Transposed input data to find the medoids of
   * @param medoid_indices Array of medoid indices corresponding to dataset entries
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
   * @brief Calculate loss for medoids
   *
   * Calculates the loss under the previously identified loss function of the
   * medoid indices.
   *
   * @param data Transposed input data to find the medoids of
   * @param medoid_indices Indices of the medoids in the dataset.
   */
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

  /**
   * @brief LP loss
   *
   * Calculates the LP loss between the datapoints at index i and j of the dataset
   *
   * @param data Transposed input data to find the medoids of
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   */
  double LP(
    const arma::mat& data,
    const size_t i,
    const size_t j) const;

  /**
   * @brief L_INFINITY loss
   *
   * Calculates the Manhattan loss between the datapoints at index i and j of the
   * dataset
   *
   * @param data Transposed input data to find the medoids of
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   */
  double LINF(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  /**
   * @brief cos loss
   *
   * Calculates the cosine loss between the datapoints at index i and j of the
   * dataset
   *
   * @param data Transposed input data to find the medoids of
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   */
  double cos(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  /**
   * @brief Manhattan loss
   *
   * Calculates the Manhattan loss between the datapoints at index i and j of the
   * dataset
   *
   * @param data Transposed input data to find the medoids of
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   */
  double manhattan(const arma::mat& data,
    const size_t i,
    const size_t j) const;

  /**
   *  @brief Checks whether algorithm input is valid
   *
   *  Checks whether the user's selected algorithm is a valid option.
   *
   *  @param algorithm Name of the algorithm input by the user.
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
