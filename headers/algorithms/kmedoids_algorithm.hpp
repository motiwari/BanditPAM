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
 * @param nMedoids Number of medoids to use and clusters to create
 * @param algorithm Algorithm used to find medoids: "BanditPAM", "PAM", or "FastPAM1"
 * @param maxIter The maximum number of SWAP steps the algorithm runs
 * @param buildConfidence Parameter that affects the width of BUILD confidence intervals
 * @param swapConfidence Parameter that affects the width of SWAP confidence intervals
 */
class KMedoids {
 public:
  KMedoids(
    size_t nMedoids = 5,
    const std::string& algorithm = "BanditPAM",
    size_t maxIter = 1000,
    size_t buildConfidence = 1000,
    size_t swapConfidence = 10000,
    size_t seed = 0);

  ~KMedoids();

  /**
   * @brief Finds medoids for the input data, given loss function.
   *
   * @param inputData Input data to cluster
   * @param loss The loss function used during medoid computation
   * 
   * @throws if the input data is empty.
   */
  void fit(const arma::fmat& inputData, const std::string& loss);

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
   * @param newNMedoids The new number, k, of medoids to use
   */
  void setNMedoids(size_t newNMedoids);

  /**
   * @brief Returns the algorithm being used for k-medoids clustering.
   * 
   * @returns The name of the algorithm being used
   */
  std::string getAlgorithm() const;

  /**
   * @brief Sets the algorithm being used for k-medoids clustering.
   *
   * @param newAlgorithm The new algorithm to use: "BanditPAM", "PAM", or "FastPAM1"
   */
  void setAlgorithm(const std::string& newAlgorithm);

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
   * @param newMaxIter New maximum number of iterations to use
   */
  void setMaxIter(size_t newMaxIter);

  /**
   * @brief Returns the buildConfidence, a parameter that affects the width
   * of the confidence intervals during the BUILD step.
   * 
   * @returns The current value of buildConfidence
   */
  size_t getBuildConfidence() const;

  /**
   * @brief Sets the buildConfidence, a parameter that affects the width
   * of the confidence intervals during the BUILD step.
   *
   * @param newBuildConfidence The new buildConfidence to use
   * 
   * @throws If attempting to set buildConfidence when not using BanditPAM
   */
  void setBuildConfidence(size_t newBuildConfidence);

  /**
   * @brief Returns the swapConfidence, a parameter that affects the width
   * of the confidence intervals during the BUILD step.
   * 
   * @returns The current value of swapConfidence
   */
  size_t getSwapConfidence() const;

  /**
   * @brief Sets the swapConfidence, a parameter that affects the width
   * of the confidence intervals during the BUILD step.
   *
   * @param newSwapConfidence The new swapConfidence to use
   * 
   * @throws If attempting to set buildConfidence when not using BanditPAM
   */
  void setSwapConfidence(size_t newSwapConfidence);

  /**
   * @brief Sets the random seed for armadillo.
   *
   * @param newSeed The new seed value to use
   */
  void setSeed(size_t newSeed);

  /**
   * @brief Gets the value of the last supplied seed used by armadillo
   *
   * @param newSeed The new seed value to use
   */
  size_t getSeed() const;

  /**
   * @brief Sets the loss function to use during clustering.
   *
   * @param loss Loss function to be used e.g. "L2"
   * 
   * @throws If the loss function is unrecognized
   */
  void setLossFn(std::string loss);

  /**
   * @brief Gets the loss function currently used by the KMedoids object
   *
   * @returns Loss function currently being recognized
   */
  std::string getLossFn() const;

  /**
   * @brief Get the average loss from the prior clustering
   *
   * @returns The average clustering loss from the prior clusteringq
   *
   * @throws If no clustering has been run yet
   */
  float getAverageLoss() const;

  /// The cache will be of size cacheMultiplier*nlogn
  size_t cacheMultiplier = 1000;

  /// The cache which stores pairwise distance computations
  float* cache;

  /// The permutation in which to sample the reference points
  arma::uvec permutation;

  /// The index of our current position in the permutated points
  size_t permutationIdx;

  /// A map from permutation index of each point to its original index
  std::unordered_map<size_t, size_t> reindex;

  /// Used for debugging only to toggle a fixed permutation of points
  bool usePerm = true;

  /// Used for debugging only to toggle use of the cache
  bool useCacheP = true;

 protected:
  /**
   * @brief Calculates the best and second best distances for each datapoint to
   * the medoids in the current set of medoids.
   *
   * @param data Transposed data to cluster
   * @param medoidIndices Array of medoid indices corresponding to dataset entries
   * @param bestDistances Array of best distances from each point to previous set
   * of medoids
   * @param secondBestDistances Array of second smallest distances from each
   * point to previous set of medoids
   * @param assignments Assignments of datapoints to their closest medoid
   */
  void calcBestDistancesSwap(
    const arma::fmat& data,
    const arma::urowvec* medoidIndices,
    arma::frowvec* bestDistances,
    arma::frowvec* secondBestDistances,
    arma::urowvec* assignments,
    const bool swapPerformed = true);

  /**
   * @brief Calculate the average loss for the given choice of medoids. 
   * 
   * @param data Transposed data to cluster
   * @param medoidIndices Indices of the medoids in the dataset
   * 
   * @returns The average loss, i.e., the average distance from each point to its
   * nearest medoid
   */
  float calcLoss(
    const arma::fmat& data,
    const arma::urowvec* medoidIndices);

  /**
   * @brief A wrapper around the given loss function that caches distances
   * between the given points.
   * 
   * NOTE: if you change useCache, also change useCacheP
   * 
   * @param data Transposed data to cluster
   * @param i Index of first datapoint
   * @param j Index of second datapoint
   * @param useCache Indices of the medoids in the dataset
   * 
   * @returns The distance between points i and j
   */
  float cachedLoss(
    const arma::fmat& data,
    const size_t i,
    const size_t j,
    const bool useCache = true);

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
  float LP(
    const arma::fmat& data,
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
  float LINF(const arma::fmat& data,
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
  float cos(const arma::fmat& data,
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
  float manhattan(const arma::fmat& data,
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
  size_t nMedoids;

  /// k-medoids algorithm to use
  std::string algorithm;

  /// Maximum number of SWAP steps to perform
  size_t maxIter;

  /// Data to be clustered
  arma::fmat data;

  /// Cluster assignments of each point
  arma::urowvec labels;

  /// Indices of the medoids after BUILD step
  arma::urowvec medoidIndicesBuild;

  /// Indices of the medoids after clustering
  arma::urowvec medoidIndicesFinal;

  /// Function pointer to the loss function to use
  float (KMedoids::*lossFn)(
    const arma::fmat& data,
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
  const float precision = 0.001;

  /// Contains the average loss at the last step of the algorithm
  float averageLoss;

  /// Number of points to sample per reference batch
  size_t batchSize = 100;

  /// The random seed with which to perform the clustering
  size_t seed = 0;
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_KMEDOIDS_ALGORITHM_HPP_
