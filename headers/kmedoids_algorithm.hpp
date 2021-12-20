#ifndef HEADERS_KMEDOIDS_ALGORITHM_HPP_
#define HEADERS_KMEDOIDS_ALGORITHM_HPP_


#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <tuple>
#include <functional>
#include <unordered_map>
#include <string>
#include <mutex>


namespace km {
  /**
   *  \brief Class implementation for running KMedoids methods.
   *
   *  KMedoids class. Creates a KMedoids object that can be used to find the medoids
   *  for a particular set of input data.
   *
   *  @param n_medoids Number of medoids/clusters to create
   *  @param algorithm Algorithm used to find medoids; options are "BanditPAM" for
   *  the "BanditPAM" algorithm, or "naive" to use PAM
   *  @param max_iter The maximum number of iterations the algorithm runs for
   *  @param buildConfidence Constant that affects the sensitivity of build confidence bounds
   *  @param swapConfidence Constant that affects the sensitiviy of swap confidence bounds
   */
class KMedoids {
 public:
      KMedoids(size_t n_medoids = 5, const std::string& algorithm = "BanditPAM", size_t max_iter = 1000,
              size_t buildConfidence = 1000, size_t swapConfidence = 10000);

      ~KMedoids();

      void fit(const arma::mat& inputData, const std::string& loss);

      // cache-related variables

      size_t cache_multiplier = 1000;

      float* cache; // array of floats

      arma::uvec permutation;

      size_t permutation_idx;

      std::unordered_map<size_t, size_t> reindex; 

      bool use_perm = true; // set to false for debugging only, to measure speedup

      bool use_cache_p = true; // set to false for debugging only, to measure speedup

      // The functions below are getters for read-only attributes

      arma::rowvec getMedoidsFinal();

      arma::rowvec getMedoidsBuild();

      arma::rowvec getLabels();

      size_t getSteps();

      // The functions below are get/set functions for attributes

      size_t getNMedoids();

      void setNMedoids(size_t new_num);

      std::string getAlgorithm();

      void setAlgorithm(const std::string& new_alg); // pass by ref

      size_t getMaxIter();

      void setMaxIter(size_t new_max);

      size_t getbuildConfidence();

      void setbuildConfidence(size_t new_buildConfidence);

      size_t getswapConfidence();

      void setswapConfidence(size_t new_swapConfidence);

      void setLossFn(std::string loss);

 protected:
      // The functions below are PAM's constituent functions
      arma::rowvec build_sigma(
        const arma::mat& data,
        arma::rowvec& best_distances,
        arma::uword batch_size,
        bool use_absolute
      );

      void calc_best_distances_swap(
        const arma::mat& data,
        arma::rowvec& medoidIndices,
        arma::rowvec& best_distances,
        arma::rowvec& second_distances,
        arma::rowvec& assignments
      );

      arma::mat swap_sigma(
        const arma::mat& data,
        size_t batch_size,
        arma::rowvec& best_distances,
        arma::rowvec& second_best_distances,
        arma::rowvec& assignments
      );

      double calc_loss(const arma::mat& data, arma::rowvec& medoidIndices);

      // Loss functions
      double cachedLoss(const arma::mat& data, size_t i, size_t j, bool use_cache = true); // if you change use_cache, also change use_cache_p

      size_t lp;

      double LP(const arma::mat& data, size_t i, size_t j) const;

      double LINF(const arma::mat& data, size_t i, size_t j) const;

      double cos(const arma::mat& data, size_t i, size_t j) const;

      double manhattan(const arma::mat& data, size_t i, size_t j) const;

      void checkAlgorithm(const std::string& algorithm);

      // Constructor params
      size_t n_medoids; ///< number of medoids identified for a given dataset

      std::string algorithm; ///< options: "naive" and "BanditPAM"

      size_t max_iter; ///< maximum number of iterations during KMedoids::fit

      // Properties of the KMedoids instance
      arma::mat data; ///< input data used during KMedoids::fit

      arma::rowvec labels; ///< assignments of each datapoint to its medoid

      arma::rowvec medoid_indices_build; ///< medoids at the end of build step

      arma::rowvec medoid_indices_final; ///< medoids at the end of the swap step

      double (KMedoids::*lossFn)(const arma::mat& data, size_t i, size_t j) const; ///< loss function used during KMedoids::fit

      size_t steps; ///< number of actual swap iterations taken by the algorithm

      // Hyperparameters
      size_t buildConfidence; ///< constant that affects the sensitivity of build confidence bounds

      size_t swapConfidence; ///< constant that affects the sensitiviy of swap confidence bounds

      const double precision = 0.001; ///< bound for double comparison precision

      const size_t batchSize = 100; ///< batch size for computation steps
};
} // namespace km
#endif // HEADERS_KMEDOIDS_ALGORITHM_HPP_
