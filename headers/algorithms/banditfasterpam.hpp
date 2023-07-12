#ifndef HEADERS_ALGORITHMS_BANDITFASTERPAM_HPP_
#define HEADERS_ALGORITHMS_BANDITFASTERPAM_HPP_

#include <omp.h>
#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>

#include "kmedoids_algorithm.hpp"


namespace km {
/**
 * @brief Contains all necessary BanditFasterPAM functions
 */
class BanditFasterPAM : public km::KMedoids {
 public:
  /**
  * @brief Runs BanditFasterPAM to identify a dataset's medoids.
  *
  * @param inputData Input data to cluster
  */
  void fitBanditFasterPAM(
  const arma::fmat& inputData,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat);

  /**
  * @brief Performs uniform random sampling to initialize the k medoids.
  *
  * @param n Number of rows in the dataset
  *
  * @returns Array of medoid indices created from uniform random sampling
   */
  arma::urowvec randomInitialization(
      size_t n);

  /**
   * @brief: TODO
   *
   * TODO
   *
   * @param assignments
   * @param bestDistances
   * @param secondBestDistances
   * @param Delta_TD_ms
   */
  arma::frowvec calcDeltaTDMs(
    arma::urowvec* assignments,
    arma::frowvec* bestDistances,
    arma::frowvec* secondBestDistances);

  float swapSigma(
    const size_t candidate,
    const arma::fmat& data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    const arma::frowvec* bestDistances,
    const arma::frowvec* secondBestDistances,
    const arma::urowvec* assignments);

  std::tuple<float, arma::frowvec> swapTarget(
    const arma::fmat& data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    const size_t candidate,
    const arma::frowvec* bestDistances,
    const arma::frowvec* secondBestDistances,
    const arma::urowvec* assignments,
    const size_t exact = 0);

    /**
    * @brief Performs the SWAP steps of BanditFasterPAM.
    *
    * @param data Transposed input data to cluster
    * @param medoidIndices Array of medoid indices created from the BUILD step
    * that is modified in place as better medoids are identified
    * @param assignments Array of containing the medoid each point is closest to
    */
  void swapBanditFasterPAM(
    const arma::fmat& data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec* medoidIndices,
    arma::urowvec* assignments);
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_BANDITFASTERPAM_HPP_
