#ifndef HEADERS_ALGORITHMS_FASTERPAM_HPP_
#define HEADERS_ALGORITHMS_FASTERPAM_HPP_

#include <omp.h>
#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>

#include "kmedoids_algorithm.hpp"


namespace km {
/**
 * @brief Contains all necessary FasterPAM functions
 */
class FasterPAM : public km::KMedoids {
 public:
  /**
  * @brief Runs FasterPAM to identify a dataset's medoids.
  *
  * @param inputData Input data to cluster
  */
  void fitFasterPAM(
  const arma::fmat& inputData,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat);

  /**
  * @brief Performs the BUILD step of FasterPAM.
  *
  * Loops over all datapoint and checks each's distance to every other
  * datapoint in the dataset, then adds the point with the lowest overall
  * loss to the set of medoids.
  *
  * @param data Transposed input data to cluster
  * @param medoidIndices Array of medoids that is modified in place
  * as medoids are identified
  */
  void buildFasterPAM(
    const arma::fmat& data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec* medoidIndices);

  /**
   * @brief: TODO
   *
   * TODO
   *
   * @param assignments
   * @param bestDistances
   * @param secondBestDistances
   */
  arma::frowvec calcDeltaTDMs(
    arma::urowvec* assignments,
    arma::frowvec* bestDistances,
    arma::frowvec* secondBestDistances);

  /**
  * @brief Performs the SWAP steps of FasterPAM.
  *
  * @param data Transposed input data to cluster
  * @param medoidIndices Array of medoid indices created from the BUILD step
  * that is modified in place as better medoids are identified
  * @param assignments Array of containing the medoid each point is closest to
  */
  void swapFasterPAM(
    const arma::fmat& data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec* medoidIndices,
    arma::urowvec* assignments);
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_FASTERPAM_HPP_
