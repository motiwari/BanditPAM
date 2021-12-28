#ifndef HEADERS_ALGORITHMS_PAM_HPP_
#define HEADERS_ALGORITHMS_PAM_HPP_

#include <omp.h>
#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>

#include "kmedoids_algorithm.hpp"



namespace km {
/**
 * @brief Contains all necessary PAM functions
 */
class PAM : public km::KMedoids {
 public:
  /**
   * @brief Runs PAM to identify a dataset's medoids.
   * 
   * @param inputData Input data to cluster
   */
  void fitPAM(const arma::fmat& inputData);

  /**
  * @brief Performs the BUILD step of PAM.
  *
  * Loops over all datapoint and checks each's distance to every other
  * datapoint in the dataset, then adds the point with the lowest overall
  * loss to the set of medoids.
  *
  * @param data Transposed input data to cluster
  * @param medoidIndices Array of medoids that is modified in place
  * as medoids are identified
  */
  void buildPAM(
    const arma::fmat& data,
    arma::urowvec* medoidIndices);

  /** 
  * @brief Performs the SWAP steps of BanditPAM.
  * 
  * Loops over all (medoid, non-medoid) pairs and computes the change in loss
  * when the points are swapped in and out of the medoid set. Then updates
  * the list of medoids by performing the swap that would lower the overall 
  * loss the most, provided at least one such swap would reduce the loss.
  *
  * @param data Transposed input data to cluster
  * @param medoidIndices Array of medoid indices created from the BUILD step
  * that is modified in place as better medoids are identified
  * @param assignments Array of containing the medoid each point is closest to
  */
  void swapPAM(
    const arma::fmat& data,
    arma::urowvec* medoidIndices,
    arma::urowvec* assignments);
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_PAM_HPP_
