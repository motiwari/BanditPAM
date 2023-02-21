#ifndef HEADERS_ALGORITHMS_BANDITPAM_HPP_
#define HEADERS_ALGORITHMS_BANDITPAM_HPP_

#include <banditpam_common.h>
#include <vector>
#include <fstream>
#include <iostream>


#include "kmedoids_algorithm.hpp"

namespace km {
/**
 * @brief Contains all necessary BanditPAM functions
 */
class BanditPAM : public km::KMedoids {
 public:
  /**
   * @brief Runs BanditPAM to identify a dataset's medoids.
   *
   * @param inputData Input data to cluster
   */
  void fitBanditPAM(
    const arma_mat& inputData,
    std::optional<std::reference_wrapper<const arma_mat>> distMat);

  /**
   * @brief Empirical estimation of standard deviation of arm returns
   * in the BUILD step.
   *
   * @param data Transposed input data to cluster
   * @param bestDistances Contains best distances from each point to medoids
   * @param useAbsolute Flag to use the absolute distance to each arm instead
   * of improvement over prior loss; necessary for the first BUILD step
   *
   * @returns Estimate of each arm's standard deviation
   */
  arma_rowvec buildSigma(
    const arma_mat& data,
    std::optional<std::reference_wrapper<const arma_mat>> distMat,
    const arma_rowvec& bestDistances,
    const bool useAbsolute);

  /**
   * @brief Estimates the mean reward for each arm in the BUILD steps.
   *
   * @param data Transposed input data to cluster
   * @param target Candidate datapoints to consider adding as medoids
   * @param bestDistances Contains best distances from each point to medoids
   * @param useAbsolute Flag to use the absolute distance to each arm instead
   * of improvement over prior loss; necessary for the first BUILD step
   * @param exact 0 if using standard batch size; size of dataset otherwise
   *
   * @returns Estimate of each arm's change in loss
   */
  arma_rowvec buildTarget(
    const arma_mat& data,
    std::optional<std::reference_wrapper<const arma_mat>> distMat,
    const arma::uvec* target,
    const arma_rowvec* bestDistances,
    const bool useAbsolute,
    const size_t exact);

  /**
   * @brief Performs the BUILD step of BanditPAM.
   *
   * Draws batch size reference points with replacement and uses the estimated
   * reward of adding candidate medoids to the set of medoids. Constructs
   * confidence intervals for expected loss when adding each candidate as a
   * medoid, and continues drawing reference points until the best candidate's
   * confidence interval is disjoint from all others.
   *
   * @param data Transposed input data to cluster
   * @param medoidIndices Array of medoids that is modified in place
   * as medoids are identified
   * @param medoids Matrix that contains the coordinates of each medoid
   */
  void build(
    const arma_mat& data,
    std::optional<std::reference_wrapper<const arma_mat>> distMat,
    arma::urowvec* medoidIndices,
    arma_mat* medoids);

  /**
   * @brief Empirical estimation of standard deviation of arm returns
   * in the SWAP step.
   *
   * @param data Transposed input data to cluster
   * @param bestDistances Contains best distances from each point to medoids
   * @param secondBestDistances Contains second best distances from each point to medoids
   * @param assignments Assignments of datapoints to their closest medoid
   *
   * @returns Estimate of each arm's standard deviation
   */
  arma_mat swapSigma(
    const arma_mat& data,
    std::optional<std::reference_wrapper<const arma_mat>> distMat,
    const arma_rowvec* bestDistances,
    const arma_rowvec* secondBestDistances,
    const arma::urowvec* assignments);

  /**
   * @brief Estimates the mean reward for each arm in SWAP step.
   *
   * @param data Transposed input data to cluster
   * @param medoidIndices Array of medoids that is modified in place
   * as medoids are swapped in and out
   * @param targets Set of potential swaps to be evaluated
   * @param bestDistances Contains best distances from each point to medoids
   * @param secondBestDistances Contains second best distances from each point to medoids
   * @param assignments Assignments of datapoints to their closest medoid
   * @param exact 0 if using standard batch size; size of dataset otherwise
   *
   * @returns Estimate of each arm's change in loss
   */
  arma_mat swapTarget(
    const arma_mat& data,
    std::optional<std::reference_wrapper<const arma_mat>> distMat,
    const arma::urowvec* medoidIndices,
    const arma::uvec* targets,
    const arma_rowvec* bestDistances,
    const arma_rowvec* secondBestDistances,
    const arma::urowvec* assignments,
    const size_t exact);

  /**
  * @brief Performs the SWAP step of BanditPAM.
  *
  * Draws batch size reference points with replacement and uses the estimated
  * reward of performing a (medoid, non-medoid) swap. Constructs
  * confidence intervals for expected loss when performing the swap,
  * and continues drawing reference points until the best candidate's
  * confidence interval is disjoint from all others.
  *
  * @param data Transposed input data to cluster
  * @param medoidIndices Array of medoid indices created from the BUILD step
  * that is modified in place as better medoids are identified
  * @param medoids Matrix of possible medoids that is updated as the bandit
  * learns which datapoints will be unlikely to be good candidates
  * @param assignments Array of containing the medoid each point is closest to
  */
  void swap(
    const arma_mat& data,
    std::optional<std::reference_wrapper<const arma_mat>> distMat,
    arma::urowvec* medoidIndices,
    arma_mat* medoids,
    arma::urowvec* assignments);
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_BANDITPAM_HPP_
