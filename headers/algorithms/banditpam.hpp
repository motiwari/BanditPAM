#ifndef HEADERS_ALGORITHMS_BANDITPAM_HPP_
#define HEADERS_ALGORITHMS_BANDITPAM_HPP_

#include <omp.h>
#include <armadillo>
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
  void fit_bpam(const arma::mat& inputData);

  /**
   * @brief Empirical estimation of standard deviation of arm returns
   * in the BUILD step.
   *
   * @param data Transposed input data to cluster
   * @param best_distances Contains best distances from each point to medoids
   * @param use_aboslute Flag to use the absolute distance to each arm instead
   * of improvement over prior loss; necessary for the first BUILD step
   * 
   * @returns Estimate of each arm's standard deviation
   */
  arma::rowvec build_sigma(
    const arma::mat& data,
    const arma::rowvec& best_distances,
    const bool use_absolute);

  /** 
   * @brief Estimates the mean reward for each arm in the BUILD steps.
   * 
   * @param data Transposed input data to cluster
   * @param target Candidate datapoints to consider adding as medoids
   * @param best_distances Contains best distances from each point to medoids
   * @param use_aboslute Flag to use the absolute distance to each arm instead
   * of improvement over prior loss; necessary for the first BUILD step
   * @param exact 0 if using standard batch size; size of dataset otherwise
   * 
   * @returns Estimate of each arm's change in loss
   */
  arma::rowvec build_target(
    const arma::mat& data,
    const arma::uvec* target,
    const arma::rowvec* best_distances,
    const bool use_absolute,
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
    const arma::mat& data,
    arma::urowvec* medoidIndices,
    arma::mat* medoids);

  /**
   * @brief Empirical estimation of standard deviation of arm returns
   * in the SWAP step.
   *
   * @param data Transposed input data to cluster
   * @param best_distances Contains best distances from each point to medoids
   * @param second_best_distances Contains second best distances from each point to medoids
   * @param assignments Assignments of datapoints to their closest medoid
   * 
   * @returns Estimate of each arm's standard deviation
   */
  arma::mat swap_sigma(
    const arma::mat& data,
    const arma::rowvec* best_distances,
    const arma::rowvec* second_best_distances,
    const arma::urowvec* assignments);

  /**
   * @brief Estimates the mean reward for each arm in SWAP step.
   *
   * @param data Transposed input data to cluster
   * @param medoidIndices Array of medoids that is modified in place
   * as medoids are swapped in and out
   * @param targets Set of potential swaps to be evaluated
   * @param best_distances Contains best distances from each point to medoids
   * @param second_best_distances Contains second best distances from each point to medoids
   * @param assignments Assignments of datapoints to their closest medoid
   * @param exact 0 if using standard batch size; size of dataset otherwise
   * 
   * @returns Estimate of each arm's change in loss
   */
  arma::vec swap_target(
    const arma::mat& data,
    const arma::urowvec* medoidIndices,
    const arma::uvec* targets,
    const arma::rowvec* best_distances,
    const arma::rowvec* second_best_distances,
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
    const arma::mat& data,
    arma::urowvec* medoidIndices,
    arma::mat* medoids,
    arma::urowvec* assignments);
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_BANDITPAM_HPP_
