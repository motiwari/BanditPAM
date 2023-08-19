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
      const arma::fmat &inputData,
      std::optional<std::reference_wrapper<const arma::fmat>> distMat);

  /**
  * @brief Performs uniform random sampling to initialize the k medoids.
  *
  * @param n Number of rows in the dataset
  * @param medoidIndices Array of medoids that is modified in place
  * as medoids are identified
  * @param medoids Matrix that contains the coordinates of each medoid
   */
  void randomInitialization(
      size_t n,
      const arma::fmat &data,
      arma::urowvec *medoidIndices,
      arma::fmat *medoids);

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
  arma::fmat swapSigma(
      const arma::fmat &data,
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      const arma::frowvec *bestDistances,
      const arma::frowvec *secondBestDistances,
      const arma::urowvec *assignments,
      const size_t activeColumn);

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
   * @param exact false if using standard batch size; true otherwise
   *
   * @returns Estimate of each arm's change in loss
   */
  arma::fmat swapTarget(
      const arma::fmat &data,
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      const arma::urowvec *medoidIndices,
      const arma::uvec *targets,
      const arma::frowvec *bestDistances,
      const arma::frowvec *secondBestDistances,
      const arma::urowvec *assignments,
      const bool exact);

  /**
   * @brief Calculates the best and second best distances for each datapoint to
   * the medoids in the current set of medoids. Also updates best and second
   * best assignments and loss.
   *
   * @param data Transposed data to cluster
   * @param medoidIndices Array of medoid indices corresponding to dataset entries
   * @param bestDistances Array of best distances from each point to previous set
   * of medoids
   * @param secondBestDistances Array of second smallest distances from each
   * point to previous set of medoids
   * @param assignments Assignments of datapoints to their closest medoid
   * @param secondAssignments Assignments of datapoints to their second closest
   * medoid
   */
  void calcBestDistancesSwapInitial(
      const arma::fmat &data,
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      const arma::urowvec *medoidIndices,
      arma::frowvec *bestDistances,
      arma::frowvec *secondBestDistances,
      arma::urowvec *assignments,
      arma::urowvec *secondAssignments,
      const bool swapPerformed);

  /**
   * @brief Updates the second best medoid and distance for a given point.
   *
   * @param medoidIndices Array of medoid indices corresponding to dataset entries
   * @param n Non-medoid that was just swapped in
   * @param k Medoid that was just swapped out
   * @param o Index of current datapoint in consideration
   * @param djo Distance from o to n
   *
   * @returns Tuple of second best medoid and distance
   */
  std::tuple<size_t, float> updateSecondNearest(
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      const arma::urowvec *medoidIndices,
      size_t n,
      size_t k,
      size_t o,
      float djo);

  /**
   * @brief Calculates the best and second best distances for each datapoint to
   * the medoids in the current set of medoids. Also updates best and second
   * best assignments and loss.
   *
   * @param medoidIndices Array of medoid indices corresponding to dataset entries
   * @param bestDistances Array of best distances from each point to previous set
   * of medoids
   * @param secondBestDistances Array of second smallest distances from each
   * point to previous set of medoids
   * @param assignments Assignments of datapoints to their closest medoid
   * @param secondAssignments Assignments of datapoints to their second closest
   * medoid
   * @param k Medoid that was just swapped out
   * @param n Non-medoid that was just swapped in
   */
  void calcBestDistancesSwapWithFPOptimizations(
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      const arma::urowvec *medoidIndices,
      arma::frowvec *bestDistances,
      arma::frowvec *secondBestDistances,
      arma::urowvec *assignments,
      arma::urowvec *secondAssignments,
      size_t k,
      size_t n);

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
      const arma::fmat &data,
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      arma::urowvec *medoidIndices,
      arma::fmat *medoids,
      arma::urowvec *assignments,
      arma::urowvec *secondAssignments);
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_BANDITFASTERPAM_HPP_
