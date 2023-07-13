#ifndef HEADERS_ALGORITHMS_FASTERPAM_HPP_
#define HEADERS_ALGORITHMS_FASTERPAM_HPP_

#include <tuple>
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
  * @brief Performs uniform random sampling to initialize the k medoids.
  *
  * @param n Number of rows in the dataset
  *
  * @returns Array of medoid indices created from uniform random sampling
  */
  arma::urowvec randomInitialization(
      size_t n);

  /**
  * @brief Performs an initialize swap if k > 1
  *
  * @param data Input data to cluster
  * @param medoidIndices Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  *
  * @returns Loss after initial assignment
  */
  float initialAssignment(
      const arma::fmat &data,
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      arma::urowvec medoidIndices,
      arma::frowvec *bestDistances,
      arma::frowvec *secondBestDistances,
      arma::urowvec *assignments,
      arma::urowvec *secondAssignments);

  /**
  * @brief Finds the single best medoid within the data
  *
  * @param assignments Array of containing the medoid each point is closest to
  * @param medoidIndices Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param m Index of medoid to start seara potential contribution: is there a way to use bandits to get the guaranteed right answer instead of doing naive uniform random sampling?ching with
  *
  * @returns Tuple of whether a better medoid was found and the loss change
  */
  std::tuple<bool, float> chooseMedoidWithinPartition(
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      arma::urowvec assignments,
      arma::urowvec& medoidIndices,
      size_t m);

  /**
  * @brief Updates the loss change of a potential swap
  *
  * @param bestDistances Array of best distances for each point
  * @param secondBestDistances Array of second best distances for each point
  * @param loss Array of losses for each medoid
  * @param assignments Array of containing the medoid each point is closest to
  */
  void updateRemovalLoss(
      arma::frowvec *bestDistances,
      arma::frowvec *secondBestDistances,
      arma::frowvec& loss,
      arma::urowvec *assignments);

  /**
  * @brief Find the index of the best swap and its loss change
  *
  * @param removal_loss Array of losses for each medoid
  * @param bestDistances Array of best distances for each point
  * @param secondBestDistances Array of second best distances for each point
  * @param j Row in distance matrix to use for distance calculations
  * @param assignments Array of containing the medoid each point is closest to
  *
  * @returns Tuple of the loss change and the index of the best swap
  */
  std::tuple<float, size_t> findBestSwap(
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      arma::frowvec& removal_loss,
      arma::frowvec *bestDistances,
      arma::frowvec *secondBestDistances,
      size_t j,
      arma::urowvec *assignments);

  /**
  * @brief Update the second nearest medoid and the distance to it
  *
  * @param medoidIndices Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param n Number of rows in the dataset
  * @param b Medoid index
  * @param o Row in distance matrix to use for distance calculations
  * @param djo Current distance to second nearest medoid
  *
  * @returns Tuple of updated second nearest medoid and distance to it
  */
  std::tuple<size_t, float> updateSecondNearest(
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      arma::urowvec medoidIndices,
      size_t n,
      size_t b,
      size_t o,
      float djo);

  /**
  * @brief Execute a swap and adjust losses and distances accordingly
  *
  * @param medoidIndices Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param bestDistances Array of best distances for each point
  * @param secondBestDistances Array of second best distances for each point
  * @param assignments Array of containing the medoid each point is closest to
  * @param secondAssignments Array of containing the medoid each point is
  * closest to after the one in assignments array
  * @param b Medoid index
  * @param j Row in distance matrix to use for distance calculations
  *
  * @returns Loss change from swap
  */
  float doSwap(
      std::optional<std::reference_wrapper<const arma::fmat>> distMat,
      arma::urowvec& medoidIndices,
      arma::frowvec *bestDistances,
      arma::frowvec *secondBestDistances,
      arma::urowvec *assignments,
      arma::urowvec *secondAssignments,
      size_t b,
      size_t j);

  /**
  * @brief Performs the SWAP steps of FasterPAM.
  *
  * @param data Input data to cluster
  * @param medoidIndices Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param assignments Array of containing the medoid each point is closest to
  * @param secondAssignments Array of containing the medoid each point is
  * closest to after the one in assignments array
  *
  * @returns Tuple of the new assignments and the number of swaps performed
  */
  std::tuple<arma::urowvec, size_t> swapFasterPAM(
    const arma::fmat &data,
    std::optional<std::reference_wrapper<const arma::fmat>> distMat,
    arma::urowvec& medoidIndices,
    arma::urowvec assignments,
    arma::urowvec secondAsignments);
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_FASTERPAM_HPP_
