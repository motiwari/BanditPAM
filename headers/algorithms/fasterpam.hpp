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
 * @brief Contains indices and distances to certain medoids
 */
class DistancePair {
public:
  size_t i;
  float d;

  /**
  * @brief Constructor that initializes i and d to the passed arguments
  *
  * @param i Index of the data point
  * @param d Distance from data point to medoid
  */
  DistancePair(size_t i, float d);

  /**
  * @brief Initialize an empty DistancePair
  */
  static DistancePair empty();
};

// TODO(@Adarsh321123): should there be space between the inside code and the left side (same with cpp file itself)?
/**
 * @brief Contains DistancePairs to nearest and second nearest medoids
 */
class Rec {
public:
  DistancePair near;
  DistancePair seco;

  /**
  * @brief Default constructor that initializes i and d with 0 and 0.0
  */
  Rec();

  /**
  * @brief Constructor that initializes i and d to the passed arguments
  *
  * This applies to both the nearest and second nearest DistancePairs
  *
  * @param i1 Index of the first data point
  * @param d1 Distance from first data point to nearest medoid
  * @param i2 Index of the second data point
  * @param d2 Distance from second data point to second nearest medoid
  */
  Rec(size_t i1, float d1, size_t i2, float d2);

  /**
  * @brief Initialize an empty Rec
  */
  static Rec empty();
};

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
  * @param k Number of medoids to initialize
  */
  std::vector<size_t> randomInitialization(
      size_t n,
      size_t k);

  /**
  * @brief Performs an initialize swap if k > 1
  *
  * @param mat Array of distances between the data points.
  * @param med Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  */
  std::tuple<float, std::vector<Rec>> initialAssignment(
      const arma::fmat& mat,
      std::vector<size_t> med);

  /**
  * @brief Checks for inconsistencies in distance assignments
  *
  * @param mat Array of distances between the data points.
  * @param med Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param data Input data to cluster
  */
  void debugAssertAssignment(
      const arma::fmat& mat,
      std::vector<size_t> med,
      std::vector<Rec>& data);

  /**
  * @brief Finds the single best medoid within the data
  *
  * @param mat Array of distances between the data points.
  * @param assi Array of containing the medoid each point is closest to
  * @param med Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param m Index of medoid to start seara potential contribution: is there a way to use bandits to get the guaranteed right answer instead of doing naive uniform random sampling?ching with
   */
  std::tuple<bool, float> chooseMedoidWithinPartition(
      const arma::fmat& mat,
      std::vector<size_t> assi,
      std::vector<size_t>& med,
      size_t m);

  /**
  * @brief Updates the loss change of a potential swap
  *
  * @param data Input data to cluster
  * @param loss Array of losses for each medoid
  */
  void updateRemovalLoss(
      std::vector<Rec>& data,
      std::vector<float>& loss);

  /**
  * @brief Find the index of the best swap and its loss change
  *
  * @param mat Array of distances between the data points.
  * @param removal_loss Array of losses for each medoid
  * @param data Input data to cluster
  * @param j Row in distance matrix to use for distance calculations
  */
  std::tuple<float, size_t> findBestSwap(
      const arma::fmat& mat,
      std::vector<float>& removal_loss,
      std::vector<Rec>& data,
      size_t j);

  /**
  * @brief Update the DistancePair with distance to the second nearest medoid
  *
  * @param mat Array of distances between the data points.
  * @param med Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param n Number of rows in the dataset
  * @param b Medoid index
  * @param o Row in distance matrix to use for distance calculations
  * @param djo Current distance to second nearest medoid
  */
  DistancePair updateSecondNearest(
      const arma::fmat& mat,
      std::vector<size_t> med,
      size_t n,
      size_t b,
      size_t o,
      float djo);

  /**
  * @brief Execute a swap and adjust losses and distances accordingly
  *
  * @param mat Array of distances between the data points.
  * @param med Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param data Input data to cluster
  * @param b Medoid index
  * @param j Row in distance matrix to use for distance calculations
  */
  float doSwap(
      const arma::fmat& mat,
      std::vector<size_t>& med,
      std::vector<Rec>& data,
      size_t b,
      size_t j);

  // TODO(@Adarsh321123): add returns to docstring (throughout)?
  /**
  * @brief Performs the SWAP steps of FasterPAM.
  *
  * @param inputData Input data to cluster
  * @param mat Array of distances between the data points.
  * @param med Array of medoid indices created from uniform random
  * sampling step that is modified in place as better medoids are identified
  * @param assi Array of containing the medoid each point is closest to
  */
  std::tuple<float, std::vector<size_t>, size_t, size_t> swapFasterPAM(
    const arma::fmat &inputData,
    const arma::fmat& mat,
    std::vector<size_t>& med,
    std::vector<size_t> assi);
  // TODO(@Adarsh321123): fix 2 vs 4 spaces
};
}  // namespace km
#endif  // HEADERS_ALGORITHMS_FASTERPAM_HPP_