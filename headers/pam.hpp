#ifndef HEADERS_PAM_HPP_
#define HEADERS_PAM_HPP_

#include "kmedoids_algorithm.hpp"

#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>

/**
 *  \brief Class implementation for PAM algorithm.
 *
 *  PAM class. Consists of all necessary functions to implement
 *  PAM algorithm.
 *
 */
class PAM : public km::KMedoids {
 public:
    /*! \brief Runs PAM algorithm.
    *
    *  Run the PAM algorithm to identify a dataset's medoids.
    *
    *  @param input_data Input data to cluster
    */
    void fit_naive(const arma::mat& inputData);

    /*! \brief Build step for the PAM algorithm
    *
    *  Runs build step for the PAM algorithm. Loops over all datapoint and
    *  checks its distance from every other datapoint in the dataset, then checks if
    *  the total cost is less than that of the medoid (if a medoid exists yet).
    *
    *  @param data Transposed input data to cluster
    *  @param medoid_indices Uninitialized array of medoids that is modified in place
    *  as medoids are identified
    */
    void build_naive(const arma::mat& data, arma::rowvec& medoidIndices);

    /*! \brief Swap step for the PAM algorithm
    *
    *  Runs build step for the PAM algorithm. Loops over all datapoint and
    *  checks its distance from every other datapoint in the dataset, then checks if
    *  the total cost is less than that of the medoid.
    *
    *  @param data Transposed input data to find the medoids of
    *  @param medoid_indices Array of medoid indices created from the build step
    *  that is modified in place as better medoids are identified
    *  @param assignments Uninitialized array of indices corresponding to each
    *  datapoint assigned the index of the medoid it is closest to
    */
    void swap_naive(const arma::mat& data, arma::rowvec& medoidIndices, arma::rowvec& assignments);
};
#endif // HEADERS_PAM_HPP_
