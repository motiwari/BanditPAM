#ifndef HEADERS_FASTPAM1_HPP_
#define HEADERS_FASTPAM1_HPP_

#include "kmedoids_algorithm.hpp"

#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>

/**
 *  \brief Class implementation for FastPAM1 algorithm.
 *
 *  FastPAM1 class. Consists of all necessary functions to implement
 *  FastPAM1 algorithm.
 *
 */
class FastPAM1 : public km::KMedoids {
 public:
    /*! \brief Runs FastPAM1 algorithm.
    *
    *  Run the FastPAM1 algorithm to identify a dataset's medoids.
    *
    *  @param input_data Input data to cluster
    */
    void fit_fastpam1(const arma::mat& input_data);

    /*! \brief Build step for the FastPAM1 algorithm
    *
    *  Runs build step for the FastPAM1 algorithm. Loops over all datapoint and
    *  checks its distance from every other datapoint in the dataset, then checks if
    *  the total cost is less than that of the medoid (if a medoid exists yet).
    *
    *  @param data Transposed input data to cluster
    *  @param medoid_indices Uninitialized array of medoids that is modified in place
    *  as medoids are identified
    */
    void build_fastpam1(const arma::mat& data, arma::rowvec& medoid_indices);

    /*! \brief Swap step for the FastPAM1 algorithm
    *
    *  Runs swap step for the FastPAM1 algorithm. Loops over all datapoint and
    *  compute the loss change when a medoid is replaced by the datapoint. The
    *  loss change is stored in an array of size n_medoids and the update is
    *  based on an if conditional outside of the loop. The best medoid is chosen
    *  according to the best loss change.
    *
    *  @param data Transposed input data to cluster
    *  @param medoid_indices Array of medoid indices created from the build step
    *  that is modified in place as better medoids are identified
    *  @param assignments Uninitialized array of indices corresponding to each
    *  datapoint assigned the index of the medoid it is closest to
    */
    void swap_fastpam1(const arma::mat& data, arma::rowvec& medoid_indices, arma::rowvec& assignments);
};
#endif // HEADERS_FASTPAM1_HPP_
