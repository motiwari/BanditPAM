#ifndef HEADERS_BANDITPAM_HPP_
#define HEADERS_BANDITPAM_HPP_

#include "kmedoids_algorithm.hpp"

#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>

/**
 *  \brief Class implementation for BanditPAM algorithm.
 *
 *  BanditPAM class. Consists of all necessary functions to implement
 *  BanditPAM algorithm.
 *
 */
class BanditPAM : public km::KMedoids {
 public:
    /*! \brief Runs BanditPAM algorithm.
    *
    *  Run the BanditPAM algorithm to identify a dataset's medoids.
    *
    *  @param input_data Input data to cluster
    */
    void fit_bpam(const arma::mat& inputData);

    /*! \brief Build step for BanditPAM
    *
    *  Runs build step for the BanditPAM algorithm. Draws batch sizes with replacement
    *  from reference set, and uses the estimated reward of the potential medoid
    *  solutions on the reference set to update the reward confidence intervals and
    *  accordingly narrow the solution set.
    *
    *  @param data Transposed input data to find the medoids of
    *  @param medoid_indices Uninitialized array of medoids that is modified in place
    *  as medoids are identified
    *  @param medoids Matrix of possible medoids that is updated as the bandit
    *  learns which datapoints will be unlikely to be good candidates
    */
    void build(const arma::mat& data, arma::rowvec& medoidIndices, arma::mat& medoids);

    /*! \brief Estimates the mean reward for each arm in build step
    *
    *  Estimates the mean reward (or loss) for each arm in the identified targets
    *  in the build step and returns a list of the estimated reward.
    *
    *  @param data Transposed input data to find the medoids of
    *  @param target Set of target datapoints to be estimated
    *  @param batch_size Number of datapoints sampled for updating confidence
    *  intervals
    *  @param best_distances Array of best distances from each point to previous set
    *  of medoids
    *  @param use_absolute Determines whether the absolute cost is added to the total
    */
    arma::rowvec build_target(
        const arma::mat& data,
        arma::uvec& target,
        size_t batch_size,
        arma::rowvec& best_distances,
        bool use_absolute
    );

    /*! \brief Swap step for BanditPAM
    *
    *  Runs Swap step for the BanditPAM algorithm. Draws batch sizes with replacement
    *  from reference set, and uses the estimated reward of the potential medoid
    *  solutions on the reference set to update the reward confidence intervals and
    *  accordingly narrow the solution set.
    *
    *  @param data Transposed input data to find the medoids of
    *  @param medoid_indices Array of medoid indices created from the build step
    *  that is modified in place as better medoids are identified
    *  @param medoids Matrix of possible medoids that is updated as the bandit
    *  learns which datapoints will be unlikely to be good candidates
    *  @param assignments Uninitialized array of indices corresponding to each
    *  datapoint assigned the index of the medoid it is closest to
    */
    void swap(
        const arma::mat& data,
        arma::rowvec& medoidIndices,
        arma::mat& medoids,
        arma::rowvec& assignments
    );

    /*! \brief Estimates the mean reward for each arm in swap step
    *
    *  Estimates the mean reward (or loss) for each arm in the identified targets
    *  in the swap step and returns a list of the estimated reward.
    *
    *  @param data Transposed input data to find the medoids of
    *  @param targets Set of target datapoints to be estimated
    *  @param batch_size Number of datapoints sampled for updating confidence
    *  intervals
    *  @param best_distances Array of best distances from each point to previous set
    *  of medoids
    *  @param second_best_distances Array of second smallest distances from each
    *  point to previous set of medoids
    *  @param assignments Assignments of datapoints to their closest medoid
    */
    arma::vec swap_target(
        const arma::mat& data,
        arma::rowvec& medoidIndices,
        arma::uvec& targets,
        size_t batch_size,
        arma::rowvec& best_distances,
        arma::rowvec& second_best_distances,
        arma::rowvec& assignments
    );
};
#endif // HEADERS_BANDITPAM_HPP_
