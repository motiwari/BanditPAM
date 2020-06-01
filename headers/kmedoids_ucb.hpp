/*
 * Things to fix
 * generic matrix type
 * generic metric type
 * error checking for number of clusters >= unique data points
 * should the mediods be index numbers or the mediods themselves?
 */

#ifndef MLPACK_METHODS_KMEANS_KMEANS_HPP
#define MLPACK_METHODS_KMEANS_KMEANS_HPP

#include <math.h>
#include <iostream>
#include <armadillo>
#include <omp.h>
#include <fstream>

// switch all naming to camelCase
// switch to armadillo typedefs
// switch to rowvec
// most typedefs are 64 bits, overkill?
class KMediods
{
public:
    KMediods(size_t maxIterations = 1000);

    void cluster(const arma::mat &data,
                 const size_t clusters,
                 arma::urowvec &assignments,
                 arma::urowvec &medoid_indicies);

private:
    void build_sigma(
        const arma::mat &data,
        arma::rowvec &best_distances,
        arma::rowvec &sigma,
        arma::uword batch_size,
        bool use_absolute);

    arma::rowvec build_target(
        const arma::mat &data,
        arma::uvec &target,
        size_t batch_size,
        arma::rowvec &best_distances,
        bool use_absolute);

    void build(const arma::mat &data,
               const size_t clusters,
               arma::urowvec &medoid_indicies,
               arma::mat &medoids);

    double cost_fn_build(const arma::mat &data, arma::uword target, arma::uvec &tmp_refs, arma::rowvec &best_distances);

    arma::vec swap_target(
        const arma::mat &data,
        arma::urowvec &medoid_indices,
        arma::uvec &targets,
        size_t batch_size,
        arma::rowvec &best_distances,
        arma::rowvec &second_best_distances,
        arma::urowvec &assignments);

    void swap(const arma::mat &data,
              const size_t clusters,
              arma::urowvec &medoid_indicies,
              arma::mat &medoids);

    double calc_loss(const arma::mat &data,
                     const size_t clusters,
                     arma::urowvec &medoid_indicies);

    void swap_sigma(
        const arma::mat &data,
        arma::mat &sigma,
        size_t batch_size,
        arma::rowvec &best_distances,
        arma::rowvec &second_best_distances,
        arma::urowvec &assignments);

    double sigma_const = 0.1;
    size_t maxIterations;
};

#endif