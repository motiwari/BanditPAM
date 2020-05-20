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
                 arma::Row<size_t> &assignments);

private:
    arma::Row<double> build_target(
        const arma::mat &data,
        arma::uvec &targets,
        size_t batch_size,
        arma::Row<double> &best_distances);

    void build(const arma::mat &data,
               const size_t clusters,
               arma::Row<size_t> &medoid_indicies,
               arma::mat &medoids);

    double cost_fn_build(const arma::mat &data, arma::uword target, arma::uvec &tmp_refs, arma::Row<double> &best_distances);

    arma::vec swap_target(
        const arma::mat &data,
        arma::Row<size_t> &medoid_indices,
        arma::uvec &targets,
        size_t batch_size,
        arma::Row<double> &best_distances,
        arma::Row<double> &second_best_distances,
        arma::urowvec &assignments);

    void swap(const arma::mat &data,
              const size_t clusters,
              arma::Row<size_t> &medoid_indicies,
              arma::mat &medoids);

    double cost_fn(const arma::mat &data, arma::uword target, arma::uvec &tmp_refs, arma::Row<double> &best_distances);

    double calc_loss(const arma::mat &data,
                     const size_t clusters,
                     arma::Row<size_t> &medoid_indicies);

    double sigma = 0.1;
    size_t maxIterations;
};

#endif