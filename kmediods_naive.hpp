/*
 * Things to fix
 * generic matrix type
 * generic metric type
 * error checking for number of clusters >= unique data points
 * should the mediods be index numbers or the mediods themselves?
 */

#ifndef MLPACK_METHODS_KMEANS_KMEANS_HPP
#define MLPACK_METHODS_KMEANS_KMEANS_HPP

#include <iostream>
#include <armadillo>

class KMediods
{
    public:

    KMediods(size_t maxIterations = 1000);

    void cluster(const arma::mat& data,
        const size_t clusters,
        arma::Row<size_t>& assignments);

    private:

    void build(const arma::mat& data,
        arma::mat& mediods,
        const size_t clusters,
        arma::Row<size_t>& assignments);

    void swap(const arma::mat& data,
        arma::mat& mediods,
        const size_t clusters,
        arma::Row<size_t>& assignments);

    size_t maxIterations;
};

#endif