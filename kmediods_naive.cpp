#include "kmediods_naive.hpp"

KMediods::KMediods(size_t maxIterations)
{
    this->maxIterations = maxIterations;
}

void KMediods::cluster(const arma::mat& data,
        const size_t clusters,
        arma::Row<size_t>& assignments) 
{
    arma::mat centroids(data.n_rows, clusters);

    // build clusters
    // 
    KMediods::build(data, centroids, clusters, assignments);
    KMediods::swap(data, centroids, clusters, assignments);
    KMediods::swap(data, centroids, clusters, assignments);
    KMediods::swap(data, centroids, clusters, assignments);

    // swap thingy

}

void KMediods::build(const arma::mat& data,
    arma::mat& mediods,
    const size_t clusters,
    arma::Row<size_t>& assignments)
{
    // select initial point
    double minDistance = std::numeric_limits<double>::infinity();
    size_t best = 0;

    for (size_t i = 0; i < data.n_cols; i++) {
        double total = 0;
        for (size_t j = 0; j < data.n_cols; j++) {
            total += arma::norm(data.col(i) - data.col(j), 2);
        }
        //std::cout << total << std::endl;
        if (total < minDistance) {
            minDistance = total;
            best = i;
        }
    }
    mediods.col(0) = data.col(best);
    std::cout << best << " and " << minDistance << std::endl;

    for (size_t k = 0; k < clusters; k++) {
        double minDistance = std::numeric_limits<double>::infinity();
        size_t best = 0;
        for (size_t i = 0; i < data.n_cols; i++) { //consider each data point as a potentail mediod
            double total = 0;
            for (size_t j = 0; j < data.n_cols; j++) {
                double cost = arma::norm(data.col(i) - data.col(j), 2); //either add distance between potential mediod and point
                for (size_t mediod = 0; mediod < k; mediod++) { // or closer mediod and point
                    if (arma::norm(mediods.col(mediod) - data.col(j), 2) < cost) {
                        cost = arma::norm(mediods.col(mediod) - data.col(j), 2);
                    }
                }
                total += cost;

            }
            if (total < minDistance) {
                minDistance = total;
                best = i;
            }
        }
        std::cout << best << " and " << minDistance << std::endl;

        mediods.col(k) = data.col(best);
    }

}

void KMediods::swap(const arma::mat& data,
    arma::mat& mediods,
    const size_t clusters,
    arma::Row<size_t>& assignments)
{
    for (size_t k = 0; k < clusters; k++) {
        double minDistance = std::numeric_limits<double>::infinity();
        size_t best = 0;
        for (size_t i = 0; i < data.n_cols; i++) { //consider each data point as a potentail mediod
            double total = 0;
            for (size_t j = 0; j < data.n_cols; j++) {
                double cost = arma::norm(data.col(i) - data.col(j), 2); //either add distance between potential mediod and point
                for (size_t mediod = 0; mediod < clusters; mediod++) { // or closer mediod and point
                    if (mediod == k) {
                        continue;
                    }
                    if (arma::norm(mediods.col(mediod) - data.col(j), 2) < cost) {
                        cost = arma::norm(mediods.col(mediod) - data.col(j), 2);
                    }
                }
                total += cost;

            }
            if (total < minDistance) {
                minDistance = total;
                best = i;
            }
        }
        std::cout << best << " mindistance is now" << minDistance << std::endl;
        if (approx_equal(mediods.col(k), data.col(best), "absdiff", 0.002)) {
            std::cout << "found a stabe mediod" << std::endl;
        }
        mediods.col(k) = data.col(best);
    }
}