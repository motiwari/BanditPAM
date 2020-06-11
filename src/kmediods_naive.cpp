#include "kmediods_naive.hpp"

KMediods::KMediods(size_t maxIterations)
{
    this->maxIterations = maxIterations;
}

void
KMediods::cluster(const arma::mat& data,
                  const size_t clusters,
                  arma::Row<size_t>& assignments)
{
    // arma::mat centroids(data.n_rows, clusters);
    arma::Row<size_t> centroid_indices(clusters);
    // the thought is that having centroids as its own structure would
    // minimize cache misses. But it is also necessary to have the index
    // of the centroids.
    //
    // Should the swap be happening with the updated clusters? yes?

    // build clusters
    KMediods::build(data, clusters, centroid_indices);
    std::cout << centroid_indices << std::endl;

    size_t i = 0;
    bool medoid_change = true;
    while (i < this->maxIterations && medoid_change) {
        auto previous(centroid_indices);
        KMediods::swap(data, clusters, assignments, centroid_indices);
        std::cout << centroid_indices << std::endl;
        medoid_change = arma::any(centroid_indices != previous);
        // medoid_change = true;
        std::cout << "mediod change is " << medoid_change << std::endl;
        i++;
    }
    // swap thingy
}

void
KMediods::build(const arma::mat& data,
                const size_t clusters,
                arma::Row<size_t>& centroid_indices)
{
    for (size_t k = 0; k < clusters; k++) {
        double minDistance = std::numeric_limits<double>::infinity();
        size_t best = 0;
        for (size_t i = 0; i < data.n_cols;
             i++) { // consider each data point as a potentail mediod
            double total = 0;
            for (size_t j = 0; j < data.n_cols; j++) {
                double cost = arma::norm(data.col(i) - data.col(j),
                                         2); // either add distance between
                                             // potential mediod and point
                for (size_t mediod = 0; mediod < k;
                     mediod++) { // or closer mediod and point
                    if (arma::norm(data.col(centroid_indices(mediod)) -
                                     data.col(j),
                                   2) < cost) {
                        cost = arma::norm(
                          data.col(centroid_indices(mediod)) - data.col(j), 2);
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
        centroid_indices(k) = best;
    }
}

void
KMediods::swap(const arma::mat& data,
               const size_t clusters,
               arma::Row<size_t>& assignments,
               arma::Row<size_t>& centroid_indices)
{
    double minDistance = std::numeric_limits<double>::infinity();
    size_t best = 0;
    size_t medoid_to_swap = 0;
    for (size_t k = 0; k < clusters; k++) {

        for (size_t i = 0; i < data.n_cols;
             i++) { // consider each data point as a potential mediod
            double total = 0;
            for (size_t j = 0; j < data.n_cols; j++) {
                double cost = arma::norm(data.col(i) - data.col(j),
                                         2); // either add distance between
                                             // potential mediod and point
                for (size_t mediod = 0; mediod < clusters;
                     mediod++) { // or closer mediod and point
                    if (mediod == k) {
                        continue;
                    }
                    if (arma::norm(data.col(centroid_indices(mediod)) -
                                     data.col(j),
                                   2) < cost) {
                        cost = arma::norm(
                          data.col(centroid_indices(mediod)) - data.col(j), 2);
                    }
                }
                total += cost;
            }
            if (total < minDistance) {
                minDistance = total;
                best = i;
                medoid_to_swap = k;
            }
        }
    }
    std::cout << best << " mindistance is now" << minDistance << std::endl;
    centroid_indices(medoid_to_swap) = best;
}

double
KMediods::calc_loss(const arma::mat& data,
                    const size_t clusters,
                    arma::Row<size_t>& centroid_indices)
{
    double total = 0;

    for (size_t i = 0; i < data.n_cols; i++) {
        double cost = std::numeric_limits<double>::infinity();
        for (size_t mediod = 0; mediod < clusters; mediod++) {
            if (arma::norm(data.col(centroid_indices(mediod)) - data.col(i),
                           2) < cost) {
                cost = arma::norm(
                  data.col(centroid_indices(mediod)) - data.col(i), 2);
            }
        }
        total += cost;
    }
    return total;
}
