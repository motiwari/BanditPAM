#include "KMedoids_naive.hpp"

KMedoids::KMedoids(size_t maxIterations)
{
    this->maxIterations = maxIterations;
}

void
KMedoids::cluster(const arma::mat& data,
                  const size_t clusters,
                  arma::Row<size_t>& assignments)
{
    arma::Row<size_t> centroidIndices(clusters);
    // build clusters
    KMedoids::build(data, clusters, centroidIndices);
    std::cout << centroidIndices << std::endl;

    size_t i = 0;
    bool medoidChange = true;
    while (i < this->maxIterations && medoidChange) {
        auto previous(centroidIndices);
        KMedoids::swap(data, clusters, assignments, centroidIndices);
        std::cout << centroidIndices << std::endl;
        medoidChange = arma::any(centroidIndices != previous);
        std::cout << "mediod change is " << medoidChange << std::endl;
        i++;
    }
}

void
KMedoids::build(const arma::mat& data,
                const size_t clusters,
                arma::Row<size_t>& centroidIndices)
{
    for (size_t k = 0; k < clusters; k++) {
        double minDistance = std::numeric_limits<double>::infinity();
        size_t best = 0;
        for (size_t i = 0; i < data.n_cols; i++) {
            double total = 0;
            for (size_t j = 0; j < data.n_cols; j++) {
                double cost = arma::norm(data.col(i) - data.col(j), 2);
                for (size_t mediod = 0; mediod < k; mediod++) {
                    double current = arma::norm(
                      data.col(centroidIndices(mediod)) - data.col(j), 2);
                    if (current < cost) {
                        cost = current;
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
        centroidIndices(k) = best;
    }
}

void
KMedoids::swap(const arma::mat& data,
               const size_t clusters,
               arma::Row<size_t>& assignments,
               arma::Row<size_t>& centroidIndices)
{
    double minDistance = std::numeric_limits<double>::infinity();
    size_t best = 0;
    size_t medoid_to_swap = 0;
    for (size_t k = 0; k < clusters; k++) {
        for (size_t i = 0; i < data.n_cols; i++) {
            double total = 0;
            for (size_t j = 0; j < data.n_cols; j++) {
                double cost = arma::norm(data.col(i) - data.col(j), 2);
                for (size_t mediod = 0; mediod < clusters; mediod++) {
                    if (mediod == k) {
                        continue;
                    }
                    double current = arma::norm(
                      data.col(centroidIndices(mediod)) - data.col(j), 2);
                    if (current < cost) {
                        cost = current;
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
    centroidIndices(medoid_to_swap) = best;
}

double
KMedoids::calc_loss(const arma::mat& data,
                    const size_t clusters,
                    arma::Row<size_t>& centroidIndices)
{
    double total = 0;
    for (size_t i = 0; i < data.n_cols; i++) {
        double cost = std::numeric_limits<double>::infinity();
        for (size_t mediod = 0; mediod < clusters; mediod++) {
            double current =
              arma::norm(data.col(centroidIndices(mediod)) - data.col(i), 2);
            if (current < cost) {
                cost = current;
            }
        }
        total += cost;
    }
    return total;
}
