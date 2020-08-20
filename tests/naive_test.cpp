#include "gtest/gtest.h"
#include "kmedoids_ucb.hpp"
#include <armadillo>

void harness(int k, std::string filename, arma::urowvec buildIndicies, arma::urowvec finalIndicies) {
    arma::mat data;
    data.load(filename);
    data = arma::trans(data);

    arma::uword n = data.n_cols;
    arma::uword d = data.n_rows;

    arma::urowvec assignments(n);
    arma::urowvec medoid_indices(k);

    KMediods kmed(data);

    arma::mat medoids(data.n_rows, k);
    kmed.build(k, medoid_indices, medoids);
    // check that all build indicies match. order is meaninful.
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(buildIndicies(i), medoid_indices(i));
    }

    kmed.cluster(k, assignments, medoid_indices);
    // check that all final indices match. order is not meaninful
    // so the vectors are sorted.
    medoid_indices = arma::sort(medoid_indices);
    finalIndicies = arma::sort(finalIndicies);
    for (int i = 0; i < k; i++) {
        EXPECT_EQ(finalIndicies(i), medoid_indices(i));
    }
}

TEST(BanditPAM, mnist_small_k_5) {
    int k = 5;
    std::string filename = "../../data/mnist.csv";
    arma::urowvec buildIndicies = {16, 32, 70, 87, 24};
    arma::urowvec finalIndicies = {70, 99, 30, 49, 23};

    harness(k, filename, buildIndicies, finalIndicies);
}

TEST(BanditPAM, mnist_small_k_10) {
    int k = 10;
    std::string filename = "../../data/mnist.csv";
    arma::urowvec buildIndicies = {16, 32, 70, 87, 24, 90, 49, 99, 82, 94};
    arma::urowvec finalIndicies = {16, 63, 70, 25, 31, 90, 49, 99, 82, 94};

    harness(k, filename, buildIndicies, finalIndicies);
}

TEST(BanditPAM, mnist_1k_k_5) {
    int k = 5;
    std::string filename = "../../data/MNIST-1k.csv";
    arma::urowvec buildIndicies = {891, 392, 354, 714, 23};
    arma::urowvec finalIndicies = {714, 694, 765, 507, 737};
    

    harness(k, filename, buildIndicies, finalIndicies);
}
