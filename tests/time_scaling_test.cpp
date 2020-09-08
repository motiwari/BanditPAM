#include "gtest/gtest.h"
#include "kmedoids_ucb.hpp"
#include <armadillo>
#include <array>

TEST(BanditPAM, time_scaling_test_MNIST) {
    int k = 5;
    std::string base_data_path = "../../data/MNIST-10k.csv";
    std::string test_data_paths [3] = {
      "../../data/MNIST-20k.csv",
      "../../data/MNIST-40k.csv",
      "../../data/MNIST-70k.csv"
    };
    std::string

    // RUN BASELINE 10K DATA AND CLUSTER
    arma::mat base_data;
    base_data.load(base_data_path);
    base_data = arma::trans(base_data);
    arma::uword n = base_data.n_cols;

    KMedoids base_med(base_data);
    arma::urowvec base_assignments(n);
    arma::urowvec base_indices(k);

    auto base_start = std::chrono::steady_clock::now();
    base_med.cluster(k, base_assignments, base_indices);
    auto base_end = std::chrono::steady_clock::now();
    auto base_diff = std::chrono::duration_cast<std::chrono::milliseconds>(base_end - base_start);

    for (int i=0; i<3; ++i) {
        arma::mat test_data;
        test_data.load(test_data_paths[i]);
        test_data = arma::trans(test_data);
        arma::uword n = test_data.n_cols;

        KMedoids test_med(test_data);
        arma::urowvec test_assignments(n);
        arma::urowvec test_indices(k);

        auto test_start = std::chrono::steady_clock::now();
        test_med.cluster(k, test_assignments, test_indices);
        auto test_end = std::chrono::steady_clock::now();
        auto test_diff = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);
        EXPECT_TRUE(test_diff < 1.2 * base_diff);
    }
}
