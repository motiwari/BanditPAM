#include "kmedoids_ucb.hpp"
#include <armadillo>
#include <chrono>
#include <fstream>

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        std::cout << "incorrect usage" << std::endl;
        return 1;
    }
    std::string input_name = argv[1];
    std::string output_name = argv[2];
    int k = std::stoi(argv[3]);

    arma::mat data;
    data.load(input_name);
    data = arma::trans(data);

    arma::uword n = data.n_cols;
    arma::uword d = data.n_rows;
    std::cout << "Read in " 
        << n 
        << " data points of dimension " 
        << d
        << std::endl;

    arma::urowvec assignments(n);
    arma::urowvec medoid_indices(k);

    KMediods kmed;
    auto start = std::chrono::steady_clock::now();
    kmed.cluster(data, k, assignments, medoid_indices);
    auto end = std::chrono::steady_clock::now();
    cout << "Took "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count()
         << " milliseconds" << endl;

}