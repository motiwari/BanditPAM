#include <armadillo>
#include "kmedoids_ucb.hpp"
using namespace arma;
int main() {
    // need to update sigma calculations
    // switch loss calculation in swap
    // how often are things resampled?
    // standardize the typedefs
    // choose between rowvec and vec
    
    arma::mat data;
    //data.load("../test_cases/data1.csv", arma::csv_ascii);
    data.load("../data/mnist.csv", arma::csv_ascii);
    data = arma::trans(data);
    std::cout << "num cols " << data.n_cols << std::endl;
    std::cout << data.col(3) << std::endl;

    arma::Row<size_t> assignments;
    KMediods kmed;

    kmed.cluster(data, 5, assignments);

    //data.load('data1.csv', csv_ascii);
}