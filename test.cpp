#include <armadillo>
#include "kmediods_naive.hpp"
using namespace arma;
int main() {
    arma::mat data = arma::randu< arma::mat >(2,200);
    std::cout << data.n_cols << std::endl;
    std::cout << data.col(3) << std::endl;

    arma::Row<size_t> assignments;
    KMediods kmed;

    kmed.cluster(data, 3, assignments);

    //data.load('data1.csv', csv_ascii);
}