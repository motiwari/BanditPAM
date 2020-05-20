#include <armadillo>
#include "kmedoids_ucb.hpp"
using namespace arma;
int main() {
    /*arma::mat data1 = arma::randu< arma::mat >(4,2);
    std::cout << data1 << std::endl;
    std::cout << data1(0) << std::endl;
    std::cout << data1(1) << std::endl;
    std::cout << data1(2) << std::endl;
    std::cout << data1(3) << std::endl;
    std::cout << data1(4) << std::endl;
    std::cout << data1(5) << std::endl;
    arma::uvec indices(3);
    indices(0) = 0;
    indices(1) = 2;
    indices(2) = 4;
    data1.elem(indices).fill(0);
    std::cout << data1 << std::endl;
    std::cout << arma::find(data1) << std::endl;
    data1.elem(arma::find(data1)).fill(10);
    std::cout << data1 << std::endl;*/
    
    
    arma::mat data;
    data.load("test_cases/data1.csv", arma::csv_ascii);
    //data.load("data/mnist.csv", arma::csv_ascii);
    data = arma::trans(data);
    std::cout << "num cols " << data.n_cols << std::endl;
    //std::cout << data.col(3) << std::endl;

    arma::Row<size_t> assignments;
    KMediods kmed;

    kmed.cluster(data, 4, assignments);

    //data.load('data1.csv', csv_ascii);
}