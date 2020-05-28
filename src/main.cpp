#include <armadillo>
#include "kmedoids_ucb.hpp"
using namespace arma;
int main(int argc, char * argv[]) {

    // how often are things resampled?
    // standardize the typedefs
    // choose between rowvec and vec
    
    std::string input_name = argv[1];
    std::string output_name = argv[2];
    int k = std::stoi(argv[3]);

    arma::mat data;
    //data.load("../test_cases/data1.csv", arma::csv_ascii);
    //data.load("../data/mnist.csv", arma::csv_ascii);
    data.load(input_name);
    data = arma::trans(data);
    arma::uword n = data.n_cols;
    arma::uword d = data.n_rows;
    std::cout << "Read in " << n << " data points of dimension " << d << std::endl;

    arma::urowvec assignments(n);
    arma::urowvec medoid_indices(k);
    
    KMediods kmed;
    kmed.cluster(data, k, assignments);

    //data.load('data1.csv', csv_ascii);
}