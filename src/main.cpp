/**
 * @file main.cpp
 * @date 2020-06-10
 *
 * Defines a command line program that can be used
 * to run the BanditPAM KMedoids algorithm.
 *
 * Usage (from home repo directory):
 * ./src/build/BanditPAM -f [path/to/input] -k [number of clusters]
 */

#include "kmedoids_ucb.hpp"

#include <armadillo>
#include <chrono>
#include <fstream>
#include <unistd.h>

int main(int argc, char* argv[])
{
    std::string input_name;
    std::string log_file_name = "KMedoidsLogfile";
    int k;
    int opt;
    int verbosity = 0;
    int max_iter = 1000;
    std::string loss = "2";

    while ((opt = getopt(argc, argv, "f:n:l:k:v:")) != -1) {
        switch (opt) {
            // path to the data file to be read in
            case 'f':
                input_name = optarg;
                break;
            // path to log output data
            case 'n':
                log_file_name = optarg;
                break;
            // number of clusters to create
            case 'k':
                k = std::stoi(optarg);
                break;
            // type of loss/distance function to use
            case 'l':
                loss = optarg;
                if (loss.at(0) == 'L') {
                  loss = loss.substr(1);
                }
                break;
            // set the verbosity of the algorithm
            case 'v':
                verbosity = std::stoi(optarg);
                break;
            case ':':
                printf("option needs a value\n");
                return 1;
            case '?':
                printf("unknown option: %c\n", optopt);
                return 1;
        }
    }
    arma::mat data;
    data.load(input_name);
    arma::uword n = data.n_cols;
    arma::uword d = data.n_rows;

    KMedoids kmed(k, "BanditPAM", verbosity, max_iter, log_file_name);
    kmed.fit(data, loss);

    if (verbosity > 0) {
      arma::rowvec meds = kmed.getMedoidsFinal();
      std::cout << "Medoids: ";
      for (size_t i = 0; i < meds.n_cols; i++) {
        if (i == (meds.n_cols - 1)) {
          std::cout << meds(i) << std::endl;
        } else {
          std::cout << meds(i) << ',';
        }
      }
    }
}
