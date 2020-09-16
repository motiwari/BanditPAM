/**
 * @file main.cpp
 * @date 2020-06-10
 *
 * This file defines a command line program that can be used
 * to run the BanditPAM KMedoids algorithim.
 *
 * Usage
 * ./pam [path/to/input] [number of clusters] -a
 */

#include "kmedoids_ucb_updated.hpp"
#include <armadillo>
#include <chrono>
#include <fstream>
#include <unistd.h>

int
main(int argc, char* argv[])
{
    std::string input_name = "./data/MNIST-1k.csv";
    int k;
    bool print_assignments = false;
    int opt;
    int verbosity = 0;
    std::string loss;

    while ((opt = getopt(argc, argv, "f:k:av:")) != -1) {
        switch (opt) {
            case 'a':
                print_assignments = true;
                break;
            case 'f':
                input_name = optarg;
                break;
            case 'k':
                k = std::stoi(optarg);
                break;
            case 'l':
                loss = optarg;
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
    data = arma::trans(data);

    arma::uword n = data.n_cols;
    arma::uword d = data.n_rows;
    if (verbosity >= 1)
    {
    	std::cout << "Read in " << n << " data points of dimension " << d
              << std::endl;
    }

    KMedoids kmed;
    kmed.fit(data);

    if (print_assignments) {
      for (size_t i = 0; i < k; i++) {
        std::cout << kmed.medoids(i) << ", ";
      }
      std::cout << std::endl;
    }
}
