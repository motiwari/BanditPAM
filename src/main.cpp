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

#include "kmedoids_algorithm.hpp"
#include "log_helper.hpp"
#include "kmedoids_pywrapper.hpp"

#include <armadillo>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <exception>
#include <regex>
#include <filesystem>

int main(int argc, char* argv[]) {
    std::string input_name;
    std::string log_file_name = "KMedoidsLogfile";
    size_t k;
    int opt;
    int prev_ind;
    size_t verbosity = 0;
    size_t max_iter = 1000;
    size_t buildConfidence = 1000;
    size_t swapConfidence = 10000;
    std::string loss = "2";
    bool f_flag = false;
    bool k_flag = false;
    const int ARGUMENT_ERROR_CODE = 1;

    while (prev_ind = optind, (opt = getopt(argc, argv, "f:l:k:v:s:")) != -1) {

        if ( optind == prev_ind + 2 && *optarg == '-' ) {
        opt = ':';
        --optind;
        }

        switch (opt) {
            // path to the data file to be read in
            case 'f':
                input_name = optarg;
                f_flag = true;
                break;
            // name of the output log file
            case 's':
                log_file_name = optarg;
                break;
            // number of clusters to create
            case 'k':
                k = std::stoi(optarg);
                k_flag = true;
                break;
            // type of loss/distance function to use
            case 'l':
                loss = optarg;
                break;
            // set the verbosity of the algorithm
            case 'v':
                verbosity = std::stoi(optarg);
                break;
            case ':':
                printf("option needs a value\n");
                return ARGUMENT_ERROR_CODE;
            case '?':
                printf("unknown option: %c\n", optopt);
                return ARGUMENT_ERROR_CODE;
        }
    }

    try {
      if (!f_flag) {
        throw std::invalid_argument("error: Must specify input file via -f flag");
      } else if (!k_flag) {
        throw std::invalid_argument("error: Must specify number of clusters via -k flag");
      } else if (!std::filesystem::exists(input_name)) {
        throw std::invalid_argument("error: The file does not exist");
      }
    } catch (std::invalid_argument& e) {
      std::cout << e.what() << std::endl;
      return ARGUMENT_ERROR_CODE;
    }

    arma::mat data;
    data.load(input_name);
    arma::uword n = data.n_cols;
    arma::uword d = data.n_rows;

    km::KMedoids kmed(k, "BanditPAM", verbosity, max_iter, buildConfidence, swapConfidence, log_file_name);
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
