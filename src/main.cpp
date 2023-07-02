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

#include <unistd.h>
#include <fstream>
#include <exception>
#include <filesystem>

#include "kmedoids_algorithm.hpp"

int main(int argc, char *argv[]) {
  std::string input_name;
  size_t k;
  int opt;
  int prev_ind;
  size_t maxIter = 100;
  size_t buildConfidence = 3;
  size_t swapConfidence = 5;
  std::string loss = "L2";
  bool f_flag = false;
  bool k_flag = false;
  const int ARGUMENT_ERROR_CODE = 1;
  bool useCache = true;
  bool usePerm = true;
  int seed = 0;
  int num_data = 0;
  bool parallelize = false;

  // TODO(@motiwari): Use a variadic function signature for this instead
  while (prev_ind = optind,
          (opt = getopt(argc, argv, "f:l:k:v:s:cpnw:")) != -1) {
    if (optind == prev_ind + 2 && *optarg == '-') {
      opt = ':';
      --optind;
    }

    switch (opt) {
      // path to the data file to be read in
      case 'f':
        input_name = optarg;
        f_flag = true;
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
      case 's':
        seed = std::stoi(optarg);
        break;
      case ':':
        printf("option needs a value\n");
        return ARGUMENT_ERROR_CODE;
      case 'c':
        useCache = true;
        break;
      case 'p':
        usePerm = true;
        break;
      case 'n':
        num_data = std::stoi(optarg);
        break;
      case 'w':
        parallelize = true;
        break;
      case '?':
        printf("unknown option: %c\n", optopt);
        return ARGUMENT_ERROR_CODE;
    }
  }

  try {
    if (!f_flag) {
      throw std::invalid_argument(
              "Error: Must specify input file via -f flag");
    } else if (!k_flag) {
      throw std::invalid_argument(
              "Error: Must specify number of clusters via -k flag");
    } else if (!std::filesystem::exists(input_name)) {
      throw std::invalid_argument(
              "Error: The file does not exist");
    }
  } catch (std::invalid_argument &e) {
    std::cout << e.what() << std::endl;
    return ARGUMENT_ERROR_CODE;
  }

  arma::fmat data;
  data.load(input_name);

  if (num_data < 0) {
    throw std::invalid_argument(
            "Error: num_data passed was less than 0");
  } else if (num_data != 0) {
    data.resize(num_data, data.n_cols);
  }

  km::KMedoids kmed(
          k,
          "BanditPAM",
          maxIter,
          buildConfidence,
          swapConfidence,
          useCache,
          usePerm,
          1000,  // Cache Width
          parallelize,
          seed);
  kmed.fit(data, loss, {});
  for (auto medoid : kmed.getMedoidsFinal()) {
    std::cout << medoid << ",";
  }
  std::cout << "Final loss: " << kmed.getAverageLoss() << "\n";
  std::cout << "Num Swap Steps: " << kmed.getSteps() << "\n";
  std::cout << "Total Swap Milliseconds: " << kmed.getTotalSwapTime() << "\n";
  std::cout << "Average Swap Milliseconds: " << kmed.getTimePerSwap() << "\n";
}
