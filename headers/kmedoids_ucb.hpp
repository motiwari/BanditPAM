#ifndef KMEDOIDS_UCB_H_
#define KMEDOIDS_UCB_H_

#include <omp.h>
#include <armadillo>
#include <fstream>
#include <iostream>
#include <chrono>

struct LogHelper {
    std::ofstream hlogFile;
    std::string filename;
    int k;

    void init(int input_k, std::string input_filename = "HKMedoidsLogfile") {
      k = input_k;
      filename = input_filename;
    }

    void close() {
      hlogFile.close();
      std::cout << "finished closing" << std::endl;
    }

    void writeProfile(arma::rowvec b_medoids, arma::rowvec f_medoids, int steps, double loss) {
      hlogFile << "Built:";
      for (size_t i = 0; i < k; i++) {
        if (i == (k - 1)) {
          hlogFile << b_medoids(i) << '\n';
        }
      }
      hlogFile << "Swapped:";
      for (size_t i = 0; i < k; i++) {
        if (i == (k - 1)) {
          hlogFile << f_medoids(i) << '\n';
        }
      }
      hlogFile << "Num Swaps: " << steps << '\n';
      hlogFile << "Final Loss: " << loss << '\n';
      std::cout << "finished writing" << std::endl;
    }
};

class KMedoids
{
  public:
    KMedoids(int n_medoids = 5, std::string algorithm = "BanditPAM", int verbosity = 0, int max_iter = 1000, std::string logFilename = "KMedoidsLogfile");

    KMedoids(const KMedoids &kmed);

    ~KMedoids();

    void setNMedoids(int k);

    void setLogFilename(std::string l);

    void fit(arma::mat input_data, std::string loss);

    arma::rowvec getMedoidsFinal();

    arma::rowvec getMedoidsBuild();

    arma::rowvec getLabels();

    void setLossFn(std::string loss);

    void checkAlgorithm(std::string algorithm);

    int getSteps();

    // Constructor parameters
    int n_medoids; // TODO (@Mo): Rename this to k, make this private

    std::string logFilename; // TODO: Make this private


  private:
    // The functions below are PAM's constituent functions
    void fit_bpam(arma::mat input_data);

    void fit_naive(arma::mat input_data);

    void build_naive(arma::rowvec& medoid_indices);

    void swap_naive(arma::rowvec& medoid_indices);

    void build(
      arma::rowvec& medoid_indices,
      arma::mat& medoids
    );

    void build_sigma(
      arma::rowvec& best_distances,
      arma::rowvec& sigma,
      arma::uword batch_size,
      bool use_absolute
    );

    arma::rowvec build_target(
      arma::uvec& target,
      size_t batch_size,
      arma::rowvec& best_distances,
      bool use_absolute
    );

    void swap(
      arma::rowvec& medoid_indices,
      arma::mat& medoids,
      arma::rowvec& assignments
    );

    void calc_best_distances_swap(
      arma::rowvec& medoid_indices,
      arma::rowvec& best_distances,
      arma::rowvec& second_distances,
      arma::rowvec& assignments
    );

    arma::vec swap_target(
      arma::rowvec& medoid_indices,
      arma::uvec& targets,
      size_t batch_size,
      arma::rowvec& best_distances,
      arma::rowvec& second_best_distances,
      arma::rowvec& assignments
    );

    void swap_sigma(
      arma::mat& sigma,
      size_t batch_size,
      arma::rowvec& best_distances,
      arma::rowvec& second_best_distances,
      arma::rowvec& assignments
    );

    double calc_loss(arma::rowvec& medoid_indices);

    // Loss functions
    double L1(int i, int j) const;

    double L2(int i, int j) const;

    double cos(int i, int j) const;

    double manhattan(int i, int j) const;

    void log(int priority);

    // Constructor params
    std::string algorithm; // options: "naive" and "BanditPAM"

    int max_iter;

    int verbosity;

    // Properties of the KMedoids instance
    arma::mat data;

    arma::rowvec labels; // assignments of each datapoint to its medoid

    arma::rowvec medoid_indices_build; // Medoids at the end of build step

    arma::rowvec medoid_indices_final;

    double (KMedoids::*lossFn)(int i, int j) const;

    void (KMedoids::*fitFn)(arma::mat input_data); // Function to use (from algorithm)

    LogHelper logHelper;

    std::ofstream logFile;

    std::stringstream logBuffer;

    std::ofstream swapLogfile;

    std::stringstream swapLogBuffer;

    int steps; // number of actual swap iterations taken by the algorithm

    // Hyperparameters
    // constant that affects the sensitivity of build confidence bounds
    static const size_t buildConfidence = 1000;

    // constant that affects the sensitiviy of swap confidence bounds
    static const size_t swapConfidence = 1000;

    // bound for double comparison precision
    const double precision = 0.001;

    const size_t batchSize = 100;
};

#endif // KMEDOIDS_UCB_H_
