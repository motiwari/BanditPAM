#ifndef KMEDOIDS_UCB_H_
#define KMEDOIDS_UCB_H_

#include <omp.h>
#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

struct LogHelper {
    std::ofstream hlogFile;
    std::string filename;
    int k;

    std::vector<double> comp_exact_build;
    std::vector<double> comp_exact_swap;

    std::vector<double> loss_build;
    std::vector<double> loss_swap;

    std::vector<double> p_build;
    std::vector<double> p_swap;

    std::vector<std::string> sigma_build;
    std::vector<std::string> sigma_swap;

    void init(int input_k, std::string input_filename = "HKMedoidsLogfile") {
      k = input_k;
      filename = input_filename;
      hlogFile.open(filename);
    }

    void close() {
      hlogFile.close();
    }

    void writeProfile(arma::rowvec b_medoids, arma::rowvec f_medoids, int steps, double loss) {
      hlogFile << "Built:";
      for (size_t i = 0; i < b_medoids.n_cols; i++) {
        if (i == (k - 1)) {
          hlogFile << b_medoids(i) << '\n';
        } else {
          hlogFile << b_medoids(i) << ',';
        }
      }
      hlogFile << "Swapped:";
      for (size_t i = 0; i < f_medoids.n_cols; i++) {
        if (i == (k - 1)) {
          hlogFile << f_medoids(i) << '\n';
        } else {
          hlogFile << f_medoids(i) << ',';
        }
      }
      hlogFile << "Num Swaps: " << steps << '\n';
      hlogFile << "Final Loss: " << loss << '\n';
      hlogFile << "Build Logstring:" << '\n';
      hlogFile << "\t\tcompute_exactly:\n";
      for (size_t i = 0; i < comp_exact_build.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << comp_exact_build.at(i) << '\n';
      }
      hlogFile << "\t\tloss:\n";
      for (size_t i = 0; i < loss_build.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << loss_build.at(i) << '\n';
      }
      hlogFile << "\t\tp:\n";
      for (size_t i = 0; i < p_build.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << p_build.at(i) << '\n';
      }
      hlogFile << "\t\tsigma:\n";
      for (size_t i = 0; i < sigma_build.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << sigma_build.at(i) << '\n';
      }
      hlogFile << "Swap Logstring:" << '\n';
      hlogFile << "\t\tcompute_exactly:\n";
      for (size_t i = 0; i < comp_exact_swap.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << comp_exact_swap.at(i) << '\n';
      }
      hlogFile << "\t\tloss:\n";
      for (size_t i = 0; i < loss_swap.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << loss_swap.at(i) << '\n';
      }
      hlogFile << "\t\tp:\n";
      for (size_t i = 0; i < p_swap.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << p_swap.at(i) << '\n';
      }
      hlogFile << "\t\tsigma:\n";
      for (size_t i = 0; i < sigma_swap.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << sigma_swap.at(i) << '\n';
      }
    }
};

/**
 *  KMedoids class. Creates a KMedoids object that can be used to find the medoids
 *  for a particular set of input data.
 *
 *  @param nMedoids Number of medoids to identify
 *  @param algorithm Algorithm to use to find medoids; options are "BanditPAM" for
 *  this paper's iplementation, or "naive" to use the naive method
 *  @param verbosity Verbosity of the algorithm, 0 will have no log file emitted, 1 will emit a log file
 *  @param maxIter The maximum number of iterations to run the algorithm for
 *  @param logFilename The name of the output log file
 */
class KMedoids {
  public:
    KMedoids(int nMedoids = 5, std::string algorithm = "BanditPAM", int verbosity = 0, int maxIter = 1000, std::string logFilename = "KMedoidsLogfile");

    KMedoids(const KMedoids &kmed);

    ~KMedoids();

    void fit(arma::mat inputData, std::string loss);

    // The functions below are "get" functions for read-only attributes

    arma::rowvec getMedoidsFinal();

    arma::rowvec getMedoidsBuild();

    arma::rowvec getLabels();

    int getSteps();

    // The functions below are get/set functions for attributes

    int getNMedoids();

    void setNMedoids(int new_num);

    std::string getAlgorithm();

    void setAlgorithm(std::string new_alg);

    int getVerbosity();

    void setVerbosity(int new_ver);

    int getMaxIter();

    void setMaxIter(int new_max);

    std::string getLogfileName();

    void setLogFilename(std::string new_lname);

    void setLossFn(std::string loss);
  private:
    // The functions below are PAM's constituent functions
    void fit_bpam(arma::mat inputData);

    void fit_naive(arma::mat inputData);

    void build_naive(arma::rowvec& medoidIndices);

    void swap_naive(arma::rowvec& medoidIndices);

    void build(
      arma::rowvec& medoidIndices,
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
      arma::rowvec& medoidIndices,
      arma::mat& medoids,
      arma::rowvec& assignments
    );

    void calc_best_distances_swap(
      arma::rowvec& medoidIndices,
      arma::rowvec& best_distances,
      arma::rowvec& second_distances,
      arma::rowvec& assignments
    );

    arma::vec swap_target(
      arma::rowvec& medoidIndices,
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

    double calc_loss(arma::rowvec& medoidIndices);

    // Loss functions
    double L1(int i, int j) const;

    double L2(int i, int j) const;

    double cos(int i, int j) const;

    double manhattan(int i, int j) const;

    void log(int priority);

    void checkAlgorithm(std::string algorithm);

    // Constructor params
    std::string algorithm; // options: "naive" and "BanditPAM"

    int maxIter;

    int verbosity;

    int nMedoids;

    std::string logFilename;

    // Properties of the KMedoids instance
    arma::mat data;

    arma::rowvec labels; // assignments of each datapoint to its medoid

    arma::rowvec medoidIndicesBuild; // Medoids at the end of build step

    arma::rowvec medoidIndicesFinal;

    double (KMedoids::*lossFn)(int i, int j) const;

    void (KMedoids::*fitFn)(arma::mat inputData); // Function to use (from algorithm)

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
