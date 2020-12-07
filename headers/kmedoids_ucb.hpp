#ifndef KMEDOIDS_UCB_H_
#define KMEDOIDS_UCB_H_

#include <armadillo>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

/**
 *  \brief Logging class for structured KMedoids logs.
 *
 *  LogHelper class. Assists the KMedoids class in structured logging.
 */
struct LogHelper {
    std::ofstream hlogFile; ///< Output stream that writes the KMedoids log

    std::vector<double> comp_exact_build; ///< Number of computations in build step
    std::vector<double> comp_exact_swap; ///< Number of computations in swap step

    std::vector<double> loss_build; ///< Loss after each iteration of build step
    std::vector<double> loss_swap; ///< Loss after each iteration of swap step

    std::vector<double> p_build; ///< Precision for each iteration of build step
    std::vector<double> p_swap; ///< Precision for each iteration of swap step

    std::vector<std::string> sigma_build; ///< Distributions for each iteration of build step
    std::vector<std::string> sigma_swap; ///< Distributions for each iteration of swap step

    /*! \brief Opens the log file.
     *
     *  Opens the log file.
     *
     *  @param input_filename Filename that log will be saved as.
     */
    void init(std::string input_filename = "KMedoidsLogfile") {
      hlogFile.open(input_filename);
    }

    /*! \brief Closes the log file.
     *
     *  Closes the log file.
     */
    void close() {
      hlogFile.close();
    }

    /*! \brief Writes a vector out for a given key
     *
     *  Writes a vector out for a given key
     *
     *  @param key Key for json-ified output structure
     *  @param vec Vector to be iterated across when writing line
     */
    void writeSummaryLine(std::string key, arma::rowvec vec) {
      hlogFile << key << ':';
      for (size_t i = 0; i < vec.n_cols; i++) {
        if (i == (vec.n_cols - 1)) {
          hlogFile << vec(i) << '\n';
        } else {
          hlogFile << vec(i) << ',';
        }
      }
    }

    /*! \brief Writes a logstring component
     *
     *  Writes a logstring line for the writeProfile function.
     *
     *  @param key Key for json-ified output structure
     *  @param vec Vector to be iterated across when writing logstring line
     */
    template <typename T>
    void writeLogStringLine(std::string key, std::vector<T> vec) {
      hlogFile << "\t\t:" << key << '\n';
      for (size_t i = 0; i < vec.size(); i++) {
        hlogFile << "\t\t\t\t" << i << ": " << vec.at(i) << '\n';
      }
    }

    /*! \brief Writes formatted summary log of a KMedoids run
     *
     *  Writes summary statistics of a KMedoids run. Statistics include medoids
     *  after the build step, medoids after the swap step, number of swap steps,
     *  the final loss, and logstrings of the number of points that had distance
     *  computations, loss, precision, and uncertainty for each iteration of
     *  both the build and swap steps.
     *
     *  @param b_medoids Medoids after the build step.
     *  @param f_medoids Medoids after the swap step (final medoids).
     *  @param steps Number of swap steps.
     *  @param loss Final loss of the KMedoids object.
     */
    void writeProfile(arma::rowvec b_medoids, arma::rowvec f_medoids, int steps, double loss) {
      writeSummaryLine("Built", b_medoids);
      writeSummaryLine("Swapped", f_medoids);
      hlogFile << "Num Swaps: " << steps << '\n';
      hlogFile << "Final Loss: " << loss << '\n';

      hlogFile << "Build Logstring:" << '\n';
      writeLogStringLine("compute_exactly", comp_exact_build);
      writeLogStringLine("loss", loss_build);
      writeLogStringLine("p", p_build);
      writeLogStringLine("sigma", sigma_build);

      hlogFile << "Swap Logstring:" << '\n';
      writeLogStringLine("compute_exactly", comp_exact_swap);
      writeLogStringLine("loss", loss_swap);
      writeLogStringLine("p", p_swap);
      writeLogStringLine("sigma", sigma_swap);
    }
};

/**
 *  \brief Class implementation for running KMedoids methods.
 *
 *  KMedoids class. Creates a KMedoids object that can be used to find the medoids
 *  for a particular set of input data.
 *
 *  @param nMedoids Number of medoids/clusters to create
 *  @param algorithm Algorithm used to find medoids; options are "BanditPAM" for
 *  the "Bandit-PAM" algorithm, or "naive" to use the naive method
 *  @param verbosity Verbosity of the algorithm, 0 will have no log file
 *  emitted, 1 will emit a log file
 *  @param maxIter The maximum number of iterations the algorithm runs for
 *  @param logFilename The name of the output log file
 */
class KMedoids {
  public:
    KMedoids(int n_medoids = 5, std::string algorithm = "BanditPAM", int verbosity = 0, int max_iter = 1000, std::string logFilename = "KMedoidsLogfile");

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

    void checkAlgorithm(std::string algorithm);

    // Constructor params
    std::string algorithm; ///< options: "naive" and "BanditPAM"

    int max_iter; ///< maximum number of iterations during KMedoids::fit

    int verbosity; ///< determines whether KMedoids::fit outputs a logfile

    int n_medoids; ///< number of medoids identified for a given dataset

    std::string logFilename; ///< name of the logfile output (verbosity permitting)

    // Properties of the KMedoids instance
    arma::mat data; ///< input data used during KMedoids::fit

    arma::rowvec labels; ///< assignments of each datapoint to its medoid

    arma::rowvec medoid_indices_build; ///< medoids at the end of build step

    arma::rowvec medoid_indices_final; ///< medoids at the end of the swap step

    double (KMedoids::*lossFn)(int i, int j) const; ///< loss function used during KMedoids::fit

    void (KMedoids::*fitFn)(arma::mat inputData); ///< function used for finding medoids (from algorithm)

    LogHelper logHelper; ///< helper object for making formatted logs

    int steps; ///< number of actual swap iterations taken by the algorithm

    // Hyperparameters
    static const size_t buildConfidence = 1000; ///< constant that affects the sensitivity of build confidence bounds

    static const size_t swapConfidence = 1000; ///< constant that affects the sensitiviy of swap confidence bounds

    const double precision = 0.001; ///< bound for double comparison precision

    const size_t batchSize = 100; ///< batch size for computation steps
};

#endif // KMEDOIDS_UCB_H_
