#include <omp.h>
#include <armadillo>
#include <fstream>
#include <iostream>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <carma/carma.h>

namespace py = pybind11;

class KMedoids
{
  public:
    // TODO: add in precompute distances as an option to initialization
    // TODO: add number of threads to use for computation to the initialization
    KMedoids(size_t n_medoids = 5, std::string algorithm = "BanditPAM", size_t max_iter = 1000, std::string loss = "L2", int verbosity = 0, std::string logFilename = "KMedoidsLogfile");

    ~KMedoids();

    void fit(arma::mat data);

    void fit_python(py::array_t<double> & arr);

    py::array_t<double> getMedoidsPython();

    void build(arma::urowvec& medoid_indices, arma::mat& medoids);

    void swap(arma::urowvec& medoid_indices, arma::mat& medoids, arma::urowvec& assignments);

    // medoids
    arma::urowvec medoids;

    // assignments
    arma::urowvec labels;
  private:
    // arma::urowvec medoids;

    void build_sigma(arma::rowvec& best_distances, arma::rowvec& sigma, arma::uword batch_size, bool use_absolute);

    arma::rowvec build_target(arma::uvec& target, size_t batch_size, arma::rowvec& best_distances, bool use_absolute);

    arma::vec swap_target(arma::urowvec& medoid_indices, arma::uvec& targets, size_t batch_size, arma::rowvec& best_distances, arma::rowvec& second_best_distances, arma::urowvec& assignments);

    double calc_loss(arma::urowvec& medoid_indices);

    void swap_sigma(arma::mat& sigma, size_t batch_size, arma::rowvec& best_distances, arma::rowvec& second_best_distances, arma::urowvec& assignments);

    void calc_best_distances_swap(arma::urowvec& medoid_indices, arma::rowvec& best_distances, arma::rowvec& second_distances, arma::urowvec& assignments);

    void log(int priority);

    double L1(int i, int j) const;

    double L2(int i, int j) const;

    double manhattan(int i, int j) const;

    double cos(int i, int j) const;

    // ############# CONSTRUCTOR PARAMETERS #############
    // number of medoids for use in algorithm
    size_t n_medoids;

    // options: "naive" and "BanditPAM"
    std::string algorithm;

    // maximum number of iterations to run algorithm for
    size_t max_iter;

    // verbosity of the algorithm. Breakdown is as follows:
    // 0: no logfile
    // 1: logfile built and final medoids printed
    // TODO: see if we want to do additional stuff?
    int verbosity;

    // name of the log file to save
    std::string logFilename;
    // ############# END CONSTRUCTOR PARAMETERS #############

    // ############# DEFAULT HYPERPARAMETERS/VARIABLES #############
    // constant that affects the sensitiviy of build confidence bounds
    static const size_t k_buildConfidence = 1000;

    // constant that affects the sensitiviy of swap confidence bounds
    static const size_t k_swapConfidence = 1000;

    // bound for double comparison
    const double k_doubleComparisonLimit = 0.01;

    // batch size for build and swap iterations
    const size_t k_batchSize = 100;

    // log buffer
    std::stringstream logBuffer;

    // log file
    std::ofstream logFile;
    // ############# END DEFAULT HYPERPARAMETERS/VARIABLES #############

    // ############# OBJECT VARIABLES #############
    // data to be fit
    arma::mat data;

    // loss function
    double (KMedoids::*lossFn)(int i, int j) const;
};
