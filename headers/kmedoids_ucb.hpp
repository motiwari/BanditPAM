#ifndef _kmeds_inc_h_
#define _kmeds_inc_h_

#include <omp.h>
#include <armadillo>
#include <fstream>
#include <iostream>
#include <chrono>

class KMedoids
{
public:
  KMedoids(int n_medoids = 5, std::string algorithm = "BanditPAM", int verbosity = 0, int max_iter = 1000, std::string logFilename = "KMedoidsLogfile");

  KMedoids(const KMedoids &kmed);

  ~KMedoids();

  void fit(arma::mat input_data, std::string loss);

  arma::rowvec getMedoidsFinal();

  arma::rowvec getMedoidsBuild();

  arma::rowvec getLabels();

  int getSteps();
private:
  // ###################### Supporting BanditPAM Functions ######################
  void fit_bpam(arma::mat input_data);

  void fit_naive(arma::mat input_data);

  void build_naive(arma::rowvec& medoid_indices);

  void swap_naive(arma::rowvec& medoid_indices);

  void build(
    arma::rowvec& medoid_indices,
    arma::mat& medoids);

  void build_sigma(
    arma::rowvec& best_distances,
    arma::rowvec& sigma,
    arma::uword batch_size,
    bool use_absolute);

  arma::rowvec build_target(
    arma::uvec& target,
    size_t batch_size,
    arma::rowvec& best_distances,
    bool use_absolute);

  void swap(
    arma::rowvec& medoid_indices,
    arma::mat& medoids,
    arma::rowvec& assignments);

  void calc_best_distances_swap(
    arma::rowvec& medoid_indices,
    arma::rowvec& best_distances,
    arma::rowvec& second_distances,
    arma::rowvec& assignments);

  arma::vec swap_target(
    arma::rowvec& medoid_indices,
    arma::uvec& targets,
    size_t batch_size,
    arma::rowvec& best_distances,
    arma::rowvec& second_best_distances,
    arma::rowvec& assignments);

  void swap_sigma(
    arma::mat& sigma,
    size_t batch_size,
    arma::rowvec& best_distances,
    arma::rowvec& second_best_distances,
    arma::rowvec& assignments);

  double calc_loss(arma::rowvec& medoid_indices);

  // ###################### Loss/misc. functions ######################
  double L1(int i, int j) const;

  double L2(int i, int j) const;

  double cos(int i, int j) const;

  double manhattan(int i, int j) const;

  void log(int priority);

  // ###################### Constructor Parameters ######################
  // number of medoids for use in algorithm
  int n_medoids;

  // options: "naive" and "BanditPAM"
  std::string algorithm;

  // maximum number of iterations to run algorithm for
  int max_iter;

  // verbosity of the algorithm. Breakdown is as follows:
  // 0: no logfile
  // 1: logfile built and final medoids printed
  // TODO: see if we want to do additional stuff?
  int verbosity;

  // name of the log file to save
  std::string logFilename;

  // ###################### Class Properties ######################
  // labels of the data to each medoid
  arma::rowvec labels;

  // post-build medoid indices
  arma::rowvec medoid_indices_build;

  // final medoid indices
  arma::rowvec medoid_indices_final;

  // data input from the user
  arma::mat data;

  // loss fucntion
  double (KMedoids::*lossFn)(int i, int j) const;

  // fit function
  void (KMedoids::*fitFn)(arma::mat input_data);

  // logfile that's being written
  std::ofstream logFile;

  // log buffer
  std::stringstream logBuffer;

  // number of steps
  int steps;

  // ###################### Hyperparameters ######################
  // constant that affects the sensitiviy of build confidence bounds
  static const size_t k_buildConfidence = 1000;

  // constant that affects the sensitiviy of swap confidence bounds
  static const size_t k_swapConfidence = 1000;

  // bound for double comparison
  const double k_doubleComparisonLimit = 0.01;

  // batch size for build and swap iterations
  const size_t k_batchSize = 100;
};

#endif
