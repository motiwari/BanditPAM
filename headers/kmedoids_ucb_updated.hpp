#include <omp.h>
#include <armadillo>
#include <fstream>
#include <iostream>
#include <chrono>

class KMediods {
  public:
    // future options potentially:
    //    initialization method
    //    number of times run with diff initialization
    //    pre-compute distances
    //    tolerance of error between consecutive iterations
    //    njobs
    KMediods(size_t clusters = 5, size_t maxIterations = 1000, std::string loss = "L2", int verbosity = 0);

    ~KMediods();

    void fit(
      arma::mat data
    );

    void build(
     const size_t clusters,
     arma::urowvec& medoid_indices,
     arma::mat& medoids);

    void swap(
      const size_t clusters,
      arma::urowvec& medoid_indices,
      arma::mat& medoids,
      arma::urowvec& assignments);

    void set_data(arma::mat input_data);

    void set_assignments(size_t n_cols);

    arma::urowvec get_medoids();
  private:
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

    double cost_fn_build(
      arma::uword target,
      arma::uvec& tmp_refs,
      arma::rowvec& best_distances
    );

    arma::vec swap_target(
      arma::urowvec& medoid_indices,
      arma::uvec& targets,
      size_t batch_size,
      arma::rowvec& best_distances,
      arma::rowvec& second_best_distances,
      arma::urowvec& assignments
    );

    double calc_loss(
      const size_t clusters,
      arma::urowvec& medoid_indices
    );

    void swap_sigma(
      arma::mat& sigma,
      size_t batch_size,
      arma::rowvec& best_distances,
      arma::rowvec& second_best_distances,
      arma::urowvec& assignments
    );

    void calc_best_distances_swap(
      arma::urowvec& medoid_indices,
      arma::rowvec& best_distances,
      arma::rowvec& second_distances,
      arma::urowvec& assignments
    );

    void log(int priority);

    double L1(int i, int j) const;

    double L2(int i, int j) const;

    double manhattan(int i, int j) const;

    double Lp(int i, int j) const;

    double cos(int i, int j) const;

    double (KMediods::*lossFn)(int i, int j) const;

    std::ofstream logFile;

    std::stringstream logBuffer;

    size_t clusters;

    size_t maxIterations;

    std::string loss;

    arma::mat data;

    arma::urowvec assignments_final;

    arma::urowvec medoid_indices_final;

    // constant that affects the sensitiviy of build confidence bounds
    static const size_t k_buildConfidence = 1000;

    // constant that affects the sensitiviy of swap confidence bounds
    static const size_t k_swapConfidence = 1000;

    // bound for double comparison
    const double k_doubleComparisonLimit = 0.01;

    // batch size for build and swap iterations
    const size_t k_batchSize = 100;

    // 0 no log file
    // 1 log build and final medoids
    // 2
    int verbosity = 0;
  // TODO: fit
  // TODO: fit_predict
  // TODO: fit_transform
  // TODO: get_params
  // TODO: predict
  // TODO: score
  // TODO: set_params
  // TODO: transform
};
