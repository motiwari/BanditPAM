/**
 * @file kmedoids_ucb.cpp
 * @date 2020-06-10
 *
 * This file contains the primary C++ implementation of the BanditPAM code.
 *
 */
#include "kmedoids_algorithm.hpp"

#include <carma.h>
#include <armadillo>
#include <unordered_map>
#include <regex>
//#include <sstream>

/**
 *  \brief Class implementation for running KMedoids methods.
 *
 *  KMedoids class. Creates a KMedoids object that can be used to find the medoids
 *  for a particular set of input data.
 *
 *  @param n_medoids Number of medoids/clusters to create
 *  @param algorithm Algorithm used to find medoids; options are "BanditPAM" for
 *  the "Bandit-PAM" algorithm, or "naive" to use the naive method
 *  @param verbosity Verbosity of the algorithm, 0 will have no log file
 *  emitted, 1 will emit a log file
 *  @param max_iter The maximum number of iterations the algorithm runs for
 *  @param buildConfidence Constant that affects the sensitivity of build confidence bounds
 *  @param swapConfidence Constant that affects the sensitiviy of swap confidence bounds
 *  @param logFilename The name of the output log file
 */
KMedoids::KMedoids(size_t n_medoids, std::string algorithm, size_t verbosity,
                   size_t max_iter, size_t buildConfidence, size_t swapConfidence,
                   std::string logFilename
    ): n_medoids(n_medoids),
       algorithm(algorithm),
       max_iter(max_iter),
       verbosity(verbosity),
       buildConfidence(buildConfidence),
       swapConfidence(swapConfidence),
       logFilename(logFilename) {
  KMedoids::checkAlgorithm(algorithm);
}

/**
 *  \brief Destroys KMedoids object.
 *
 *  Destructor for the KMedoids class.
 */
KMedoids::~KMedoids() {;}

/**
 *  \brief Checks whether algorithm input is valid
 *
 *  Checks whether the user's selected algorithm is a valid option.
 *
 *  @param algorithm Name of the algorithm input by the user.
 */
void KMedoids::checkAlgorithm(const std::string& algorithm) {
  if (algorithm == "BanditPAM") {
    fitFn = &KMedoids::fit_bpam;
  } else if (algorithm == "naive") {
    fitFn = &KMedoids::fit_naive;
  } else {
    throw "unrecognized algorithm";
  }
}

/**
 *  \brief Returns the final medoids
 *
 *  Returns the final medoids at the end of the SWAP step after KMedoids::fit
 *  has been called.
 */
arma::rowvec KMedoids::getMedoidsFinal() {
  return medoid_indices_final;
}

/**
 *  \brief Returns the build medoids
 *
 *  Returns the build medoids at the end of the BUILD step after KMedoids::fit
 *  has been called.
 */
arma::rowvec KMedoids::getMedoidsBuild() {
  return medoid_indices_build;
}

/**
 *  \brief Returns the medoid assignments for each datapoint
 *
 *  Returns the medoid each input datapoint is assigned to after KMedoids::fit
 *  has been called and the final medoids have been identified
 */
arma::rowvec KMedoids::getLabels() {
  return labels;
}

/**
 *  \brief Returns the number of swap steps
 *
 *  Returns the number of SWAP steps completed during the last call to
 *  KMedoids::fit
 */
size_t KMedoids::getSteps() {
  return steps;
}

/**
 *  \brief Sets the loss function
 *
 *  Sets the loss function used during KMedoids::fit
 *
 *  @param loss Loss function to be used e.g. L2
 */
void KMedoids::setLossFn(std::string loss) {
  if (std::regex_match(loss, std::regex("L\\d*"))) {
      loss = loss.substr(1);
  }
  try {
    if (loss == "manhattan") {
        lossFn = &KMedoids::manhattan;
    } else if (loss == "cos") {
        lossFn = &KMedoids::cos;
    } else if (loss == "inf") {
        lossFn = &KMedoids::LINF;
    } else if (std::isdigit(loss.at(0))) {
        lossFn = &KMedoids::LP;
        lp     = atoi(loss.c_str());
    } else {
        throw std::invalid_argument("error: unrecognized loss function");
    }
  } catch (std::invalid_argument& e) {
      std::cout << e.what() << std::endl;
    }
}

/**
 *  \brief Returns the number of medoids
 *
 *  Returns the number of medoids to be identified during KMedoids::fit
 */
size_t KMedoids::getNMedoids() {
  return n_medoids;
}

/**
 *  \brief Sets the number of medoids
 *
 *  Sets the number of medoids to be identified during KMedoids::fit
 */
void KMedoids::setNMedoids(size_t new_num) {
  n_medoids = new_num;
}

/**
 *  \brief Returns the algorithm for KMedoids
 *
 *  Returns the algorithm used for identifying the medoids during KMedoids::fit
 */
std::string KMedoids::getAlgorithm() {
  return algorithm;
}

/**
 *  \brief Sets the algorithm for KMedoids
 *
 *  Sets the algorithm used for identifying the medoids during KMedoids::fit
 *
 *  @param new_alg New algorithm to use
 */
void KMedoids::setAlgorithm(const std::string& new_alg) {
  algorithm = new_alg;
}

/**
 *  \brief Returns the verbosity for KMedoids
 *
 *  Returns the verbosity used during KMedoids::fit, with 0 not creating a
 *  logfile, and >0 creating a detailed logfile.
 */
size_t KMedoids::getVerbosity() {
  return verbosity;
}

/**
 *  \brief Sets the verbosity for KMedoids
 *
 *  Sets the verbosity used during KMedoids::fit, with 0 not creating a
 *  logfile, and >0 creating a detailed logfile.
 *
 *  @param new_ver New verbosity to use
 */
void KMedoids::setVerbosity(size_t new_ver) {
  verbosity = new_ver;
}

/**
 *  \brief Returns the maximum number of iterations for KMedoids
 *
 *  Returns the maximum number of iterations that can be run during
 *  KMedoids::fit
 */
size_t KMedoids::getMaxIter() {
  return max_iter;
}

/**
 *  \brief Sets the maximum number of iterations for KMedoids
 *
 *  Sets the maximum number of iterations that can be run during KMedoids::fit
 *
 *  @param new_max New maximum number of iterations to use
 */
void KMedoids::setMaxIter(size_t new_max) {
  max_iter = new_max;
}

/**
 *  \brief Returns the constant buildConfidence 
 *
 *  Returns the constant that affects the sensitivity of build confidence bounds 
 *  that can be run during KMedoids::fit
 */
size_t KMedoids::getbuildConfidence() {
  return buildConfidence;
}

/**
 *  \brief Sets the constant buildConfidence
 *
 *  Sets the constant that affects the sensitivity of build confidence bounds 
 *  that can be run during KMedoids::fit
 *
 *  @param new_buildConfidence New buildConfidence 
 */
void KMedoids::setbuildConfidence(size_t new_buildConfidence) {
  buildConfidence = new_buildConfidence;
}

/**
 *  \brief Returns the constant swapConfidence 
 *
 *  Returns the constant that affects the sensitivity of swap confidence bounds 
 *  that can be run during KMedoids::fit
 */
size_t KMedoids::getswapConfidence() {
  return swapConfidence;
}

/**
 *  \brief Sets the constant swapConfidence
 *
 *  Sets the constant that affects the sensitivity of swap confidence bounds
 *  that can be run during KMedoids::fit
 *
 *  @param new_swapConfidence New swapConfidence 
 */
void KMedoids::setswapConfidence(size_t new_swapConfidence) {
  swapConfidence = new_swapConfidence;
}

/**
 *  \brief Returns the log filename for KMedoids
 *
 *  Returns the name of the logfile that will be output at the end of
 *  KMedoids::fit if verbosity is >0
 */
std::string KMedoids::getLogfileName() {
  return logFilename;
}

/**
 *  \brief Sets the log filename for KMedoids
 *
 *  Sets the name of the logfile that will be output at the end of
 *  KMedoids::fit if verbosity is >0
 *
 *  @param new_lname New logfile name
 */
void KMedoids::setLogFilename(const std::string& new_lname) {
  logFilename = new_lname;
}

/**
 * \brief Finds medoids for the input data under identified loss function
 *
 * Primary function of the KMedoids class. Identifies medoids for input dataset
 * after both the SWAP and BUILD steps, and outputs logs if verbosity is >0
 *
 * @param input_data Input data to find the medoids of
 * @param loss The loss function used during medoid computation
 */
void KMedoids::fit(const arma::mat& input_data, const std::string& loss) {
  KMedoids::setLossFn(loss);
  KMedoids::checkAlgorithm(algorithm);
  (this->*fitFn)(input_data);
  if (verbosity > 0) {
      logHelper.init(logFilename);
      logHelper.writeProfile(medoid_indices_build, medoid_indices_final, steps,
                                                        logHelper.loss_swap.back());
      logHelper.close();
  }
}

/**
 * \brief Calculates confidence intervals in build step
 *
 * Calculates the confidence intervals about the reward for each arm
 *
 * @param data Transposed input data to find the medoids of
 * @param sigma Dispersion paramater for each datapoint
 * @param batch_size Number of datapoints sampled for updating confidence
 * intervals
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param use_aboslute Determines whether the absolute cost is added to the total
 */
void KMedoids::build_sigma(
  const arma::mat& data,
  arma::rowvec& best_distances,
  arma::rowvec& sigma,
  arma::uword batch_size,
  bool use_absolute)
{
    size_t N = data.n_cols;
    // without replacement, requires updated version of armadillo
    arma::uvec tmp_refs = arma::randperm(N, batch_size);
    arma::vec sample(batch_size);
// for each possible swap
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        // gather a sample of points
        for (size_t j = 0; j < batch_size; j++) {
            double cost = (this->*lossFn)(data, i, tmp_refs(j));
            if (use_absolute) {
                sample(j) = cost;
            } else {
                sample(j) = cost < best_distances(tmp_refs(j))
                              ? cost
                              : best_distances(tmp_refs(j));
                sample(j) -= best_distances(tmp_refs(j));
            }
        }
        sigma(i) = arma::stddev(sample);
    }
    arma::rowvec P = {0.25, 0.5, 0.75};
    arma::rowvec Q = arma::quantile(sigma, P);
    std::ostringstream sigma_out;
    sigma_out << "min: " << arma::min(sigma)
              << ", 25th: " << Q(0)
              << ", median: " << Q(1)
              << ", 75th: " << Q(2)
              << ", max: " << arma::max(sigma)
              << ", mean: " << arma::mean(sigma);
    logHelper.sigma_build.push_back(sigma_out.str());
}

/**
 * \brief Calculates distances in swap step
 *
 * Calculates the best and second best distances for each datapoint to one of
 * the medoids in the current medoid set.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Array of medoid indices corresponding to dataset entries
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param second_best_distances Array of second smallest distances from each
 * point to previous set of medoids
 * @param assignments Assignments of datapoints to their closest medoid
 */
void KMedoids::calc_best_distances_swap(
  const arma::mat& data,
  arma::rowvec& medoid_indices,
  arma::rowvec& best_distances,
  arma::rowvec& second_distances,
  arma::rowvec& assignments)
{
#pragma omp parallel for
    for (size_t i = 0; i < data.n_cols; i++) {
        double best = std::numeric_limits<double>::infinity();
        double second = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < medoid_indices.n_cols; k++) {
            double cost = (this->*lossFn)(data, medoid_indices(k), i);
            if (cost < best) {
                assignments(i) = k;
                second = best;
                best = cost;
            } else if (cost < second) {
                second = cost;
            }
        }
        best_distances(i) = best;
        second_distances(i) = second;
    }
}

/**
 * \brief Calculates confidence intervals in swap step
 *
 * Calculates the confidence intervals about the reward for each arm
 *
 * @param data Transposed input data to find the medoids of
 * @param sigma Dispersion paramater for each datapoint
 * @param batch_size Number of datapoints sampled for updating confidence
 * intervals
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param second_best_distances Array of second smallest distances from each
 * point to previous set of medoids
 * @param assignments Assignments of datapoints to their closest medoid
 */
void KMedoids::swap_sigma(
  const arma::mat& data,
  arma::mat& sigma,
  size_t batch_size,
  arma::rowvec& best_distances,
  arma::rowvec& second_best_distances,
  arma::rowvec& assignments)
{
    size_t N = data.n_cols;
    size_t K = sigma.n_rows;
    arma::uvec tmp_refs = arma::randperm(N,
                                   batch_size); // without replacement, requires
                                                // updated version of armadillo

    arma::vec sample(batch_size);
// for each considered swap
#pragma omp parallel for
    for (size_t i = 0; i < K * N; i++) {
        // extract data point of swap
        size_t n = i / K;
        size_t k = i % K;

        // calculate change in loss for some subset of the data
        for (size_t j = 0; j < batch_size; j++) {
            double cost = (this->*lossFn)(data, n,tmp_refs(j));

            if (k == assignments(tmp_refs(j))) {
                if (cost < second_best_distances(tmp_refs(j))) {
                    sample(j) = cost;
                } else {
                    sample(j) = second_best_distances(tmp_refs(j));
                }
            } else {
                if (cost < best_distances(tmp_refs(j))) {
                    sample(j) = cost;
                } else {
                    sample(j) = best_distances(tmp_refs(j));
                }
            }
            sample(j) -= best_distances(tmp_refs(j));
        }
        sigma(k, n) = arma::stddev(sample);
    }
}

/**
* \brief Write the sigma distribution into logfile
*
* Calculates the statistical measures of the sigma distribution
* and writes the results to the log file. 
*
* @param sigma Dispersion paramater for each datapoint
*/
void KMedoids::sigma_log(arma::mat& sigma) {
  arma::rowvec flat_sigma = sigma.as_row(); 
  arma::rowvec P = {0.25, 0.5, 0.75};
  arma::rowvec Q = arma::quantile(flat_sigma, P);
  std::ostringstream sigma_out;
  sigma_out << "min: " << arma::min(flat_sigma)
            << ", 25th: " << Q(0)
            << ", median: " << Q(1)
            << ", 75th: " << Q(2)
            << ", max: " << arma::max(flat_sigma)
            << ", mean: " << arma::mean(flat_sigma);
  logHelper.sigma_swap.push_back(sigma_out.str());
}

/**
 * \brief Calculate loss for medoids
 *
 * Calculates the loss under the previously identified loss function of the
 * medoid indices.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Indices of the medoids in the dataset.
 */
double KMedoids::calc_loss(
  const arma::mat& data,
  arma::rowvec& medoid_indices)
{
    double total = 0;

    for (size_t i = 0; i < data.n_cols; i++) {
        double cost = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < n_medoids; k++) {
            double currCost = (this->*lossFn)(data, medoid_indices(k), i);
            if (currCost < cost) {
                cost = currCost;
            }
        }
        total += cost;
    }
    return total;
}

// Loss and miscellaneous functions

/**
 * \brief LP loss
 *
 * Calculates the LP loss between the datapoints at index i and j of the dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double KMedoids::LP(const arma::mat& data, size_t i, size_t j) const {
    return arma::norm(data.col(i) - data.col(j), lp);
}


/**
 * \brief cos loss
 *
 * Calculates the cosine loss between the datapoints at index i and j of the
 * dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double KMedoids::cos(const arma::mat& data, size_t i, size_t j) const {
    return arma::dot(data.col(i), data.col(j)) / (arma::norm(data.col(i))
                                                    * arma::norm(data.col(j)));
}

/**
 * \brief Manhattan loss
 *
 * Calculates the Manhattan loss between the datapoints at index i and j of the
 * dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double KMedoids::manhattan(const arma::mat& data, size_t i, size_t j) const {
    return arma::accu(arma::abs(data.col(i) - data.col(j)));
}

/**
 * \brief L_INFINITY loss
 *
 * Calculates the Manhattan loss between the datapoints at index i and j of the
 * dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double KMedoids::LINF(const arma::mat& data, size_t i, size_t j) const {
    return arma::max(arma::abs(data.col(i) - data.col(j)));
}
