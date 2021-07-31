/**
 * @file kmedoids_ucb.cpp
 * @date 2020-06-10
 *
 * This file contains the primary C++ implementation of the BanditPAM code.
 *
 */
#include "kmedoids_ucb.hpp"

#include <carma.h>
#include <armadillo>
#include <unordered_map>
#include <regex>
//#include <sstream>
#include <signal.h>

/**
 *  \brief Kills the process.
 * 
 *  Terminates the process if ctrl+c signal is received
 * 
 */
 
void signal_callback_handler(int signum) {
   std::cout << "Caught signal " << signum << std::endl;
   exit(signum);
}

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
 *  @param logFilename The name of the output log file
 */
KMedoids::KMedoids(int n_medoids, std::string algorithm, int verbosity,
                                          int max_iter, std::string logFilename
    ): n_medoids(n_medoids),
       algorithm(algorithm),
       max_iter(max_iter),
       verbosity(verbosity),
       logFilename(logFilename) {
  KMedoids::checkAlgorithm(algorithm);
  signal(SIGINT, signal_callback_handler);
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
void KMedoids::checkAlgorithm(std::string algorithm) {
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
int KMedoids::getSteps() {
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
int KMedoids::getNMedoids() {
  return n_medoids;
}

/**
 *  \brief Sets the number of medoids
 *
 *  Sets the number of medoids to be identified during KMedoids::fit
 */
void KMedoids::setNMedoids(int new_num) {
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
void KMedoids::setAlgorithm(std::string new_alg) {
  algorithm = new_alg;
}

/**
 *  \brief Returns the verbosity for KMedoids
 *
 *  Returns the verbosity used during KMedoids::fit, with 0 not creating a
 *  logfile, and >0 creating a detailed logfile.
 */
int KMedoids::getVerbosity() {
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
void KMedoids::setVerbosity(int new_ver) {
  verbosity = new_ver;
}

/**
 *  \brief Returns the maximum number of iterations for KMedoids
 *
 *  Returns the maximum number of iterations that can be run during
 *  KMedoids::fit
 */
int KMedoids::getMaxIter() {
  return max_iter;
}

/**
 *  \brief Sets the maximum number of iterations for KMedoids
 *
 *  Sets the maximum number of iterations that can be run during KMedoids::fit
 *
 *  @param new_max New maximum number of iterations to use
 */
void KMedoids::setMaxIter(int new_max) {
  max_iter = new_max;
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
void KMedoids::setLogFilename(std::string new_lname) {
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
void KMedoids::fit(arma::mat input_data, std::string loss) {
  KMedoids::setLossFn(loss);
  (this->*fitFn)(input_data);
  if (verbosity > 0) {
      logHelper.init(logFilename);
      logHelper.writeProfile(medoid_indices_build, medoid_indices_final, steps,
                                                        logHelper.loss_swap.back());
      logHelper.close();
  }
}


/**
 * \brief Runs naive PAM algorithm.
 *
 * Run the naive PAM algorithm to identify a dataset's medoids.
 *
 * @param input_data Input data to find the medoids of
 */
void KMedoids::fit_naive(arma::mat input_data) {
  data = input_data;
  data = arma::trans(data);
  arma::rowvec medoid_indices(n_medoids);
  // runs build step
  KMedoids::build_naive(data, medoid_indices);
  steps = 0;

  medoid_indices_build = medoid_indices;
  arma::rowvec assignments(data.n_cols);
  size_t i = 0;
  bool medoidChange = true;
  while (i < max_iter && medoidChange) {
    auto previous(medoid_indices);
    // runs swa step as necessary
    KMedoids::swap_naive(data, medoid_indices, assignments);
    medoidChange = arma::any(medoid_indices != previous);
    i++;
  }
  medoid_indices_final = medoid_indices;
  labels = assignments;
  steps = i;
}

/**
 * \brief Build step for the naive algorithm
 *
 * Runs build step for the naive PAM algorithm. Loops over all datapoint and
 * checks its distance from every other datapoint in the dataset, then checks if
 * the total cost is less than that of the medoid (if a medoid exists yet).
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Uninitialized array of medoids that is modified in place
 * as medoids are identified
 */
void KMedoids::build_naive(
  arma::mat& data, 
  arma::rowvec& medoid_indices)
{ 
  size_t N = data.n_cols;
  int p = (buildConfidence * N); // reciprocal
  bool use_absolute = true;
  arma::rowvec estimates(N, arma::fill::zeros);
  arma::rowvec best_distances(N);
  best_distances.fill(std::numeric_limits<double>::infinity());
  arma::rowvec sigma(N); // standard deviation of induced losses on reference points
  for (size_t k = 0; k < n_medoids; k++) {
    double minDistance = std::numeric_limits<double>::infinity();
    int best = 0;
    KMedoids::build_sigma(
           data, best_distances, sigma, batchSize, use_absolute); // computes std dev amongst batch of reference points
    // fixes a base datapoint
    for (int i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // computes distance between base and all other points
        double cost = (this->*lossFn)(data, i, j);
        for (size_t medoid = 0; medoid < k; medoid++) {
          double current = (this->*lossFn)(data, medoid_indices(medoid), j);
          // compares this for cost of the medoid
          if (current < cost) {
            cost = current;
          }
        }
        total += cost;
      }
      if (total < minDistance) {
        minDistance = total;
        best = i;
      }
    }
    // updates the medoid index for that of lowest cost.
    medoid_indices(k) = best;

    // don't need to do this on final iteration
    for (size_t l = 0; l < N; l++) {
        double cost = (this->*lossFn)(data, l, medoid_indices(k));
        if (cost < best_distances(l)) {
            best_distances(l) = cost;
        }
    }
    use_absolute = false; // use difference of loss for sigma and sampling,
                          // not absolute
    logHelper.loss_build.push_back(minDistance/N);
    logHelper.p_build.push_back((float)1/(float)p);
    logHelper.comp_exact_build.push_back(N);
  }
}

/**
 * \brief Swap step for the naive algorithm
 *
 * Runs build step for the naive PAM algorithm. Loops over all datapoint and
 * checks its distance from every other datapoint in the dataset, then checks if
 * the total cost is less than that of the medoid.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Array of medoid indices created from the build step
 * that is modified in place as better medoids are identified
 * @param assignments Uninitialized array of indices corresponding to each
 * datapoint assigned the index of the medoid it is closest to
 */
void KMedoids::swap_naive(
  arma::mat& data, 
  arma::rowvec& medoid_indices,
  arma::rowvec& assignments)
{
  double minDistance = std::numeric_limits<double>::infinity();
  size_t best = 0;
  size_t medoid_to_swap = 0;
  size_t N = data.n_cols;
  int p = (N * n_medoids * swapConfidence); // reciprocal
  arma::mat sigma(n_medoids, N, arma::fill::zeros);
  arma::rowvec best_distances(N);
  arma::rowvec second_distances(N);

  // calculate quantities needed for swap, best_distances and sigma
  calc_best_distances_swap(
    data, medoid_indices, best_distances, second_distances, assignments);

  swap_sigma(data,
              sigma,
              batchSize,
              best_distances,
              second_distances,
              assignments);
  
  // write the sigma distribution to logfile
  sigma_log(sigma);

  // iterate across the current medoids
  for (size_t k = 0; k < n_medoids; k++) {
    // for every point in our dataset, let it serve as a "base" point
    for (size_t i = 0; i < data.n_cols; i++) {
      double total = 0;
      for (size_t j = 0; j < data.n_cols; j++) {
        // compute distance between base point and every other datapoint
        double cost = (this->*lossFn)(data, i, j);
        for (size_t medoid = 0; medoid < n_medoids; medoid++) {
          if (medoid == k) {
            continue;
          }
          double current = (this->*lossFn)(data, medoid_indices(medoid), j);
          if (current < cost) {
            cost = current;
          }
        }
        total += cost;
      }
      // if total distance for new base point is better than that of the medoid,
      // update the best index identified so far
      if (total < minDistance) {
        minDistance = total;
        best = i;
        medoid_to_swap = k;
      }
    }
  }
  medoid_indices(medoid_to_swap) = best;
  logHelper.loss_swap.push_back(minDistance/N);
  logHelper.p_swap.push_back((float)1/(float)p);
  logHelper.comp_exact_swap.push_back(N*n_medoids);
}

/**
 * \brief Runs BanditPAM algorithm.
 *
 * Run the BanditPAM algorithm to identify a dataset's medoids.
 *
 * @param input_data Input data to find the medoids of
 */
void KMedoids::fit_bpam(arma::mat input_data) {
  data = input_data;
  data = arma::trans(data);
  arma::mat medoids_mat(data.n_rows, n_medoids);
  arma::rowvec medoid_indices(n_medoids);
  // runs build step
  KMedoids::build(data, medoid_indices, medoids_mat);
  steps = 0;

  medoid_indices_build = medoid_indices;
  arma::rowvec assignments(data.n_cols);
  // runs swap step
  KMedoids::swap(data, medoid_indices, medoids_mat, assignments);
  medoid_indices_final = medoid_indices;
  labels = assignments;
}

/**
 * \brief Build step for BanditPAM
 *
 * Runs build step for the BanditPAM algorithm. Draws batch sizes with replacement
 * from reference set, and uses the estimated reward of the potential medoid
 * solutions on the reference set to update the reward confidence intervals and
 * accordingly narrow the solution set.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Uninitialized array of medoids that is modified in place
 * as medoids are identified
 * @param medoids Matrix of possible medoids that is updated as the bandit
 * learns which datapoints will be unlikely to be good candidates
 */
void KMedoids::build(
  arma::mat& data,
  arma::rowvec& medoid_indices,
  arma::mat& medoids)
{
    // Parameters
    size_t N = data.n_cols;
    arma::rowvec N_mat(N);
    N_mat.fill(N);
    int p = (buildConfidence * N); // reciprocal of
    bool use_absolute = true;
    arma::rowvec estimates(N, arma::fill::zeros);
    arma::rowvec best_distances(N);
    best_distances.fill(std::numeric_limits<double>::infinity());
    arma::rowvec sigma(N); // standard deviation of induced losses on reference points
    arma::urowvec candidates(
      N,
      arma::fill::ones); // one hot encoding of candidates -- points not filtered out yet
    arma::rowvec lcbs(N);
    arma::rowvec ucbs(N);
    arma::rowvec T_samples(N, arma::fill::zeros); // number of times calculating induced loss for reference point
    arma::rowvec exact_mask(N, arma::fill::zeros); // computed the loss exactly for this datapoint

    for (size_t k = 0; k < n_medoids; k++) {
        // instantiate medoids one-by-online
        size_t step_count = 0;
        candidates.fill(1);
        T_samples.fill(0);
        exact_mask.fill(0);
        estimates.fill(0);
        KMedoids::build_sigma(
           data, best_distances, sigma, batchSize, use_absolute); // computes std dev amongst batch of reference points

        while (arma::sum(candidates) > precision) { // while some candidates exist
            arma::umat compute_exactly =
              ((T_samples + batchSize) >= N_mat) != exact_mask;
            if (arma::accu(compute_exactly) > 0) {
                arma::uvec targets = find(compute_exactly);
                logHelper.comp_exact_build.push_back(targets.n_rows);
                arma::rowvec result =
                  build_target(data, targets, N, best_distances, use_absolute); // induced loss for these targets over all reference points
                estimates.cols(targets) = result;
                ucbs.cols(targets) = result;
                lcbs.cols(targets) = result;
                exact_mask.cols(targets).fill(1);
                T_samples.cols(targets) += N;
                candidates.cols(targets).fill(0);
            }
            if (arma::sum(candidates) < precision) {
                break;
            }
            arma::uvec targets = arma::find(candidates);
            arma::rowvec result = build_target(
              data, targets, batchSize, best_distances, use_absolute); // induced loss for the targets (sample)
            estimates.cols(targets) =
              ((T_samples.cols(targets) % estimates.cols(targets)) +
               (result * batchSize)) /
              (batchSize + T_samples.cols(targets)); // update the running average
            T_samples.cols(targets) += batchSize;
            arma::rowvec adjust(targets.n_rows);
            adjust.fill(p);
            adjust = arma::log(adjust);
            arma::rowvec cb_delta =
              sigma.cols(targets) %
              arma::sqrt(adjust / T_samples.cols(targets));
            ucbs.cols(targets) = estimates.cols(targets) + cb_delta;
            lcbs.cols(targets) = estimates.cols(targets) - cb_delta;
            candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            step_count++;
        }

        medoid_indices.at(k) = lcbs.index_min();
        medoids.unsafe_col(k) = data.unsafe_col(medoid_indices(k));

        // don't need to do this on final iteration
        for (size_t i = 0; i < N; i++) {
            double cost = (this->*lossFn)(data, i, medoid_indices(k));
            if (cost < best_distances(i)) {
                best_distances(i) = cost;
            }
        }
        use_absolute = false; // use difference of loss for sigma and sampling,
                              // not absolute
        logHelper.loss_build.push_back(arma::mean(arma::mean(best_distances)));
        logHelper.p_build.push_back((float)1/(float)p);
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
  arma::mat& data,
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
 * \brief Estimates the mean reward for each arm in build step
 *
 * Estimates the mean reward (or loss) for each arm in the identified targets
 * in the build step and returns a list of the estimated reward.
 *
 * @param data Transposed input data to find the medoids of
 * @param target Set of target datapoints to be estimated
 * @param batch_size Number of datapoints sampled for updating confidence
 * intervals
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param use_absolute Determines whether the absolute cost is added to the total
 */
arma::rowvec KMedoids::build_target(
  arma::mat& data,
  arma::uvec& target,
  size_t batch_size,
  arma::rowvec& best_distances,
  bool use_absolute)
{
    size_t N = data.n_cols;
    arma::rowvec estimates(target.n_rows, arma::fill::zeros);
    arma::uvec tmp_refs = arma::randperm(N,
                                   batch_size); // without replacement, requires
                                                // updated version of armadillo
#pragma omp parallel for
    for (size_t i = 0; i < target.n_rows; i++) {
        double total = 0;
        for (size_t j = 0; j < tmp_refs.n_rows; j++) {
            double cost =
              (this->*lossFn)(data, tmp_refs(j), target(i));
            if (use_absolute) {
                total += cost;
            } else {
                total += cost < best_distances(tmp_refs(j))
                           ? cost
                           : best_distances(tmp_refs(j));
                total -= best_distances(tmp_refs(j));
            }
        }
        estimates(i) = total / batch_size;
    }
    return estimates;
}

/**
 * \brief Swap step for BanditPAM
 *
 * Runs Swap step for the BanditPAM algorithm. Draws batch sizes with replacement
 * from reference set, and uses the estimated reward of the potential medoid
 * solutions on the reference set to update the reward confidence intervals and
 * accordingly narrow the solution set.
 *
 * @param data Transposed input data to find the medoids of
 * @param medoid_indices Array of medoid indices created from the build step
 * that is modified in place as better medoids are identified
 * @param medoids Matrix of possible medoids that is updated as the bandit
 * learns which datapoints will be unlikely to be good candidates
 * @param assignments Uninitialized array of indices corresponding to each
 * datapoint assigned the index of the medoid it is closest to
 */
void KMedoids::swap(
  arma::mat& data,
  arma::rowvec& medoid_indices,
  arma::mat& medoids,
  arma::rowvec& assignments)
{
    size_t N = data.n_cols;
    int p = (N * n_medoids * swapConfidence); // reciprocal

    arma::mat sigma(n_medoids, N, arma::fill::zeros);

    arma::rowvec best_distances(N);
    arma::rowvec second_distances(N);
    size_t iter = 0;
    bool swap_performed = true;
    arma::umat candidates(n_medoids, N, arma::fill::ones);
    arma::umat exact_mask(n_medoids, N, arma::fill::zeros);
    arma::mat estimates(n_medoids, N, arma::fill::zeros);
    arma::mat lcbs(n_medoids, N);
    arma::mat ucbs(n_medoids, N);
    arma::umat T_samples(n_medoids, N, arma::fill::zeros);

    // continue making swaps while loss is decreasing
    while (swap_performed && iter < max_iter) {
        iter++;

        // calculate quantities needed for swap, best_distances and sigma
        calc_best_distances_swap(
          data, medoid_indices, best_distances, second_distances, assignments);

        swap_sigma(data,
                   sigma,
                   batchSize,
                   best_distances,
                   second_distances,
                   assignments);

        candidates.fill(1);
        exact_mask.fill(0);
        estimates.fill(0);
        T_samples.fill(0);

        // while there is at least one candidate (double comparison issues)
        while (arma::accu(candidates) > 0.5) {
            calc_best_distances_swap(
              data, medoid_indices, best_distances, second_distances, assignments);

            // compute exactly if it's been samples more than N times and hasn't
            // been computed exactly already
            arma::umat compute_exactly =
              ((T_samples + batchSize) >= N) != (exact_mask);
            arma::uvec targets = arma::find(compute_exactly);

            if (targets.size() > 0) {
                logHelper.comp_exact_swap.push_back(targets.size());
                arma::vec result = swap_target(data,
                                               medoid_indices,
                                               targets,
                                               N,
                                               best_distances,
                                               second_distances,
                                               assignments);
                estimates.elem(targets) = result;
                ucbs.elem(targets) = result;
                lcbs.elem(targets) = result;
                exact_mask.elem(targets).fill(1);
                T_samples.elem(targets) += N;

                candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            }
            if (arma::accu(candidates) < precision) {
                break;
            }
            targets = arma::find(candidates);
            arma::vec result = swap_target(data,
                                           medoid_indices,
                                           targets,
                                           batchSize,
                                           best_distances,
                                           second_distances,
                                           assignments);
            estimates.elem(targets) =
              ((T_samples.elem(targets) % estimates.elem(targets)) +
               (result * batchSize)) /
              (batchSize + T_samples.elem(targets));
            T_samples.elem(targets) += batchSize;
            arma::vec adjust(targets.n_rows);
            adjust.fill(p);
            adjust = arma::log(adjust);
            arma::vec cb_delta = sigma.elem(targets) %
                                 arma::sqrt(adjust / T_samples.elem(targets));

            ucbs.elem(targets) = estimates.elem(targets) + cb_delta;
            lcbs.elem(targets) = estimates.elem(targets) - cb_delta;
            candidates = (lcbs < ucbs.min()) && (exact_mask == 0);
            targets = arma::find(candidates);
        }
        // now switch medoids
        arma::uword new_medoid = lcbs.index_min();
        // extract medoid of swap
        size_t k = new_medoid % medoids.n_cols;

        // extract data point of swap
        size_t n = new_medoid / medoids.n_cols;
        swap_performed = medoid_indices(k) != n;
        steps++;

        medoid_indices(k) = n;
        medoids.col(k) = data.col(medoid_indices(k));
        calc_best_distances_swap(
          data, medoid_indices, best_distances, second_distances, assignments);
        sigma_log(sigma);
        logHelper.loss_swap.push_back(arma::mean(arma::mean(best_distances)));
        logHelper.p_swap.push_back((float)1/(float)p);
    }
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
  arma::mat& data,
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
 * \brief Estimates the mean reward for each arm in swap step
 *
 * Estimates the mean reward (or loss) for each arm in the identified targets
 * in the swap step and returns a list of the estimated reward.
 *
 * @param data Transposed input data to find the medoids of
 * @param sigma Dispersion paramater for each datapoint
 * @param targets Set of target datapoints to be estimated
 * @param batch_size Number of datapoints sampled for updating confidence
 * intervals
 * @param best_distances Array of best distances from each point to previous set
 * of medoids
 * @param second_best_distances Array of second smallest distances from each
 * point to previous set of medoids
 * @param assignments Assignments of datapoints to their closest medoid
 */
arma::vec KMedoids::swap_target(
  arma::mat& data,
  arma::rowvec& medoid_indices,
  arma::uvec& targets,
  size_t batch_size,
  arma::rowvec& best_distances,
  arma::rowvec& second_best_distances,
  arma::rowvec& assignments)
{
    size_t N = data.n_cols;
    arma::vec estimates(targets.n_rows, arma::fill::zeros);
    arma::uvec tmp_refs = arma::randperm(N,
                                   batch_size); // without replacement, requires
                                                // updated version of armadillo

// for each considered swap
#pragma omp parallel for
    for (size_t i = 0; i < targets.n_rows; i++) {
        double total = 0;
        // extract data point of swap
        size_t n = targets(i) / medoid_indices.n_cols;
        size_t k = targets(i) % medoid_indices.n_cols;
        // calculate total loss for some subset of the data
        for (size_t j = 0; j < batch_size; j++) {
            double cost = (this->*lossFn)(data, n, tmp_refs(j));
            if (k == assignments(tmp_refs(j))) {
                if (cost < second_best_distances(tmp_refs(j))) {
                    total += cost;
                } else {
                    total += second_best_distances(tmp_refs(j));
                }
            } else {
                if (cost < best_distances(tmp_refs(j))) {
                    total += cost;
                } else {
                    total += best_distances(tmp_refs(j));
                }
            }
            total -= best_distances(tmp_refs(j));
        }
        estimates(i) = total / tmp_refs.n_rows;
    }
    return estimates;
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
  arma::mat& data,
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
};

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
  arma::mat& data,
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
double KMedoids::LP(arma::mat& data, int i, int j) const {
    return arma::norm(data.col(i) - data.col(j), lp);
}

/**
 * \brief L2 loss
 *
 * Calculates the L2 loss between the datapoints at index i and j of the dataset
 *
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
//double KMedoids::L2(int i, int j) const {
//    return arma::norm(data.col(i) - data.col(j), 2);
//}

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
double KMedoids::cos(arma::mat& data, int i, int j) const {
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
double KMedoids::manhattan(arma::mat& data, int i, int j) const {
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
double KMedoids::LINF(arma::mat& data, int i, int j) const {
    return arma::max(arma::abs(data.col(i) - data.col(j)));
}
