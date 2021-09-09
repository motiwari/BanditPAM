/**
 * @file kmedoids_ucb.cpp
 * @date 2020-06-10
 *
 * This file contains the primary C++ implementation of the BanditPAM code.
 *
 */
#include "kmedoids_algorithm.hpp"
#include "log_helper.hpp"

#include <carma>
#include <armadillo>
#include <unordered_map>
#include <regex>
#include <string>
#include <cstring>
#include "Python.h"
#include "numpy/arrayobject.h"

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
 *  @param modPath The Path of the custom loss function
 *  @param dist_mat The name of the custom loss function
 */
km::KMedoids::KMedoids(size_t n_medoids, const std::string& algorithm, size_t verbosity,
                   size_t max_iter, size_t buildConfidence, size_t swapConfidence,
                   std::string logFilename, std::string modPath, std::string dist_mat
    ): n_medoids(n_medoids),
       algorithm(algorithm),
       max_iter(max_iter),
       buildConfidence(buildConfidence),
       swapConfidence(swapConfidence),
       verbosity(verbosity),
       logFilename(logFilename),
       modPath(modPath),
       dist_mat(dist_mat)
       {
  km::KMedoids::checkAlgorithm(algorithm);
}

/**
 *  \brief Destroys KMedoids object.
 *
 *  Destructor for the KMedoids class.
 */
km::KMedoids::~KMedoids() {;}

/**
 *  \brief Checks whether algorithm input is valid
 *
 *  Checks whether the user's selected algorithm is a valid option.
 *
 *  @param algorithm Name of the algorithm input by the user.
 */
void km::KMedoids::checkAlgorithm(const std::string& algorithm) {
  if (algorithm == "BanditPAM") {
    fitFn = &km::KMedoids::fit_bpam;
  } else if (algorithm == "naive") {
    fitFn = &km::KMedoids::fit_naive;
  } else {
    throw "unrecognized algorithm";
  }
}

/**
 *  \brief Returns the final medoids
 *
 *  Returns the final medoids at the end of the SWAP step after km::KMedoids::fit
 *  has been called.
 */
arma::rowvec km::KMedoids::getMedoidsFinal() {
  return medoid_indices_final;
}

/**
 *  \brief Returns the build medoids
 *
 *  Returns the build medoids at the end of the BUILD step after km::KMedoids::fit
 *  has been called.
 */
arma::rowvec km::KMedoids::getMedoidsBuild() {
  return medoid_indices_build;
}

/**
 *  \brief Returns the medoid assignments for each datapoint
 *
 *  Returns the medoid each input datapoint is assigned to after km::KMedoids::fit
 *  has been called and the final medoids have been identified
 */
arma::rowvec km::KMedoids::getLabels() {
  return labels;
}

/**
 *  \brief Returns the number of swap steps
 *
 *  Returns the number of SWAP steps completed during the last call to
 *  km::KMedoids::fit
 */
size_t km::KMedoids::getSteps() {
  return steps;
}

/**
 *  \brief Sets the loss function
 *
 *  Sets the loss function used during km::KMedoids::fit
 *
 *  @param loss Loss function to be used e.g. L2
 */
void km::KMedoids::setModPath(std::string mod_path) {
  if (!mod_path.empty())
  {
     modPath = mod_path;
  }

}


/**
 *  \brief Sets the loss function
 *
 *  Sets the loss function used during km::KMedoids::fit
 *
 *  @param loss Loss function to be used e.g. L2
 */
void km::KMedoids::setDistMat(std::string dist_matrics) {
  if (!dist_matrics.empty())
  {
     dist_mat = dist_matrics;
  }
}

/**
 *  \brief Sets the loss function
 *
 *  Sets the loss function used during km::KMedoids::fit
 *
 *  @param loss Loss function to be used e.g. L2
 */
void km::KMedoids::setLossFn(std::string loss) {
  if (std::regex_match(loss, std::regex("L\\d*"))) {
      loss = loss.substr(1);
  }
  try {
    if (loss == "manhattan") {
        lossFn = &km::KMedoids::manhattan;
    } else if (loss == "cos") {
        lossFn = &km::KMedoids::cos;
    } else if (loss == "inf") {
        lossFn = &km::KMedoids::LINF;
    }else if (loss == "custom") {
        lossFn = &km::KMedoids::custom_loss;
    } else if (std::isdigit(loss.at(0))) {
        lossFn = &km::KMedoids::LP;
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
 *  Returns the number of medoids to be identified during km::KMedoids::fit
 */
size_t km::KMedoids::getNMedoids() {
  return n_medoids;
}

/**
 *  \brief Sets the number of medoids
 *
 *  Sets the number of medoids to be identified during km::KMedoids::fit
 */
void km::KMedoids::setNMedoids(size_t new_num) {
  n_medoids = new_num;
}

/**
 *  \brief Returns the algorithm for KMedoids
 *
 *  Returns the algorithm used for identifying the medoids during km::KMedoids::fit
 */
std::string km::KMedoids::getAlgorithm() {
  return algorithm;
}

/**
 *  \brief Sets the algorithm for KMedoids
 *
 *  Sets the algorithm used for identifying the medoids during km::KMedoids::fit
 *
 *  @param new_alg New algorithm to use
 */
void km::KMedoids::setAlgorithm(const std::string& new_alg) {
  algorithm = new_alg;
}

/**
 *  \brief Returns the verbosity for KMedoids
 *
 *  Returns the verbosity used during km::KMedoids::fit, with 0 not creating a
 *  logfile, and >0 creating a detailed logfile.
 */
size_t km::KMedoids::getVerbosity() {
  return verbosity;
}

/**
 *  \brief Sets the verbosity for KMedoids
 *
 *  Sets the verbosity used during km::KMedoids::fit, with 0 not creating a
 *  logfile, and >0 creating a detailed logfile.
 *
 *  @param new_ver New verbosity to use
 */
void km::KMedoids::setVerbosity(size_t new_ver) {
  verbosity = new_ver;
}

/**
 *  \brief Returns the maximum number of iterations for KMedoids
 *
 *  Returns the maximum number of iterations that can be run during
 *  km::KMedoids::fit
 */
size_t km::KMedoids::getMaxIter() {
  return max_iter;
}

/**
 *  \brief Sets the maximum number of iterations for KMedoids
 *
 *  Sets the maximum number of iterations that can be run during km::KMedoids::fit
 *
 *  @param new_max New maximum number of iterations to use
 */
void km::KMedoids::setMaxIter(size_t new_max) {
  max_iter = new_max;
}

/**
 *  \brief Returns the constant buildConfidence 
 *
 *  Returns the constant that affects the sensitivity of build confidence bounds 
 *  that can be run during km::KMedoids::fit
 */
size_t km::KMedoids::getbuildConfidence() {
  return buildConfidence;
}

/**
 *  \brief Sets the constant buildConfidence
 *
 *  Sets the constant that affects the sensitivity of build confidence bounds 
 *  that can be run during km::KMedoids::fit
 *
 *  @param new_buildConfidence New buildConfidence 
 */
void km::KMedoids::setbuildConfidence(size_t new_buildConfidence) {
  buildConfidence = new_buildConfidence;
}

/**
 *  \brief Returns the constant swapConfidence 
 *
 *  Returns the constant that affects the sensitivity of swap confidence bounds 
 *  that can be run during km::KMedoids::fit
 */
size_t km::KMedoids::getswapConfidence() {
  return swapConfidence;
}

/**
 *  \brief Sets the constant swapConfidence
 *
 *  Sets the constant that affects the sensitivity of swap confidence bounds
 *  that can be run during km::KMedoids::fit
 *
 *  @param new_swapConfidence New swapConfidence 
 */
void km::KMedoids::setswapConfidence(size_t new_swapConfidence) {
  swapConfidence = new_swapConfidence;
}

/**
 *  \brief Returns the log filename for KMedoids
 *
 *  Returns the name of the logfile that will be output at the end of
 *  km::KMedoids::fit if verbosity is >0
 */
std::string km::KMedoids::getLogfileName() {
  return logFilename;
}

/**
 *  \brief Returns the module path for custom loss
 *
 *  Returns the path of the module that will have the
 *  custom loss function
 */
std::string km::KMedoids::getModPath() {
  return modPath;
}

/**
 *  \brief Returns the name of the custom loss function
 *
 *  Returns the name of the custom log function that will be used to
 *  calculate the distance between two data points
 */
std::string km::KMedoids::getDistMat() {
  return dist_mat;
}

/**
 *  \brief Sets the log filename for KMedoids
 *
 *  Sets the name of the logfile that will be output at the end of
 *  km::KMedoids::fit if verbosity is >0
 *
 *  @param new_lname New logfile name
 */
void km::KMedoids::setLogFilename(const std::string& new_lname) {
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
void km::KMedoids::fit(const arma::mat& input_data, const std::string& loss, std::string mod_path, std::string dist_mat) {
  std::cout<<"Inside fittt "<<std::endl;
  km::KMedoids::setLossFn(loss);
  km::KMedoids::setModPath(mod_path);
  km::KMedoids::setDistMat(dist_mat);
  km::KMedoids::checkAlgorithm(algorithm);
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
void km::KMedoids::build_sigma(
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
            double cost = 0;
            PyGILState_STATE gil_state = PyGILState_Ensure();
            {
              cost= (this->*lossFn)(data, i, tmp_refs(j));
            }
            PyGILState_Release(gil_state);
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
void km::KMedoids::calc_best_distances_swap(
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
            double cost=0;
            PyGILState_STATE gil_state = PyGILState_Ensure();
            { 
              cost= (this->*lossFn)(data, medoid_indices(k), i);
            }
            PyGILState_Release(gil_state);
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
void km::KMedoids::swap_sigma(
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
            double cost = 0;
            PyGILState_STATE gil_state = PyGILState_Ensure();
            { 
              cost = (this->*lossFn)(data, n,tmp_refs(j));
            }
            PyGILState_Release(gil_state);
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
void km::KMedoids::sigma_log(arma::mat& sigma) {
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
double km::KMedoids::calc_loss(
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
double km::KMedoids::LP(const arma::mat& data, size_t i, size_t j) const {
   std::cout<<"INSIDE LPPPPPP"<<std::endl;
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
double km::KMedoids::cos(const arma::mat& data, size_t i, size_t j) const {
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
double km::KMedoids::manhattan(const arma::mat& data, size_t i, size_t j) const {
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
double km::KMedoids::LINF(const arma::mat& data, size_t i, size_t j) const {
    return arma::max(arma::abs(data.col(i) - data.col(j)));
}

/**
 * \brief custom loss
 *
 * Calculates the custom loss between the datapoints at index i and j of the
 * dataset
 *
 * @param data Transposed input data to find the medoids of
 * @param i Index of first datapoint
 * @param j Index of second datapoint
 */
double km::KMedoids::custom_loss(const arma::mat& data, size_t i, size_t j) const {
  //return arma::max(arma::abs(data.col(i) - data.col(j)));
  double result =0;
  std::cout<< "inside custom lost"<<std::endl;
  PyGILState_STATE gil_state = PyGILState_Ensure();
  {
    setenv("PYTHONPATH",".",1);
    Py_Initialize();
    
    PyObject *pName, *sys, *path;
    std::string modPath = "multiply";
    std::string dist_mat = "multiply";
    char* mod = (char*) modPath.c_str();
    char* distmat = (char*) dist_mat.c_str();
    sys  = PyImport_ImportModule("sys");
    path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyUnicode_DecodeFSDefault("."));
    PyObject *pModule = PyImport_ImportModule("multiply");
    if (!pModule)
    {
      PyErr_Print();
      printf("ERROR in pModule\n");
      exit(1);
    }
    PyObject *pFunc = PyObject_GetAttrString(pModule, distmat);
    
    std::cout<< "Works fine till here\n";
    PyObject *py_args_tuple, *pResult;
    py_args_tuple = PyTuple_New(2);
    
    arma::vec v1= data.col(i);
    arma::vec v2= data.col(j);
    
    PyObject *pylist, *item;
    pylist = PyList_New(v1.size());
    if (pylist != NULL){ 
        for (int i = 0; i < v1.size(); i++) {
          item = PyLong_FromLong(v1[i]);
            PyList_SET_ITEM(pylist, i, item);
        }
    }
    PyObject *pylist1, *item1;
    pylist1 = PyList_New(v2.size());
    //py::array_t<double> cv1 = carma::col_to_arr(v1);
    pylist1 = PyList_New(v2.size());
    if (pylist1 != NULL){ 
        for (int j = 0; j < v2.size(); j++) {
            item1 = PyLong_FromLong(v2[j]);
            PyList_SET_ITEM(pylist1, j, item1);
        }
    }
  
    PyTuple_SetItem(py_args_tuple, 0, pylist); 
    PyTuple_SetItem(py_args_tuple, 1, pylist1 ); 
    
    pResult=PyObject_CallObject(pFunc, py_args_tuple);
    result = PyFloat_AsDouble(pResult); 
    std::cout<< "results--"<<result<<std::endl;
  }
  PyGILState_Release(gil_state);
  std::cout<< "END ** custom lost"<<std::endl;
  return result;

}
