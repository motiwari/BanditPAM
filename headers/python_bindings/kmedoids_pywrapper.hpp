#ifndef HEADERS_PYTHON_BINDINGS_KMEDOIDS_PYWRAPPER_HPP_
#define HEADERS_PYTHON_BINDINGS_KMEDOIDS_PYWRAPPER_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>
#include <string>

#include "kmedoids_algorithm.hpp"

namespace km {
/**
 *  @brief Python wrapper for KMedoids class. Allows Python code to call
 *  the C++ code.
 */
class KMedoidsWrapper : public km::KMedoids {
 public:
  using km::KMedoids::KMedoids;  // TODO(@motiwari): fix?
  /**
   * @brief Python binding for fitting a KMedoids object to the
   *
   * This is the primary function of the KMedoids module: this finds the build and swap
   * medoids for the desired data
   *
   * @param inputData Input data to find the medoids of
   * @param loss The loss function used during medoid computation
   * @param k The number of medoids to compute
   */
  void fitPython(
    const pybind11::array_t<float>& inputData,
    const std::string& loss,
    pybind11::kwargs kw);

  /**
   * @brief Returns the build medoids
   *
   * Returns as a numpy array the build medoids at the end of the BUILD step
   * after KMedoids::fit has been called.
   */
  pybind11::array_t<arma::uword> getMedoidsBuildPython();

  /**
   * @brief Returns the final medoids
   *
   * Returns as a numpy array the final medoids at the end of the SWAP step
   * after KMedoids::fit has been called.
   */
  pybind11::array_t<arma::uword> getMedoidsFinalPython();

  /**
   * @brief Returns the medoid assignments for each datapoint
   *
   * Returns as a numpy array the medoid each input datapoint is assigned to
   * after KMedoids::fit is called and the final medoids have been identified
   */
  pybind11::array_t<arma::uword> getLabelsPython();

  /**
   * @brief Returns the number of swap steps
   *
   * Returns the number of SWAP steps completed during the last call to
   * KMedoids::fit
   */
  int getStepsPython();

  /**
   * @brief Returns the average clustering loss
   *
   * The average loss, i.e., the average distance from each point to its
   * nearest medoid
   */
  float getLossPython();
};

// TODO(@motiwari): Encapsulate these

/**
 * @brief Binding for the C++ function KMedoids::fit
 */
void fit_python(pybind11::class_<km::KMedoidsWrapper> *);

/**
 * @brief Binding for the C++ function KMedoids::getMedoidsBuild()
 */
void build_medoids_python(pybind11::class_<km::KMedoidsWrapper> *);

/**
 * @brief Binding for the C++ function KMedoids::getMedoidsFinal()
 */
void medoids_python(pybind11::class_<km::KMedoidsWrapper> *);

/**
 * @brief Binding for the C++ function KMedoids::getLabels()
 */
void labels_python(pybind11::class_<km::KMedoidsWrapper> *);

/**
 * @brief Binding for the C++ function KMedoids::getSteps()
 */
void steps_python(pybind11::class_<km::KMedoidsWrapper> *);

/**
 * @brief Binding for the C++ function KMedoids::calcLoss()
 */
void loss_python(pybind11::class_<km::KMedoidsWrapper> *);
}  // namespace km
#endif  // HEADERS_PYTHON_BINDINGS_KMEDOIDS_PYWRAPPER_HPP_
