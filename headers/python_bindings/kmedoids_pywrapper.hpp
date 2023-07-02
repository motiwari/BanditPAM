#ifndef HEADERS_PYTHON_BINDINGS_KMEDOIDS_PYWRAPPER_HPP_
#define HEADERS_PYTHON_BINDINGS_KMEDOIDS_PYWRAPPER_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
          const pybind11::array_t<float> &inputData,
          const std::string &loss,
          pybind11::kwargs kw);

  /**
   * @brief Returns the build medoids
   *
   * Returns as a numpy array the build medoids at the end of the BUILD step
   * after KMedoids::fit has been called.
   */
  pybind11::array_t <arma::uword> getMedoidsBuildPython();

  /**
   * @brief Returns the final medoids
   *
   * Returns as a numpy array the final medoids at the end of the SWAP step
   * after KMedoids::fit has been called.
   */
  pybind11::array_t <arma::uword> getMedoidsFinalPython();

  /**
   * @brief Returns the medoid assignments for each datapoint
   *
   * Returns as a numpy array the medoid each input datapoint is assigned to
   * after KMedoids::fit is called and the final medoids have been identified
   */
  pybind11::array_t <arma::uword> getLabelsPython();

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

  /**
   * @brief Returns the loss after the BUILD step
   *
   * The loss after the BUILD step, i.e., the average distance from each point to its
   * nearest medoid after the BUILD step
   */
  float getBuildLossPython();

  /**
   * @brief Returns the number of distance computations (sample complexity) used by .fit()
   *
   * The total number of distance computations (sample complexity) used by .fit()
   */
  size_t getDistanceComputationsPython(const bool includeMisc);

  // TODO(@motiwari): Add docstring
  size_t getMiscDistanceComputationsPython();

  // TODO(@motiwari): Add docstring
  size_t getBuildDistanceComputationsPython();

  // TODO(@motiwari): Add docstring
  size_t getSwapDistanceComputationsPython();

  /**
   * @brief Returns the number of cache writes done by .fit()
   *
   * The number of cache writes performed by the last call to .fit()
   */
  size_t getCacheWritesPython();

  /**
   * @brief Returns the number of cache hits from the last call to .fit()
   *
   * The number of cache hits from the last call to .fit()
  */
  size_t getCacheHitsPython();

  /**
   * @brief Returns the number of cache misses by the last call to .fit()
   *
   * The number of cache misses from the last call to .fit()
   */
  size_t getCacheMissesPython();

  /**
   * @brief Returns the total time for the entire SWAP procedure by the last call to .fit()
   *
   * The total time for the entire SWAP procedure by the last call to .fit()
   */
  size_t getTotalSwapTimePython();

  /**
   * @brief Returns the average time per swap step by the last call to .fit()
   *
   * The average time per swap step by the last call to .fit()
   */
  float getTimePerSwapPython();
};

// TODO(@motiwari): Encapsulate these

  /**
  * @brief Binding for the C++ function KMedoids::fit
  */
  void fit_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getMedoidsBuild()
  */
  void build_medoids_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getMedoidsFinal()
  */
  void medoids_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getLabels()
  */
  void labels_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getSteps()
  */
  void steps_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::calcLoss()
  */
  void loss_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::calcLoss() (after BUILD step)
  */
  void build_loss_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getDistanceComputations()
  */
  void
  distance_computations_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  // TODO(@motiwari): Add docstring
  void misc_distance_computations_python(
        pybind11::class_ <km::KMedoidsWrapper> *cls);

  // TODO(@motiwari): Add docstring
  void build_distance_computations_python(
        pybind11::class_ <km::KMedoidsWrapper> *cls);

  // TODO(@motiwari): Add docstring
  void swap_distance_computations_python(
        pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getCacheWrites()
  */
  void cache_writes_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getCacheHits()
  */
  void cache_hits_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getCacheMisses()
  */
  void cache_misses_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getTotalSwapTime()
  */
  void total_swap_time_python(pybind11::class_ <km::KMedoidsWrapper> *cls);

  /**
  * @brief Binding for the C++ function KMedoids::getTimePerSwap()
  */
  void time_per_swap_python(pybind11::class_ <km::KMedoidsWrapper> *cls);
}  // namespace km
#endif  // HEADERS_PYTHON_BINDINGS_KMEDOIDS_PYWRAPPER_HPP_
