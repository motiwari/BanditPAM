/**
 * @file cache_python.cpp
 * @date 2021-08-16
 *
 * Defines the functions to cache statistics in the Python bindings.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace km {
  size_t km::KMedoidsWrapper::getDistanceComputationsPython(
          const bool includeMisc) {
    return KMedoids::getDistanceComputations(includeMisc);
  }

  size_t km::KMedoidsWrapper::getMiscDistanceComputationsPython() {
    return KMedoids::getMiscDistanceComputations();
  }

  size_t km::KMedoidsWrapper::getBuildDistanceComputationsPython() {
    return KMedoids::getBuildDistanceComputations();
  }

  size_t km::KMedoidsWrapper::getSwapDistanceComputationsPython() {
    return KMedoids::getSwapDistanceComputations();
  }

  size_t km::KMedoidsWrapper::getCacheWritesPython() {
    return KMedoids::getCacheWrites();
  }

  size_t km::KMedoidsWrapper::getCacheHitsPython() {
    return KMedoids::getCacheHits();
  }

  size_t km::KMedoidsWrapper::getCacheMissesPython() {
    return KMedoids::getCacheMisses();
  }

  void distance_computations_python(
          pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def("getDistanceComputations",
             &KMedoidsWrapper::getDistanceComputationsPython);
  }

  void misc_distance_computations_python(
          pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "misc_distance_computations",
            &KMedoidsWrapper::getMiscDistanceComputationsPython);
  }

  void build_distance_computations_python(
          pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "build_distance_computations",
            &KMedoidsWrapper::getBuildDistanceComputationsPython);
  }

  void swap_distance_computations_python(
          pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "swap_distance_computations",
            &KMedoidsWrapper::getSwapDistanceComputationsPython);
  }

  void cache_writes_python(pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "cache_writes",
            &KMedoidsWrapper::getCacheWritesPython);
  }

  void cache_hits_python(pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "cache_hits",
            &KMedoidsWrapper::getCacheHitsPython);
  }

  void cache_misses_python(pybind11::class_ <km::KMedoidsWrapper> *cls) {
    cls->def_property_readonly(
            "cache_misses",
            &KMedoidsWrapper::getCacheMisses);
  }
}  // namespace km
