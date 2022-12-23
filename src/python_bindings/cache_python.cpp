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
size_t km::KMedoidsWrapper::getDistanceComputationsPython() {
    return KMedoids::getDistanceComputations();
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

void distance_computations_python(pybind11::class_<km::KMedoidsWrapper> *cls) {
  cls->def_property_readonly("distance_computations", &KMedoidsWrapper::getDistanceComputationsPython);
}

void cache_writes_python(pybind11::class_<km::KMedoidsWrapper> *cls) {
  cls->def_property_readonly("cache_writes", &KMedoidsWrapper::getCacheWritesPython);
}

void cache_hits_python(pybind11::class_<km::KMedoidsWrapper> *cls) {
  cls->def_property_readonly("cache_hits", &KMedoidsWrapper::getCacheHitsPython);
}

void cache_misses_python(pybind11::class_<km::KMedoidsWrapper> *cls) {
  cls->def_property_readonly("cache_misses", &KMedoidsWrapper::getCacheMisses);
}
}  // namespace km
