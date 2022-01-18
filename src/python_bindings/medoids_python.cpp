/**
 * @file medoids_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getMedoidsFinalPython in KMedoidsWrapper class 
 * which is used in Python bindings.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace km {
pybind11::array_t<arma::uword> km::KMedoidsWrapper::getMedoidsFinalPython() {
  if (KMedoids::getMedoidsFinal().size() > 1) {
    return carma::row_to_arr<arma::uword>(
      KMedoids::getMedoidsFinal()).squeeze();
  } else {
    return carma::row_to_arr<arma::uword>(KMedoids::getMedoidsFinal());
  }
}

void medoids_python(pybind11::class_<km::KMedoidsWrapper> *cls) {
  cls->def_property_readonly("medoids",
    &KMedoidsWrapper::getMedoidsFinalPython);
}
}  // namespace km
