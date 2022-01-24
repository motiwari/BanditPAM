/**
 * @file build_medoids_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getMedoidsBuildPython in KMedoidsWrapper class 
 * which is used in Python bindings.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace km {
pybind11::array_t<arma::uword> km::KMedoidsWrapper::getMedoidsBuildPython() {
  if (KMedoids::getMedoidsBuild().size() > 1) {
    return
      carma::row_to_arr<arma::uword>(KMedoids::getMedoidsBuild()).squeeze();
  } else {
      return carma::row_to_arr<arma::uword>(KMedoids::getMedoidsBuild());
  }
}

void build_medoids_python(pybind11::class_<KMedoidsWrapper> *cls) {
  cls->def_property_readonly("build_medoids",
    &KMedoidsWrapper::getMedoidsBuildPython);
}
}  // namespace km
