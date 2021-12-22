/**
 * @file build_medoids_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getMedoidsBuildPython in KMedoidsWrapper class 
 * which is used in Python bindings.
 * 
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace py = pybind11;

/**
 *  \brief Returns the build medoids
 *
 *  Returns as a numpy array the build medoids at the end of the BUILD step
 *  after KMedoids::fit has been called.
 */
py::array_t<arma::uword> KMedoidsWrapper::getMedoidsBuildPython() {
  if (KMedoids::getMedoidsBuild().size() > 1) {
    return
      carma::row_to_arr<arma::uword>(KMedoids::getMedoidsBuild()).squeeze();
  } else {
      return carma::row_to_arr<arma::uword>(KMedoids::getMedoidsBuild());
  }
}

void build_medoids_python(py::class_<KMedoidsWrapper> *cls) {
    cls->def_property_readonly("build_medoids",
        &KMedoidsWrapper::getMedoidsBuildPython);
}
