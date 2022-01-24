/**
 * @file fit_python.cpp
 * @date 2021-08-16
 *
 * Defines the function fitPython in KMedoidsWrapper class 
 * which is used in Python bindings.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace km {
void km::KMedoidsWrapper::fitPython(
  const pybind11::array_t<float>& inputData,
  const std::string& loss,
  pybind11::kwargs kw) {
  // throw an error if the number of medoids is not specified in either
  // the KMedoids object or the fitPython function
  try {
    if (KMedoids::getNMedoids() == 0) {  // Check for 0 as NULL
      if (kw.size() == 0) {
        throw pybind11::value_error("Error: must specify number of medoids.");
      }
    }
  } catch (pybind11::value_error &e) {
    // Throw it again (pybind11 will raise ValueError)
    // TODO(@motiwari): Make this more informative
    throw;
  }
  // if k is specified here, then
  // we set the number of medoids as k and override previous value
  if ((kw.size() != 0) && (kw.contains("k"))) {
    KMedoids::setNMedoids(pybind11::cast<int>(kw["k"]));
  }
  KMedoids::fit(carma::arr_to_mat<float>(inputData), loss);
}

void fit_python(pybind11::class_<KMedoidsWrapper> *cls) {
  cls->def("fit", &KMedoidsWrapper::fitPython);
}
}  // namespace km
