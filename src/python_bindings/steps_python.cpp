/**
 * @file steps_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getStepsPython in KMedoidsWrapper class 
 * which is used in Python bindings.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace km {
int km::KMedoidsWrapper::getStepsPython() {
  return KMedoids::getSteps();
}

void steps_python(pybind11::class_<KMedoidsWrapper> *cls) {
  cls->def_property_readonly("steps", &KMedoidsWrapper::getStepsPython);
}
}  // namespace km
