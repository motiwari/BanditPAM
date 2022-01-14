/**
 * @file loss_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getLossPython in KMedoidsWrapper class 
 * which is used in Python bindings.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_pywrapper.hpp"

namespace km {
float km::KMedoidsWrapper::getLossPython() {
  return KMedoids::getAverageLoss();
}

void loss_python(pybind11::class_<KMedoidsWrapper> *cls) {
  cls->def_property_readonly("average_loss", &KMedoidsWrapper::getLossPython);
}
}  // namespace km
