/**
 * @file loss_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getLossPython and getBuildLossPython in KMedoidsWrapper
 * class which is used in Python bindings.
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

float km::KMedoidsWrapper::getBuildLossPython() {
  return KMedoids::getBuildLoss();
}

void loss_python(pybind11::class_<KMedoidsWrapper> *cls) {
  cls->def_property_readonly("average_loss", &KMedoidsWrapper::getLossPython);
}

void build_loss_python(pybind11::class_<KMedoidsWrapper> *cls) {
  cls->def_property_readonly("build_loss",
                             &KMedoidsWrapper::getBuildLossPython);
}
}  // namespace km
