/**
 * @file steps_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getStepsPython in KMedoidsWrapper class 
 * which is used in Python bindings.
 * 
 */

#include "kmedoids_pywrapper.hpp"

#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 *  \brief Returns the number of swap steps
 *
 *  Returns the number of SWAP steps completed during the last call to
 *  KMedoids::fit
 */
int KMedoidsWrapper::getStepsPython() {
    return KMedoids::getSteps();
}

void steps_python(py::class_<KMedoidsWrapper> &cls) {
    cls.def_property_readonly("steps", &KMedoidsWrapper::getStepsPython);
}
