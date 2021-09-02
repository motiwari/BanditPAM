/**
 * @file medoids_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getMedoidsFinalPython in KMedoidsWrapper class 
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
 *  \brief Returns the final medoids
 *
 *  Returns as a numpy array the final medoids at the end of the SWAP step
 *  after KMedoids::fit has been called.
 */
py::array_t<double> KMedoidsWrapper::getMedoidsFinalPython() {
    return carma::row_to_arr<double>(KMedoids::getMedoidsFinal()).squeeze();
}

void medoids_python(py::class_<KMedoidsWrapper> &cls) {
    cls.def_property_readonly("medoids", &KMedoidsWrapper::getMedoidsFinalPython);
}
