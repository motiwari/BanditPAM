/**
 * @file build_medoids_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getMedoidsBuildPython in KMedoidsWrapper class 
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
 *  \brief Returns the build medoids
 *
 *  Returns as a numpy array the build medoids at the end of the BUILD step
 *  after KMedoids::fit has been called.
 */
py::array_t<double> KMedoidsWrapper::getMedoidsBuildPython() {
    return carma::row_to_arr<double>(KMedoids::getMedoidsBuild()).squeeze();
}

void build_medoids_python(py::class_<KMedoidsWrapper> &cls) {
    cls.def_property_readonly("build_medoids", &KMedoidsWrapper::getMedoidsBuildPython);
}
