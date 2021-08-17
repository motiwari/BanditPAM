/**
 * @file labels_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getLabelsPython in KMedoidsWrapper class 
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
 *  \brief Returns the medoid assignments for each datapoint
 *
 *  Returns as a numpy array the medoid each input datapoint is assigned to
 *  after KMedoids::fit is called and the final medoids have been identified
 */
py::array_t<double> KMedoidsWrapper::getLabelsPython() {
    return carma::row_to_arr<double>(KMedoids::getLabels()).squeeze();
}

void labels_python(py::class_<KMedoidsWrapper> &cls) {
    cls.def_property_readonly("labels", &KMedoidsWrapper::getLabelsPython);
}

