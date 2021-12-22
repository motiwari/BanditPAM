/**
 * @file labels_python.cpp
 * @date 2021-08-16
 *
 * Defines the function getLabelsPython in KMedoidsWrapper class 
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
 *  \brief Returns the medoid assignments for each datapoint
 *
 *  Returns as a numpy array the medoid each input datapoint is assigned to
 *  after KMedoids::fit is called and the final medoids have been identified
 */
py::array_t<arma::uword> KMedoidsWrapper::getLabelsPython() {
    if (KMedoids::getLabels().size() > 1) {
        return carma::row_to_arr<arma::uword>(KMedoids::getLabels()).squeeze();
    } else {
        return carma::row_to_arr<arma::uword>(KMedoids::getLabels());
    }
}

void labels_python(py::class_<KMedoidsWrapper> *cls) {
    cls->def_property_readonly("labels", &KMedoidsWrapper::getLabelsPython);
}

