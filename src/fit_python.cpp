/**
 * @file fit_python.cpp
 * @date 2021-08-16
 *
 * Defines the function fitPython in KMedoidsWrapper class 
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
 * \brief Python binding for fitting a KMedoids object to the
 *
 * This is the primary function of the KMedoids module: this finds the build and swap
 * medoids for the desired data
 *
 * @param inputData Input data to find the medoids of
 * @param loss The loss function used during medoid computation
 * @param k The number of medoids to compute
 */
void KMedoidsWrapper::fitPython(const py::array_t<double>& inputData, 
                                const std::string& loss,
                                py::kwargs kw
    ) 
{
    // throw an error if the number of medoids is not specified in either 
    // the KMedoids object or the fitPython function
    try {
        if (KMedoids::getNMedoids() == 0) { // Check for 0 as NULL
            if (kw.size() == 0) {
                throw py::value_error("Must specify number of medoids via n_medoids in KMedoids or k in fit function.");
            }
        }
    } catch (py::value_error &e) {
        // Throw it again (pybind11 will raise ValueError)
        // TODO: Make this more informative
        throw;
    }
    // if k is specified here, we set the number of medoids as k and override previous value 
    if ((kw.size() != 0) && (kw.contains("k"))) {
        KMedoids::setNMedoids(py::cast<int>(kw["k"]));
    }
    KMedoids::fit(carma::arr_to_mat<double>(inputData), loss);
}

void fit_python(py::class_<KMedoidsWrapper> &cls) {
    cls.def("fit", &KMedoidsWrapper::fitPython);
}