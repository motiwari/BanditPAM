/**
 * @file kmedoids_pywrapper.cpp
 * @date 2020-06-10
 *
 * Creates the Python bindings for the C++ code that
 * allows it to be called in Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_algorithm.hpp"
#include "kmedoids_pywrapper.hpp"

namespace km {
PYBIND11_MODULE(banditpam, m) {
  m.doc() = "BanditPAM Python library, implemented in C++";
  m.def("get_max_threads",
    &omp_get_max_threads, "Returns max number of threads");
  m.def("set_num_threads",
    &omp_set_num_threads, "Set the maximum number of threads");
  pybind11::class_<KMedoidsWrapper> cls(m, "KMedoids");
  cls.def(pybind11::init<int, std::string, int, int, int>(),
          pybind11::arg("n_medoids") = NULL,
          pybind11::arg("algorithm") = "BanditPAM",
          pybind11::arg("max_iter") = 1000,
          pybind11::arg("build_confidence") = 1000,
          pybind11::arg("swap_confidence") = 10000);
  cls.def_property("n_medoids",
    &KMedoidsWrapper::getNMedoids, &KMedoidsWrapper::setNMedoids);
  cls.def_property("algorithm",
    &KMedoidsWrapper::getAlgorithm, &KMedoidsWrapper::setAlgorithm);
  cls.def_property("max_iter",
    &KMedoidsWrapper::getMaxIter, &KMedoidsWrapper::setMaxIter);
  cls.def_property("build_confidence",
    &KMedoidsWrapper::getBuildConfidence, &KMedoidsWrapper::setBuildConfidence);
  cls.def_property("swap_confidence",
    &KMedoidsWrapper::getSwapConfidence, &KMedoidsWrapper::setSwapConfidence);
  cls.def_property("loss_function",
    &KMedoidsWrapper::getLossFn, &KMedoidsWrapper::setLossFn);
  cls.def_property("seed",
    &KMedoidsWrapper::getSeed, &KMedoidsWrapper::setSeed);
  medoids_python(&cls);
  build_medoids_python(&cls);
  labels_python(&cls);
  steps_python(&cls);
  fit_python(&cls);
  loss_python(&cls);
  m.attr("__version__") = VERSION_INFO;
}
}  // namespace km
