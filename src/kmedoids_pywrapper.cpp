/**
 * @file kmedoids_pywrapper.cpp
 * @date 2020-06-10
 *
 * Creates the Python bindings for the C++ code that
 * allows it to be called in Python.
 *
 */

/* 
 * We perform these imports first because kmedoids_pywrapper is compiled first
 * when building the python package, and carma must be 'include'd before armadillo
*/
#include <carma>
#include <armadillo>

#include "kmedoids_algorithm.hpp"
#include "kmedoids_pywrapper.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

PYBIND11_MODULE(banditpam, m) {
  m.doc() = "BanditPAM Python library, implemented in C++";
  m.def("get_max_threads", &omp_get_max_threads, "Returns max number of threads");
  m.def("set_num_threads", &omp_set_num_threads, "Set the maximum number of threads");
  py::class_<KMedoidsWrapper> cls(m, "KMedoids");
  cls.def(py::init<int, std::string, int, int, int>(),
          py::arg("n_medoids") = NULL,
          py::arg("algorithm") = "BanditPAM",
          py::arg("maxIter") = 1000,
          py::arg("buildConfidence") = 1000,
          py::arg("swapConfidence") = 10000
  );
  cls.def_property("n_medoids", &KMedoidsWrapper::getNMedoids, &KMedoidsWrapper::setNMedoids);
  cls.def_property("algorithm", &KMedoidsWrapper::getAlgorithm, &KMedoidsWrapper::setAlgorithm);
  cls.def_property("maxIter", &KMedoidsWrapper::getMaxIter, &KMedoidsWrapper::setMaxIter);
  cls.def_property("buildConfidence", &KMedoidsWrapper::getbuildConfidence, &KMedoidsWrapper::setbuildConfidence);
  cls.def_property("swapConfidence", &KMedoidsWrapper::getswapConfidence, &KMedoidsWrapper::setswapConfidence);
  medoids_python(cls);
  build_medoids_python(cls);
  labels_python(cls);
  steps_python(cls);
  fit_python(cls);
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
