/**
 * @file kmedoids_pywrapper.cpp
 * @date 2020-06-10
 *
 * Creates the Python bindings for the C++ code that
 * allows it to be called in Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <carma>
#include <armadillo>

#include "kmedoids_algorithm.hpp"
#include "kmedoids_pywrapper.hpp"

// from https://github.com/pybind/python_example/blob/master/src/main.cpp
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace km {
PYBIND11_MODULE(banditpam, m) {
  // Module functions
  m.doc() = "BanditPAM Python library, implemented in C++";

  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  m.def("get_max_threads",  // TODO(@motiwari): change to get_num_threads
        &omp_get_max_threads, "Returns max number of threads");
  m.def("set_num_threads", &omp_set_num_threads,
        "Set the maximum number of threads");

  // Class functions
  pybind11::class_<KMedoidsWrapper> cls(m, "KMedoids");

  // Constructor
  // NOTE: The order of these matters! Otherwise you can get warnings about
  //  variables not being intialized in the same order as their declarations,
  //  which can lead to undefined behavior (variables being intialized with
  //  each others' values). The order here much also match that of the
  //  constructor in kmedoids_algorithm.*pp
  cls.def(
    pybind11::init<int, std::string, int, int, int, bool, bool, int, bool>(),
    pybind11::arg("n_medoids") = 5, pybind11::arg("algorithm") = "BanditPAM",
    pybind11::arg("max_iter") = 100,
    pybind11::arg("build_confidence") = 10,  // 100 fixes stochasticity issues
    pybind11::arg("swap_confidence") = 5,
    // TODO(@motiwari): Verify these options are re-used correctly on reset
    pybind11::arg("use_cache") = true, pybind11::arg("use_perm") = true,
    pybind11::arg("cache_width") = 1000, pybind11::arg("parallelize") = true);

  // Properties
  cls.def_property("n_medoids", &KMedoidsWrapper::getNMedoids,
                   &KMedoidsWrapper::setNMedoids);
  cls.def_property("algorithm", &KMedoidsWrapper::getAlgorithm,
                   &KMedoidsWrapper::setAlgorithm);
  cls.def_property("max_iter", &KMedoidsWrapper::getMaxIter,
                   &KMedoidsWrapper::setMaxIter);
  cls.def_property("build_confidence", &KMedoidsWrapper::getBuildConfidence,
                   &KMedoidsWrapper::setBuildConfidence);
  cls.def_property("swap_confidence", &KMedoidsWrapper::getSwapConfidence,
                   &KMedoidsWrapper::setSwapConfidence);
  cls.def_property("use_cache", &KMedoidsWrapper::getUseCache,
                   &KMedoidsWrapper::setUseCache);
  cls.def_property("use_perm", &KMedoidsWrapper::getUsePerm,
                   &KMedoidsWrapper::setUsePerm);
  cls.def_property("cache_width", &KMedoidsWrapper::getCacheWidth,
                   &KMedoidsWrapper::setCacheWidth);
  cls.def_property("parallelize", &KMedoidsWrapper::getParallelize,
                   &KMedoidsWrapper::setParallelize);
  cls.def_property("loss_function", &KMedoidsWrapper::getLossFn,
                   &KMedoidsWrapper::setLossFn);
  cls.def_property("seed", &KMedoidsWrapper::getSeed,
                   &KMedoidsWrapper::setSeed);

  // Other functions
  medoids_python(&cls);
  build_medoids_python(&cls);
  labels_python(&cls);
  steps_python(&cls);
  fit_python(&cls);
  loss_python(&cls);
  build_loss_python(&cls);

  // Cache functions
  distance_computations_python(&cls);
  misc_distance_computations_python(&cls);
  build_distance_computations_python(&cls);
  swap_distance_computations_python(&cls);
  cache_writes_python(&cls);
  cache_hits_python(&cls);
  cache_misses_python(&cls);

  // Swap timing functions
  time_per_swap_python(&cls);
  total_swap_time_python(&cls);
}
}  // namespace km
