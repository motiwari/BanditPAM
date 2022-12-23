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
  pybind11::class_<KMedoidsWrapper> cls(m, "KMedoids");
  cls.def(pybind11::init<int, std::string, bool, bool, int, int, int, int, int>(),
          pybind11::arg("n_medoids") = NULL,
          pybind11::arg("algorithm") = "BanditPAM",
          pybind11::arg("useCacheP") = true,
          pybind11::arg("usePerm") = true,
          pybind11::arg("cacheMultiplier") = 1000,
          pybind11::arg("max_iter") = 1000,
          pybind11::arg("build_confidence") = 1000,
          pybind11::arg("swap_confidence") = 10000,
          pybind11::arg("seed") = 0);
  cls.def_property("n_medoids",
    &KMedoidsWrapper::getNMedoids, &KMedoidsWrapper::setNMedoids);
  cls.def_property("algorithm",
    &KMedoidsWrapper::getAlgorithm, &KMedoidsWrapper::setAlgorithm);
  cls.def_property("useCacheP",
    &KMedoidsWrapper::getUseCacheP, &KMedoidsWrapper::setUseCacheP);
  cls.def_property("usePerm",
    &KMedoidsWrapper::getUsePerm, &KMedoidsWrapper::setUsePerm);
  cls.def_property("cacheMultiplier",
    &KMedoidsWrapper::getCacheMultiplier, &KMedoidsWrapper::setCacheMultiplier);
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
