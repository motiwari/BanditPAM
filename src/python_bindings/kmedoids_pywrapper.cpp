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
  cls.def("fit", &KMedoidsWrapper::fitPython, "Fit K-Medoids model to data");
  cls.def_property_readonly("medoids", &KMedoidsWrapper::getMedoidsFinalPython,
                            "Final medoids");
  cls.def_property_readonly(
    "build_medoids", &KMedoidsWrapper::getMedoidsBuildPython, "Build medoids");
  cls.def_property_readonly("labels", &KMedoidsWrapper::getLabelsPython,
                            "Cluster assignments");
  cls.def_property_readonly("steps", &KMedoidsWrapper::getStepsPython,
                            "Number of swap steps");
  cls.def_property_readonly("loss", &KMedoidsWrapper::getLossPython,
                            "Average loss");
  cls.def_property_readonly("build_loss", &KMedoidsWrapper::getBuildLossPython,
                            "Loss after build step");
  cls.def_property_readonly("distance_computations",
                            &KMedoidsWrapper::getDistanceComputationsPython,
                            "Number of distance computations");
  cls.def_property_readonly("misc_distance_computations",
                            &KMedoidsWrapper::getMiscDistanceComputationsPython,
                            "Number of misc distance computations");
  cls.def_property_readonly(
    "build_distance_computations",
    &KMedoidsWrapper::getBuildDistanceComputationsPython,
    "Number of build distance computations");
  cls.def_property_readonly("swap_distance_computations",
                            &KMedoidsWrapper::getSwapDistanceComputationsPython,
                            "Number of swap distance computations");
  cls.def_property_readonly("cache_writes",
                            &KMedoidsWrapper::getCacheWritesPython,
                            "Number of cache writes");
  cls.def_property_readonly("cache_hits", &KMedoidsWrapper::getCacheHitsPython,
                            "Number of cache hits");
  cls.def_property_readonly("cache_misses",
                            &KMedoidsWrapper::getCacheMissesPython,
                            "Number of cache misses");
  cls.def_property_readonly(
    "time_per_swap", &KMedoidsWrapper::getTimePerSwapPython, "Time per swap");
  cls.def_property_readonly("total_swap_time",
                            &KMedoidsWrapper::getTotalSwapTimePython,
                            "Total swap time");

  // Predict function binding
  cls.def(
    "predict",
    [](KMedoidsWrapper& self, const pybind11::array_t<float>& X_new) {
      // Convert numpy array to Armadillo float matrix
      auto X_new_mat = carma::arr_to_mat<float>(X_new);
      self.predict(X_new_mat);
    },
    "Predict cluster labels for new data points", pybind11::arg("X_new"));

  cls.def_property_readonly("labels_predict",
                            &KMedoidsWrapper::get_predict_labels,
                            "Cluster labels for predicted data points");

  // Sparse matrix support
  cls.def("fit_sparse", &KMedoidsWrapper::fit_sparse,
          "Fit K-Medoids model to sparse data");
  cls.def("predict_sparse", &KMedoidsWrapper::predict_sparse,
          "Predict cluster labels for new sparse data points");
}
void KMedoidsWrapper::fitPython(
  const pybind11::array_t<float>& inputData,
  const std::string& loss,
  pybind11::kwargs kw) {
    try {
        fit(carma::arr_to_mat<float>(inputData), loss, std::nullopt);
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

pybind11::array_t<arma::uword> KMedoidsWrapper::getMedoidsBuildPython() {
    return carma::row_to_arr(getMedoidsBuild());
}

pybind11::array_t<arma::uword> KMedoidsWrapper::getMedoidsFinalPython() {
    return carma::row_to_arr(getMedoidsFinal());
}

pybind11::array_t<arma::uword> KMedoidsWrapper::getLabelsPython() {
    return carma::row_to_arr(getLabels());
}

int KMedoidsWrapper::getStepsPython() {
    return getSteps();
}

float KMedoidsWrapper::getLossPython() {
    return getAverageLoss();
}

float KMedoidsWrapper::getBuildLossPython() {
    return getBuildLoss();
}

size_t KMedoidsWrapper::getDistanceComputationsPython(const bool includeMisc) {
    return getDistanceComputations(includeMisc);
}

size_t KMedoidsWrapper::getMiscDistanceComputationsPython() {
    return getMiscDistanceComputations();
}

size_t KMedoidsWrapper::getBuildDistanceComputationsPython() {
    return getBuildDistanceComputations();
}

size_t KMedoidsWrapper::getSwapDistanceComputationsPython() {
    return getSwapDistanceComputations();
}

size_t KMedoidsWrapper::getCacheWritesPython() {
    return getCacheWrites();
}

size_t KMedoidsWrapper::getCacheHitsPython() {
    return getCacheHits();
}

size_t KMedoidsWrapper::getCacheMissesPython() {
    return getCacheMisses();
}

size_t KMedoidsWrapper::getTotalSwapTimePython() {
    return getTotalSwapTime();
}

float KMedoidsWrapper::getTimePerSwapPython() {
    return getTimePerSwap();
}
}  // namespace km
