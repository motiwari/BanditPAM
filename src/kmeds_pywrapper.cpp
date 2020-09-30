#include <armadillo>
#include <carma/carma.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kmedoids_ucb.hpp"

namespace py = pybind11;

class KMedsWrapper : public KMedoids {
public:
  using KMedoids::KMedoids;

  void fitPython(py::array_t<double> input_data, std::string loss) {
    KMedoids::fit(carma::arr_to_mat<double>(input_data), loss);
  }

  py::array_t<double> getMedoidsFinalPython() {
    return carma::row_to_arr<double>(KMedoids::getMedoidsFinal()).squeeze();
  }

  py::array_t<double> getMedoidsBuildPython() {
    return carma::row_to_arr<double>(KMedoids::getMedoidsBuild()).squeeze();
  }

  py::array_t<double> getLabelsPython() {
    return carma::row_to_arr<double>(KMedoids::getLabels()).squeeze();
  }

  int getStepsPython() {
    return KMedoids::getSteps();
  }
};

PYBIND11_MODULE(BanditPAM, m) {
  m.doc() = "BanditPAM Python library, implemented in C++";
  py::class_<KMedsWrapper>(m, "KMedoids")
      .def(py::init<int, std::string, int, int, std::string>(),
        py::arg("n_medoids") = 5,
        py::arg("algorithm") = "BanditPAM",
        py::arg("verbosity") = 0,
        py::arg("max_iter") = 1000,
        py::arg("logFilename") = "KMedoidsLogfile"
      )
      .def_property_readonly("final_medoids", &KMedsWrapper::getMedoidsFinalPython)
      .def_property_readonly("build_medoids", &KMedsWrapper::getMedoidsBuildPython)
      .def_property_readonly("labels", &KMedsWrapper::getLabelsPython)
      .def_property_readonly("steps", &KMedsWrapper::getStepsPython)
      .def("fit", &KMedsWrapper::fitPython);
}
