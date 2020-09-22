#include <armadillo>
#include <carma/carma.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kmedoids_ucb.hpp"

namespace py = pybind11;

class KMedsWrapper : public KMedoids {
public:
  using KMedoids::KMedoids;

  void fitPython(py::array_t<double> input_data) {
    KMedoids::fit(carma::arr_to_mat<double>(input_data));
  }

  py::array_t<int> getMedoidsFinalPython() {
    return carma::row_to_arr<double>(KMedoids::getMedoidsFinal()).squeeze();
  }

  py::array_t<int> getMedoidsBuildPython() {
    return carma::row_to_arr<double>(KMedoids::getMedoidsBuild()).squeeze();
  }

  py::array_t<int> getLabelsPython() {
    return carma::row_to_arr<double>(KMedoids::getLabels()).squeeze();
  }
};

PYBIND11_MODULE(banditPAM, m) {
  m.doc() = "BanditPAM Test";
  py::class_<KMedsWrapper>(m, "KMedoids")
      .def(py::init<int, std::string, int, std::string, int, std::string>(),
        py::arg("n_medoids") = 5,
        py::arg("algorithm") = "BanditPAM",
        py::arg("max_iter") = 1000,
        py::arg("loss") = "L2",
        py::arg("verbosity") = 0,
        py::arg("logFilename") = "KMedoidsLogfile"
      )
      .def_property_readonly("final_medoids", &KMedsWrapper::getMedoidsFinalPython)
      .def_property_readonly("build_medoids", &KMedsWrapper::getMedoidsBuildPython)
      .def_property_readonly("labels", &KMedsWrapper::getLabelsPython)
      .def("fit", &KMedsWrapper::fitPython);
}
