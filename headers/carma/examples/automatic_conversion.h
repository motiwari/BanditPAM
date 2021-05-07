#include <armadillo>
#include <carma>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

arma::Mat<double> automatic_example(arma::Mat<double> & mat);
void bind_automatic_example(py::module &m);
