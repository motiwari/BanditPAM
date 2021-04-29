#include <armadillo>
#include <carma>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void update_example(py::array_t<double> & arr);
py::array_t<double> manual_example(py::array_t<double> & arr);

void bind_manual_example(py::module &m);
void bind_update_example(py::module &m);
