#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>

#include <carma>

namespace py = pybind11;

#ifndef TESTS_SRC_TEST_MAT_TO_ARR_H_
#define TESTS_SRC_TEST_MAT_TO_ARR_H_

namespace carma {
namespace tests {
py::array_t<double> test_mat_to_arr(bool copy);
py::array_t<double> test_row_to_arr(bool copy);
py::array_t<double> test_col_to_arr(bool copy);
py::array_t<double> test_cube_to_arr(bool copy);
py::array_t<double> test_to_numpy_mat(bool copy);
py::array_t<double> test_to_numpy_row(bool copy);
py::array_t<double> test_to_numpy_col(bool copy);
py::array_t<double> test_to_numpy_cube(bool copy);
void test_update_array_mat(py::array_t<double>& arr, int cols);
void test_update_array_row(py::array_t<double>& arr, int cols);
void test_update_array_col(py::array_t<double>& arr, int cols);
void test_update_array_cube(py::array_t<double>& arr, int cols);

py::array_t<double> test_mat_to_arr_plus_one(const py::array_t<double>& arr, bool copy);
py::array_t<double> test_row_to_arr_plus_one(const py::array_t<double>& arr, bool copy);
py::array_t<double> test_col_to_arr_plus_one(const py::array_t<double>& arr, bool copy);
py::array_t<double> test_cube_to_arr_plus_one(const py::array_t<double>& arr, bool copy);
}  // namespace tests
}  // namespace carma

void bind_test_mat_to_arr(py::module& m);
void bind_test_row_to_arr(py::module& m);
void bind_test_col_to_arr(py::module& m);
void bind_test_cube_to_arr(py::module& m);
void bind_test_to_numpy_mat(py::module& m);
void bind_test_to_numpy_row(py::module& m);
void bind_test_to_numpy_col(py::module& m);
void bind_test_to_numpy_cube(py::module& m);
void bind_test_update_array_mat(py::module& m);
void bind_test_update_array_row(py::module& m);
void bind_test_update_array_col(py::module& m);
void bind_test_update_array_cube(py::module& m);
void bind_test_mat_to_arr_plus_one(py::module& m);
void bind_test_row_to_arr_plus_one(py::module& m);
void bind_test_col_to_arr_plus_one(py::module& m);
void bind_test_cube_to_arr_plus_one(py::module& m);

#endif  // TESTS_SRC_TEST_MAT_TO_ARR_H_
