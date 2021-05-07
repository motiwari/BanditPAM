#include "test_mat_to_arr.h"

namespace carma {
namespace tests {

// ------------------------------ Mat -----------------------------------------
py::array_t<double> test_mat_to_arr(bool copy) {
    arma::mat mat = arma::mat(4, 5, arma::fill::randu);
    return mat_to_arr<double>(mat, copy);
} /* test_mat_to_arr */

py::array_t<double> test_to_numpy_mat(bool copy) {
    arma::mat mat = arma::mat(4, 5, arma::fill::randu);
    return to_numpy<arma::mat>(mat, copy);
} /* test_to_numpy_mat */

py::array_t<double> test_mat_to_arr_plus_one(const py::array_t<double>& arr, bool copy) {
    arma::Mat<double> ones = arma::ones(arr.shape(0), arr.shape(1));
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
    arma::Mat<double> out = ones + mat;
    return mat_to_arr<double>(out, copy);
} /* test_mat_to_arr_plus_one */

void test_update_array_mat(py::array_t<double>& arr, int cols) {
    arma::mat mat = arr_to_mat<double>(arr, false, false);
    mat.insert_cols(0, cols, true);
    update_array(mat, arr);
}

// ------------------------------ Row -----------------------------------------
py::array_t<double> test_row_to_arr(bool copy) {
    arma::Row<double> mat = arma::Row<double>(20, arma::fill::randu);
    return row_to_arr<double>(mat, copy);
} /* test_row_to_arr */

py::array_t<double> test_to_numpy_row(bool copy) {
    arma::Row<double> mat = arma::Row<double>(20, arma::fill::randu);
    return to_numpy<arma::Row<double>>(mat, copy);
} /* test_to_numpy_row */

py::array_t<double> test_row_to_arr_plus_one(const py::array_t<double>& arr, bool copy) {
    arma::Row<double> ones = arma::Row<double>(arr.size(), arma::fill::ones);
    arma::Row<double> mat = carma::arr_to_row<double>(arr);
    arma::Row<double> out = ones + mat;
    return row_to_arr<double>(out, copy);
} /* test_row_to_arr_plus_one */

void test_update_array_row(py::array_t<double>& arr, int cols) {
    arma::Row<double> mat = arr_to_row<double>(arr, false, false);
    mat.insert_cols(0, cols, true);
    update_array(mat, arr);
}

// ------------------------------ Col -----------------------------------------
py::array_t<double> test_col_to_arr(bool copy) {
    arma::Col<double> mat = arma::Col<double>(100, arma::fill::randu);
    return col_to_arr<double>(mat, copy);
} /* test_col_to_arr */

py::array_t<double> test_to_numpy_col(bool copy) {
    arma::Col<double> mat = arma::Col<double>(20, arma::fill::randu);
    return to_numpy<arma::Col<double>>(mat, copy);
} /* test_to_numpy_col */

py::array_t<double> test_col_to_arr_plus_one(const py::array_t<double>& arr, bool copy) {
    arma::Col<double> ones = arma::ones(arr.size());
    arma::Col<double> mat = carma::arr_to_col<double>(arr);
    arma::Col<double> out = ones + mat;
    return col_to_arr<double>(out, copy);
} /* test_col_to_arr_plus_one */

void test_update_array_col(py::array_t<double>& arr, int cols) {
    arma::Col<double> mat = arr_to_col<double>(arr, false, false);
    mat.insert_rows(0, cols, true);
    update_array(mat, arr);
}

// ------------------------------ Cube ----------------------------------------
py::array_t<double> test_cube_to_arr(bool copy) {
    arma::Cube<double> mat = arma::Cube<double>(100, 2, 4, arma::fill::randu);
    return cube_to_arr<double>(mat, copy);
} /* test_cube_to_arr */

py::array_t<double> test_to_numpy_cube(bool copy) {
    arma::Cube<double> mat = arma::Cube<double>(100, 2, 4, arma::fill::randu);
    return to_numpy<arma::Cube<double>>(mat, copy);
} /* test_to_numpy_cube */

py::array_t<double> test_cube_to_arr_plus_one(const py::array_t<double>& arr, bool copy) {
    arma::Cube<double> ones = arma::ones(arr.shape(0), arr.shape(1), arr.shape(2));
    arma::Cube<double> mat = carma::arr_to_cube<double>(arr);
    arma::Cube<double> out = ones + mat;
    return cube_to_arr<double>(out, copy);
} /* test_mat_to_arr_plus_one */

void test_update_array_cube(py::array_t<double>& arr, int cols) {
    arma::Cube<double> mat = arr_to_cube<double>(arr, false, false);
    mat.insert_cols(0, cols, true);
    update_array(mat, arr);
}
}  // namespace tests
}  // namespace carma

void bind_test_mat_to_arr(py::module& m) {
    m.def("mat_to_arr", &carma::tests::test_mat_to_arr, "Test mat_to_arr");
}

void bind_test_row_to_arr(py::module& m) {
    m.def("row_to_arr", &carma::tests::test_row_to_arr, "Test mat_to_arr");
}

void bind_test_col_to_arr(py::module& m) {
    m.def("col_to_arr", &carma::tests::test_col_to_arr, "Test col_to_arr");
}

void bind_test_cube_to_arr(py::module& m) {
    m.def("cube_to_arr", &carma::tests::test_cube_to_arr, "Test cube_to_arr");
}

void bind_test_to_numpy_mat(py::module& m) {
    m.def("to_numpy_mat", &carma::tests::test_to_numpy_mat, "Test to_numpy");
}

void bind_test_to_numpy_row(py::module& m) {
    m.def("to_numpy_row", &carma::tests::test_to_numpy_row, "Test to_numpy");
}

void bind_test_to_numpy_col(py::module& m) {
    m.def("to_numpy_col", &carma::tests::test_to_numpy_col, "Test to_numpy");
}

void bind_test_to_numpy_cube(py::module& m) {
    m.def("to_numpy_cube", &carma::tests::test_to_numpy_cube, "Test to_numpy");
}

void bind_test_update_array_mat(py::module& m) {
    m.def("update_array_mat", &carma::tests::test_update_array_mat, "Test update_array");
}

void bind_test_update_array_cube(py::module& m) {
    m.def("update_array_cube", &carma::tests::test_update_array_cube, "Test update_array");
}

void bind_test_update_array_row(py::module& m) {
    m.def("update_array_row", &carma::tests::test_update_array_row, "Test update_array");
}

void bind_test_update_array_col(py::module& m) {
    m.def("update_array_col", &carma::tests::test_update_array_col, "Test update_array");
}

void bind_test_mat_to_arr_plus_one(py::module& m) {
    m.def("mat_to_arr_plus_one", &carma::tests::test_mat_to_arr_plus_one, "Test mat_to_arr");
}

void bind_test_row_to_arr_plus_one(py::module& m) {
    m.def("row_to_arr_plus_one", &carma::tests::test_row_to_arr_plus_one, "Test row_to_arr");
}

void bind_test_col_to_arr_plus_one(py::module& m) {
    m.def("col_to_arr_plus_one", &carma::tests::test_col_to_arr_plus_one, "Test col_to_arr");
}

void bind_test_cube_to_arr_plus_one(py::module& m) {
    m.def("cube_to_arr_plus_one", &carma::tests::test_cube_to_arr_plus_one, "Test cube_to_arr");
}
