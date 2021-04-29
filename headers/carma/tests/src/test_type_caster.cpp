#include "test_type_caster.h"

namespace carma {
namespace tests {

double test_tc_in_mat(arma::Mat<double>& mat) {
    return arma::accu(mat);
}

double test_tc_in_row(arma::Row<double>& mat) {
    return arma::accu(mat);
}

double test_tc_in_col(arma::Col<double>& mat) {
    return arma::accu(mat);
}

double test_tc_in_cube(arma::Cube<double>& mat) {
    return arma::accu(mat);
}

arma::Mat<double> test_tc_out_mat(const py::array_t<double>& arr) {
    arma::Mat<double> ones = arma::ones(arr.shape(0), arr.shape(1));
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
    arma::Mat<double> out = ones + mat;
    return out;
}

arma::Mat<double> test_tc_out_mat_rvalue(const py::array_t<double>& arr) {
    arma::Mat<double> ones = arma::ones(arr.shape(0), arr.shape(1));
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
    return ones + mat;
}

arma::Row<double> test_tc_out_row(const py::array_t<double>& arr) {
    arma::Row<double> ones = arma::Row<double>(arr.size(), arma::fill::ones);
    arma::Row<double> mat = carma::arr_to_row<double>(arr);
    arma::Row<double> out = ones + mat;
    return out;
}

arma::Row<double> test_tc_out_row_rvalue(const py::array_t<double>& arr) {
    arma::Row<double> ones = arma::Row<double>(arr.size(), arma::fill::ones);
    arma::Row<double> mat = carma::arr_to_row<double>(arr);
    return ones + mat;
}

arma::Col<double> test_tc_out_col(const py::array_t<double>& arr) {
    arma::Col<double> ones = arma::Col<double>(arr.size(), arma::fill::ones);
    arma::Col<double> mat = carma::arr_to_col<double>(arr);
    arma::Col<double> out = ones + mat;
    return out;
}

arma::Col<double> test_tc_out_col_rvalue(const py::array_t<double>& arr) {
    arma::Col<double> ones = arma::Col<double>(arr.size(), arma::fill::ones);
    arma::Col<double> mat = carma::arr_to_col<double>(arr);
    return ones + mat;
}

arma::Cube<double> test_tc_out_cube(const py::array_t<double>& arr) {
    arma::Cube<double> ones = arma::Cube<double>(arr.shape(0), arr.shape(1), arr.shape(2), arma::fill::ones);
    arma::Cube<double> mat = carma::arr_to_cube<double>(arr);
    arma::Cube<double> out = ones + mat;
    return out;
}

arma::Cube<double> test_tc_out_cube_rvalue(const py::array_t<double>& arr) {
    arma::Cube<double> ones = arma::Cube<double>(arr.shape(0), arr.shape(1), arr.shape(2), arma::fill::ones);
    arma::Cube<double> mat = carma::arr_to_cube<double>(arr);
    return ones + mat;
}

}  // namespace tests
}  // namespace carma

void bind_test_tc_in_mat(py::module& m) {
    m.def("tc_in_mat", &carma::tests::test_tc_in_mat, "Test type caster");
}

void bind_test_tc_in_row(py::module& m) {
    m.def("tc_in_row", &carma::tests::test_tc_in_row, "Test type caster");
}

void bind_test_tc_in_col(py::module& m) {
    m.def("tc_in_col", &carma::tests::test_tc_in_col, "Test type caster");
}

void bind_test_tc_in_cube(py::module& m) {
    m.def("tc_in_cube", &carma::tests::test_tc_in_cube, "Test type caster");
}

void bind_test_tc_out_mat(py::module& m) {
    m.def("tc_out_mat", &carma::tests::test_tc_out_mat, "Test type caster");
}

void bind_test_tc_out_mat_rvalue(py::module& m) {
    m.def("tc_out_mat_rvalue", &carma::tests::test_tc_out_mat_rvalue, "Test type caster");
}

void bind_test_tc_out_row(py::module& m) {
    m.def("tc_out_row", &carma::tests::test_tc_out_row, "Test type caster");
}

void bind_test_tc_out_row_rvalue(py::module& m) {
    m.def("tc_out_row_rvalue", &carma::tests::test_tc_out_row_rvalue, "Test type caster");
}

void bind_test_tc_out_col(py::module& m) {
    m.def("tc_out_col", &carma::tests::test_tc_out_col, "Test type caster");
}

void bind_test_tc_out_col_rvalue(py::module& m) {
    m.def("tc_out_col_rvalue", &carma::tests::test_tc_out_col_rvalue, "Test type caster");
}

void bind_test_tc_out_cube(py::module& m) {
    m.def("tc_out_cube", &carma::tests::test_tc_out_cube, "Test type caster");
}

void bind_test_tc_out_cube_rvalue(py::module& m) {
    m.def("tc_out_cube_rvalue", &carma::tests::test_tc_out_cube_rvalue, "Test type caster");
}
