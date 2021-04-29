#include "test_nparray.h"

void bind_test_is_f_contiguous(py::module& m) {
    m.def(
        "is_f_contiguous",
        [](const py::array_t<double>& arr) { return carma::is_f_contiguous(arr); },
        "Test is F contiguous");
}

void bind_test_is_c_contiguous(py::module& m) {
    m.def(
        "is_c_contiguous",
        [](const py::array_t<double>& arr) { return carma::is_c_contiguous(arr); },
        "Test is C contiguous");
}

void bind_test_is_writeable(py::module& m) {
    m.def(
        "is_writeable", [](const py::array_t<double>& arr) { return carma::is_writeable(arr); }, "Test is writeable");
}

void bind_test_is_owndata(py::module& m) {
    m.def(
        "is_owndata", [](const py::array_t<double>& arr) { return carma::is_owndata(arr); }, "Test is owndata");
}

void bind_test_is_aligned(py::module& m) {
    m.def(
        "is_aligned", [](const py::array_t<double>& arr) { return carma::is_aligned(arr); }, "Test is aligned");
}

void bind_test_set_not_owndata(py::module& m) {
    m.def(
        "set_not_owndata", [](py::array_t<double>& arr) { carma::set_not_owndata(arr); }, "Test is set_not_owndata");
}

void bind_test_set_not_writeable(py::module& m) {
    m.def(
        "set_not_writeable",
        [](py::array_t<double>& arr) { carma::set_not_writeable(arr); },
        "Test is set_not_writeable");
}

void bind_test_is_well_behaved(py::module& m) {
    m.def(
        "is_well_behaved",
        [](const py::array_t<double>& arr) { return carma::is_well_behaved(arr); },
        "Test is_well_behaved"
    );
}
