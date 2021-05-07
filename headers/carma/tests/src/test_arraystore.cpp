#include "test_arraystore.h"
#include <limits>

namespace carma {
namespace tests {

double test_ArrayStore_get_mat() {
    arma::mat mat_in = arma::ones(100, 1);
    ArrayStore<arma::mat> store = ArrayStore<arma::mat>(mat_in, true);
    arma::mat mat_out = store.mat;
    return std::fabs(arma::accu(mat_out) - arma::accu(mat_in));
} /* test_ArrayStore_get_mat */

double test_ArrayStore_get_mat_rvalue() {
    arma::mat mat_in = arma::ones(100, 1);
    arma::mat ref_mat = arma::mat(mat_in.memptr(), 100, 1);
    ArrayStore<arma::mat> store = ArrayStore<arma::mat>(std::move(mat_in));
    arma::mat mat_out = store.mat;
    return std::fabs(arma::accu(mat_out) - arma::accu(ref_mat));
} /* test_ArrayStore_get_mat */

py::array_t<double> test_ArrayStore_get_view(bool writeable) {
    arma::mat mat_in = arma::ones(100, 1);
    ArrayStore<arma::mat> store = ArrayStore<arma::mat>(mat_in, false);
    return store.get_view(writeable);
} /* test_ArrayStore_get_mat_const */

}  // namespace tests
}  // namespace carma

void bind_test_ArrayStore_get_mat(py::module& m) {
    m.def("test_ArrayStore_get_mat", &carma::tests::test_ArrayStore_get_mat, "Test ArrayStore");
}

void bind_test_ArrayStore_get_mat_rvalue(py::module& m) {
    m.def("test_ArrayStore_get_mat_rvalue", &carma::tests::test_ArrayStore_get_mat_rvalue, "Test ArrayStore");
}

void bind_test_ArrayStore_get_view(py::module& m) {
    m.def("test_ArrayStore_get_view", &carma::tests::test_ArrayStore_get_view, "Test ArrayStore");
}
