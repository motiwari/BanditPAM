#include <armadillo>
#include <carma>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

py::array_t<double> manual_example(py::array_t<double> & arr) {
    // convert to armadillo matrix without copying.
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);

    // normally you do something useful here ...
    arma::Mat<double> result = arma::Mat<double>(arr.shape(0), arr.shape(1), arma::fill::randu);

    // convert to Numpy array and return
    return carma::mat_to_arr(result);
}

PYBIND11_MODULE(example, m) {
    m.def(
        "manual_example",
        &manual_example,
        R"pbdoc(
            Example function for manual conversion.

            Parameters
            ----------
            mat : np.array
                input array
        )pbdoc",
        py::arg("arr")
    );
}
