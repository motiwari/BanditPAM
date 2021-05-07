#include "automatic_conversion.h"

arma::Mat<double> automatic_example(arma::Mat<double> & mat) {
    // normally you do something useful here with mat ...
    arma::Mat<double> rand = arma::Mat<double>(mat.n_rows, mat.n_cols, arma::fill::randu);

    arma::Mat<double> result = mat + rand;
    // type caster will take care of casting `result` to a Numpy array.
    return result;
}

// Create binding, see pybind11 documentation for details
void bind_automatic_example(py::module &m) {
    m.def(
        "automatic_example",
        &automatic_example,
        R"pbdoc(
            Example function for automatic conversion.

            Parameters
            ----------
            mat : np.array
                input array

            Returns
            -------
            result : np.array
                output array
        )pbdoc",
        py::arg("mat")
    );
}
