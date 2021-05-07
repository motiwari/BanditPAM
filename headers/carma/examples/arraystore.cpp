#include "arraystore.h"

/* This is an example of a class where the data is stored
 * in C++. An example use-cae would be a regression where
 * you only return the underlying arrays when requested.
 *
 * Additional functionality exists for setting data directly
 * from a arma::Mat or retrieving the matrix.
 */

class ExampleClass {
    private:
        carma::ArrayStore<double> _x;
        carma::ArrayStore<double> _y;

    public:
        ExampleClass(py::array_t<double> & x, py::array_t<double> & y) :
        // steal the array, mark it mutable and store it as an
        // Armadillo array
        _x{carma::ArrayStore<double>(x, true)},
        // copy the array, mark it read-only and store it as an
        // Armadillo array
        _y{carma::ArrayStore<double>(y, false)} {}

        py::array_t<double> member_func() {
            // normallly you would something useful here
            _x.mat += _y.mat;
            // return mutable view off arma matrix
            return _x.get_view(true);
        }
};

void bind_exampleclass(py::module &m) {
    py::class_<ExampleClass>(m, "ExampleClass")
        .def(py::init<py::array_t<double> &, py::array_t<double> &>(), R"pbdoc(
            Initialise ExampleClass.

            Parameters
            ----------
            arr1: np.ndarray
                array to be stored in armadillo matrix
            arr2: np.ndarray
                array to be stored in armadillo matrix
        )pbdoc")
        .def("member_func", &ExampleClass::member_func, R"pbdoc(
            Compute ....
        )pbdoc");
}
