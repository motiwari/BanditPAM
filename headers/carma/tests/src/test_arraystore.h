#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <carma>

#include <string>

namespace py = pybind11;

#ifndef TESTS_SRC_TEST_ARRAYSTORE_H_
#define TESTS_SRC_TEST_ARRAYSTORE_H_

void bind_test_ArrayStore_get_mat(py::module& m);
void bind_test_ArrayStore_get_mat_rvalue(py::module& m);
void bind_test_ArrayStore_get_view(py::module& m);

template <typename armaT>
void bind_ArrayStore(py::module& m, std::string&& typestr) {
    using T = typename armaT::elem_type;
    using Class = carma::ArrayStore<armaT>;
    std::string pyclass_name = typestr + std::string("ArrayStore");
    py::class_<Class>(m, pyclass_name.c_str())
        .def(py::init<py::array_t<T>&, bool>(), R"pbdoc(
            Initialise ArrayStore class.

            The class store Numpy arrays as Armadillo matrices.
            This class is intended to inherited from or used
            as attribute for another class.

            Parameters
            ----------
            arr: np.ndarray
                array to be stored in armadillo matrix
            copy : bool
                if false steal the memory of the numpy array,
                note that the previous array should
                no longer be used.
        )pbdoc")
        .def("get_view", &Class::get_view, R"pbdoc(
            Get view of matrix as numpy array.

            Parameters
            ----------
            writeable : bool
                mark view as read-only if True

            Returns
            -------
            np.ndarray
                view on armadillo matrix

            Raises
            ------
            RuntimeError : if writeable is True
            but writeable was set to false as initialization
        )pbdoc")
        .def("set_array", &Class::set_array, R"pbdoc(
            Store numpy array in armadillo matrix.

            Parameters
            ----------
            arr : np.ndarray
                array to be stored in armadillo matrix
        )pbdoc");
}

#endif  // TESTS_SRC_TEST_ARRAYSTORE_H_
