#include <carma/carma/converters.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

#ifndef CARMA_ARRAYSTORE
#define CARMA_ARRAYSTORE

namespace carma {

template <typename T>
class ArrayStore {
   protected:
    constexpr static ssize_t tsize = sizeof(T);
    bool _steal;
    T* _ptr;
    py::capsule _base;

    void _convert_to_arma(py::array_t<T>& arr) {
        if (_steal) {
            mat = arr_to_mat<T>(arr, false);
            _ptr = mat.memptr();
            Data<T> data{_ptr, Deallocator::Free};
            _base = create_capsule(data);
            // inform numpy it no longer owns the data
            set_not_owndata(arr);
        } else {
            mat = arr_to_mat<T>(arr, true);
            _ptr = mat.memptr();
            // create a dummy capsule as armadillo will be responsible
            // for destruction of the memory
            // We need a capsule to prevent a copy on the way out.
            _base = create_dummy_capsule(_ptr);
        }
    }

   public:
    arma::Mat<T> mat;

    ArrayStore(py::array_t<T>& arr, bool steal) : _steal{steal} {
        /* Constructor
         *
         * Takes numpy array and converters to Armadillo matrix.
         * If the array should be stolen we set owndata false for
         * numpy array.
         *
         * We store a capsule to serve as a reference for the
         * views on the data
         *
         */
        _convert_to_arma(arr);
    }

    ArrayStore(arma::Mat<T>& mat) : _steal{false}, _ptr{mat.memptr()}, _base{create_dummy_capsule(_ptr)}, mat{mat} {}

    ArrayStore(arma::Mat<T>&& mat)
        : _steal{true}, _ptr{mat.memptr()}, _base{create_dummy_capsule(_ptr)}, mat{std::move(mat)} {}

    void set_data(py::array_t<T>& arr, bool steal) {
        _steal = steal;
        _convert_to_arma(arr);
    }

    void set_mat(arma::Mat<T>& src) {
        _steal = false;
        _ptr = mat.memptr();
        _base = create_dummy_capsule(_ptr);
        mat = src;
    }

    void set_mat(arma::Mat<T>&& src) {
        _steal = true;
        _ptr = mat.memptr();
        _base = create_dummy_capsule(_ptr);
        mat = std::move(src);
    }

    py::array_t<T> get_view(bool writeable) {
        ssize_t nrows = static_cast<ssize_t>(mat.n_rows);
        ssize_t ncols = static_cast<ssize_t>(mat.n_cols);

        // create the array
        py::array_t<T> arr = py::array_t<T>(
            {nrows, ncols},          // shape
            {tsize, nrows * tsize},  // F-style contiguous strides
            _ptr,                    // the data pointer
            _base                    // numpy array references this parent
        );

        // inform numpy it does not own the buffer
        set_not_owndata(arr);

        if (!writeable)
            set_not_writeable(arr);
        return arr;
    }
};

} /* namespace carma */

#endif /* CARMA_ARRAYSTORE */
