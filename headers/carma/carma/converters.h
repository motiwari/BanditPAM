/*  carma/carma.h: Coverter of Numpy arrays and Armadillo matrices
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 *
 *  Adapated from:
 *
 *      pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices
 *      Copyright (c) 2016 Wolf Vollprecht <w.vollprecht@gmail.com>
 *                         Wenzel Jakob <wenzel.jakob@epfl.ch>
 *      All rights reserved. Use of this source code is governed by a
 *      BSD-style license that can be found in the pybind11/LICENSE file.
 *
 *      arma_wrapper/arma_wrapper.h:
 *      Copyright (C) 2019 Paul Sangrey governed by Apache 2.0 License
 */
#include <memory>
#include <type_traits>
#include <utility>

/* External headers */
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>

/* carma headers */
#include <carma/carma/nparray.h>
#include <carma/carma/utils.h>

namespace py = pybind11;

#ifndef ARMA_CONVERTERS
#define ARMA_CONVERTERS

namespace carma {

/*****************************************************************************************
 *                                   Numpy to Armadillo                                   *
 *****************************************************************************************/
template <typename T>
arma::Mat<T> arr_to_mat(py::handle src, bool copy = false, bool strict = false) {
    /* Convert numpy array to Armadillo Matrix
     *
     * The default behaviour is to avoid copying, we copy if:
     * - ndim == 2 && not F contiguous memory
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     *
     * If the array is 1D we create a column oriented matrix (N, 1)
     */
    // set as array buffer
    py::array_t<T> buffer = py::array_t<T>::ensure(src);
    if (!buffer) {
        throw std::runtime_error("armadillo matrix conversion failed");
    }

    auto dims = buffer.ndim();
    if (dims < 1 || dims > 2) {
        throw std::runtime_error("Number of dimensions must be 1 <= ndim <= 2");
    }

    py::buffer_info info = buffer.request();
    if (info.ptr == nullptr) {
        throw std::runtime_error("armadillo matrix conversion failed, nullptr");
    }

    if (dims == 1) {
        if (requires_copy(buffer)) {
            copy = true;
            strict = false;
        }
        return arma::Mat<T>(static_cast<T*>(info.ptr), buffer.size(), 1, copy, strict);
    }

#ifdef CARMA_DONT_REQUIRE_F_CONTIGUOUS
    if (requires_copy(buffer)) {
        copy = false;
        strict = false;
    }
#else
    if (requires_copy(buffer) || !is_f_contiguous(buffer)) {
        // If not F-contiguous or writeable or numpy's data let pybind handle the copy
        buffer = py::array_t<T, py::array::f_style | py::array::forcecast>::ensure(src);
        info = buffer.request();
        copy = false;
        strict = false;
    }
#endif
    return arma::Mat<T>(static_cast<T*>(info.ptr), info.shape[0], info.shape[1], copy, strict);
} /* arr_to_mat */

template <typename T>
arma::Col<T> arr_to_col(py::handle src, bool copy = false, bool strict = false) {
    /* Convert numpy array to Armadillo Column
     *
     * The default behaviour is to avoid copying, we copy if:
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     */
    // set as array buffer
    py::array_t<T> buffer = py::array_t<T>::ensure(src);
    if (!buffer) {
        throw std::runtime_error("armadillo matrix conversion failed");
    }

    auto dims = buffer.ndim();
    if ((dims != 1) && (buffer.shape(1) != 1)) {
        throw std::runtime_error("Number of columns must <= 1");
    }

    py::buffer_info info = buffer.request();
    if (info.ptr == nullptr) {
        throw std::runtime_error("armadillo matrix conversion failed, nullptr");
    }

    if (requires_copy(buffer)) {
        copy = true;
        strict = false;
    }
    return arma::Col<T>(static_cast<T*>(info.ptr), buffer.size(), copy, strict);
} /* arr_to_col */

template <typename T>
arma::Row<T> arr_to_row(py::handle src, bool copy = false, bool strict = false) {
    /* Convert numpy array to Armadillo Row
     *
     * The default behaviour is to avoid copying, we copy if:
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     */
    // set as array buffer
    py::array_t<T> buffer = py::array_t<T>::ensure(src);
    if (!buffer) {
        throw std::runtime_error("armadillo matrix conversion failed");
    }

    auto dims = buffer.ndim();
    if ((dims != 1) && (buffer.shape(0) != 1)) {
        throw std::runtime_error("Number of rows must <= 1");
    }

    py::buffer_info info = buffer.request();
    if (info.ptr == nullptr) {
        throw std::runtime_error("armadillo matrix conversion failed, nullptr");
    }

    if (requires_copy(buffer)) {
        copy = true;
        strict = false;
    }
    return arma::Row<T>(static_cast<T*>(info.ptr), buffer.size(), copy, strict);
} /* arr_to_row */

template <typename T>
arma::Cube<T> arr_to_cube(py::handle src, bool copy = false, bool strict = false) {
    /* Convert numpy array to Armadillo Cube
     *
     * The default behaviour is to avoid copying, we copy if:
     * - ndim == 2 && not F contiguous memory
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     *
     */
    // set as array buffer
    py::array_t<T> buffer = py::array_t<T>::ensure(src);
    if (!buffer) {
        throw std::runtime_error("armadillo matrix conversion failed");
    }

    auto dims = buffer.ndim();
    if (dims != 3) {
        throw std::runtime_error("Number of dimensions must be 3");
    }

    py::buffer_info info = buffer.request();
    if (info.ptr == nullptr) {
        throw std::runtime_error("armadillo matrix conversion failed, nullptr");
    }

#ifdef CARMA_DONT_REQUIRE_F_CONTIGUOUS
    if (requires_copy(buffer)) {
        copy = false;
        strict = false;
    }
#else
    if (requires_copy(buffer) || !is_f_contiguous(buffer)) {
        // If not F-contiguous or writeable or numpy's data let pybind handle the copy
        buffer = py::array_t<T, py::array::f_style | py::array::forcecast>::ensure(src);
        info = buffer.request();
        copy = false;
        strict = false;
    }
#endif

    return arma::Cube<T>(static_cast<T*>(info.ptr), info.shape[0], info.shape[1], info.shape[2], copy, strict);
} /* arr_to_mat */

/* The below functor approach is ported from:
 *     Arma_Wrapper - Paul Sangrey 2019
 *     Apache 2.0 License
 * This is a templated functor that has overloads that convert the various
 * types that I want to pass from Python to C++.
 */
template <typename returnT, typename SFINAE = std::true_type>
struct _to_arma {
    static_assert(!SFINAE::value, "The general case is not defined.");
    template <typename innerT>
    static returnT from(innerT&&);
}; /* to_arma */

template <typename returnT>
struct _to_arma<returnT, typename is_row<returnT>::type> {
    /* Overload concept on return type; convert to row */
    template <typename T>
    static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
        return arr_to_row<T>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct _to_arma<returnT, typename is_col<returnT>::type> {
    /* Overload concept on return type; convert to col */
    template <typename T>
    static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
        return arr_to_col<T>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct _to_arma<returnT, typename is_mat<returnT>::type> {
    /* Overload concept on return type; convert to matrix */
    template <typename T>
    static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
        return arr_to_mat<T>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct _to_arma<returnT, typename is_cube<returnT>::type> {
    /* Overload concept on return type; convert to cube */
    template <typename T>
    static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
        return arr_to_cube<T>(arr, copy, strict);
    }
}; /* to_arma */

/*****************************************************************************************
 *                                   Armadillo to Numpy                                   *
 *****************************************************************************************/
template <typename T>
inline py::array_t<T> _row_to_arr(arma::Row<T>* src, bool copy) {
    /* Convert armadillo row to numpy array */
    ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t ncols = static_cast<ssize_t>(src->n_cols);

    auto data = get_data<arma::Row<T>>(src, copy);
    py::capsule base = create_capsule<T>(data);

    return py::array_t<T>(
        {static_cast<ssize_t>(1), ncols},  // shape
        {ncols * tsize, tsize},            // F-style contiguous strides
        data.data,                         // the data pointer
        base                               // numpy array references this parent
    );
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>&& src, bool copy = true) {
    /* Convert armadillo row to numpy array */
    return _row_to_arr<T>(&src, copy);
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>& src, bool copy = true) {
    /* Convert armadillo row to numpy array */
    return _row_to_arr<T>(&src, copy);
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>* src, bool copy = true) {
    /* Convert armadillo row to numpy array */
    return _row_to_arr<T>(src, copy);
} /* row_to_arr */

template <typename T>
inline void update_array(arma::Row<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(src.n_cols)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Row<T>& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(src.n_cols)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Row<T>* src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(src->n_cols)}, false);
} /* update_array */

/* ######################################## Col ######################################## */
template <typename T>
inline py::array_t<T> _col_to_arr(arma::Col<T>* src, bool copy) {
    /* Convert armadillo col to numpy array */
    ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(src->n_rows);

    auto data = get_data<arma::Col<T>>(src, copy);
    py::capsule base = create_capsule(data);

    return py::array_t<T>(
        {nrows, static_cast<ssize_t>(1)},  // shape
        {tsize, nrows * tsize},            // F-style contiguous strides
        data.data,                         // the data pointer
        base                               // numpy array references this parent
    );
} /* _col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>&& src, bool copy = true) {
    /* Convert armadillo col to numpy array */
    return _col_to_arr<T>(&src, copy);
} /* col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>& src, bool copy = true) {
    /* Convert armadillo col to numpy array */
    return _col_to_arr<T>(&src, copy);
} /* col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>* src, bool copy = true) {
    /* Convert armadillo col to numpy array */
    return _col_to_arr<T>(src, copy);
} /* col_to_arr */

template <typename T>
inline void update_array(arma::Col<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(1)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Col<T>& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(1)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Col<T>* src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src->n_rows), static_cast<ssize_t>(1)}, false);
} /* update_array */

/* ######################################## Mat ######################################## */
template <typename T>
inline py::array_t<T> _mat_to_arr(arma::Mat<T>* src, bool copy) {
    /* Convert armadillo matrix to numpy array */
    ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(src->n_rows);
    ssize_t ncols = static_cast<ssize_t>(src->n_cols);

    auto data = get_data<arma::Mat<T>>(src, copy);
    py::capsule base = create_capsule(data);

    return py::array_t<T>(
        {nrows, ncols},          // shape
        {tsize, nrows * tsize},  // F-style contiguous strides
        data.data,               // the data pointer
        base                     // numpy array references this parent
    );
} /* _mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>&& src, bool copy = false) {
    return _mat_to_arr<T>(&src, copy);
} /* mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>& src, bool copy = false) {
    return _mat_to_arr<T>(&src, copy);
} /* mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>* src, bool copy = false) {
    return _mat_to_arr<T>(src, copy);
} /* mat_to_arr */

template <typename T>
inline void update_array(arma::Mat<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Mat<T>& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Mat<T>* src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src->n_rows), static_cast<ssize_t>(src->n_cols)}, false);
} /* update_array */

/* ######################################## Cube ######################################## */
template <typename T>
inline py::array_t<T> _cube_to_arr(arma::Cube<T>* src, bool copy) {
    /* Convert armadillo matrix to numpy array */
    ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(src->n_rows);
    ssize_t ncols = static_cast<ssize_t>(src->n_cols);
    ssize_t nslices = static_cast<ssize_t>(src->n_slices);

    auto data = get_data<arma::Cube<T>>(src, copy);
    py::capsule base = create_capsule(data);

    return py::array_t<T>(
        {nslices, nrows, ncols},                        // shape
        {tsize * nrows * ncols, tsize, nrows * tsize},  // F-style contiguous strides
        data.data,                                      // the data pointer
        base                                            // numpy array references this parent
    );
} /* _cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>&& src, bool copy = true) {
    return _cube_to_arr<T>(&src, copy);
} /* cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>& src, bool copy = true) {
    return _cube_to_arr<T>(&src, copy);
} /* cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>* src, bool copy = true) {
    return _cube_to_arr<T>(src, copy);
} /* cube_to_arr */

template <typename T>
inline void update_array(arma::Cube<T>&& src, py::array_t<T>& arr) {
    arr.resize(
        {static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols), static_cast<ssize_t>(src.n_slices)},
        false);
} /* update_array */

template <typename T>
inline void update_array(arma::Cube<T>& src, py::array_t<T>& arr) {
    arr.resize(
        {static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols), static_cast<ssize_t>(src.n_slices)},
        false);
} /* update_array */

template <typename T>
inline void update_array(arma::Cube<T>* src, py::array_t<T>& arr) {
    arr.resize(
        {static_cast<ssize_t>(src->n_rows), static_cast<ssize_t>(src->n_cols), static_cast<ssize_t>(src->n_slices)},
        false);
} /* update_array */

/* ---------------------------------- to_numpy ---------------------------------- */
template <typename T>
inline py::array_t<T> to_numpy(arma::Mat<T>* src, bool copy = false) {
    return _mat_to_arr<T>(src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Mat<T>& src, bool copy = false) {
    return _mat_to_arr<T>(&src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Mat<T>&& src, bool copy = false) {
    return _mat_to_arr<T>(&src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Row<T>* src, bool copy = true) {
    return _row_to_arr<T>(src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Row<T>& src, bool copy = true) {
    return _row_to_arr<T>(&src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Row<T>&& src, bool copy = true) {
    return _row_to_arr<T>(&src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Col<T>* src, bool copy = true) {
    return _col_to_arr<T>(src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Col<T>& src, bool copy = true) {
    return _col_to_arr<T>(&src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Col<T>&& src, bool copy = true) {
    return _col_to_arr<T>(&src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Cube<T>* src, bool copy = true) {
    return _cube_to_arr<T>(src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Cube<T>& src, bool copy = true) {
    return _cube_to_arr<T>(&src, copy);
}

template <typename T>
inline py::array_t<T> to_numpy(arma::Cube<T>&& src, bool copy = true) {
    return _cube_to_arr<T>(&src, copy);
}

}  // namespace carma

namespace pybind11 {
namespace detail {

template <typename armaT>
struct type_caster<armaT, enable_if_t<carma::is_convertible<armaT>::value>> {
    using T = typename armaT::elem_type;

    /* Convert numpy array to Armadillo Matrix
     *
     * The default behaviour is to avoid copying, we copy if:
     * - ndim == 2 && not F contiguous memory
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     *
     * If the array is 1D we create a column oriented matrix (N, 1) */
    bool load(handle src, bool) {
        // set as array buffer
        bool copy = false;
        bool strict = true;

        py::array_t<T> buffer = py::array_t<T>::ensure(src);
        if (!buffer) {
            return false;
        }

        auto dims = buffer.ndim();
        if (dims < 1 || dims > 3) {
            return false;
        }

        py::buffer_info info = buffer.request();
        if (info.ptr == nullptr) {
            return false;
        }

        value = carma::_to_arma<armaT>::from(buffer, copy, strict);
        return true;
    }

   private:
    // Cast implementation
    template <typename CType>
    static handle cast_impl(CType* src, return_value_policy policy, handle) {
        switch (policy) {
            case return_value_policy::move:
                return carma::to_numpy<T>(src).release();
            case return_value_policy::automatic:
                return carma::to_numpy<T>(src).release();
            case return_value_policy::take_ownership:
                return carma::to_numpy<T>(src).release();
            case return_value_policy::copy:
                return carma::to_numpy<T>(src, true).release();
            default:
                throw cast_error("unhandled return_value_policy");
        };
    }

   public:
    // Normal returned non-reference, non-const value: we steal
    static handle cast(armaT&& src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }
    // If you return a non-reference const; we copy
    static handle cast(const armaT&& src, return_value_policy policy, handle parent) {
        policy = return_value_policy::copy;
        return cast_impl(&src, policy, parent);
    }
    // lvalue reference return; default (automatic) becomes steal
    static handle cast(armaT& src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }
    // const lvalue reference return; default (automatic) becomes copy
    static handle cast(const armaT& src, return_value_policy policy, handle parent) {
        policy = return_value_policy::copy;
        return cast_impl(&src, policy, parent);
    }
    // non-const pointer return; we steal
    static handle cast(armaT* src, return_value_policy policy, handle parent) { return cast_impl(src, policy, parent); }
    // const pointer return; we copy
    static handle cast(const armaT* src, return_value_policy policy, handle parent) {
        policy = return_value_policy::copy;
        return cast_impl(src, policy, parent);
    }

    PYBIND11_TYPE_CASTER(armaT, _("Numpy.ndarray[") + npy_format_descriptor<T>::name + _("]"));
};
} /* namespace detail */
} /* namespace pybind11 */
#endif /* ARMA_CONVERTERS */
