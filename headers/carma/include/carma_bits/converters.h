/*  carma/converters.h: Coverter of Numpy arrays and Armadillo matrices
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
#include <armadillo>  // NOLINT
#include <pybind11/buffer_info.h>  // NOLINT
#include <pybind11/detail/common.h>  // NOLINT
#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT

/* carma headers */
#include <carma_bits/utils.h>  // NOLINT
#include <carma_bits/cnumpy.h>  // NOLINT
#include <carma_bits/nparray.h>  // NOLINT
#include <carma_bits/numpytoarma.h>  // NOLINT
#include <carma_bits/armatonumpy.h>  // NOLINT
#include <carma_bits/config.h> // NOLINT

namespace py = pybind11;

#ifndef INCLUDE_CARMA_BITS_CONVERTERS_H_
#define INCLUDE_CARMA_BITS_CONVERTERS_H_

namespace carma {

/*****************************************************************************************
 *                                   Numpy to Armadillo                                   *
 *****************************************************************************************/

/* Convert numpy array to Armadillo Matrix with copy
 * If the array is 1D we create a column oriented matrix (N, 1)
 */
template <typename T>
arma::Mat<T> arr_to_mat(const py::array_t<T>& src) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_mat<T>(info);
    // copy and ensure fortran order
    data = steal_copy_array<T>(src.ptr());
    return p_arr_to_mat(info, data, true, false);
} /* arr_to_mat */

/* Convert numpy array to Armadillo Matrix by stealing the data
 *
 * We copy the array if:
 * - ndim == 2 && (optional) not F contiguous memory
 * - writeable is false
 * - owndata is false
 * - memory is not aligned
 * - NOTE if platform is windows
 *
 * If the array is 1D we create a column oriented matrix (N, 1)
 */
template <typename T>
arma::Mat<T> arr_to_mat(py::array_t<T>&& src) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_mat<T>(info);
    // steal memory and copy if needed
    data = steal_andor_copy(src.ptr(), data);
    return p_arr_to_mat(info, data, true, false);
} /* arr_to_mat */

/* Convert numpy array to Armadillo Matrix
 *
 * The default behaviour is to borrow the array, we copy if:
 * - copy is true
 * - ndim == 2 && (optional) not F contiguous memory
 * - writeable is false
 * - owndata is false
 * - memory is not aligned
 * Note that the user set behaviour is overridden is one of the above conditions
 * is true
 *
 * If the array is 1D we create a column oriented matrix (N, 1)
 */
template <typename T>
arma::Mat<T> arr_to_mat(py::array_t<T>& src, bool copy = false, bool strict = false) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_mat<T>(info);
    PyObject* obj = src.ptr();
#ifdef CARMA_EXTRA_DEBUG
    if (!well_behaved(obj)) {
        debug::print_copy_of_data<T>(data);
    }
#endif
    if (!well_behaved(obj) || copy) {
        // copy and ensure fortran order
        data = steal_copy_array<T>(obj);
        return p_arr_to_mat(info, data, true, strict);
    }
    return p_arr_to_mat(info, data, false, strict);
} /* arr_to_mat */


// #########################################################################
//                                   COL
// #########################################################################

/* Convert numpy array to Armadillo Column with copy */
template <typename T>
arma::Col<T> arr_to_col(const py::array_t<T>& src) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_col<T>(info);
    // copy and ensure fortran order
    data = steal_copy_array<T>(src.ptr());
    return p_arr_to_col(info, data, true, false);
} /* arr_to_col */

/* Convert numpy array to Armadillo Column by stealing the data
 *
 * We copy the array if:
 * - writeable is false
 * - owndata is false
 * - memory is not aligned
 * - NOTE if platform is windows
 */
template <typename T>
arma::Col<T> arr_to_col(py::array_t<T>&& src) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_col<T>(info);
    // steal memory and copy if needed
    data = steal_andor_copy(src.ptr(), data);
    return p_arr_to_col(info, data, true, false);
} /* arr_to_col */

/* Convert numpy array to Armadillo Col
 *
 * The default behaviour is to borrow the array, we copy if:
 * - copy is true
 * - writeable is false
 * - owndata is false
 * - memory is not aligned
 * Note that the user set behaviour is overridden is one of the above conditions
 * is true
 */
template <typename T>
arma::Col<T> arr_to_col(py::array_t<T>& src, bool copy = false, bool strict = false) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_col<T>(info);
    PyObject* obj = src.ptr();
#ifdef CARMA_EXTRA_DEBUG
    if (!well_behaved(obj)) {
        debug::print_copy_of_data<T>(data);
    }
#endif
    if (!well_behaved(obj) || copy) {
        // copy and ensure fortran order
        data = steal_copy_array<T>(obj);
        return p_arr_to_col(info, data, true, strict);
    }
    return p_arr_to_col(info, data, false, strict);
} /* arr_to_col */

// #########################################################################
//                                   ROW
// #########################################################################

/* Convert numpy array to Armadillo Row with copy */
template <typename T>
arma::Row<T> arr_to_row(const py::array_t<T>& src) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_row<T>(info);
    // copy and ensure fortran order
    data = steal_copy_array<T>(src.ptr());
    return p_arr_to_row(info, data, true, false);
} /* arr_to_row */

/* Convert numpy array to Armadillo Row by stealing the data
 *
 * We copy the array if:
 * - writeable is false
 * - owndata is false
 * - memory is not aligned
 * - NOTE if platform is windows
 */
template <typename T>
arma::Row<T> arr_to_row(py::array_t<T>&& src) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_row<T>(info);
    // steal memory and copy if needed
    data = steal_andor_copy(src.ptr(), data);
    return p_arr_to_row(src, info, data, true, false);
} /* arr_to_row */

/* Convert numpy array to Armadillo Row
 *
 * The default behaviour is to borrow the array, we copy if:
 * - copy is true
 * - writeable is false
 * - owndata is false
 * - memory is not aligned
 * Note that the user set behaviour is overridden is one of the above conditions
 * is true
 */
template <typename T>
arma::Row<T> arr_to_row(py::array_t<T>& src, bool copy = false, bool strict = false) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_row<T>(info);
    PyObject* obj = src.ptr();
#ifdef CARMA_EXTRA_DEBUG
    if (!well_behaved(obj)) {
        debug::print_copy_of_data<T>(data);
    }
#endif
    if (!well_behaved(obj) || copy) {
        // copy and ensure fortran order
        data = steal_copy_array<T>(obj);
        return p_arr_to_row(info, data, true, strict);
    }
    return p_arr_to_row(info, data, false, strict);
} /* arr_to_row */

// #########################################################################
//                                   Cube
// #########################################################################

/* Convert numpy array to Armadillo Cube with copy */
template <typename T>
arma::Cube<T> arr_to_cube(const py::array_t<T>& src) {

    py::buffer_info info = src.request();
    T* data = p_validate_from_array_cube<T>(info);
    // copy and ensure fortran order
    data = steal_copy_array<T>(src.ptr());
    return p_arr_to_cube(info, data, true, false);
} /* arr_to_cube */

/* Convert numpy array to Armadillo Cube by stealing
 *
 * We copy if:
 * - (optional) not F contiguous memory
 * - writeable is false
 * - owndata is false
 * - memory is not aligned
 * - NOTE if platform is windows
 * Note that the user set behaviour is overridden is one of the above conditions
 * is true
 */
template <typename T>
arma::Cube<T> arr_to_cube(py::array_t<T>&& src) {

    py::buffer_info info = src.request();
    T* data = p_validate_from_array_cube<T>(info);
    // steal memory and copy if needed
    data = steal_andor_copy(src.ptr(), data);
    return p_arr_to_cube(info, data, true, false);
} /* arr_to_cube */

/* Convert numpy array to Armadillo Cube
 *
 * The default behaviour is to borrow the array, we copy if:
 * - copy is true
 * - (optional) not F contiguous memory
 * - writeable is false
 * - owndata is false
 * - memory is not aligned
 * Note that the user set behaviour is overridden is one of the above conditions
 * is true
 */
template <typename T>
arma::Cube<T> arr_to_cube(py::array_t<T>& src, bool copy = false, bool strict = false) {
    py::buffer_info info = src.request();
    T* data = p_validate_from_array_cube<T>(info);
    PyObject* obj = src.ptr();
#ifdef CARMA_EXTRA_DEBUG
    if (!well_behaved(obj)) {
        debug::print_copy_of_data<T>(data);
    }
#endif
    if (!well_behaved(obj) || copy) {
        // copy and ensure fortran order
        data = steal_copy_array<T>(obj);
        return p_arr_to_cube(info, data, true, strict);
    }
    return p_arr_to_cube(info, data, false, strict);
} /* arr_to_cube */

// #########################################################################
//                                   TO_ARMA
// #########################################################################

/* The below functor approach is ported from:
 *     Arma_Wrapper - Paul Sangrey 2019
 *     Apache 2.0 License
 * This is a templated functor that has overloads that convert the various
 * types that I want to pass from Python to C++.
 */
template <typename returnT, typename SFINAE = std::true_type>
struct p_to_arma {
    static_assert(!SFINAE::value, "The general case is not defined.");
    template <typename innerT>
    static returnT from(innerT&&);
}; /* to_arma */

template <typename returnT>
struct p_to_arma<returnT, typename is_row<returnT>::type> {
    /* Overload concept on return type; convert to row */
    static returnT from(py::array_t<typename returnT::elem_type>& arr, int copy, int strict) {
        return arr_to_row<typename returnT::elem_type>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct p_to_arma<returnT, typename is_col<returnT>::type> {
    /* Overload concept on return type; convert to col */
    static returnT from(py::array_t<typename returnT::elem_type> arr, int copy, int strict) {
        return arr_to_col<typename returnT::elem_type>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct p_to_arma<returnT, typename is_mat<returnT>::type> {
    /* Overload concept on return type; convert to matrix */
    static returnT from(py::array_t<typename returnT::elem_type>& arr, int copy, int strict) {
        return arr_to_mat<typename returnT::elem_type>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct p_to_arma<returnT, typename is_cube<returnT>::type> {
    /* Overload concept on return type; convert to cube */
    static returnT from(py::array_t<typename returnT::elem_type>& arr, int copy, int strict) {
        return arr_to_cube<typename returnT::elem_type>(arr, copy, strict);
    }
}; /* to_arma */

/*****************************************************************************************
 *                                   Armadillo to Numpy                                   *
 *****************************************************************************************/

/* ######################################## Row ######################################## */

template <typename T>
inline py::array_t<T> row_to_arr(const arma::Row<T>& src) {
    /* Convert armadillo row to numpy array */
    arma::Row<T>* data = new arma::Row<T>(src);
    return p_construct_array<T>(data);
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>&& src) {
    /* Convert armadillo row to numpy array */
    arma::Row<T>* data = new arma::Row<T>(std::move(src));
    return p_construct_array<T>(data);
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>& src, int copy = false) {
    /* Convert armadillo row to numpy array */
    arma::Row<T>* data;
    if (!copy) {
        data = new arma::Row<T>(std::move(src));
    } else {
        data = new arma::Row<T>(src.memptr(), src.n_elem, true);
    }
    return p_construct_array<T>(data);
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>* src, int copy = false) {
    /* Convert armadillo row to numpy array */
    arma::Row<T>* data;
    if (!copy) {
        data = new arma::Row<T>(std::move(*src));
    } else {
        data = new arma::Row<T>(src->memptr(), src->n_elem, true);
    }
    return p_construct_array<T>(data);
} /* row_to_arr */

template <typename T>
inline void update_array(arma::Row<T>& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(src.n_elem)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Row<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arma::Row<T> row = std::move(src);
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(row.n_elem)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Row<T>* src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(src->n_elem)}, false);
} /* update_array */

/* ######################################## Col ######################################## */
template <typename T>
inline py::array_t<T> col_to_arr(const arma::Col<T>& src) {
    /* Convert armadillo col to numpy array */
    arma::Col<T>* data = new arma::Col<T>(src);
    return p_construct_array<T>(data);
} /* col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>&& src) {
    /* Convert armadillo col to numpy array */
    arma::Col<T>* data = new arma::Col<T>(std::move(src));
    return p_construct_array<T>(data);
} /* col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>& src, int copy = false) {
    /* Convert armadillo col to numpy array */
    arma::Col<T>* data;
    if (!copy) {
        data = new arma::Col<T>(std::move(src));
    } else {
        data = new arma::Col<T>(src.memptr(), src.n_elem, true);
    }
    return p_construct_array<T>(data);
} /* col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>* src, int copy = 0) {
    /* Convert armadillo col to numpy array */
    arma::Col<T>* data;
    if (!copy) {
        data = new arma::Col<T>(std::move(*src));
    } else {
        data = new arma::Col<T>(src->memptr(), src->n_elem, true);
    }
    return p_construct_array<T>(data);
} /* col_to_arr */

template <typename T>
inline void update_array(arma::Col<T>& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_elem), static_cast<ssize_t>(1)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Col<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arma::Col<T> col = std::move(src);
    arr.resize({static_cast<ssize_t>(col.n_elem), static_cast<ssize_t>(1)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Col<T>* src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src->n_elem), static_cast<ssize_t>(1)}, false);
} /* update_array */

/* ######################################## Mat ######################################## */
template <typename T>
inline py::array_t<T> mat_to_arr(const arma::Mat<T>& src) {
    return p_construct_array<T>(new arma::Mat<T>(src));
} /* mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>&& src) {
    return p_construct_array<T>(new arma::Mat<T>(std::move(src)));
} /* mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>& src, int copy = 0) {
    if (!copy) {
        return p_construct_array<T>(new arma::Mat<T>(std::move(src)));
    }
    return p_construct_array<T>(
        new arma::Mat<T>(src.memptr(), src.n_rows, src.n_cols, true)
    );
} /* mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>* src, int copy = 0) {
    if (!copy) {
        return p_construct_array<T>(new arma::Mat<T>(std::move(*src)));
    }
    return p_construct_array(
        new arma::Mat<T>(src->memptr(), src->n_rows, src->n_cols, true)
    );
} /* mat_to_arr */

template <typename T>
inline void update_array(arma::Mat<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arma::Mat<T> mat(std::move(src));
    arr.resize({static_cast<ssize_t>(mat.n_rows), static_cast<ssize_t>(mat.n_cols)}, false);
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
inline py::array_t<T> cube_to_arr(const arma::Cube<T>& src) {
    arma::Cube<T>* data = new arma::Cube<T>(src);
    return p_construct_array<T>(data);
} /* cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>&& src) {
    arma::Cube<T>* data = new arma::Cube<T>(std::move(src));
    return p_construct_array<T>(data);
} /* cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>& src, int copy = 0) {
    arma::Cube<T>* data;
    if (!copy) {
        data = new arma::Cube<T>(std::move(src));
    } else {
        data = new arma::Cube<T>(src.memptr(), src.n_rows, src.n_cols, src.n_slices, true);
    }
    return p_construct_array<T>(data);
} /* cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>* src, int copy = 0) {
    arma::Cube<T>* data;
    if (!copy) {
        data = new arma::Cube<T>(std::move(*src));
    } else {
        data = new arma::Cube<T>(src->memptr(), src->n_rows, src->n_cols, src->n_slices, true);
    }
    return p_construct_array<T>(data);
} /* cube_to_arr */

template <typename T>
inline void update_array(arma::Cube<T>&& src, py::array_t<T>& arr) {
    arma::Cube<T> cube = std::move(src);
    arr.resize(
        {static_cast<ssize_t>(cube.n_rows), static_cast<ssize_t>(cube.n_cols), static_cast<ssize_t>(cube.n_slices)},
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
template <typename armaT, typename T = typename armaT::elem_type, is_Cube<armaT> = 0>
inline py::array_t<T> to_numpy(const armaT& src) {
    arma::Cube<T>* data = new arma::Cube<T>(src);
    return p_construct_array<T>(data);
} /* cube_to_arr */

template <typename armaT, typename T = typename armaT::elem_type, is_Cube<armaT> = 0>
inline py::array_t<T> to_numpy(armaT&& src) {
    arma::Cube<T>* data = new arma::Cube<T>(std::forward<arma::Cube<T>>(src));
    return p_construct_array<T>(data);
} /* cube_to_arr */

template <typename armaT, typename T = typename armaT::elem_type, is_Cube<armaT> = 0>
inline py::array_t<T> to_numpy(armaT& src, int copy = 0) {
    arma::Cube<T>* data;
    if (!copy) {
        data = new arma::Cube<T>(std::move(src));
    } else {
        data = new arma::Cube<T>(src.memptr(), src.n_rows, src.n_cols, src.n_slices, true);
    }
    return p_construct_array<T>(data);
} /* cube_to_arr */

template <typename armaT, typename T = typename armaT::elem_type, is_Cube<armaT> = 0>
inline py::array_t<T> to_numpy(armaT* src, int copy = 0) {
    arma::Cube<T>* data;
    if (!copy) {
        data = new arma::Cube<T>(std::move(*src));
    } else {
        data = new arma::Cube<T>(src->memptr(), src->n_rows, src->n_cols, src->n_slices, true);
    }
    return p_construct_array<T>(data);
} /* cube_to_arr */

template <typename armaT, typename T = typename armaT::elem_type, is_Mat<armaT> = 1>
inline py::array_t<T> to_numpy(const armaT& src) {
    // use armadillo copy constructor
    armaT* data = new armaT(src);
    return p_construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Mat<armaT> = 1>
inline py::array_t<T> to_numpy(armaT&& src) {
    // steal mem
    armaT* data = new armaT(std::forward<armaT>(src));
    return p_construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Mat_only<armaT> = 2>
inline py::array_t<T> to_numpy(armaT& src, int copy = 0) {
    // if not copy we steal
    armaT* data;
    if (!copy) {
        data = new armaT(std::move(src));
    } else {
        data = new armaT(src.memptr(), src.n_rows, src.n_cols, true);
    }
    return p_construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Mat_only<armaT> = 2>
inline py::array_t<T> to_numpy(armaT* src, int copy = 0) {
    // if not copy we steal
    armaT* data;
    if (!copy) {
        data = new armaT(std::move(*src));
    } else {
        data = new armaT(src->memptr(), src->n_rows, src->n_cols, true);
    }
    return p_construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Vec<armaT> = 3>
inline py::array_t<T> to_numpy(armaT& src, int copy = 0) {
    // if not copy we steal
    armaT* data;
    if (!copy) {
        data = new armaT(std::move(src));
    } else {
        data = new armaT(src.memptr(), src.n_elem, true);
    }
    return p_construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Vec<armaT> = 3>
inline py::array_t<T> to_numpy(armaT* src, int copy = 0) {
    // if not copy we steal
    armaT* data;
    if (!copy) {
        data = new armaT(std::move(*src));
    } else {
        data = new armaT(src->memptr(), src->n_elem, true);
    }
    return p_construct_array<T>(data);
} /* to_numpy */

}  // namespace carma

namespace pybind11 {
namespace detail {

template <typename armaT>
struct type_caster<armaT, enable_if_t<carma::is_convertible<armaT>::value>> {
    using T = typename armaT::elem_type;

    /* Convert numpy array to Armadillo Matrix
     *
     * The default behaviour is to avoid copying, we copy if:
     * - ndim == 2 && (optional) not F contiguous memory
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     *
     * If the array is 1D we create a column oriented matrix (N, 1) */
    bool load(handle src, bool) {
        // set as array buffer
        //
        py::array_t<T> buffer = py::array_t<T>::ensure(src);
        if (!buffer) {
            throw carma::conversion_error("invalid object passed");
        }
        value = carma::p_to_arma<armaT>::from(buffer, 0, 1);
        return true;
    }

 private:
    // Cast implementation
    static handle cast_impl(armaT&& src, return_value_policy policy, handle) {
        switch (policy) {
            case return_value_policy::move:
                return carma::to_numpy<armaT>(std::move(src)).release();
            case return_value_policy::automatic:
                return carma::to_numpy<armaT>(std::move(src)).release();
            case return_value_policy::take_ownership:
                return carma::to_numpy<armaT>(std::move(src)).release();
            case return_value_policy::copy:
                return carma::to_numpy<armaT>(src, true).release();
            default:
                throw cast_error("unhandled return_value_policy");
        }
    }

    static handle cast_impl(armaT* src, return_value_policy policy, handle) {
        switch (policy) {
            case return_value_policy::move:
                return carma::to_numpy<armaT>(src).release();
            case return_value_policy::automatic:
                return carma::to_numpy<armaT>(src).release();
            case return_value_policy::take_ownership:
                return carma::to_numpy<armaT>(src).release();
            case return_value_policy::copy:
                return carma::to_numpy<armaT>(src, true).release();
            default:
                throw cast_error("unhandled return_value_policy");
        }
    }

 public:
    // Normal returned non-reference, non-const value: we steal
    static handle cast(armaT&& src, return_value_policy policy, handle parent) {
        return cast_impl(std::move(src), policy, parent);
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
    static handle cast(armaT* src, return_value_policy policy, handle parent) {
        return cast_impl(src, policy, parent);
    }
    // const pointer return; we copy
    static handle cast(const armaT* src, return_value_policy policy, handle parent) {
        policy = return_value_policy::copy;
        return cast_impl(src, policy, parent);
    }

    PYBIND11_TYPE_CASTER(armaT, _("Numpy.ndarray[") + npy_format_descriptor<T>::name + _("]"));
};
} /* namespace detail */
} /* namespace pybind11 */
#endif  // INCLUDE_CARMA_BITS_CONVERTERS_H_
