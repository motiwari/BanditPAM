/*  carma/nparray.h: Condition checks numpy arrays
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 *
 *  Adapated from:
 *      pybind11/numpy.h: Basic NumPy support, vectorize() wrapper
 *
 *      Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>
 *      All rights reserved. Use of this source code is governed by a
 *      BSD-style license that can be found in the LICENSE file.
 */

/* External headers */
#include <memory>
#include <type_traits>
#include <utility>

#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT
#include <carma_bits/cnumpy.h> // NOLINT
#include <carma_bits/config.h> // NOLINT
namespace py = pybind11;

#ifndef INCLUDE_CARMA_BITS_NPARRAY_H_
#define INCLUDE_CARMA_BITS_NPARRAY_H_

namespace carma {

template <typename T>
inline bool is_f_contiguous(const py::array_t<T>& arr) {
    return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_);
}

template <typename T>
inline bool is_c_contiguous(const py::array_t<T>& arr) {
    return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_);
}

template <typename T>
inline bool is_c_contiguous_2d(const py::array_t<T>& arr) {
    if (arr.ndim() > 1) {
        return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_);
    }
    return false;
}

template <typename T>
inline bool is_contiguous(const py::array_t<T>& arr) {
    return is_f_contiguous(arr) || is_c_contiguous(arr);
}

template <typename T>
inline bool is_writeable(const py::array_t<T>& arr) {
    return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_WRITEABLE_);
}

template <typename T>
inline bool is_owndata(const py::array_t<T>& arr) {
    return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_OWNDATA_);
}

template <typename T>
inline bool is_aligned(const py::array_t<T>& arr) {
    return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_ALIGNED_);
}

template <typename T>
inline bool is_well_behaved(const py::array_t<T>& arr) {
    return well_behaved(arr.ptr());
}

template <typename T>
inline void set_owndata(py::array_t<T>& arr) {
    py::detail::array_proxy(arr.ptr())->flags &= py::detail::npy_api::NPY_ARRAY_OWNDATA_;
}

template <typename T>
inline void set_not_owndata(py::array_t<T>& arr) {
    py::detail::array_proxy(arr.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_OWNDATA_;
}

template <typename T>
inline void set_writeable(py::array_t<T>& arr) {
    py::detail::array_proxy(arr.ptr())->flags &= py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
}

template <typename T>
inline void set_not_writeable(py::array_t<T>& arr) {
    py::detail::array_proxy(arr.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
}

template <typename T>
inline void set_f_contiguous(py::array_t<T>& arr) {
    py::detail::array_proxy(arr.ptr())->flags &= py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_;
}

template <typename T>
inline void set_c_contiguous(py::array_t<T>& arr) {
    py::detail::array_proxy(arr.ptr())->flags &= py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_;
}

}  // namespace carma

#endif  // INCLUDE_CARMA_BITS_NPARRAY_H_
