/*  carma/armatonumpy.h: Coverter of Armadillo matrices to numpy arrays
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
#include <utility>

/* External headers */
#include <armadillo>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT
#include <pybind11/numpy.h>  // NOLINT
#include <carma_bits/config.h> // NOLINT

namespace py = pybind11;

#ifndef INCLUDE_CARMA_BITS_ARMATONUMPY_H_
#define INCLUDE_CARMA_BITS_ARMATONUMPY_H_

namespace carma {

template <typename armaT>
inline py::capsule create_capsule(armaT* data) {
    return py::capsule(data, [](void* f) {
        armaT* mat = reinterpret_cast<armaT*>(f);
#ifdef CARMA_EXTRA_DEBUG
        std::cout << "\n-----------\nCARMA DEBUG\n-----------" << "\n";
        // if in debug mode let us know what pointer is being freed
        std::cerr << "Freeing memory @" << mat->memptr() << std::endl;
        std::cout << "-----------" << "\n";
#endif
        delete mat;
    });
} /* create_capsule */

template <typename T>
inline py::capsule create_dummy_capsule(T* data) {
    return py::capsule(data, [](void* f) {
#ifdef CARMA_EXTRA_DEBUG
        std::cout << "\n-----------\nCARMA DEBUG\n-----------" << "\n";
        // if in debug mode let us know what pointer is being freed
        std::cerr << "Destructing view on memory @" << f << std::endl;
        std::cout << "-----------" << "\n";
#endif
    });
} /* create_capsule */


template <typename T>
inline py::array_t<T> p_construct_array(arma::Row<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);

    py::capsule base = create_capsule<arma::Row<T>>(data);

    return py::array_t<T>(
        {static_cast<ssize_t>(1), ncols},  // shape
        {tsize, tsize},                    // F-style contiguous strides
        data->memptr(),                    // the data pointer
        base                               // numpy array references this parent
    );
} /* p_construct_array */

template <typename T>
inline py::array_t<T> p_construct_array(arma::Col<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);

    py::capsule base = create_capsule<arma::Col<T>>(data);

    return py::array_t<T>(
        {nrows, static_cast<ssize_t>(1)},  // shape
        {tsize, nrows * tsize},            // F-style contiguous strides
        data->memptr(),                    // the data pointer
        base                               // numpy array references this parent
    );
} /* p_construct_array */

template <typename T>
inline py::array_t<T> p_construct_array(arma::Mat<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);

    py::capsule base = create_capsule<arma::Mat<T>>(data);

    return py::array_t<T>(
        {nrows, ncols},          // shape
        {tsize, nrows * tsize},  // F-style contiguous strides
        data->memptr(),          // the data pointer
        base                     // numpy array references this parent
    );
} /* p_construct_array */

template <typename T>
inline py::array_t<T> p_construct_array(arma::Cube<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);
    ssize_t nslices = static_cast<ssize_t>(data->n_slices);

    py::capsule base = create_capsule<arma::Cube<T>>(data);

    return py::array_t<T>(
        {nslices, nrows, ncols},                        // shape
        {tsize * nrows * ncols, tsize, nrows * tsize},  // F-style contiguous strides
        data->memptr(),                                 // the data pointer
        base                                            // numpy array references this parent
    );
} /* p_construct_array */

}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_ARMATONUMPY_H_
