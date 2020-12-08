/*  carma/utils.h: Utility functions for arma converters
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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
namespace py = pybind11;

#ifndef ARMA_UTILS
#define ARMA_UTILS

namespace carma {

// Base template:
template <typename T>
struct is_convertible : std::false_type {};

// Specialisations:
template <typename T>
struct is_convertible<arma::Mat<T>> : std::true_type {};
template <typename T>
struct is_convertible<arma::Col<T>> : std::true_type {};
template <typename T>
struct is_convertible<arma::Row<T>> : std::true_type {};
template <typename T>
struct is_convertible<arma::Cube<T>> : std::true_type {};

template <typename T>
struct is_mat : std::false_type {};
template <typename T>
struct is_mat<arma::Mat<T>> : std::true_type {};

template <typename T>
struct is_col : std::false_type {};
template <typename T>
struct is_col<arma::Col<T>> : std::true_type {};

template <typename T>
struct is_row : std::false_type {};
template <typename T>
struct is_row<arma::Row<T>> : std::true_type {};

template <typename T>
struct is_cube : std::false_type {};
template <typename T>
struct is_cube<arma::Cube<T>> : std::true_type {};

enum class Deallocator { Undefined, Arma, Free, Delete, None };

// Not a struct to force all fields initialization
template <typename armaT>
class Data {
   public:
    Data(armaT* data, Deallocator deallocator) : data(data), deallocator(deallocator) {}
    armaT* data;
    Deallocator deallocator;
};

template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
inline Data<typename std::decay_t<typename armaT::elem_type>> copy_mem(armaT& src) {
    using T = typename armaT::elem_type;
    size_t N = src.n_elem;
    T* data = new T[N];
    std::memcpy(data, src.memptr(), sizeof(T) * N);
    return {data, Deallocator::Delete};
}

template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
inline Data<typename std::decay_t<typename armaT::elem_type>> steal_mem(armaT* src) {
    using T = typename armaT::elem_type;
    T* data = src->memptr();
    arma::access::rw(src->mem) = 0;
    return {data, Deallocator::Arma};
}

template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
inline Data<typename armaT::elem_type> get_data(armaT* src, bool copy) {
    using T = typename armaT::elem_type;
    if (copy) {
        size_t N = src->n_elem;
        T* data = new T[N];
        std::memcpy(data, src->memptr(), sizeof(T) * N);
        return {data, Deallocator::Delete};
    } else {
        T* data = src->memptr();
        arma::access::rw(src->mem) = 0;
        return {data, Deallocator::Arma};
    }
} /* get_data */

template <typename T>
inline py::capsule create_capsule(Data<T>& data) {
    /* Create a Python object that will free the allocated
     * memory when destroyed:
     */
    switch (data.deallocator) {
        case Deallocator::Arma:
            return py::capsule(data.data, [](void* f) {
                T* data = reinterpret_cast<T*>(f);
#ifndef NDEBUG
                // if in debug mode let us know what pointer is being freed
                std::cerr << "freeing copy memory @ " << f << std::endl;
#endif
                arma::memory::release(data);
            });
        case Deallocator::Free:
            return py::capsule(data.data, [](void* f) {
                T* data = reinterpret_cast<T*>(f);
#ifndef NDEBUG
                // if in debug mode let us know what pointer is being freed
                std::cerr << "freeing copy memory @ " << f << std::endl;
#endif
                free(data);
            });
        case Deallocator::Delete:
            return py::capsule(data.data, [](void* f) {
                T* data = reinterpret_cast<T*>(f);
#ifndef NDEBUG
                // if in debug mode let us know what pointer is being freed
                std::cerr << "freeing copy memory @ " << f << std::endl;
#endif
                delete[] data;
            });
        case Deallocator::Undefined:
            assert(false);
        case Deallocator::None:
        default:
            return py::capsule(data.data, [](void* f) {
                T* data = reinterpret_cast<T*>(f);
#ifndef NDEBUG
                // if in debug mode let us know what pointer is being freed
                std::cerr << "freeing copy memory @ " << f << std::endl;
#endif
            });
    }
} /* create_capsule */

template <typename T>
inline py::capsule create_dummy_capsule(T* data) {
    /* Create a Python object that will free the allocated
     * memory when destroyed:
     */
    return py::capsule(data, [](void* f) {
#ifndef NDEBUG
        // if in debug mode let us know what pointer is being freed
        std::cerr << "freeing memory @ " << f << std::endl;
#endif
    });
} /* create_dummy_capsule */

}  // namespace carma
#endif /* ARMA_UTILS */
