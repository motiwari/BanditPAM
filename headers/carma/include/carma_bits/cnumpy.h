/*  carma/cnumpy.h: Code to steal the memory from Numpy arrays
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */

#ifndef INCLUDE_CARMA_BITS_CNUMPY_H_
#define INCLUDE_CARMA_BITS_CNUMPY_H_
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION

/* C headers */
#include <Python.h>
#include <pymem.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

/* STD header */
#include <limits>
#include <iostream>
#include <algorithm>

#include <armadillo> // NOLINT

/* External */
#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT
#include <carma_bits/numpyapi.h> // NOLINT
#include <carma_bits/config.h> // NOLINT

namespace py = pybind11;

extern "C" {
/* well behaved is defined as:
 *   - aligned
 *   - writeable
 *   - Fortran contiguous (column major)
 *   - owndata (optional, on by default)
 * The last check can be disabled by setting `-DCARMA_DONT_REQUIRE_OWNDATA`
 */
static inline bool well_behaved(PyObject* src) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(src);
#if defined CARMA_DONT_REQUIRE_OWNDATA && defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
    return PyArray_CHKFLAGS(
        arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE
    );
#elif defined CARMA_DONT_REQUIRE_OWNDATA
    return PyArray_CHKFLAGS(
        arr,
        NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_F_CONTIGUOUS
    );
#elif defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
    return PyArray_CHKFLAGS(
        arr,
        NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_OWNDATA
    );
#else
    return PyArray_CHKFLAGS(
        arr,
        NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA
    );
#endif
}

/* well behaved is defined as:
 *   - aligned
 *   - writeable
 *   - Fortran contiguous (column major)
 *   - owndata (optional, on by default)
 * The last check can be disabled by setting `-DCARMA_DONT_REQUIRE_OWNDATA`
 */
static inline bool well_behaved_arr(PyArrayObject* arr) {
#if defined CARMA_DONT_REQUIRE_OWNDATA && defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
    return PyArray_CHKFLAGS(
        arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE
    );
#elif defined CARMA_DONT_REQUIRE_OWNDATA
    return PyArray_CHKFLAGS(
        arr,
        NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_F_CONTIGUOUS
    );
#elif defined CARMA_DONT_REQUIRE_F_CONTIGUOUS
    return PyArray_CHKFLAGS(
        arr,
        NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_OWNDATA
    );
#else
    return PyArray_CHKFLAGS(
        arr,
        NPY_ARRAY_ALIGNED| NPY_ARRAY_WRITEABLE | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA
    );
#endif
}
}  // extern "C"

namespace carma {

/* ---- steal_memory ----
 * The default behaviour is to turn off the owndata flag, numpy will no longer
 * free the allocated resources.
 * Benefit of this approach is that it's doesn't rely on deprecated access.
 * However, it can result in hard to detect bugs
 *
 * If CARMA_SOFT_STEAL is defined, the stolen array is replaced with an array
 * containing a single NaN and set the appropriate dimensions and strides.
 * This means the original references can be accessed but no longer should.
 *
 * Alternative is to define CARMA_HARD_STEAL which sets a nullptr and decreases
 * the reference count. NOTE, accessing the original reference when using
 * CARMA_HARD_STEAL will trigger a segfault.
 *
 * Note this function makes use of PyArrayObject_fields which is internal
 * and is noted with:
 *
 * "The main array object structure. It has been recommended to use the inline
 * functions defined below (PyArray_DATA and friends) to access fields here
 * for a number of releases. Direct access to the members themselves is
 * deprecated. To ensure that your code does not use deprecated access,
 * #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION (or
 * NPY_1_8_API_VERSION or higher as required).
 * This struct will be moved to a private header in a future release"
 */
template <typename T>
static inline void steal_memory(PyObject* src) {
#ifdef CARMA_EXTRA_DEBUG
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(src);
    std::cout << "\n-----------\nCARMA DEBUG\n-----------" << "\n";
    T* data = reinterpret_cast<T*>(PyArray_DATA(arr));
    std::cout << "Array with adress: " << data << "is being stolen" << "\n";
    int ndim = PyArray_NDIM(arr);
    npy_intp * dims = PyArray_DIMS(arr);
    bool first = true;
    std::cout << "the array has shape: ";
    for (int i = 0; i < ndim; i++) {
        std::cout << (first ? "(" : ", ") << dims[i];
        first = false;
    }
    std::cout << ")" << "\n";
    std::cout << "with first element: " << data[0] << "\n";
    std::cout << "-----------" << "\n";
#endif
#if defined CARMA_HARD_STEAL
    reinterpret_cast<PyArrayObject_fields *>(src)->data = nullptr;
#elif defined CARMA_SOFT_STEAL
    PyArrayObject_fields* obj = reinterpret_cast<PyArrayObject_fields *>(src);
    double* data = reinterpret_cast<double *>(
            carman::npy_api::get().PyDataMem_NEW_(sizeof(double))
    );
    if (data == NULL) throw std::bad_alloc();
    data[0] = NAN;
    obj->data = reinterpret_cast<char*>(data);

    size_t ndim = obj->nd;
    obj->nd = 1;
    if (ndim == 1) {
        obj->dimensions[0] = static_cast<npy_int>(1);
    } else if (ndim == 2) {
        obj->dimensions[0] = static_cast<npy_int>(1);
        obj->dimensions[1] = static_cast<npy_int>(0);
    } else {
        obj->dimensions[0] = static_cast<npy_int>(1);
        obj->dimensions[1] = static_cast<npy_int>(0);
        obj->dimensions[2] = static_cast<npy_int>(0);
    }
#else
    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(src), NPY_ARRAY_OWNDATA);
#endif
}  // steal_memory


/* Use Numpy's api to account for stride, order and steal the memory */

template <typename T>
inline static T* steal_copy_array(PyObject* obj) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(obj);
    auto& api = carman::npy_api::get();

    PyArray_Descr* dtype = PyArray_DESCR(src);
    // NewFromDescr steals a reference
    Py_INCREF(dtype);
    // dimension checks have been done prior so array should
    // not have more than 3 dimensions
    int ndim = PyArray_NDIM(src);
    npy_intp const* dims = PyArray_DIMS(src);

    // data will be freed by arma::memory::release<T>
    T* data = arma::memory::acquire<T>(api.PyArray_Size_(obj));
    if (data == NULL) throw std::bad_alloc();

    // build an PyArray to do F-order copy
    PyArrayObject* dest = reinterpret_cast<PyArrayObject*>(api.PyArray_NewFromDescr_(
        Py_TYPE(src),
        dtype,
        ndim,
        dims,
        NULL,
        data,
        NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_BEHAVED,
        NULL
    ));

    // copy the array to a well behaved F-order
    api.PyArray_CopyInto_(dest, src);

    // free the array
    PyArray_CLEARFLAGS(dest, NPY_ARRAY_OWNDATA);
    api.PyArray_Free_(dest, static_cast<void*>(nullptr));
    return data;
}  // steal_copy_array

}  // namespace carma

#endif  // INCLUDE_CARMA_BITS_CNUMPY_H_
