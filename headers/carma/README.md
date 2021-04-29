<div align="center">
  <img src="docs/source/carma_logo_small.png" alt="carma_logo"/>
</div>

<br/>

<div align="center">
  <!-- Build status -->
  <a href="https://travis-ci.com/RUrlus/carma">
    <img src="https://img.shields.io/travis/rurlus/carma/master.svg?style=for-the-badge" alt="Build Status"/>
  </a>
  <!-- Coverage status -->
  <a href="https://coveralls.io/github/RUrlus/carma?branch=stable">
    <img src="https://img.shields.io/coveralls/github/RUrlus/carma?style=for-the-badge" alt="Coveralls github" >
  </a>
  <!-- Documentation status -->
  <a href="https://carma.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/carma/badge/?version=latest&style=for-the-badge" alt="Documentation Status" />
  </a>
  <!-- Release version -->
  <a href="https://github.com/RUrlus/carma/releases">
    <img src="https://img.shields.io/github/v/release/RUrlus/carma.svg?style=for-the-badge" alt="Release Version" />
  </a>
  <!-- License -->
  <a href="https://github.com/RUrlus/carma/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/RUrlus/carma.svg?style=for-the-badge" alt="license"/>
  </a>
</div>

<br/>

CARMA is a header only library providing conversions between Numpy arrays and Armadillo matrices. Examples and reference documentation can be found at [carma.readthedocs.io](https://carma.readthedocs.io/).

## Introduction

CARMA provides fast bidirectional conversions between [Numpy](https://numpy.org) arrays and [Armadillo](http://arma.sourceforge.net/docs.html) matrices, vectors and cubes, much like [RcppArmadillo](https://github.com/RcppCore/RcppArmadillo) does for R and Armadillo.

The library relies heavily on the impressive [pybind11](https://pybind11.readthedocs.io/en/stable/intro.html) library and is largely inspired by their Eigen conversion albeit with a less conservative approach to memory management.
For details on pybind11 and Armadillo refer to their respective documentation.

### Installation

Carma is a header only library that relies on two other header only libraries, Armadillo and Pybind11.
Both libraries are linked as submodule in the `third_party` directory.

### Considerations

In order to achieve fast conversions the default behaviour is avoid copying both from and to Numpy whenever possible and reasonable.
This allows very low overhead conversions but it impacts memory safety and requires user vigilance.

A second consideration is memory layout. Armadillo is optimised for column-major (Fortran order) memory whereas Numpy defaults to row-major (C order).
The default behaviour is to automatically convert, read copy, C-order arrays to F-order arrays upon conversion to Armadillo. Users should note that the library will not convert back to C-order when returning, this has consequences for matrices and cubes.

For details see the documentation section Memory Management.

### Examples

On a high level Carma provides three ways to work Numpy arrays in Armadillo:

#### Manual conversion


The easiest way to use Carma is manual conversion, it gives you the most control over when to copy or not.
You pass a Numpy array as an argument and/or as the return type and call the respective conversion function.

```cpp
#include <armadillo>
#include <carma/carma.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

py::array_t<double> manual_example(py::array_t<double> & arr) {
    // convert to armadillo matrix without copying.
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);

    // normally you do something useful here ...
    arma::Mat<double> result = arma::Mat<double>(
        arr.shape(0),
        arr.shape(1),
        arma::fill::randu
    );

    // convert to Numpy array and return
    return carma::mat_to_arr(result);
}
```

#### Update array

```cpp
#include <armadillo>
#include <carma/carma.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void update_example(py::array_t<double> & arr) {
    // convert to armadillo matrix without copying.
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);

    // normally you do something useful here with mat ...
    mat += arma::Mat<double>(arr.shape(0), arr.shape(1), arma::fill::randu);

    // update Numpy array buffer
    carma::update_array(mat, arr);
}
```

#### Automatic conversion

```cpp
#include <armadillo>
#include <carma/carma.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

arma::Mat<double> automatic_example(arma::Mat<double> & mat) {
    // normally you do something useful here with mat ...
    arma::Mat<double> rand = arma::Mat<double>(mat.n_rows, mat.n_cols, arma::fill::randu);

    arma::Mat<double> result = mat + rand;
    // type caster will take care of casting `result` to a Numpy array.
    return result;
}
```

The repository contains tests, examples and CMake build instructions that can be used as an reference.
For manual compilation see the documentation section Usage.

### Compatibility

`carma` has been tested with:

* armadillo-9.800.1
* pybind11-2.4.3

The repository contains tests, examples and CMake build instructions that can be used as an reference.
For manual compilation see the documentation section Usage.

**Compiler requirements through pybind11**

1. Clang/LLVM 3.3 or newer (for Apple Xcode's clang, this is 5.0.0 or newer)
2. GCC 4.8 or newer
3. Microsoft Visual Studio 2015 Update 3 or newer
4. Intel C++ compiler 17 or newer
5. Cygwin/GCC (tested on 2.5.1)

### About

This project was created by Ralph Urlus. Significant improvements to the project have been contributed by [Pascal H.](https://github.com/hpwxf)

### License

carma is provided under a Apache 2.0 license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.
