Introduction
############

CARMA provides fast bidirectional conversions between Numpy_ arrays and Armadillo_ matrices, vectors and cubes, much like RcppArmadillo_ does for R and Armadillo.

The library relies heavily on the impressive pybind11_ library and is largely inspired by their Eigen conversion albeit with a less conservative approach to memory management.
For details on pybind11_ and Armadillo_ refer to their respective documentation.

Installation
++++++++++++

`carma` is a header only library that relies on two other header only libraries, Armadillo and Pybind11.
Both libraries are linked as submodule in the `third_party` directory.

Considerations
++++++++++++++

In order to achieve fast conversions the default behaviour is avoid copying both from and to Numpy whenever possible and reasonable.
This allows very low overhead conversions but it impacts memory safety and requires user vigilance.

A second consideration is memory layout. Armadillo is optimised for column-major (Fortran order) memory whereas Numpy defaults to row-major (C order).
The default behaviour is to automatically convert, read copy, C-order arrays to F-order arrays upon conversion to Armadillo. Users should note that the library will not convert back to C-order when returning, this has consequences for matrices and cubes.

For details see the :doc:`Memory Management <memory_management>` section.

Compatibility
+++++++++++++

`carma` has been tested with:

* armadillo-9.800.1
* pybind11-2.4.3

The repository contains tests and CMake build instructions that can be used as an example.
For manual compilation see the :doc:`Usage section <basic_usage>`.

**Compiler requirements through pybind11**

1. Clang/LLVM 3.3 or newer (for Apple Xcode's clang, this is 5.0.0 or newer)
2. GCC 4.8 or newer
3. Microsoft Visual Studio 2015 Update 3 or newer
4. Intel C++ compiler 17 or newer
5. Cygwin/GCC (tested on 2.5.1)

About
#####

This project was created by Ralph Urlus.

License
+++++++

`carma` is provided under a Apache 2.0 license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

.. _numpy: https://numpy.org
.. _rcpparmadillo: https://github.com/RcppCore/RcppArmadillo
.. _pybind11: https://pybind11.readthedocs.io/en/stable/intro.html
.. _armadillo: http://arma.sourceforge.net/docs.html
