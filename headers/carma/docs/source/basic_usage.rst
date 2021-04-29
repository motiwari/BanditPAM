.. role:: bash(code)
   :language: bash

First steps
###########

Carma relies on Pybind11 for the generation of the bindings and casting of the arguments from Python to C++.
Make sure you are familiar with `Pybind11 <https://pybind11.readthedocs.io/en/stable/intro.html>`__ before continuing on.

Requirements
************

Before starting you should check that your environment is set up properly.

Carma has two requirements:

* `Pybind11 <https://github.com/pybind/pybind11>`__
* `Armadillo <http://arma.sourceforge.net/download.html>`__

Carma provides both tests and examples that can be compiled without any additional libraries, although you will need additional libraries to use Armadillo in practice.

.. note:: The Pybind11 and Armadillo libraries are linked with `carma` as submodule in `third_party` directory. To get them using :bash:`git clone`, don't forget to use :bash:`--recursive` option.

Manual compilation
******************

Although using a build system is suggested, the :bash:`examples/example.cpp` can be compiled with:

**on Linux**

.. code-block:: bash
    
    c++ -O3 -Wall -shared -std=c++14 -fPIC -larmadillo `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

**on MacOS**

.. code-block:: bash
    
    c++ -O3 -Wall -shared -std=c++14 -larmadillo -undefined dynamic_lookup `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

Manual compilation requires that Pybind11 and Armadillo are discoverable.

Build systems 
*************

The tests and examples can be compiled using CMake.
CMake can be installed with :bash:`pip install cmake`, your package manager or directly from `cmake <http://cmake.org/download/>`__.

.. code-block:: bash

   git submodule update --init
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=. -DBUILD_EXAMPLES=true  -DBUILD_TESTS=true .. && make install

To run the tests you need to install `pytest`:

.. code-block:: bash

   pip install pytest

and run:

.. code-block:: bash

   ctest

To install `carma`, you have to define 

.. code-block:: bash
    
    -DCMAKE_INSTALL_PREFIX=/installation/path/directory

(default value is ``/usr/local``)

Installation directory contains

.. code-block::

    include   # carma headers
    tests     # carma python tests with python module (if enabled using -DBUILD_TESTS=on)
    examples  # carma python examples with python module (if enabled using -DBUILD_EXAMPLES=on)

Advanced build system configuration
***********************************

`Carma` requirements can be provided out of :bash:`third_party` directory.

To do so, you have to define locations of `armadillo` or/and `pybind11` by setting:

.. code-block:: bash
    
    -DARMADILLO_ROOT_DIR=/path/to/armadillo-code/root/directory

.. code-block:: bash
    
    -DPYBIND11_ROOT_DIR=/path/to/pybind11/root/directory

Sometimes, if you have multiple python interpret available in your system, 
you may want to specify the one you want. Python detection is delegated to pybind11 dependency 
and you can drive it using

.. code-block:: bash

    -DPYTHON_PREFIX_PATH=/path/to/directory/containing/your/favorite/python/interpret
    -DPYBIND11_PYTHON_VERSION=/version/of/your/favorite/python/interpret

e.g.:

.. code-block:: bash

    -DPYTHON_PREFIX_PATH=/usr/bin
    -DPYBIND11_PYTHON_VERSION=3.7

Carma as an embedded CMake project 
++++++++++++++++++++++++++++++++++

You can embed `Carma` using CMake command

.. code-block::

    add_subdirectory(/path/to/carma/root/directory)

If you do so, you can use :code:`ARMADILLO_ROOT_DIR` and :code:`PYBIND11_ROOT_DIR` to define requirements 
(as CMake variables in main project).

Nevertheless, if :code:`armadillo` or/and :code:`pybind11` CMake targets already exist, `carma` will use them 
(to avoid conflict with already existing targets in your main project).       

Moreover, it could be useful to define 

.. code-block::

    set(CARMA_DEV_TARGET false)

to disable carma development targets (e.g. ``clang-format``).

Examples
########

On a high level `carma` provides four ways to work with Numpy arrays and Armadillo:
See the :doc:`Function specifications <carma>` section for details about the available functions and the examples directory for runnable examples.

Manual conversion
*****************

The easiest way to use `carma` is manual conversion, it gives you the most control over when to copy or not.
You pass a Numpy array as an argument and/or as the return type and call the respective conversion function.

.. warning:: Carma will avoid copying by default so make sure not to return the memory of the input array without copying or use `update_array`.

.. code-block:: c++

    #include <armadillo>
    #include <carma/carma.h>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    py::array_t<double> manual_example(py::array_t<double> & arr) {
        // convert to armadillo matrix without copying.
        arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
    
        // normally you do something useful here ...
        arma::Mat<double> result = arma::Mat<double>(arr.shape(0), arr.shape(1), arma::fill::randu);
    
        // convert to Numpy array and return
        return carma::mat_to_arr(result);
    }

Update array
************

.. code-block:: c++

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

Transfer ownership
******************

If you want to transfer ownership to the C++ side you can use:

.. code-block:: c++

    #include <armadillo>
    #include <carma/carma.h>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    arma::Mat<double> steal_array(py::array_t<double> & arr) {
        // convert to armadillo matrix
        arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
        // inform numpy it no longer owns the data
        carma::set_not_owndata<double>(arr);
        return mat;
    }

    py::array_t<double> numpy_view(arma::Mat<double> & mat) {
        /* Return view on the buffer */
        py::array_t<double> arr = carma::mat_to_arr<double>(mat);
        // inform numpy it that it doesn't own the data
        carma::set_not_owndata<double>(arr)
        return arr;
    }

    py::array_t<double> numpy_view(const arma::Mat<double> & mat) {
        /* Return read only view on the buffer */
        py::array_t<double> arr = carma::mat_to_arr<double>(mat);
        carma::set_not_owndata<double>(arr)
        carma::test_set_not_writeable<double>(arr)
        return arr;
    }

Automatic conversion
********************

For automatic conversion you specify the desired Armadillo type for either or both the return type and the function parameter.
When calling the function from Python, Pybind11 will call `carma`'s type caster when a Numpy array is passed or returned.

.. warning:: Make sure to include `carma` in every compilation unit that makes use of the type caster, not including it results in undefined behaviour.

.. code-block:: c++

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

.. warning::
    
    The automatic conversion will **not** copy the Numpy array's memory when converting to Armadillo objects.
    When converting back to Numpy arrays the memory will **not** be copied when converting back from matrices but **will be** copied from a vector or cube.
    See :doc:`Memory Management <memory_management>` for details.

ArrayStore
**********

There are use-cases where you would want to keep the data in C++ and only return when requested.
For example, you write an Ordinary Least Squares (OLS) class and you want to store the residuals, covariance matrix, ... in C++ for when additional tests need to be run on the values without converting back and forth.

ArrayStore is a convenience class that provides conversion methods back and forth.
It is intended to used as an attribute such as below:

.. code-block:: c++

    #include <armadillo>
    #include <carma/carma.h>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    class ExampleClass {
        private:
            carma::ArrayStore<double> _x;
            carma::ArrayStore<double> _y;
    
        public:
            ExampleClass(py::array_t<double> & x, py::array_t<double> & y) :
            // steal the arrayand store it as an Armadillo matrix
            _x{carma::ArrayStore<double>(x, true)},
            // copy the arrayand store it as an Armadillo matrix
            _y{carma::ArrayStore<double>(y, false)} {}
    
            py::array_t<double> member_func() {
                // normallly you would something useful here
                _x.mat += _y.mat;
                // return mutable view off arma matrix
                return _x.get_view(true);
            }
    };

    void bind_exampleclass(py::module &m) {
        py::class_<ExampleClass>(m, "ExampleClass")
            .def(py::init<py::array_t<double> &, py::array_t<double> &>(), R"pbdoc(
                Initialise ExampleClass.
    
                Parameters
                ----------
                arr1: np.ndarray
                    array to be stored in armadillo matrix
                arr2: np.ndarray
                    array to be stored in armadillo matrix
            )pbdoc")
            .def("member_func", &ExampleClass::member_func, R"pbdoc(
                Compute ....
            )pbdoc");
    }

.. warning::
    
    The ArrayStore owns the data, the returned numpy arrays are views that
    are tied to the lifetime of ArrayStore.
