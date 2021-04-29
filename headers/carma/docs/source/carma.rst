#################
Manual conversion
#################

CARMA provides a set of functions for manual conversion of Numpy arrays and Armadillo matrices.
Manual conversion should be used when fine grained control of memory is required.

Numpy to Armadillo
******************

Functions to convert Numpy arrays to Armadillo matrices, vectors or cubes

Matrix
------

.. function:: arma::Mat<T> arr_to_mat(py::array<T> & arr, bool copy=false, bool strict=false)

   Convert Numpy array to Armadillo matrix.

   If the array is 1D we create a column oriented matrix (N, 1)

   :param arr: numpy array to be converted
   :param copy: copy the memory of the array, default is false
   :param strict: the memory of the array cannot be changed in size, default is true.
    Parameter is ignored if copy is true.
   :exception: raises `runtime_error` if `n_dims` > 2 or memory is not initialised (nullptr)

Vector
------

.. function:: arma::Col<T> arr_to_col(py::array<T> & arr, bool copy=false, bool strict=false)

   Convert Numpy array to Armadillo column.

   :param arr: numpy array to be converted
   :param copy: copy the memory of the array, default is false
   :param strict: the memory of the array cannot be changed in size, default is true.
    Parameter is ignored if copy is true.
   :exception: raises `runtime_error` if `n_cols` > 1 or memory is not initialised (nullptr)

.. function:: arma::Row<T> arr_to_row(py::array<T> & arr, bool copy=false, bool strict=false)

   Convert Numpy array to Armadillo row.

   :param arr: numpy array to be converted
   :param copy: copy the memory of the array, default is false
   :param strict: the memory of the array cannot be changed in size, default is true.
    Parameter is ignored if copy is true.
   :exception: raises `runtime_error` if `n_rows` > 1 or memory is not initialised (nullptr)

Cube
----

.. function:: arma::Cube<T> arr_to_cube(py::array<T> & arr, bool copy=false, bool strict=false)

   Convert Numpy array to Armadillo Cube.

   :param arr: numpy array to be converted
   :param copy: copy the memory of the array, default is false
   :param strict: the memory of the array cannot be changed in size, default is true.
    Parameter is ignored if copy is true.
   :exception: raises `runtime_error` if `n_dims` < 3 or memory is not initialised (nullptr)

Armadillo to Numpy
******************

This section documents the functions to convert Armadillo matrices, vectors or cubes to Numpy arrays. 

.. warning:: default for copy is `false` for matrices and `true` for all other armadillo types. See :ref:`memsafe` for details.

.. note:: All functions for conversion to Numpy arrays overloads for {pointer, lvalue, rvalue} Armadillo objects of type {Mat, Col, Row, Cube}.

Matrix
------

.. function:: py::array_t<T> to_numpy(arma::Mat<T> & src, bool copy=false)

   Convert Armadillo matrix to Numpy array.

   :note: the returned array will have F order memory.
   :param src: armadillo object to be converted.
   :type src: `*, &&, &`
   :param copy: copy the memory of the array, default is false

.. function:: py::array_t<T> mat_to_arr(arma::Mat<T> & src, bool copy=false)

   Convert Armadillo matrix to Numpy array.

   :note: the returned array will have F order memory.
   :param src: armadillo object to be converted.
   :type src: `*, &&, &`
   :param copy: copy the memory of the array, default is false

Vector
------

.. function:: py::array_t<T> to_numpy(arma::Col<T> & src, bool copy=true)

   Convert Armadillo column to Numpy array.

   :param src: armadillo object to be converted.
   :type src: `*, &&, &`
   :param copy: copy the memory of the array, default is true

.. function:: py::array_t<T> to_numpy(arma::Row<T> & src, bool copy=true)

   Convert Armadillo row to Numpy array.

   :param src: armadillo object to be converted.
   :type src: `*, &&, &`
   :param copy: copy the memory of the array, default is true

.. function:: py::array_t<T> col_to_arr(arma::Col<T> & src, bool copy=true)

   Convert Armadillo column to Numpy array.

   :param src: armadillo object to be converted.
   :type src: `*, &&, &`
   :param copy: copy the memory of the array, default is true

.. function:: py::array_t<T> row_to_arr(arma::Row<T> & src, bool copy=true)

   Convert Armadillo row to Numpy array.

   :param src: armadillo object to be converted.
   :type src: `*, &&, &`
   :param copy: copy the memory of the array, default is true

Cube
----

.. function:: py::array_t<T> to_numpy(arma::Cube<T> & src, bool copy=false)

   Convert Armadillo cube to Numpy array.

   :note: the returned array will have F-order memory and the axis are ordered as `[slices, rows, columns]`
   :param src: armadillo object to be converted.
   :type src: `*, &&, &`
   :param copy: copy the memory of the array, default is false

.. function:: py::array_t<T> cube_to_arr(arma::Cube<T> & src, bool copy=false)

   Convert Armadillo cube to Numpy array.

   :note: the returned array will have F-order memory and the axis are ordered as `[slices, rows, columns]`
   :param src: armadillo object to be converted.
   :type src: `*, &&, &`
   :param copy: copy the memory of the array, default is false

Update Array
************

The `update_array` function should be used to update Numpy array attributes to reflect state of the memory based on the Armadillo object.

**Note that:**

* `update_array` is overloaded for all supported armadillo types `[Mat, Col, Row, Cube]`.
* `update_array` accepts `pointer`, `lvalue` and `rvalue` armadillo objects as `src` argument.

.. function:: void update_array(arma::Mat<T> & src, py::array_t<T> & arr)

   Update Numpy array attributes to reflect state of the memory based on the Armadillo object.

   :param src: armadillo object containing the memory reference.
   :type src: `*, &&, &`
   :param arr: numpy array for which to update the memory reference.

.. function:: void update_array(arma::Col<T> & src, py::array_t<T> & arr)

   Update Numpy array attributes to reflect state of the memory based on the Armadillo object.

   :param src: armadillo object containing the memory reference.
   :type src: `*, &&, &`
   :param arr: numpy array for which to update the memory reference.

.. function:: void update_array(arma::Row<T> & src, py::array_t<T> & arr)

   Update Numpy array attributes to reflect state of the memory based on the Armadillo object.

   :param src: armadillo object containing the memory reference.
   :type src: `*, &&, &`
   :param arr: numpy array for which to update the memory reference.

.. function:: void update_array(arma::Cube<T> & src, py::array_t<T> & arr)

   Update Numpy array attributes to reflect state of the memory based on the Armadillo object.

   :param src: armadillo object containing the memory reference.
   :type src: `*, &&, &`
   :param arr: numpy array for which to update the memory reference.

###################################
Automatic conversion -- Type caster
###################################

CARMA provides a type caster which enables automatic conversion using pybind11.

.. warning:: `carma.h` should included in every compilation unit where automated type casting occurs, otherwise undefined behaviour will occur.

.. note:: The underlying casting function has overloads for `{pointer, lvalue, rvalue}` Armadillo objects of type `{Mat, Col, Row, Cube}`.

Return policies
***************

Pybind11 provides a number of return value policies of which a subset is supported:

.. function:: return_value_policy::move

   * `Mat`: copy is false
   * `Col, Row, Cube`: copy is true

.. function:: return_value_policy::automatic

   * `Mat`: copy is false
   * `Col, Row, Cube`: copy is true

.. function:: return_value_policy::take_ownership

   * `Mat`: copy is false
   * `Col, Row, Cube`: copy is true

.. function:: return_value_policy::copy

   * `Mat`: copy is true
   * `Col, Row, Cube`: copy is true

For arguments why vectors and cubes are returned by copying see :ref:`memsafe` for details.

To pass the return value policy set it in the binding function:

.. code-block:: c++

    m.def("example_function", &example_function, return_value_policy::copy);

##########
ArrayStore
##########

ArrayStore is a convenience class for storing the memory in a C++ class.

.. warning:: 
    The ArrayStore owns the data, the returned numpy arrays are views that
    are tied to the lifetime of ArrayStore.

.. class:: ArrayStore

       .. attribute:: mat arma::Matrix<T>
           
           Matrix containing the memory of the array.

       .. method:: ArrayStore(py::array_t<T> & arr, bool steal)

           Class constructor

           :param arr: Numpy array to be stored as Armadillo matrix
           :param steal: Take ownership of the array, else copy

       .. method:: ArrayStore(arma::Mat<T> & mat)

           Class constructor, matrix is copied

           :param mat: Armadillo matrix to be stored

       .. method:: ArrayStore(arma::Mat<T> && mat)

           Class constructor, matrix is moved

           :param mat: Armadillo matrix to be stored

       .. method:: get_view(bool writeable)

           Obtain a view of the memory as Numpy array.

           :param writeable: Mark array as writeable

       .. method:: set_data(py::array_t<T> & arr, bool steal)

           Store new array in the ArrayStore.

           :param arr: Numpy array to be stored as Armadillo matrix
           :param steal: Take ownership of the array, else copy

       .. method:: set_mat(arma::Mat<T> & src)

           Store new matrix in the ArrayStore.

           :param src: Armadillo matrix to be stored, matrix is copied

       .. method:: set_mat(arma::Mat<T> && src)

           Store new matrix in the ArrayStore.

           :param src: Armadillo matrix to be stored, matrix is moved

#############
NdArray flags
#############

Utility functions to check flags of numpy arrays.

.. function:: bool is_f_contiguous(const py::array_t<T> & arr)

   Check if Numpy array's  memory is Fotran contiguous.

   :param arr: numpy array to be checked

.. function:: bool is_c_contiguous(const py::array_t<T> & arr)

   Check if Numpy array's  memory is C contiguous.

   :param arr: numpy array to be checked

.. function:: bool is_writable(const py::array_t<T> & arr)

   Check if Numpy array's memory is mutable.

   :param arr: numpy array to be checked

.. function:: bool is_owndata(const py::array_t<T> & arr)

   Check if Numpy array's memory is owned by numpy.

   :param arr: numpy array to be checked

.. function:: bool is_aligned(const py::array_t<T> & arr)

   Check if Numpy array's memory is aligned.

   :param arr: numpy array to be checked

.. function:: bool requires_copy(const py::array_t<T> & arr)

   Check if Numpy array memory needs to be copied out, is true
   when either not writable, owndata or is not aligned.

   :param arr: numpy array to be checked

.. function:: void set_not_owndata(py::array_t<T> & arr)

   Set Numpy array's flag OWNDATA to false.

   :param arr: numpy array to be changed

.. function:: void set_not_writeable(py::array_t<T> & arr)

   Set Numpy array's flag WRITEABLE to false.

   :param arr: numpy array to be changed
