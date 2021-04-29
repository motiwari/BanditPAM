"""Test numpy array to matrix conversion function."""
import numpy as np

import test_carma as carma

test_flags = {
    1: 'Number of elements between array and matrix are not the same',
    2: 'Number of rows between array and matrix are not the same',
    3: 'Number of columns between array and matrix are not the same',
    4: 'Sum of elements between array and matrix is not aproximately equal',
    5: 'Pointer to memory is not as expected',
}


def test_arr_to_mat_double():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_mat_double(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_double_large():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(1000, 1000)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_mat_double(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_double_small():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(3, 3)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_mat_double(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_mat_long():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.int64, order='F'
    )
    flag = carma.arr_to_mat_long(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_double_c_contiguous():
    """Test arr_to_mat."""
    sample = np.asarray(np.random.normal(size=(50, 2)), dtype=np.float64)
    flag = carma.arr_to_mat_double(sample, False, False)
    assert flag == 5, test_flags[flag]

def test_arr_to_mat_double_c_contiguous_large():
    """Test arr_to_mat."""
    sample = np.asarray(np.random.normal(size=(1000, 1000)), dtype=np.float64)
    flag = carma.arr_to_mat_double(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_mat_long_c_contiguous():
    """Test arr_to_mat."""
    sample = np.asarray(np.random.normal(size=(50, 2)), dtype=np.int64)
    flag = carma.arr_to_mat_long(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_mat_double_copy():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_mat_double_copy(sample)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_double_copy_c_contiguous():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.float64, order='C'
    )
    flag = carma.arr_to_mat_double_copy(sample)
    assert flag == 0, test_flags[flag]


# #############################################################################
#                                   N-DIM 1                                   #
# #############################################################################
def test_arr_to_mat_1d():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(100)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_mat_1d(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_1d_small():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=5), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_mat_1d(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_mat_1d_copy():
    """Test arr_to_mat."""
    sample = np.asarray(
        np.random.normal(size=(100)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_mat_1d(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_col():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='F')
    flag = carma.arr_to_col(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_col_small():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=5), dtype=np.float64, order='F')
    flag = carma.arr_to_col(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_col_2d():
    """Test arr_to_col."""
    sample = np.asarray(
        np.random.normal(size=(100, 1)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_col(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_col_C():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='C')
    flag = carma.arr_to_col(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_col_writeable():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='F')
    sample.setflags(write=0)
    flag = carma.arr_to_col(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_col_copy():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='F')
    flag = carma.arr_to_col(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_col_copy_C():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='C')
    flag = carma.arr_to_col(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_row():
    """Test arr_to_row."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='F')
    flag = carma.arr_to_row(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_row_small():
    """Test arr_to_row."""
    sample = np.asarray(np.random.normal(size=5), dtype=np.float64, order='F')
    flag = carma.arr_to_row(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_row_2d():
    """Test arr_to_row."""
    sample = np.asarray(
        np.random.normal(size=(1, 100)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_row(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_row_C():
    """Test arr_to_row."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='C')
    flag = carma.arr_to_row(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_row_writeable():
    """Test arr_to_row."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='F')
    sample.setflags(write=0)
    flag = carma.arr_to_row(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_row_copy():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='F')
    flag = carma.arr_to_row(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_row_copy_C():
    """Test arr_to_col."""
    sample = np.asarray(np.random.normal(size=100), dtype=np.float64, order='C')
    flag = carma.arr_to_row(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_mat_cube():
    """Test arr_to_cube."""
    sample = np.asarray(
        np.random.normal(size=(25, 2, 2)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_cube(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_arr_to_mat_cube_small():
    """Test arr_to_cube."""
    sample = np.asarray(
        np.random.normal(size=(2, 2, 2)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_cube(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_cube_double_c_contiguous():
    """Test arr_to_mat."""
    sample = np.asarray(np.random.normal(size=(25, 2, 2)), dtype=np.float64)
    flag = carma.arr_to_cube(sample, False, False)
    assert flag == 5, test_flags[flag]


def test_arr_to_mat_cube_copy():
    """Test arr_to_cube."""
    sample = np.asarray(
        np.random.normal(size=(25, 2, 2)), dtype=np.float64, order='F'
    )
    flag = carma.arr_to_cube(sample, True, False)
    assert flag == 5, test_flags[flag]


def test_to_arma_mat():
    """Test private implementation of to_arma for matrix."""
    sample = np.asarray(
        np.random.normal(size=(50, 2)), dtype=np.float64, order='F'
    )
    flag = carma.to_arma_mat(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_to_arma_cube():
    """Test private implementation of to_arma for matrix."""
    sample = np.asarray(
        np.random.normal(size=(25, 2, 2)), dtype=np.float64, order='F'
    )
    flag = carma.to_arma_cube(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_to_arma_col():
    """Test private implementation of to_arma for matrix."""
    sample = np.asarray(
        np.random.normal(size=100), dtype=np.float64, order='F'
    )
    flag = carma.to_arma_col(sample, False, False)
    assert flag == 0, test_flags[flag]


def test_to_arma_row():
    """Test private implementation of to_arma for matrix."""
    sample = np.asarray(
        np.random.normal(size=100), dtype=np.float64, order='F'
    )
    flag = carma.to_arma_row(sample, False, False)
    assert flag == 0, test_flags[flag]
