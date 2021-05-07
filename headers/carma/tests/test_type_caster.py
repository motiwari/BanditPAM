"""Test armadillo matrix to numpy array type caster."""
import numpy as np

import test_carma as carma


def test_type_caster_in_mat():
    """Test type caster input handling of matrix."""
    sample = np.random.normal(size=(25, 2))
    npsum = sample.sum()
    accu = carma.tc_in_mat(sample)
    assert np.isclose(accu, npsum)

    sample = np.random.normal(size=(2, 25))
    npsum = sample.sum()
    accu = carma.tc_in_mat(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_row():
    """Test type caster input handling of row."""
    sample = np.random.normal(size=(25))
    npsum = sample.sum()
    accu = carma.tc_in_row(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_row_2d():
    """Test type caster input handling of 2d row."""
    sample = np.random.normal(size=(1, 25))
    npsum = sample.sum()
    accu = carma.tc_in_row(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_col():
    """Test type caster input handling of column."""
    sample = np.random.normal(size=(25))
    npsum = sample.sum()
    accu = carma.tc_in_col(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_col_2d():
    """Test type caster input handling of 2d column."""
    sample = np.random.normal(size=(25, 1))
    npsum = sample.sum()
    accu = carma.tc_in_col(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_in_cube():
    """Test type caster input handling of cube."""
    sample = np.random.normal(size=(25, 2, 3))
    npsum = sample.sum()
    accu = carma.tc_in_cube(sample)
    assert np.isclose(accu, npsum)


def test_type_caster_out_mat():
    """Test type caster output handling of matrix."""
    sample = np.random.normal(size=(25, 2))
    mat = carma.tc_out_mat(sample)
    assert np.allclose(mat, sample + 1)


def test_type_caster_out_mat_rvalue():
    """Test type caster output handling of matrix rvalue."""
    sample = np.random.normal(size=(25, 2))
    mat = carma.tc_out_mat_rvalue(sample)
    assert np.allclose(mat, sample + 1)


def test_type_caster_out_row():
    """Test type caster input handling of matrix."""
    sample = np.random.normal(size=(25))
    mat = carma.tc_out_row(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_row_2d():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(1, 25)), order='F')
    mat = carma.tc_out_row(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_row_rvalue():
    """Test type caster input handling of matrix."""
    sample = np.random.normal(size=(25))
    mat = carma.tc_out_row_rvalue(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_col():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25)), order='F')
    mat = carma.tc_out_col(sample)
    assert np.allclose(mat.flatten(), 1 + sample)


def test_type_caster_out_col_2d():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25, 1)), order='F')
    mat = carma.tc_out_col(sample)
    assert np.allclose(mat, 1 + sample)


def test_type_caster_out_col_rvalue():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25)), order='F')
    mat = carma.tc_out_col_rvalue(sample)
    assert np.allclose(mat.flatten(), 1 + sample)


def test_type_caster_out_cube():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25, 3, 2)), order='F')
    mat = carma.tc_out_cube(sample)
    sw_mat = np.moveaxis(mat, [0, 1, 2], [2, 0, 1])
    assert np.allclose(sw_mat, 1 + sample)


def test_type_caster_out_cube_rvalue():
    """Test type caster input handling of matrix."""
    sample = np.asarray(np.random.normal(size=(25, 3, 2)), order='F')
    mat = carma.tc_out_cube_rvalue(sample)
    sw_mat = np.moveaxis(mat, [0, 1, 2], [2, 0, 1])
    assert np.allclose(sw_mat, 1 + sample)
