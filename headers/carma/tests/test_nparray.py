"""Test nparray.h."""
import numpy as np

import test_carma as carma


def test_is_f_contiguous():
    """Test is_f_contiguous."""
    m = 'F order array should be F contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_f_contiguous(sample) is True, m

    m = 'C order array should not be F contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='C')
    assert carma.is_f_contiguous(sample) is False, m


def test_is_c_contiguous():
    """Test is_c_contiguous."""
    m = 'C order array should be C contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='C')
    assert carma.is_c_contiguous(sample) is True, m

    m = 'F order array should not be C contiguous'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_c_contiguous(sample) is False, m


def test_is_writeable():
    """Test is_writeable."""
    m = 'Array should be writeable'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_writeable(sample) is True, m

    m = 'Array should not be writeable'
    sample.setflags(write=0)
    assert carma.is_writeable(sample) is False, m


def test_is_owndata():
    """Test is_writable."""
    m = 'Array should own the data'
    sample = np.ones((10, 2), dtype=np.float64, order='F')
    assert carma.is_owndata(sample) == sample.flags['OWNDATA'], m

    m = 'Array should not own the data'
    view = sample.reshape(20, 1)
    assert carma.is_owndata(view) == view.flags['OWNDATA'], m


def test_is_aligned():
    """Test is_aligned."""
    m = 'Array should be aligned'
    sample = np.arange(200, dtype=np.float64)
    assert carma.is_aligned(sample) == sample.flags['ALIGNED'], m

    m = 'Array should not be aligned'
    alt = np.frombuffer(sample.data, offset=2, count=100, dtype=np.float64)
    alt.shape = 10, 10
    assert carma.is_aligned(alt) == alt.flags['ALIGNED'], m


def test_set_not_owndata():
    """Test set_not_owndata."""
    m = 'Array should have owndata is false'
    sample = np.arange(100, dtype=np.float64)
    carma.set_not_owndata(sample)
    assert sample.flags['OWNDATA'] is False, m


def test_set_not_writeable():
    """Test set_not_writeable."""
    m = 'Array should have writeable is false'
    sample = np.arange(100, dtype=np.float64)
    carma.set_not_writeable(sample)
    assert sample.flags['WRITEABLE'] is False, m
