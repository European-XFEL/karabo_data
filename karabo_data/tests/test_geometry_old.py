"""Tests for deprecated karabo_data.geometry module."""

import h5py
from matplotlib.figure import Figure
import numpy as np
from os.path import abspath, dirname, join as pjoin

from karabo_data.geometry import LPDGeometry

tests_dir = dirname(abspath(__file__))

# These coordinates are backwards - x increasing beam-right, y increasing
# downwards. Use karabo_data.geometry2 instead, which expects coordinates
# in the standard axis directions.
quadpos = [(-11.4, -299), (11.5, -8), (-254.5, 16), (-278.5, -275)]


def test_inspect():
    with h5py.File(pjoin(tests_dir, 'lpd_mar_18.h5')) as f:
        geom = LPDGeometry.from_h5_file_and_quad_positions(f, quadpos)

    # Smoketest
    fig = geom.inspect()
    assert isinstance(fig, Figure)


def test_position_all_modules():
    with h5py.File(pjoin(tests_dir, 'lpd_mar_18.h5')) as f:
        geom = LPDGeometry.from_h5_file_and_quad_positions(f, quadpos)

    stacked_data = np.zeros((16, 256, 256))
    img, centre = geom.position_all_modules(stacked_data)
    assert 1000 < img.shape[0] < 2000
    assert 1000 < img.shape[1] < 2000
    assert np.isnan(img[0, 0])
    assert img[200, 200] == 0
