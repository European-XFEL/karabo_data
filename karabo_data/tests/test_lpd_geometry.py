import h5py
from matplotlib.axes import Axes
import numpy as np
from os.path import abspath, dirname, join as pjoin

from karabo_data.geometry2 import LPD_1MGeometry, invert_xfel_lpd_geom

tests_dir = dirname(abspath(__file__))


def test_inspect():
    geom = LPD_1MGeometry.from_quad_positions(
        [(11.4, 299), (-11.5, 8), (254.5, -16), (278.5, 275)]
    )
    # Smoketest
    ax = geom.inspect()
    assert isinstance(ax, Axes)


def test_snap_assemble_data():
    geom = LPD_1MGeometry.from_quad_positions(
        [(11.4, 299), (-11.5, 8), (254.5, -16), (278.5, 275)]
    )

    stacked_data = np.zeros((16, 256, 256))
    img, centre = geom.position_modules_fast(stacked_data)
    assert img.shape == (1202, 1104)
    assert tuple(centre) == (604, 547)
    assert np.isnan(img[0, 0])
    assert img[50, 50] == 0

def test_to_distortion_array():
    geom = LPD_1MGeometry.from_quad_positions(
        [(11.4, 299), (-11.5, 8), (254.5, -16), (278.5, 275)]
    )
    # Smoketest
    distortion = geom.to_distortion_array()
    assert isinstance(distortion, np.ndarray)
    assert distortion.shape == (4096, 256, 4, 3)

    # Coordinates in m, origin at corner; max x & y should be ~ 50cm
    assert 0.40 < distortion[..., 1].max() < 0.70
    assert 0.40 < distortion[..., 2].max() < 0.70
    assert 0.0 <= distortion[..., 1].min() < 0.01
    assert 0.0 <= distortion[..., 2].min() < 0.01

def test_invert_xfel_lpd_geom(tmpdir):
    src_file = pjoin(tests_dir, 'lpd_mar_18.h5')
    dst_file = pjoin(str(tmpdir), 'lpd_inverted.h5')
    invert_xfel_lpd_geom(src_file, dst_file)
    with h5py.File(src_file, 'r') as fsrc, h5py.File(dst_file, 'r') as fdst:
        np.testing.assert_array_equal(
            fsrc['Q1/M1/Position'][:], -1 * fdst['Q1/M1/Position'][:]
        )
        np.testing.assert_array_equal(
            fsrc['Q1/M1/T07/Position'][:], -1 * fdst['Q1/M1/T07/Position'][:]
        )
