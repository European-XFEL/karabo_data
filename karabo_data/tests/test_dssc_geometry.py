import h5py
from matplotlib.axes import Axes
import numpy as np
from os.path import abspath, dirname, join as pjoin

from karabo_data.geometry2 import DSSC_1MGeometry

tests_dir = dirname(abspath(__file__))
sample_xfel_geom = pjoin(tests_dir, 'dssc_geo_june19.h5')

# Made up numbers!
QUAD_POS = [
    (-130, 5),
    (-130, -125),
    (5, -125),
    (5, 5),
]


def test_inspect():
    geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
        sample_xfel_geom, QUAD_POS
    )
    # Smoketest
    ax = geom.inspect()
    assert isinstance(ax, Axes)


def test_snap_assemble_data():
    geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
        sample_xfel_geom, QUAD_POS
    )

    stacked_data = np.zeros((16, 128, 512))
    img, centre = geom.position_modules_fast(stacked_data)
    assert img.shape == (1281, 1099)
    assert tuple(centre) == (656, 552)
    assert np.isnan(img[0, 0])
    assert img[50, 50] == 0

def test_to_distortion_array():
    geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
        sample_xfel_geom, QUAD_POS
    )
    # Smoketest
    distortion = geom.to_distortion_array()
    assert isinstance(distortion, np.ndarray)
    assert distortion.shape == (2048, 512, 6, 3)

    # Coordinates in m, origin at corner; max x & y should be ~ 25cm
    assert 0.20 < distortion[..., 1].max() < 0.35
    assert 0.20 < distortion[..., 2].max() < 0.35
    assert 0.0 <= distortion[..., 1].min() < 0.01
    assert 0.0 <= distortion[..., 2].min() < 0.01

def test_get_pixel_positions():
    geom = DSSC_1MGeometry.from_h5_file_and_quad_positions(
        sample_xfel_geom, QUAD_POS
    )

    pixelpos = geom.get_pixel_positions()
    assert pixelpos.shape == (16, 128, 512, 3)

    px = pixelpos[..., 0]
    py = pixelpos[..., 1]

    assert -0.15 < px.min() < -0.1
    assert  0.15 > px.max() >  0.1
    assert -0.2 < py.min() < -0.12
    assert  0.2 > py.max() >  0.12

    # Odd-numbered rows in Q1 & Q2 should have 0.5 pixel higher x than the even.
    np.testing.assert_allclose(px[0, 1::2, 0] - px[0, 0::2, 0], 236e-6/2)
