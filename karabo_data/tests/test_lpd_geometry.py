import os
from cfelpyutils.crystfel_utils import load_crystfel_geometry
import h5py
from matplotlib.axes import Axes
import numpy as np
from os.path import abspath, dirname, join as pjoin

from karabo_data.geometry2 import LPD_1MGeometry, invert_xfel_lpd_geom

tests_dir = dirname(abspath(__file__))


def test_write_read_crystfel_file(tmpdir):
    geom = LPD_1MGeometry.from_quad_positions(
        [(11.4, 299), (-11.5, 8), (254.5, -16), (278.5, 275)]
    )
    path = str(tmpdir / 'test.geom')
    geom.write_crystfel_geom(filename=path)

    with open(path, 'r') as f:
        contents = f.read()
    with open(path, 'w') as f:
        f.write('clen = 0.119\n')
        f.write('adu_per_eV = 0.0075\n')

        f.write(contents)
    # Load the geometry file with cfelpyutils and test the ridget groups
    loaded = LPD_1MGeometry.from_crystfel_geom(path)
    np.testing.assert_allclose(
        loaded.modules[0][0].corner_pos, geom.modules[0][0].corner_pos
    )
    np.testing.assert_allclose(loaded.modules[0][0].fs_vec, geom.modules[0][0].fs_vec)


    geom_dict = load_crystfel_geometry(path)
    quad_gr0 = ['p0a0', 'p0a1', 'p0a2', 'p0a3', 'p0a4', 'p0a5', 'p0a6', 'p0a7',
                'p0a8', 'p0a9', 'p0a10','p0a11', 'p0a12', 'p0a13', 'p0a14',
                'p0a15', 'p1a0', 'p1a1','p1a2', 'p1a3','p1a4','p1a5','p1a6',
                'p1a7', 'p1a8', 'p1a9', 'p1a10', 'p1a11', 'p1a12', 'p1a13',
                'p1a14', 'p1a15', 'p2a0', 'p2a1', 'p2a2', 'p2a3', 'p2a4', 'p2a5',
                'p2a6', 'p2a7', 'p2a8', 'p2a9', 'p2a10', 'p2a11', 'p2a12', 'p2a13',
                'p2a14','p2a15', 'p3a0', 'p3a1','p3a2', 'p3a3', 'p3a4', 'p3a5',
                'p3a6', 'p3a7', 'p3a8','p3a9', 'p3a10', 'p3a11', 'p3a12', 'p3a13',
                'p3a14', 'p3a15']
    assert geom_dict['rigid_groups']['p0'] == quad_gr0[:16]
    assert geom_dict['rigid_groups']['p3'] == quad_gr0[-16:]
    assert geom_dict['rigid_groups']['q0'] == quad_gr0

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

def test_data_coords_to_positions():
    geom = LPD_1MGeometry.from_quad_positions(
        [(11.4, 299), (-11.5, 8), (254.5, -16), (278.5, 275)]
    )

    module_no = np.zeros(16, dtype=np.int16)
    # Points near the centre of each tile
    slow_scan = np.tile(np.linspace(16, 240, num=8, dtype=np.float32), 2)
    fast_scan = np.array([64, 192], dtype=np.float32).repeat(8)

    tileno, tile_ss, tile_fs = geom._module_coords_to_tile(slow_scan, fast_scan)
    np.testing.assert_allclose(tileno,
                       [7, 6, 5, 4, 3, 2, 1, 0, 8, 9, 10, 11, 12, 13, 14, 15])
    np.testing.assert_allclose(tile_ss, 16)
    np.testing.assert_allclose(tile_fs, 64)

    res = geom.data_coords_to_positions(module_no, slow_scan, fast_scan)

    assert res.shape == (16, 3)

    resx, resy, resz = res.T

    np.testing.assert_allclose(resz, 0)

    assert (np.diff(resy[:8]) > 0).all()  # T1-T8 Monotonically increasing
    assert (np.diff(resy[8:]) > 0).all()  # T9-T16 Monotonically increasing
    assert -0.128 > resx.max() > resx.min() > -0.280

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
