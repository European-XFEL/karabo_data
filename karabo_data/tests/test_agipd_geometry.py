
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from itertools import product
from matplotlib.axes import Axes
import numpy as np

from karabo_data.geometry2 import AGIPD_1MGeometry


def test_snap_assemble_data():
    geom = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )

    stacked_data = np.zeros((16, 512, 128))
    img, centre = geom.position_modules_fast(stacked_data)
    assert img.shape == (1256, 1092)
    assert tuple(centre) == (631, 550)
    assert np.isnan(img[0, 0])
    assert img[50, 50] == 0


def test_write_read_crystfel_file(tmpdir):
    geom = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )
    path = str(tmpdir / 'test.geom')
    geom.write_crystfel_geom(filename=path, photon_energy=9000,
                             adu_per_ev=0.0075, clen=0.2)

    loaded = AGIPD_1MGeometry.from_crystfel_geom(path)
    np.testing.assert_allclose(
        loaded.modules[0][0].corner_pos, geom.modules[0][0].corner_pos
    )
    np.testing.assert_allclose(loaded.modules[0][0].fs_vec, geom.modules[0][0].fs_vec)

    # Load the geometry file with cfelpyutils and test the rigid groups
    geom_dict = load_crystfel_geometry(path)
    quad_gr0 = [  # 1st quadrant: p0a0 ... p3a7
        'p{}a{}'.format(p, a) for p, a in product(range(4), range(8))
    ]
    assert geom_dict['rigid_groups']['p0'] == quad_gr0[:8]
    assert geom_dict['rigid_groups']['p3'] == quad_gr0[-8:]
    assert geom_dict['rigid_groups']['q0'] == quad_gr0
    assert geom_dict['panels']['p0a0']['res'] == 5000  # 5000 pixels/metre
    p3a7 = geom_dict['panels']['p3a7']
    assert p3a7['min_ss'] == 448
    assert p3a7['max_ss'] == 511
    assert p3a7['min_fs'] == 0
    assert p3a7['max_fs'] == 127


def test_write_read_crystfel_file_2d(tmpdir):
    geom = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )
    path = str(tmpdir / 'test.geom')
    geom.write_crystfel_geom(filename=path, dims=('frame', 'ss', 'fs'),
                             adu_per_ev=0.0075, clen=0.2)

    loaded = AGIPD_1MGeometry.from_crystfel_geom(path)
    np.testing.assert_allclose(
        loaded.modules[0][0].corner_pos, geom.modules[0][0].corner_pos
    )
    np.testing.assert_allclose(loaded.modules[0][0].fs_vec, geom.modules[0][0].fs_vec)

    # Load the geometry file with cfelpyutils and check some values
    geom_dict = load_crystfel_geometry(path)

    p3a7 = geom_dict['panels']['p3a7']
    assert p3a7['dim_structure'] == ['%', 'ss', 'fs']
    assert p3a7['min_ss'] == (3 * 512) + 448
    assert p3a7['max_ss'] == (3 * 512) + 511
    assert p3a7['min_fs'] == 0
    assert p3a7['max_fs'] == 127


def test_inspect():
    geom = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )
    # Smoketest
    ax = geom.inspect()
    assert isinstance(ax, Axes)


def test_compare():
    geom1 = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )
    geom2 = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-527, 625), (-548, -10), (520, -162), (542.5, 473)]
    )
    # Smoketest
    ax = geom1.compare(geom2)
    assert isinstance(ax, Axes)


def test_to_distortion_array():
    geom = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )
    # Smoketest
    distortion = geom.to_distortion_array()
    assert isinstance(distortion, np.ndarray)
    assert distortion.shape == (8192, 128, 4, 3)

    # Coordinates in m, origin at corner; max x & y should be ~ 25cm
    assert 0.20 < distortion[..., 1].max() < 0.30
    assert 0.20 < distortion[..., 2].max() < 0.30
    assert 0.0 <= distortion[..., 1].min() < 0.01
    assert 0.0 <= distortion[..., 2].min() < 0.01

def test_get_pixel_positions():
    geom = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )

    pixelpos = geom.get_pixel_positions()
    assert pixelpos.shape == (16, 512, 128, 3)
    px = pixelpos[..., 0]
    py = pixelpos[..., 1]

    assert -0.12 < px.min() < -0.1
    assert  0.12 > px.max() > 0.1
    assert -0.14 < py.min() < -0.12
    assert  0.14 > py.max() >  0.12

def test_data_coords_to_positions():
    geom = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )

    module_no = np.zeros(16, dtype=np.int16)
    slow_scan = np.linspace(0, 500, num=16, dtype=np.float32)
    fast_scan = np.zeros(16, dtype=np.float32)

    res = geom.data_coords_to_positions(module_no, slow_scan, fast_scan)

    assert res.shape == (16, 3)

    resx, resy, resz = res.T

    np.testing.assert_allclose(resz, 0)
    np.testing.assert_allclose(resy, 625 * geom.pixel_size)

    assert (np.diff(resx) > 0).all()   # Monotonically increasing
    np.testing.assert_allclose(resx[0], -525 * geom.pixel_size)
    assert -0.01 < resx[-1] < 0.01
