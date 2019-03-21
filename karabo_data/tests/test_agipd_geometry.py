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
    geom.write_crystfel_geom(path)

    # We need to add some experiment details before cfelpyutils will read the
    # file
    with open(path, 'r') as f:
        contents = f.read()
    with open(path, 'w') as f:
        f.write('clen = 0.119\n')
        f.write('adu_per_eV = 0.0075\n')
        f.write(contents)

    loaded = AGIPD_1MGeometry.from_crystfel_geom(path)
    np.testing.assert_allclose(
        loaded.modules[0][0].corner_pos, geom.modules[0][0].corner_pos
    )
    np.testing.assert_allclose(loaded.modules[0][0].fs_vec, geom.modules[0][0].fs_vec)


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
    assert -0.01 < distortion[..., 1].min() < 0.01
    assert -0.01 < distortion[..., 2].min() < 0.01
