from matplotlib.figure import Figure
import numpy as np

from karabo_data.geometry2 import LPD_1MGeometry

def test_inspect():
    geom = LPD_1MGeometry.from_quad_positions([
        (11.4, 299),
        (-11.5, 8),
        (254.5, -16),
        (278.5, 275)
    ])
    # Smoketest
    fig = geom.inspect()
    assert isinstance(fig, Figure)

def test_snap_assemble_data():
    geom = LPD_1MGeometry.from_quad_positions([
        (11.4, 299),
        (-11.5, 8),
        (254.5, -16),
        (278.5, 275)
    ])

    stacked_data = np.zeros((16, 256, 256))
    img, centre = geom.position_modules_fast(stacked_data)
    assert img.shape == (1202, 1104)
    assert tuple(centre) == (604, 547)
    assert np.isnan(img[0, 0])
    assert img[50, 50] == 0
