import numpy as np

from karabo_data.geometry2 import AGIPD_1MGeometry

def test_snap_assemble_data():
    geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
        (-525, 625),
        (-550, -10),
        (520, -160),
        (542.5, 475),
    ])
    snap_geom = geom.snap()

    stacked_data = np.zeros((16, 512, 128))
    img, centre = snap_geom.position_all_modules(stacked_data)
    assert img.shape == (1296, 1132)
    assert tuple(centre) == (651, 570)
    assert np.isnan(img[0, 0])
    assert img[50, 50] == 0
