from cfelpyutils.crystfel_utils import load_crystfel_geometry
import numpy as np

def _crystfel_format_vec(vec):
    """Convert an array of 3 numbers to CrystFEL format like "+1.0x -0.1y"
    """
    s = '{:+}x {:+}y'.format(*vec[:2])
    if vec[2] != 0:
        s += ' {:+}z'.format(vec[2])
    return s

class AGIPDGeometryFragment:
    ss_pixels = 64
    fs_pixels = 128

    def __init__(self, corner_pos, ss_vec, fs_vec):
        self.corner_pos = corner_pos
        self.ss_vec = ss_vec
        self.fs_vec = fs_vec

    @classmethod
    def from_panel_dict(cls, d):
        corner_pos = np.array([d['cnx'], d['cny'], d['coffset']])
        ss_vec = np.array([d['ssx'], d['ssy'], d['ssz']])
        fs_vec = np.array([d['fsx'], d['fsy'], d['fsz']])
        return cls(corner_pos, ss_vec, fs_vec)

    def corners(self):
        return np.stack([
            self.corner_pos,
            self.corner_pos + (self.fs_vec * self.fs_pixels),
            self.corner_pos + (self.ss_vec * self.ss_pixels) + (self.fs_vec * self.fs_pixels),
            self.corner_pos + (self.ss_vec * self.ss_pixels),
        ])

    def centre(self):
        return self.corner_pos + (.5 * self.ss_vec * self.ss_pixels) \
                               + (.5 * self.fs_vec * self.fs_pixels)

    def to_crystfel_geom(self, p, a):
        name = 'p{}a{}'.format(p, a)
        c = self.corner_pos
        return CRYSTFEL_PANEL_TEMPLATE.format(
            name=name, p=p,
            min_ss=(a * 64), max_ss=((a * 64) + 63),
            ss_vec=_crystfel_format_vec(self.ss_vec),
            fs_vec=_crystfel_format_vec(self.fs_vec),
            corner_x=c[0], corner_y=c[1], coffset=c[2],
        )


class AGIPD_1MGeometry:
    pixel_size = 2e-7
    def __init__(self, modules):
        self.modules = modules  # List of 16 lists of 8 fragments

    @classmethod
    def from_quad_positions(cls, quad_pos, asic_gap=2, panel_gap=29):
        """Generate an AGIPD-1M geometry from quadrant positions.

        This produces an idealised geometry, assuming all modules are perfectly
        flat, aligned and equally spaced within their quadrant.

        The quadrant positions are given in pixel units, referring to the first
        pixel of the first module in each quadrant.
        """
        quads_x_orientation = [1, 1, -1, -1]
        quads_y_orientation = [-1, -1, 1, 1]
        modules = []
        for p in range(16):
            quad = p // 4
            quad_corner = quad_pos[quad]
            x_orient = quads_x_orientation[quad]
            y_orient = quads_y_orientation[quad]
            p_in_quad = p % 4
            corner_y = quad_corner[1] - (p_in_quad * (128 + panel_gap))

            tiles = []
            modules.append(tiles)

            for a in range(8):
                corner_x = quad_corner[0] + x_orient * (64 + asic_gap) * a
                tiles.append(AGIPDGeometryFragment(
                    corner_pos=np.array([corner_x, corner_y, 0.]),
                    ss_vec=np.array([x_orient, 0, 0]),
                    fs_vec=np.array([0, y_orient, 0]),
                ))
        return cls(modules)

    @classmethod
    def from_crystfel_geom(cls, filename):
        geom_dict = load_crystfel_geometry(filename)
        modules = []
        for p in range(16):
            tiles = []
            modules.append(tiles)
            for a in range(8):
                d = geom_dict['panels']['p{}a{}'.format(p, a)]
                tiles.append(AGIPDGeometryFragment.from_panel_dict(d))
        return cls(modules)

    def write_crystfel_geom(self, filename):
        from . import __version__

        panel_chunks = []
        for p, module in enumerate(self.modules):
            for a, fragment in enumerate(module):
                panel_chunks.append(fragment.to_crystfel_geom(p, a))

        with open(filename, 'w') as f:
            f.write(CRYSTFEL_HEADER_TEMPLATE.format(version=__version__))
            for chunk in panel_chunks:
                f.write(chunk)

    def inspect(self):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Figure object.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import PatchCollection
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon

        fig = Figure((10, 10))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)

        rects = []
        for p, module in enumerate(self.modules):
            for a, fragment in enumerate(module):
                corners = fragment.corners()[:, :2]  # Drop the Z dimension

                rects.append(Polygon(corners))

                if a in {0, 7}:
                    ax.text(*fragment.centre()[:2], str(a),
                            verticalalignment='center',
                            horizontalalignment='center')
                elif a == 4:
                    ax.text(*fragment.centre()[:2], 'p{}'.format(p),
                            verticalalignment='center',
                            horizontalalignment='center')

        pc = PatchCollection(rects, facecolor=(0.75, 1., 0.75), edgecolor=None)
        ax.add_collection(pc)

        ax.hlines(0, -100, +100, colors='0.75', linewidths=2)
        ax.vlines(0, -100, +100, colors='0.75', linewidths=2)

        ax.set_title('AGIPD-1M detector geometry')
        return fig


CRYSTFEL_HEADER_TEMPLATE = """\
; AGIPD-1M geometry file written by karabo_data {version}
; You may need to edit this file to add:
; - data and mask locations in the file
; - mask_good & mask_bad values to interpret the mask
; - adu_per_eV & photon_energy
; - clen (detector distance)
;
; See: http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html

dim0 = %
res = 5000 ; 200 um pixels

rigid_group_q0 = p0a0,p0a1,p0a2,p0a3,p0a4,p0a5,p0a6,p0a7,p1a0,p1a1,p1a2,p1a3,p1a4,p1a5,p1a6,p1a7,p2a0,p2a1,p2a2,p2a3,p2a4,p2a5,p2a6,p2a7,p3a0,p3a1,p3a2,p3a3,p3a4,p3a5,p3a6,p3a7
rigid_group_q1 = p4a0,p4a1,p4a2,p4a3,p4a4,p4a5,p4a6,p4a7,p5a0,p5a1,p5a2,p5a3,p5a4,p5a5,p5a6,p5a7,p6a0,p6a1,p6a2,p6a3,p6a4,p6a5,p6a6,p6a7,p7a0,p7a1,p7a2,p7a3,p7a4,p7a5,p7a6,p7a7
rigid_group_q2 = p8a0,p8a1,p8a2,p8a3,p8a4,p8a5,p8a6,p8a7,p9a0,p9a1,p9a2,p9a3,p9a4,p9a5,p9a6,p9a7,p10a0,p10a1,p10a2,p10a3,p10a4,p10a5,p10a6,p10a7,p11a0,p11a1,p11a2,p11a3,p11a4,p11a5,p11a6,p11a7
rigid_group_q3 = p12a0,p12a1,p12a2,p12a3,p12a4,p12a5,p12a6,p12a7,p13a0,p13a1,p13a2,p13a3,p13a4,p13a5,p13a6,p13a7,p14a0,p14a1,p14a2,p14a3,p14a4,p14a5,p14a6,p14a7,p15a0,p15a1,p15a2,p15a3,p15a4,p15a5,p15a6,p15a7

rigid_group_p0 = p0a0,p0a1,p0a2,p0a3,p0a4,p0a5,p0a6,p0a7
rigid_group_p1 = p1a0,p1a1,p1a2,p1a3,p1a4,p1a5,p1a6,p1a7
rigid_group_p2 = p2a0,p2a1,p2a2,p2a3,p2a4,p2a5,p2a6,p2a7
rigid_group_p3 = p3a0,p3a1,p3a2,p3a3,p3a4,p3a5,p3a6,p3a7
rigid_group_p4 = p4a0,p4a1,p4a2,p4a3,p4a4,p4a5,p4a6,p4a7
rigid_group_p5 = p5a0,p5a1,p5a2,p5a3,p5a4,p5a5,p5a6,p5a7
rigid_group_p6 = p6a0,p6a1,p6a2,p6a3,p6a4,p6a5,p6a6,p6a7
rigid_group_p7 = p7a0,p7a1,p7a2,p7a3,p7a4,p7a5,p7a6,p7a7
rigid_group_p8 = p8a0,p8a1,p8a2,p8a3,p8a4,p8a5,p8a6,p8a7
rigid_group_p9 = p9a0,p9a1,p9a2,p9a3,p9a4,p9a5,p9a6,p9a7
rigid_group_p10 = p10a0,p10a1,p10a2,p10a3,p10a4,p10a5,p10a6,p10a7
rigid_group_p11 = p11a0,p11a1,p11a2,p11a3,p11a4,p11a5,p11a6,p11a7
rigid_group_p12 = p12a0,p12a1,p12a2,p12a3,p12a4,p12a5,p12a6,p12a7
rigid_group_p13 = p13a0,p13a1,p13a2,p13a3,p13a4,p13a5,p13a6,p13a7
rigid_group_p14 = p14a0,p14a1,p14a2,p14a3,p14a4,p14a5,p14a6,p14a7
rigid_group_p15 = p15a0,p15a1,p15a2,p15a3,p15a4,p15a5,p15a6,p15a7

rigid_group_collection_quadrants = q0,q1,q2,q3
rigid_group_collection_asics = p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15

"""



CRYSTFEL_PANEL_TEMPLATE = """
{name}/dim1 = {p}
{name}/dim2 = ss
{name}/dim3 = fs
{name}/min_fs = 0
{name}/min_ss = {min_ss}
{name}/max_fs = 127
{name}/max_ss = {max_ss}
{name}/fs = {fs_vec}
{name}/ss = {ss_vec}
{name}/corner_x = {corner_x}
{name}/corner_y = {corner_y}
{name}/coffset = {coffset}
"""

if __name__ == '__main__':
    geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
        (-525, 625),
        (-550, -10),
        (520, -160),
        (542.5, 475),
    ])
    geom.write_crystfel_geom('sample.geom')
