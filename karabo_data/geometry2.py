from cfelpyutils.crystfel_utils import load_crystfel_geometry
from copy import copy
import numpy as np
from scipy.ndimage import affine_transform
import warnings

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

    def snap(self):
        corner_pos = np.around(self.corner_pos[:2]).astype(np.int32)
        ss_vec = np.around(self.ss_vec[:2]).astype(np.int32)
        fs_vec = np.around(self.fs_vec[:2]).astype(np.int32)
        assert {tuple(np.abs(ss_vec)), tuple(np.abs(fs_vec))} == {(0, 1), (1, 0)}
        # Convert xy coordinates to yx indexes
        return GridGeometryFragment(corner_pos[::-1], ss_vec[::-1], fs_vec[::-1])

class GridGeometryFragment:
    ss_pixels = 64
    fs_pixels = 128

    # These coordinates are all (y, x), suitable for indexing a numpy array.
    def __init__(self, corner_pos, ss_vec, fs_vec):
        self.ss_vec = ss_vec
        self.fs_vec = fs_vec
        if fs_vec[0] == 0:
            # Flip without transposing
            fs_order = fs_vec[1]
            ss_order = ss_vec[0]
            self.transform = lambda arr: arr[..., ::ss_order, ::fs_order]
            corner_shift = np.array([
                min(ss_order, 0) * self.ss_pixels,
                min(fs_order, 0) * self.fs_pixels
            ])
            self.pixel_dims = np.array([self.ss_pixels, self.fs_pixels])
        else:
            # Transpose and then flip
            fs_order = fs_vec[0]
            ss_order = ss_vec[1]
            self.transform = lambda arr: arr.swapaxes(-1, -2)[..., ::fs_order, ::ss_order]
            corner_shift = np.array([
                min(fs_order, 0) * self.fs_pixels,
                min(ss_order, 0) * self.ss_pixels
            ])
            self.pixel_dims = np.array([self.fs_pixels, self.ss_pixels])
        self.corner_idx = corner_pos + corner_shift
        self.opp_corner_idx = self.corner_idx + self.pixel_dims


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

    def position_all_modules(self, data):
        """Assemble data from this detector according to where the pixels are.

        Parameters
        ----------

        data : ndarray
          The three dimensions should be channelno, pixel_y, pixel_x
          (lengths 16, 512, 128).

        Returns
        -------
        out : ndarray
          Array with the one dimension fewer than the input.
          The last two dimensions represent pixel y and x in the detector space.
        centre : ndarray
          (x, y) pixel location of the detector centre in this geometry.
        """
        assert data.shape == (16, 512, 128)
        size_xy, centre = self._plotting_dimensions()
        size_yx = size_xy[::-1]
        tmp = np.empty((16 * 8,) + size_yx, dtype=data.dtype)

        for i, (module, mod_data) in enumerate(zip(self.modules, data)):
            tiles_data = np.split(mod_data, 8)
            for j, (tile, tile_data) in enumerate(zip(module, tiles_data)):
                # We store (x, y, z), but numpy indexing, and hence affine_transform,
                # work like [y, x]. Rearrange the numbers:
                fs_vec_yx = tile.fs_vec[:2][::-1]
                ss_vec_yx = tile.ss_vec[:2][::-1]

                # Offset by centre to make all coordinates positive
                corner_pos = (tile.corner_pos[:2] + centre)
                corner_pos_yx = corner_pos[::-1]

                # Make the rotation matrix
                rotn = np.stack((ss_vec_yx, fs_vec_yx), axis=-1)

                # affine_transform takes a mapping from *output* to *input*.
                # So we reverse the forward transformation.
                transform = np.linalg.inv(rotn)
                offset = np.dot(rotn, corner_pos_yx)  # this seems to work, but is it right?

                affine_transform(tile_data, transform, offset=offset, cval=np.nan,
                                 output_shape=size_yx, output=tmp[i * 8 + j])

        # Silence warnings about nans - we expect gaps in the result
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.nanmax(tmp, axis=0)

        return out, centre

    def _plotting_dimensions(self):
        """Calculate appropriate dimensions for plotting assembled data

        Returns (size_x, size_y), (centre_x, centre_y)
        """
        corners = []
        for module in self.modules:
            for tile in module:
                corners.append(tile.corners())
        corners = np.concatenate(corners)[:, :2]

        # Find extremes, add 20 px margin
        min_xy = corners.min(axis=0).astype(int) - 20
        max_xy = corners.max(axis=0).astype(int) + 20

        size = max_xy - min_xy
        centre = -min_xy
        return tuple(size), centre

    def plot_data(self, modules_data):
        """Plot data from the detector using this geometry.

        Returns a matplotlib figure.

        Parameters
        ----------

        modules_data : ndarray
          Should have exactly 3 dimensions: channelno, pixel_y, pixel_x
          (lengths 16, 512, 128).
        """
        from matplotlib.cm import viridis
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure((10, 10))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        my_viridis = copy(viridis)
        # Use a dark grey for missing data
        my_viridis.set_bad('0.25', 1.)

        res, centre = self.position_all_modules(modules_data)
        ax.imshow(res, origin='lower', cmap=my_viridis)

        cx, cy = centre
        ax.hlines(cy, cx - 20, cx + 20, colors='w', linewidths=1)
        ax.vlines(cx, cy - 20, cy + 20, colors='w', linewidths=1)
        return fig

    def snap(self):
        """Snap geometry to a 2D pixel grid

        This returns a new geometry object. The 'snapped' geometry is
        less accurate, but can assemble data into a 2D array more efficiently,
        because it doesn't do any interpolation.
        """
        new_modules = []
        for module in self.modules:
            new_tiles = [t.snap() for t in module]
            new_modules.append(new_tiles)
        return AGIPD_1M_SnappedGeometry(new_modules)

class AGIPD_1M_SnappedGeometry:
    """AGIPD geometry approximated to align modules to a 2D grid

    The coordinates used in this class are (y, x) suitable for indexing a
    Numpy array; this does not match the (x, y) coordinates in the more
    precise geometry above.
    """
    def __init__(self, modules):
        self.modules = modules

    def position_all_modules(self, data):
        """Assemble data from this detector according to where the pixels are.

        Parameters
        ----------

        data : ndarray
          The three dimensions should be channelno, pixel_y, pixel_x
          (lengths 16, 512, 128).

        Returns
        -------
        out : ndarray
          Array with one dimension fewer than the input.
          The last two dimensions represent pixel y and x in the detector space.
        centre : ndarray
          (y, x) pixel location of the detector centre in this geometry.
        """
        assert data.shape == (16, 512, 128)
        size_yx, centre = self._plotting_dimensions()
        out = np.full(size_yx, np.nan, dtype=data.dtype)

        for i, (module, mod_data) in enumerate(zip(self.modules, data)):
            tiles_data = np.split(mod_data, 8)
            for j, (tile, tile_data) in enumerate(zip(module, tiles_data)):

                # Offset by centre to make all coordinates positive
                y, x = tile.corner_idx + centre
                h, w = tile.pixel_dims

                out[y:y+h, x:x+w] = tile.transform(tile_data)

        return out, centre

    def _plotting_dimensions(self):
        """Calculate appropriate dimensions for plotting assembled data

        Returns (size_y, size_x), (centre_y, centre_x)
        """
        corners = []
        for module in self.modules:
            for tile in module:
                corners.append(tile.corner_idx)
                corners.append(tile.opp_corner_idx)
        corners = np.stack(corners)

        # Find extremes, add 20 px margin
        min_yx = corners.min(axis=0) - 20
        max_yx = corners.max(axis=0) + 20

        size = max_yx - min_yx
        centre = -min_yx
        return tuple(size), centre

    def plot_data(self, modules_data):
        """Plot data from the detector using this geometry.

        Returns a matplotlib figure.

        Parameters
        ----------

        modules_data : ndarray
          Should have exactly 3 dimensions: channelno, pixel_y, pixel_x
          (lengths 16, 256, 256).
        """
        from matplotlib.cm import viridis
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure((10, 10))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        my_viridis = copy(viridis)
        # Use a dark grey for missing data
        my_viridis.set_bad('0.25', 1.)

        res, centre = self.position_all_modules(modules_data)
        ax.imshow(res, origin='lower', cmap=my_viridis)

        cy, cx = centre
        ax.hlines(cy, cx - 20, cx + 20, colors='w', linewidths=1)
        ax.vlines(cx, cy - 20, cy + 20, colors='w', linewidths=1)
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
