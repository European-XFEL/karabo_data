"""AGIPD & LPD geometry handling."""
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from copy import copy
import h5py
from itertools import product
import numpy as np
from scipy.ndimage import affine_transform
import warnings

__all__ = ['AGIPD_1MGeometry', 'LPD_1MGeometry']


def _crystfel_format_vec(vec):
    """Convert an array of 3 numbers to CrystFEL format like "+1.0x -0.1y"
    """
    s = '{:+}x {:+}y'.format(*vec[:2])
    if vec[2] != 0:
        s += ' {:+}z'.format(vec[2])
    return s


class GeometryFragment:
    """Holds the 3D position & orientation of one detector tile

    corner_pos refers to the corner of the detector tile where the first pixel
    stored is located. The tile is assumed to be a rectangle of ss_pixels in
    the slow scan dimension and fs_pixels in the fast scan dimension.
    ss_vec and fs_vec are vectors for a step of one pixel in each dimension.

    The coordinates in this class are (x, y, z), in pixel units, so the
    magnitude of fs_vec and ss_vec should be 1.
    """

    def __init__(self, corner_pos, ss_vec, fs_vec, ss_pixels, fs_pixels):
        self.corner_pos = corner_pos
        self.ss_vec = ss_vec
        self.fs_vec = fs_vec
        self.ss_pixels = ss_pixels
        self.fs_pixels = fs_pixels

    @classmethod
    def from_panel_dict(cls, d):
        corner_pos = np.array([d['cnx'], d['cny'], d['coffset']])
        ss_vec = np.array([d['ssx'], d['ssy'], d['ssz']])
        fs_vec = np.array([d['fsx'], d['fsy'], d['fsz']])
        ss_pixels = d['max_ss'] - d['min_ss'] + 1
        fs_pixels = d['max_fs'] - d['min_fs'] + 1
        return cls(corner_pos, ss_vec, fs_vec, ss_pixels, fs_pixels)

    def corners(self):
        return np.stack([
            self.corner_pos,
            self.corner_pos + (self.fs_vec * self.fs_pixels),
            self.corner_pos + (self.ss_vec * self.ss_pixels) + (self.fs_vec * self.fs_pixels),
            self.corner_pos + (self.ss_vec * self.ss_pixels),
        ])

    def centre(self):
        return (
            self.corner_pos
            + (0.5 * self.ss_vec * self.ss_pixels)
            + (0.5 * self.fs_vec * self.fs_pixels)
        )

    def to_crystfel_geom(self, p, a):
        name = 'p{}a{}'.format(p, a)
        c = self.corner_pos
        return CRYSTFEL_PANEL_TEMPLATE.format(
            name=name,
            p=p,
            min_ss=(a * self.ss_pixels),
            max_ss=(((a + 1) * self.ss_pixels) - 1),
            ss_vec=_crystfel_format_vec(self.ss_vec),
            fs_vec=_crystfel_format_vec(self.fs_vec),
            corner_x=c[0],
            corner_y=c[1],
            coffset=c[2],
        )

    def snap(self):
        # Round positions and vectors to integers, drop z dimension
        corner_pos = np.around(self.corner_pos[:2]).astype(np.int32)
        ss_vec = np.around(self.ss_vec[:2]).astype(np.int32)
        fs_vec = np.around(self.fs_vec[:2]).astype(np.int32)

        # We should have one vector in the x direction and one in y, but
        # we don't know which is which.
        assert {tuple(np.abs(ss_vec)), tuple(np.abs(fs_vec))} == {(0, 1), (1, 0)}

        # Convert xy coordinates to yx indexes
        return GridGeometryFragment(
            corner_pos[::-1], ss_vec[::-1], fs_vec[::-1], self.ss_pixels, self.fs_pixels
        )


class GridGeometryFragment:
    """Holds the 2D axis-aligned position and orientation of one detector tile.

    This is used in 'snapped' geometry which efficiently assembles a detector
    image into a 2D array.

    These coordinates are all (y, x), suitable for indexing a numpy array.

    ss_vec and fs_vec must be length 1 vectors in either positive or negative
    x or y direction. In the output array, the fast scan dimension is always x.
    So if the input data is oriented with fast-scan vertical, we need to
    transpose it first.

    Regardless of transposition, we may also need to flip the data on one or
    both axes; the fs_order and ss_order variables handle this.
    """
    def __init__(self, corner_pos, ss_vec, fs_vec, ss_pixels, fs_pixels):
        self.ss_vec = ss_vec
        self.fs_vec = fs_vec
        self.ss_pixels = ss_pixels
        self.fs_pixels = fs_pixels

        if fs_vec[0] == 0:
            # Fast scan is x dimension: Flip without transposing
            fs_order = fs_vec[1]
            ss_order = ss_vec[0]
            self.transform = lambda arr: arr[..., ::ss_order, ::fs_order]
            corner_shift = np.array([
                min(ss_order, 0) * self.ss_pixels,
                min(fs_order, 0) * self.fs_pixels
            ])
            self.pixel_dims = np.array([self.ss_pixels, self.fs_pixels])
        else:
            # Fast scan is y : Transpose so fast scan -> x and then flip
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


class DetectorGeometryBase:
    """Base class for detector geometry. Subclassed for specific detectors."""
    # Define in subclasses:
    pixel_size = 0.0
    frag_ss_pixels = 0
    frag_fs_pixels = 0
    expected_data_shape = ()

    def __init__(self, modules, filename='No file'):
        # List of lists (1 per module) of fragments (1 per tile)
        self.modules = modules
        # self.filename is metadata for plots, we don't read/write the file.
        # There are separate methods for reading and writing.
        self.filename = filename
        self._snapped_cache = None

    def inspect(self, frontview=True):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        rects = []
        for module in self.modules:
            for fragment in module:
                corners = fragment.corners()[:, :2]  # Drop the Z dimension
                rects.append(Polygon(corners))

        pc = PatchCollection(rects, facecolor=(0.75, 1.0, 0.75), edgecolor=None)
        ax.add_collection(pc)

        # Draw cross in the centre.
        ax.hlines(0, -100, +100, colors='0.75', linewidths=2)
        ax.vlines(0, -100, +100, colors='0.75', linewidths=2)

        if frontview:
            ax.invert_xaxis()

        return ax

    def _snapped(self):
        """Snap geometry to a 2D pixel grid

        This returns a new geometry object. The 'snapped' geometry is
        less accurate, but can assemble data into a 2D array more efficiently,
        because it doesn't do any interpolation.
        """
        if self._snapped_cache is None:
            new_modules = []
            for module in self.modules:
                new_tiles = [t.snap() for t in module]
                new_modules.append(new_tiles)
            self._snapped_cache = SnappedGeometry(new_modules, self)
        return self._snapped_cache

    @staticmethod
    def split_tiles(module_data):
        """Split data from a detector module into tiles.

        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def position_modules_fast(self, data):
        """Assemble data from this detector according to where the pixels are.

        This approximates the geometry to align all pixels to a 2D grid.

        Parameters
        ----------

        data : ndarray
          The last three dimensions should match the modules, then the
          slow scan and fast scan pixel dimensions.

        Returns
        -------
        out : ndarray
          Array with one dimension fewer than the input.
          The last two dimensions represent pixel y and x in the detector space.
        centre : ndarray
          (y, x) pixel location of the detector centre in this geometry.
        """
        return self._snapped().position_modules(data)

    def position_all_modules(self, data):
        """Deprecated alias for :meth:`position_modules_fast`"""
        return self.position_modules_fast(data)

    def plot_data_fast(self, data, axis_units='px', frontview=True):
        """Plot data from the detector using this geometry.

        This approximates the geometry to align all pixels to a 2D grid.

        Returns a matplotlib axes object.

        Parameters
        ----------

        data : ndarray
          Should have exactly 3 dimensions, for the modules, then the
          slow scan and fast scan pixel dimensions.
        axis_units : str
          Show the detector scale in pixels ('px') or metres ('m').
        frontview : bool
          If True (the default), x increases to the left, as if you were looking
          along the beam. False gives a 'looking into the beam' view.
        """
        return self._snapped().plot_data(
            data, axis_units=axis_units, frontview=frontview
        )


class AGIPD_1MGeometry(DetectorGeometryBase):
    """Detector layout for AGIPD-1M

    The coordinates used in this class are 3D (x, y, z), and represent multiples
    of the pixel size.

    You won't normally instantiate this class directly:
    use one of the constructor class methods to create or load a geometry.
    """
    pixel_size = 2e-4  # 2e-4 metres == 0.2 mm
    frag_ss_pixels = 64
    frag_fs_pixels = 128
    expected_data_shape = (16, 512, 128)

    @classmethod
    def from_quad_positions(cls, quad_pos, asic_gap=2, panel_gap=29,
                            unit=pixel_size):
        """Generate an AGIPD-1M geometry from quadrant positions.

        This produces an idealised geometry, assuming all modules are perfectly
        flat, aligned and equally spaced within their quadrant.

        The quadrant positions are given in pixel units, referring to the first
        pixel of the first module in each quadrant, corresponding to data
        channels 0, 4, 8 and 12.

        The origin of the coordinates is in the centre of the detector.
        Coordinates increase upwards and to the left (looking along the beam).

        To give positions in units other than pixels, pass the *unit* parameter
        as the length of the unit in metres.
        E.g. ``unit=1e-3`` means the coordinates are in millimetres.
        """
        px_conversion = unit / cls.pixel_size
        asic_gap *= px_conversion
        panel_gap *= px_conversion

        quads_x_orientation = [1, 1, -1, -1]
        quads_y_orientation = [-1, -1, 1, 1]
        modules = []
        for p in range(16):
            quad = p // 4
            quad_corner = quad_pos[quad]
            x_orient = quads_x_orientation[quad]
            y_orient = quads_y_orientation[quad]
            p_in_quad = p % 4
            corner_y = (quad_corner[1] * px_conversion)\
                       - (p_in_quad * (cls.frag_fs_pixels + panel_gap))

            tiles = []
            modules.append(tiles)

            for a in range(8):
                corner_x = (quad_corner[0] * px_conversion)\
                           + x_orient * (cls.frag_ss_pixels + asic_gap) * a
                tiles.append(GeometryFragment(
                    corner_pos=np.array([corner_x, corner_y, 0.]),
                    ss_vec=np.array([x_orient, 0, 0]),
                    fs_vec=np.array([0, y_orient, 0]),
                    ss_pixels=cls.frag_ss_pixels,
                    fs_pixels=cls.frag_fs_pixels,
                ))
        return cls(modules)

    @classmethod
    def from_crystfel_geom(cls, filename):
        """Read a CrystFEL format (.geom) geometry file.

        Returns a new geometry object.
        """
        geom_dict = load_crystfel_geometry(filename)
        modules = []
        for p in range(16):
            tiles = []
            modules.append(tiles)
            for a in range(8):
                d = geom_dict['panels']['p{}a{}'.format(p, a)]
                tiles.append(GeometryFragment.from_panel_dict(d))
        return cls(modules, filename=filename)

    def write_crystfel_geom(self, filename):
        """Write this geometry to a CrystFEL format (.geom) geometry file."""
        from . import __version__

        panel_chunks = []
        for p, module in enumerate(self.modules):
            for a, fragment in enumerate(module):
                panel_chunks.append(fragment.to_crystfel_geom(p, a))

        with open(filename, 'w') as f:
            f.write(CRYSTFEL_HEADER_TEMPLATE.format(version=__version__))
            for chunk in panel_chunks:
                f.write(chunk)

        if self.filename == 'No file':
            self.filename = filename

    def inspect(self, frontview=True):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Axes object.

        Parameters
        ----------

        frontview : bool
          If True (the default), x increases to the left, as if you were looking
          along the beam. False gives a 'looking into the beam' view.
        """
        ax = super().inspect(frontview=frontview)

        # Label modules and tiles
        for ch, module in enumerate(self.modules):
            s = 'Q{Q}M{M}'.format(Q=(ch // 4) + 1, M=(ch % 4) + 1)
            cx, cy, _ = module[4].centre()
            ax.text(cx, cy, s, fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='center')

            for t in [0, 7]:
                cx, cy, _ = module[t].centre()
                ax.text(cx, cy, 'T{}'.format(t + 1),
                        verticalalignment='center',
                        horizontalalignment='center')

        ax.set_title('AGIPD-1M detector geometry ({})'.format(self.filename))
        return ax

    def compare(self, other, scale=1.0):
        """Show a comparison of this geometry with another in a 2D plot.

        This shows the current geometry like :meth:`inspect`, with the addition
        of arrows showing how each panel is shifted in the other geometry.

        Parameters
        ----------

        other : AGIPD_1MGeometry
          A second geometry object to compare with this one.
        scale : float
          Scale the arrows showing the difference in positions.
          This is useful to show small differences clearly.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon, FancyArrow

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        rects = []
        arrows = []
        for p, module in enumerate(self.modules):
            for a, fragment in enumerate(module):
                corners = fragment.corners()[:, :2]  # Drop the Z dimension
                corner1, corner1_opp = corners[0], corners[2]

                rects.append(Polygon(corners))
                if a in {0, 7}:
                    cx, cy, _ = fragment.centre()
                    ax.text(cx, cy, str(a),
                            verticalalignment='center',
                            horizontalalignment='center')
                elif a == 4:
                    cx, cy, _ = fragment.centre()
                    ax.text(cx, cy, 'p{}'.format(p),
                            verticalalignment='center',
                            horizontalalignment='center')

                panel2 = other.modules[p][a]
                corners2 = panel2.corners()[:, :2]
                corner2, corner2_opp = corners2[0], corners2[2]
                dx, dy = corner2 - corner1
                if not (dx == dy == 0):
                    sx, sy = corner1
                    arrows.append(FancyArrow(
                        sx, sy, scale * dx, scale * dy, width=5, head_length=4
                    ))

                dx, dy = corner2_opp - corner1_opp
                if not (dx == dy == 0):
                    sx, sy = corner1_opp
                    arrows.append(FancyArrow(
                        sx, sy, scale * dx, scale * dy, width=5, head_length=5
                    ))

        pc = PatchCollection(rects, facecolor=(0.75, 1.0, 0.75), edgecolor=None)
        ax.add_collection(pc)
        ac = PatchCollection(arrows)
        ax.add_collection(ac)

        # Set axis limits to fit all shapes, with some margin
        all_x = np.concatenate([s.xy[:, 0] for s in arrows + rects])
        all_y = np.concatenate([s.xy[:, 1] for s in arrows + rects])
        ax.set_xlim(all_x.min() - 20, all_x.max() + 20)
        ax.set_ylim(all_y.min() - 40, all_y.max() + 20)

        ax.set_title('Geometry comparison: {} → {}'
                     .format(self.filename, other.filename))
        ax.text(1, 0, 'Arrows scaled: {}×'.format(scale),
                horizontalalignment="right", verticalalignment="bottom",
                transform=ax.transAxes)
        return ax

    def position_modules_interpolate(self, data):
        """Assemble data from this detector according to where the pixels are.

        This performs interpolation, which is very slow.
        Use :meth:`position_modules_fast` to get a pixel-aligned approximation
        of the geometry.

        Parameters
        ----------

        data : ndarray
          The three dimensions should be channelno, pixel_ss, pixel_fs
          (lengths 16, 512, 128). ss/fs are slow-scan and fast-scan.

        Returns
        -------
        out : ndarray
          Array with the one dimension fewer than the input.
          The last two dimensions represent pixel y and x in the detector space.
        centre : ndarray
          (y, x) pixel location of the detector centre in this geometry.
        """
        assert data.shape == (16, 512, 128)
        size_yx, centre = self._get_dimensions()
        tmp = np.empty((16 * 8,) + size_yx, dtype=data.dtype)

        for i, (module, mod_data) in enumerate(zip(self.modules, data)):
            tiles_data = np.split(mod_data, 8)
            for j, (tile, tile_data) in enumerate(zip(module, tiles_data)):
                # We store (x, y, z), but numpy indexing, and hence affine_transform,
                # work like [y, x]. Rearrange the numbers:
                fs_vec_yx = tile.fs_vec[:2][::-1]
                ss_vec_yx = tile.ss_vec[:2][::-1]

                # Offset by centre to make all coordinates positive
                corner_pos_yx = tile.corner_pos[:2][::-1] + centre

                # Make the rotation matrix
                rotn = np.stack((ss_vec_yx, fs_vec_yx), axis=-1)

                # affine_transform takes a mapping from *output* to *input*.
                # So we reverse the forward transformation.
                transform = np.linalg.inv(rotn)
                offset = np.dot(rotn, corner_pos_yx)  # this seems to work, but is it right?

                affine_transform(
                    tile_data,
                    transform,
                    offset=offset,
                    cval=np.nan,
                    output_shape=size_yx,
                    output=tmp[i * 8 + j],
                )

        # Silence warnings about nans - we expect gaps in the result
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out = np.nanmax(tmp, axis=0)

        return out, centre

    def _get_dimensions(self):
        """Calculate appropriate array dimensions for assembling data.

        Returns (size_y, size_x), (centre_y, centre_x)
        """
        corners = []
        for module in self.modules:
            for tile in module:
                corners.append(tile.corners())
        corners = np.concatenate(corners)[:, :2]

        # Find extremes, add 1 px margin to allow for rounding errors
        min_xy = corners.min(axis=0).astype(int) - 1
        max_xy = corners.max(axis=0).astype(int) + 1

        size = max_xy - min_xy
        centre = -min_xy
        # Switch xy -> yx
        return tuple(size[::-1]), centre[::-1]

    @staticmethod
    def split_tiles(module_data):
        # Split into 8 tiles along the slow-scan axis
        return np.split(module_data, 8, axis=-2)

    def to_distortion_array(self):
        """Return distortion matrix for AGIPD detector, suitable for pyFAI.

        Returns
        -------
        out: ndarray
            Array of float 32 with shape (8192, 128, 4, 3).
            The dimensions mean:

            - 8192 = 16 modules * 512 pixels (slow scan axis)
            - 128 pixels (fast scan axis)
            - 4 corners of each pixel
            - 3 numbers for z, y, x
        """
        distortion = np.zeros((8192, 128, 4, 3), dtype=np.float32)

        # Prepare some arrays to use inside the loop
        pixel_ss_index, pixel_fs_index = np.meshgrid(
            np.arange(0, 64), np.arange(0, 128), indexing='ij'
        )
        corner_ss_offsets = np.array([0, 1, 1, 0])
        corner_fs_offsets = np.array([0, 0, 1, 1])

        for m, mod in enumerate(self.modules, start=0):
            # module offset along first dimension of distortion array
            module_offset = m * 512

            for t, tile in enumerate(mod, start=0):
                corner_x, corner_y, corner_z = tile.corner_pos * self.pixel_size
                ss_unit_x, ss_unit_y, ss_unit_z = tile.ss_vec * self.pixel_size
                fs_unit_x, fs_unit_y, fs_unit_z = tile.fs_vec * self.pixel_size

                # Calculate coordinates of each pixel's first corner
                # 2D arrays, shape: (64, 128)
                pixel_corner1_x = (
                    corner_x
                    + pixel_ss_index * ss_unit_x
                    + pixel_fs_index * fs_unit_x
                )
                pixel_corner1_y = (
                    corner_y
                    + pixel_ss_index * ss_unit_y
                    + pixel_fs_index * fs_unit_y
                )
                pixel_corner1_z = (
                    corner_z
                    + pixel_ss_index * ss_unit_z +
                    + pixel_fs_index * fs_unit_z
                )

                # Calculate corner coordinates for each pixel
                # 3D arrays, shape: (64, 128, 4)
                corners_x = (
                    pixel_corner1_x[:, :, np.newaxis]
                    + corner_ss_offsets * ss_unit_x
                    + corner_fs_offsets * fs_unit_x
                )
                corners_y = (
                    pixel_corner1_y[:, :, np.newaxis]
                    + corner_ss_offsets * ss_unit_y
                    + corner_fs_offsets * fs_unit_y
                )
                corners_z = (
                    pixel_corner1_z[:, :, np.newaxis]
                    + corner_ss_offsets * ss_unit_z
                    + corner_fs_offsets * fs_unit_z
                )

                # Which part of the array is this tile?
                tile_offset = module_offset + (t * 64)
                tile_slice = slice(tile_offset, tile_offset + tile.ss_pixels)

                # Insert the data into the array
                distortion[tile_slice, :, :, 0] = corners_z
                distortion[tile_slice, :, :, 1] = corners_y
                distortion[tile_slice, :, :, 2] = corners_x

        # Shift the x & y origin from the centre to the corner
        min_yx = distortion[..., 1:].min(axis=(0, 1, 2))
        distortion[..., 1:] -= min_yx

        return distortion


class SnappedGeometry:
    """Detector geometry approximated to align modules to a 2D grid

    The coordinates used in this class are (y, x) suitable for indexing a
    Numpy array; this does not match the (x, y, z) coordinates in the more
    precise geometry above.
    """
    def __init__(self, modules, geom: DetectorGeometryBase):
        self.modules = modules
        self.geom = geom

    def position_modules(self, data):
        """Implementation for position_modules_fast
        """
        assert data.shape[-3:] == self.geom.expected_data_shape
        size_yx, centre = self._get_dimensions()
        out = np.full(data.shape[:-3] + size_yx, np.nan, dtype=data.dtype)
        for i, module in enumerate(self.modules):
            mod_data = data[..., i, :, :]
            tiles_data = self.geom.split_tiles(mod_data)
            for j, tile in enumerate(module):
                tile_data = tiles_data[j]
                # Offset by centre to make all coordinates positive
                y, x = tile.corner_idx + centre
                h, w = tile.pixel_dims
                out[..., y : y + h, x : x + w] = tile.transform(tile_data)

        return out, centre

    def _get_dimensions(self):
        """Calculate appropriate array dimensions for assembling data.

        Returns (size_y, size_x), (centre_y, centre_x)
        """
        corners = []
        for module in self.modules:
            for tile in module:
                corners.append(tile.corner_idx)
                corners.append(tile.opp_corner_idx)
        corners = np.stack(corners)

        # Find extremes
        min_yx = corners.min(axis=0)
        max_yx = corners.max(axis=0)

        size = max_yx - min_yx
        centre = -min_yx
        return tuple(size), centre

    def plot_data(self, modules_data, axis_units='px', frontview=True):
        """Implementation for plot_data_fast
        """
        from matplotlib.cm import viridis
        import matplotlib.pyplot as plt

        if axis_units not in {'px', 'm'}:
            raise ValueError("axis_units must be 'px' or 'm', not {!r}"
                             .format(axis_units))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        my_viridis = copy(viridis)
        # Use a dark grey for missing data
        my_viridis.set_bad('0.25', 1.0)

        res, centre = self.position_modules(modules_data)
        min_y, min_x = -centre
        max_y, max_x = np.array(res.shape) - centre

        extent = np.array((min_x - 0.5, max_x + 0.5, min_y - 0.5, max_y + 0.5))
        cross_size = 20
        if axis_units == 'm':
            extent *= self.geom.pixel_size
            cross_size *= self.geom.pixel_size

        ax.imshow(res, origin='lower', cmap=my_viridis, extent=extent)
        ax.set_xlabel('metres' if axis_units == 'm' else 'pixels')
        ax.set_ylabel('metres' if axis_units == 'm' else 'pixels')

        if frontview:
            ax.invert_xaxis()

        # Draw a cross at the centre
        ax.hlines(0, -cross_size, +cross_size, colors='w', linewidths=1)
        ax.vlines(0, -cross_size, +cross_size, colors='w', linewidths=1)
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


class LPD_1MGeometry(DetectorGeometryBase):
    """Detector layout for LPD-1M

    The coordinates used in this class are 3D (x, y, z), and represent multiples
    of the pixel size.

    You won't normally instantiate this class directly:
    use one of the constructor class methods to create or load a geometry.
    """
    pixel_size = 5e-4  # 5e-4 metres == 0.5 mm
    frag_ss_pixels = 32
    frag_fs_pixels = 128
    expected_data_shape = (16, 256, 256)

    @classmethod
    def from_quad_positions(cls, quad_pos, *, unit=1e-3, asic_gap=None,
                            panel_gap=None):
        """Generate an LPD-1M geometry from quadrant positions.

        This produces an idealised geometry, assuming all modules are perfectly
        flat, aligned and equally spaced within their quadrant.

        The quadrant positions refer to the corner of each quadrant
        where module 4, tile 16 is positioned.
        This is the corner of the last pixel as the data is stored.
        In the initial detector layout, the corner positions are for the top
        left corner of the quadrant, looking along the beam.

        The origin of the coordinates is in the centre of the detector.
        Coordinates increase upwards and to the left (looking along the beam).

        Parameters
        ----------
        quad_pos: list of 2-tuples
          (x, y) coordinates of the last corner (the one by module 4) of each
          quadrant.
        unit: float, optional
          The conversion factor to put the coordinates into metres.
          The default 1e-3 means the numbers are in millimetres.
        asic_gap: float, optional
          The gap between adjacent tiles/ASICs. The default is 4 pixels.
        panel_gap: float, optional
          The gap between adjacent modules/panels. The default is 4 pixels.
        """
        px_conversion = unit / cls.pixel_size
        asic_gap_px = 4 if (asic_gap is None) else asic_gap * px_conversion
        panel_gap_px = 4 if (panel_gap is None) else panel_gap * px_conversion

        # How much space one panel/module takes up, including the 'panel gap'
        # separating it from its neighbour.
        # In the x dimension, we have only one asic gap (down the centre)
        panel_width = 256 + asic_gap_px + panel_gap_px
        # In y, we have 7 gaps between the 8 ASICs in each column.
        panel_height = 256 + (7 * asic_gap_px) + panel_gap_px

        tile_size = np.array([cls.frag_fs_pixels, cls.frag_ss_pixels, 0])

        panels_across = [-1, -1, 0, 0]
        panels_up = [0, -1, -1, 0]
        modules = []
        for p in range(16):
            quad = p // 4
            quad_corner_x = quad_pos[quad][0] * px_conversion
            quad_corner_y = quad_pos[quad][1] * px_conversion

            p_in_quad = p % 4
            # Top beam-left corner of panel
            panel_corner_x = (quad_corner_x +
                              (panels_across[p_in_quad] * panel_width))
            panel_corner_y = (quad_corner_y +
                              (panels_up[p_in_quad] * panel_height))

            tiles = []
            modules.append(tiles)

            for a in range(16):
                if a < 8:
                    up = -a
                    across = -1
                else:
                    up = -(15 - a)
                    across = 0

                tile_last_corner = (
                    np.array([panel_corner_x, panel_corner_y, 0.0])
                    + np.array([across, 0, 0]) * (cls.frag_fs_pixels + asic_gap_px)
                    + np.array([0, up, 0]) * (cls.frag_ss_pixels + asic_gap_px)
                )
                tile_first_corner = tile_last_corner - tile_size

                tiles.append(GeometryFragment(
                    corner_pos=tile_first_corner,
                    ss_vec=np.array([0, 1, 0]),
                    fs_vec=np.array([1, 0, 0]),
                    ss_pixels=cls.frag_ss_pixels,
                    fs_pixels=cls.frag_fs_pixels,
                ))
        return cls(modules)

    @classmethod
    def from_h5_file_and_quad_positions(cls, path, positions, unit=1e-3):
        """Load an LPD-1M geometry from an XFEL HDF5 format geometry file

        The quadrant positions are not stored in the file, and must be provided
        separately. By default, both the quadrant positions and the positions
        in the file are measured in millimetres; the unit parameter controls
        this.

        The origin of the coordinates is in the centre of the detector.
        Coordinates increase upwards and to the left (looking along the beam).

        This version of the code only handles x and y translation,
        as this is all that is recorded in the initial LPD geometry file.

        Parameters
        ----------

        path : str
          Path of an EuXFEL format (HDF5) geometry file for LPD.
        positions : list of 2-tuples
          (x, y) coordinates of the last corner (the one by module 4) of each
          quadrant.
        unit : float, optional
          The conversion factor to put the coordinates into metres.
          The default 1e-3 means the numbers are in millimetres.
        """
        assert len(positions) == 4
        modules = []
        with h5py.File(path, 'r') as f:
            for Q, M in product(range(1, 5), range(1, 5)):
                quad_pos = np.array(positions[Q - 1])
                mod_grp = f['Q{}/M{}'.format(Q, M)]
                mod_offset = mod_grp['Position'][:2]

                tiles = []
                for T in range(1, 17):
                    corner_pos = np.zeros(3)
                    tile_offset = mod_grp['T{:02}/Position'.format(T)][:2]
                    corner_pos[:2] = quad_pos + mod_offset + tile_offset

                    # Convert units (mm) to pixels
                    corner_pos *= unit / cls.pixel_size

                    # LPD geometry is measured to the last pixel of each tile.
                    # Subtract tile dimensions for the position of 1st pixel.
                    ss_vec, fs_vec = np.array([0, 1, 0]), np.array([1, 0, 0])
                    first_px_pos = (corner_pos
                                    - (ss_vec * cls.frag_ss_pixels)
                                    - (fs_vec * cls.frag_fs_pixels))

                    tiles.append(GeometryFragment(
                        corner_pos=first_px_pos,
                        ss_vec=ss_vec,
                        fs_vec=fs_vec,
                        ss_pixels=cls.frag_ss_pixels,
                        fs_pixels=cls.frag_fs_pixels,
                    ))
                modules.append(tiles)

        return cls(modules, filename=path)

    def inspect(self, frontview=True):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Axes object.

        Parameters
        ----------

        frontview : bool
          If True (the default), x increases to the left, as if you were looking
          along the beam. False gives a 'looking into the beam' view.
        """
        ax = super().inspect(frontview=frontview)

        # Label modules and tiles
        for ch, module in enumerate(self.modules):
            s = 'Q{Q}M{M}'.format(Q=(ch // 4) + 1, M=(ch % 4) + 1)
            cx, cy, _ = module[0].centre()
            ax.text(cx, cy, s, fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='center')

            for t in [7, 8, 15]:
                cx, cy, _ = module[t].centre()
                ax.text(cx, cy, 'T{}'.format(t + 1),
                        verticalalignment='center',
                        horizontalalignment='center')

        ax.set_title('LPD-1M detector geometry ({})'.format(self.filename))
        return ax

    @staticmethod
    def split_tiles(module_data):
        half1, half2 = np.split(module_data, 2, axis=-1)
        # Tiles 1-8 (half1) are numbered top to bottom, whereas the array
        # starts at the bottom. So we reverse their order after splitting.
        return np.split(half1, 8, axis=-2)[::-1] + np.split(half2, 8, axis=-2)


def invert_xfel_lpd_geom(path_in, path_out):
    """Invert the coordinates in an XFEL geometry file (HDF5)

    The initial geometry file for LPD was recorded with the coordinates
    increasing down and to the right (looking in the beam direction), but the
    standard XFEL coordinate scheme is the opposite, increasing upwards and to
    the left (looking in beam direction).

    This utility function reads one file, and writes a second with the
    coordinates inverted.
    """
    with h5py.File(path_in, 'r') as fin, h5py.File(path_out, 'x') as fout:
        src_ds = fin['DetectorDescribtion']
        dst_ds = fout.create_dataset('DetectorDescription', data=src_ds)
        for k, v in src_ds.attrs.items():
            dst_ds.attrs[k] = v

        for Q, M in product(range(1, 5), range(1, 5)):
            path = 'Q{}/M{}/Position'.format(Q, M)
            fout[path] = -fin[path][:]
            for T in range(1, 17):
                path = 'Q{}/M{}/T{:02}/Position'.format(Q, M, T)
                fout[path] = -fin[path][:]


if __name__ == '__main__':
    geom = AGIPD_1MGeometry.from_quad_positions(
        quad_pos=[(-525, 625), (-550, -10), (520, -160), (542.5, 475)]
    )
    geom.write_crystfel_geom('sample.geom')
