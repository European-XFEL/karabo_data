"""AGIPD & LPD geometry handling."""
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from copy import copy
import h5py
from itertools import product
import numpy as np
from scipy.ndimage import affine_transform
import warnings

from .crystfel_fmt import write_crystfel_geom

__all__ = ['AGIPD_1MGeometry', 'LPD_1MGeometry']


class GeometryFragment:
    """Holds the 3D position & orientation of one detector tile

    corner_pos refers to the corner of the detector tile where the first pixel
    stored is located. The tile is assumed to be a rectangle of ss_pixels in
    the slow scan dimension and fs_pixels in the fast scan dimension.
    ss_vec and fs_vec are vectors for a step of one pixel in each dimension.

    The coordinates in this class are (x, y, z), in metres.
    """

    def __init__(self, corner_pos, ss_vec, fs_vec, ss_pixels, fs_pixels):
        self.corner_pos = corner_pos
        self.ss_vec = ss_vec
        self.fs_vec = fs_vec
        self.ss_pixels = ss_pixels
        self.fs_pixels = fs_pixels

    @classmethod
    def from_panel_dict(cls, d):
        res = d['res']
        corner_pos = np.array([d['cnx'], d['cny'], d['coffset']]) / res
        ss_vec = np.array([d['ssx'], d['ssy'], d['ssz']]) / res
        fs_vec = np.array([d['fsx'], d['fsy'], d['fsz']]) / res
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

    def snap(self, px_shape):
        # Round positions and vectors to integers, drop z dimension
        corner_pos = np.around(self.corner_pos[:2] / px_shape).astype(np.int32)
        ss_vec = np.around(self.ss_vec[:2] / px_shape).astype(np.int32)
        fs_vec = np.around(self.fs_vec[:2] / px_shape).astype(np.int32)

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
    n_modules = 0
    n_tiles_per_module = 0
    expected_data_shape = (0, 0, 0)
    _pixel_corners = np.array([  # pixel units; overridden for DSSC
        [0, 1, 1, 0],  # slow-scan
        [0, 0, 1, 1]   # fast-scan
    ])
    _draw_first_px_on_tile = 1  # Tile num of 1st pixel - overridden for LPD

    @property
    def _pixel_shape(self):
        """Pixel (x, y) shape. Overridden for DSSC."""
        return np.array([1., 1.], dtype=np.float64) * self.pixel_size

    def __init__(self, modules, filename='No file'):
        # List of lists (1 per module) of fragments (1 per tile)
        self.modules = modules
        # self.filename is metadata for plots, we don't read/write the file.
        # There are separate methods for reading and writing.
        self.filename = filename
        self._snapped_cache = None

    def _get_plot_scale_factor(self, axis_units):
        if axis_units == 'm':
            return 1
        elif axis_units == 'px':
            return 1 / self.pixel_size
        else:
            raise ValueError("axis_units must be 'px' or 'm', not {!r}"
                             .format(axis_units))

    def inspect(self, axis_units='px', frontview=True):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection, LineCollection
        from matplotlib.patches import Polygon

        scale = self._get_plot_scale_factor(axis_units)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        rects = []
        first_rows = []
        for module in self.modules:
            for t, fragment in enumerate(module, start=1):
                corners = fragment.corners()[:, :2]  # Drop the Z dimension
                rects.append(Polygon(corners * scale))

                if t == self._draw_first_px_on_tile:
                    # Find the ends of the first row in reading order
                    c1 = fragment.corner_pos * scale
                    c2 = c1 + (fragment.fs_vec * fragment.fs_pixels * scale)
                    first_rows.append((c1[:2], c2[:2]))

        # Add tile shapes
        pc = PatchCollection(rects, facecolor=(0.75, 1.0, 0.75), edgecolor=None)
        ax.add_collection(pc)

        # Add markers for first pixels & lines for first row
        first_rows = np.array(first_rows)
        first_px_x, first_px_y = first_rows[:, 0, 0], first_rows[:, 0, 1]

        ax.scatter(first_px_x, first_px_y, marker='x', label='First pixel')
        ax.add_collection(LineCollection(
            first_rows, linestyles=':', color='k', label='First row'
        ))
        ax.legend()

        cross_size = 0.02 * scale

        # Draw cross in the centre.
        ax.hlines(0, -cross_size, +cross_size, colors='0.75', linewidths=2)
        ax.vlines(0, -cross_size, +cross_size, colors='0.75', linewidths=2)

        if frontview:
            ax.invert_xaxis()

        ax.set_xlabel('metres' if axis_units == 'm' else 'pixels')
        ax.set_ylabel('metres' if axis_units == 'm' else 'pixels')

        return ax

    @classmethod
    def from_crystfel_geom(cls, filename):
        """Read a CrystFEL format (.geom) geometry file.

        Returns a new geometry object.
        """
        geom_dict = load_crystfel_geometry(filename)
        modules = []
        for p in range(cls.n_modules):
            tiles = []
            modules.append(tiles)
            for a in range(cls.n_tiles_per_module):
                d = geom_dict['panels']['p{}a{}'.format(p, a)]
                tiles.append(GeometryFragment.from_panel_dict(d))
        return cls(modules, filename=filename)

    def write_crystfel_geom(self, filename, *,
                            data_path='/entry_1/instrument_1/detector_1/data',
                            mask_path=None, dims=('frame', 'modno', 'ss', 'fs'),
                            adu_per_ev=None, clen=None, photon_energy=None):
        """Write this geometry to a CrystFEL format (.geom) geometry file.

        Parameters
        ----------

        filename : str
            Filename of the geometry file to write.
        data_path : str
            Path to the group that contains the data array in the hdf5 file.
            Default: ``'/entry_1/instrument_1/detector_1/data'``.
        mask_path : str
            Path to the group that contains the mask array in the hdf5 file.
        dims : tuple
            Dimensions of the data. Extra dimensions, except for the defaults,
            should be added by their index, e.g.
            ('frame', 'modno', 0, 'ss', 'fs') for raw data.
            Default: ``('frame', 'modno', 'ss', 'fs')``.
            Note: the dimensions must contain frame, ss, fs.
        adu_per_ev : float
            ADU (analog digital units) per electron volt for the considered
            detector.
        clen : float
            Distance between sample and detector in meters
        photon_energy : float
            Beam wave length in eV
        """
        write_crystfel_geom(
            self, filename, data_path=data_path, mask_path=mask_path, dims=dims,
            adu_per_ev=adu_per_ev, clen=clen, photon_energy=photon_energy,
        )

        if self.filename == 'No file':
            self.filename = filename

    def _snapped(self):
        """Snap geometry to a 2D pixel grid

        This returns a new geometry object. The 'snapped' geometry is
        less accurate, but can assemble data into a 2D array more efficiently,
        because it doesn't do any interpolation.
        """
        if self._snapped_cache is None:
            new_modules = []
            for module in self.modules:
                new_tiles = [t.snap(px_shape=self._pixel_shape) for t in module]
                new_modules.append(new_tiles)
            self._snapped_cache = SnappedGeometry(new_modules, self)
        return self._snapped_cache

    @staticmethod
    def split_tiles(module_data):
        """Split data from a detector module into tiles.

        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def output_array_for_position_fast(self, extra_shape=(), dtype=np.float32):
        """Make an empty output array to use with position_modules_fast

        You can speed up assembling images by reusing the same output array:
        call this once, and then pass the array as the ``out=`` parameter to
        :meth:`position_modules_fast()`. By default, it allocates a new array on
        each call, which can be slow.

        Parameters
        ----------

        extra_shape : tuple, optional
          By default, a 2D output array is generated, to assemble a single
          detector image. If you are assembling multiple pulses at once, pass
          ``extra_shape=(nframes,)`` to get a 3D output array.
        dtype : optional (Default: np.float32)
        """
        return self._snapped().make_output_array(extra_shape=extra_shape,
                                                 dtype=dtype)

    def position_modules_fast(self, data, out=None):
        """Assemble data from this detector according to where the pixels are.

        This approximates the geometry to align all pixels to a 2D grid.

        Parameters
        ----------

        data : ndarray
          The last three dimensions should match the modules, then the
          slow scan and fast scan pixel dimensions.
        out : ndarray, optional
          An output array to assemble the image into. By default, a new
          array is allocated. Use :meth:`output_array_for_position_fast` to
          create a suitable array.
          If an array is passed in, it must match the dtype of the data and the
          shape of the array that would have been allocated.
          Parts of the array not covered by detector tiles are not overwritten.
          In general, you can reuse an output array if you are assembling
          similar pulses or pulse trains with the same geometry.

        Returns
        -------
        out : ndarray
          Array with one dimension fewer than the input.
          The last two dimensions represent pixel y and x in the detector space.
        centre : ndarray
          (y, x) pixel location of the detector centre in this geometry.
        """
        return self._snapped().position_modules(data, out=out)

    def position_all_modules(self, data, out=None):
        """Deprecated alias for :meth:`position_modules_fast`"""
        return self.position_modules_fast(data, out=out)

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

    @classmethod
    def _distortion_array_slice(cls, m, t):
        """Which part of distortion array each tile is.
        """
        # _tile_slice gives the slice for the tile within its module.
        # The distortion array joins the modules along the slow-scan axis, so
        # we need to offset the slow-scan slice to land in the correct module.
        ss_slice_inmod, fs_slice = cls._tile_slice(t)
        mod_px_ss = cls.expected_data_shape[1]
        mod_offset = m * mod_px_ss
        ss_slice = slice(
            ss_slice_inmod.start + mod_offset, ss_slice_inmod.stop + mod_offset
        )
        return ss_slice, fs_slice

    def to_distortion_array(self, allow_negative_xy=False):
        """Generate a distortion array for pyFAI from this geometry.
        """
        nmods, mod_px_ss, mod_px_fs = self.expected_data_shape
        ncorners = self._pixel_corners.shape[1]
        distortion = np.zeros((nmods * mod_px_ss, mod_px_fs, ncorners, 3),
                              dtype=np.float32)

        pixpos = self.get_pixel_positions(centre=False).reshape(
            (nmods * mod_px_ss, mod_px_fs, 3)
        )
        px, py, pz = np.moveaxis(pixpos, -1, 0)

        corner_ss_offsets = self._pixel_corners[0]
        corner_fs_offsets = self._pixel_corners[1]

        for m, mod in enumerate(self.modules, start=0):
            for t, tile in enumerate(mod, start=0):
                ss_unit_x, ss_unit_y, ss_unit_z = tile.ss_vec
                fs_unit_x, fs_unit_y, fs_unit_z = tile.fs_vec

                # Which part of the array is this tile?
                tile_ss_slice, tile_fs_slice = self._distortion_array_slice(m, t)

                # Get coordinates of each pixel's first corner
                # 2D arrays, shape: (64, 128)
                pixel_corner1_x = px[tile_ss_slice,  tile_fs_slice]
                pixel_corner1_y = py[tile_ss_slice,  tile_fs_slice]
                pixel_corner1_z = pz[tile_ss_slice,  tile_fs_slice]

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

                # Insert the data into the array
                distortion[tile_ss_slice, tile_fs_slice, :, 0] = corners_z
                distortion[tile_ss_slice, tile_fs_slice, :, 1] = corners_y
                distortion[tile_ss_slice, tile_fs_slice, :, 2] = corners_x

        if not allow_negative_xy:
            # Shift the x & y origin from the centre to the corner
            min_yx = distortion[..., 1:].min(axis=(0, 1, 2))
            distortion[..., 1:] -= min_yx

        return distortion

    @classmethod
    def _tile_slice(cls, tileno):
        """Implement in subclass: which part of module array each tile is.
        """
        raise NotImplementedError

    def _module_coords_to_tile(self, slow_scan, fast_scan):
        """Implement in subclass: positions in module to tile numbers & pos in tile
        """
        raise NotImplementedError

    @classmethod
    def _adjust_pixel_coords(cls, ss_coords, fs_coords, centre):
        """Called by get_pixel_positions; overridden by DSSC"""
        if centre:
            # A pixel is from n to n+1 in each axis, so centres are at n+0.5.
            ss_coords += 0.5
            fs_coords += 0.5

    def get_pixel_positions(self, centre=True):
        """Get the physical coordinates of each pixel in the detector

        The output is an array with shape like the data, with an extra dimension
        of length 3 to hold (x, y, z) coordinates. Coordinates are in metres.

        If centre=True, the coordinates are calculated for the centre of each
        pixel. If not, the coordinates are for the first corner of the pixel
        (the one nearest the [0, 0] corner of the tile in data space).
        """
        out = np.zeros(self.expected_data_shape + (3,), dtype=np.float64)

        # Prepare some arrays to use inside the loop
        pixel_ss_coord, pixel_fs_coord = np.meshgrid(
            np.arange(0, self.frag_ss_pixels, dtype=np.float64),
            np.arange(0, self.frag_fs_pixels, dtype=np.float64),
            indexing='ij'
        )

        # Shift coordinates from corner to centre if requested.
        # This is also where the DSSC subclass shifts odd rows by half a pixel
        self._adjust_pixel_coords(pixel_ss_coord, pixel_fs_coord, centre)

        for m, mod in enumerate(self.modules, start=0):
            for t, tile in enumerate(mod, start=0):
                corner_x, corner_y, corner_z = tile.corner_pos
                ss_unit_x, ss_unit_y, ss_unit_z = tile.ss_vec
                fs_unit_x, fs_unit_y, fs_unit_z = tile.fs_vec

                # Calculate coordinates of each pixel's first corner
                # 2D arrays, shape: (64, 128)
                pixels_x = (
                        corner_x
                        + pixel_ss_coord * ss_unit_x
                        + pixel_fs_coord * fs_unit_x
                )
                pixels_y = (
                        corner_y
                        + pixel_ss_coord * ss_unit_y
                        + pixel_fs_coord * fs_unit_y
                )
                pixels_z = (
                        corner_z
                        + pixel_ss_coord * ss_unit_z
                        + pixel_fs_coord * fs_unit_z
                )

                # Which part of the array is this tile?
                tile_ss_slice, tile_fs_slice = self._tile_slice(t)

                # Insert the data into the array
                out[m, tile_ss_slice, tile_fs_slice, 0] = pixels_x
                out[m, tile_ss_slice, tile_fs_slice, 1] = pixels_y
                out[m, tile_ss_slice, tile_fs_slice, 2] = pixels_z

        return out

    def data_coords_to_positions(self, module_no, slow_scan, fast_scan):
        """Convert data array coordinates to physical positions

        Data array coordinates are how you might refer to a pixel in an array
        of detector data: module number, and indices in the slow-scan and
        fast-scan directions. But coordinates in the two pixel dimensions aren't
        necessarily integers, e.g. if they refer to the centre of a peak.

        module_no, fast_scan and slow_scan should all be numpy arrays of the
        same shape. module_no should hold integers, starting from 0,
        so 0: Q1M1, 1: Q1M2, etc.

        slow_scan and fast_scan describe positions within that module.
        They may hold floats for sub-pixel positions. In both, 0.5 is the centre
        of the first pixel.

        Returns an array of similar shape with an extra dimension of length 3,
        for (x, y, z) coordinates in metres.

        .. seealso::

           :doc:`agipd_geometry` demonstrates using this method.
        """
        assert module_no.shape == slow_scan.shape == fast_scan.shape

        # We want to avoid iterating over the positions in Python.
        # So we assemble arrays of the corner position and step vectors for all
        # tiles, and then use numpy indexing to select the relevant ones for
        # each set of coordinates.
        tiles_corner_pos = np.stack([
            t.corner_pos for m in self.modules for t in m
        ])
        tiles_ss_vec = np.stack([
            t.ss_vec for m in self.modules for t in m
        ])
        tiles_fs_vec = np.stack([
            t.fs_vec for m in self.modules for t in m
        ])

        # Convert coordinates within each module to coordinates in a tile
        tilenos, tile_ss, tile_fs = self._module_coords_to_tile(slow_scan, fast_scan)

        # The indexes of the relevant tiles in the arrays assembled above
        all_tiles_ix = (module_no * self.n_tiles_per_module) + tilenos

        # Select the relevant tile geometry for each set of coordinates
        coords_tile_corner = tiles_corner_pos[all_tiles_ix]
        coords_ss_vec = tiles_ss_vec[all_tiles_ix]
        coords_fs_vec = tiles_fs_vec[all_tiles_ix]

        # Calculate the physical coordinate for each data coordinate
        return coords_tile_corner \
            + (np.expand_dims(tile_ss, -1) * coords_ss_vec) \
            + (np.expand_dims(tile_fs, -1) * coords_fs_vec)


class AGIPD_1MGeometry(DetectorGeometryBase):
    """Detector layout for AGIPD-1M

    The coordinates used in this class are 3D (x, y, z), and represent metres.

    You won't normally instantiate this class directly:
    use one of the constructor class methods to create or load a geometry.
    """
    pixel_size = 2e-4  # 2e-4 metres == 0.2 mm
    frag_ss_pixels = 64
    frag_fs_pixels = 128
    expected_data_shape = (16, 512, 128)
    n_modules = 16
    n_tiles_per_module = 8

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
        asic_gap_px = asic_gap * unit / cls.pixel_size
        panel_gap_px = panel_gap * unit / cls.pixel_size

        # How much space one tile takes up, including the gaps
        # separating it from its neighbour.
        # In the y dimension, 128 px + gap between modules
        module_height = (cls.frag_fs_pixels + panel_gap_px) * cls.pixel_size
        # In x, 64 px + gap between tiles (asics)
        tile_width = (cls.frag_ss_pixels + asic_gap_px) * cls.pixel_size

        quads_x_orientation = [1, 1, -1, -1]
        quads_y_orientation = [-1, -1, 1, 1]
        modules = []
        for p in range(16):
            quad = p // 4
            quad_corner = quad_pos[quad]
            x_orient = quads_x_orientation[quad]
            y_orient = quads_y_orientation[quad]
            p_in_quad = p % 4
            corner_y = (quad_corner[1] * unit)\
                       - (p_in_quad * module_height)

            tiles = []
            modules.append(tiles)

            for a in range(8):
                corner_x = (quad_corner[0] * unit)\
                           + x_orient * tile_width * a
                tiles.append(GeometryFragment(
                    corner_pos=np.array([corner_x, corner_y, 0.]),
                    ss_vec=np.array([x_orient, 0, 0]) * unit,
                    fs_vec=np.array([0, y_orient, 0]) * unit,
                    ss_pixels=cls.frag_ss_pixels,
                    fs_pixels=cls.frag_fs_pixels,
                ))
        return cls(modules)

    def inspect(self, axis_units='px', frontview=True):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Axes object.

        Parameters
        ----------

        axis_units : str
          Show the detector scale in pixels ('px') or metres ('m').
        frontview : bool
          If True (the default), x increases to the left, as if you were looking
          along the beam. False gives a 'looking into the beam' view.
        """
        ax = super().inspect(axis_units=axis_units, frontview=frontview)
        scale = self._get_plot_scale_factor(axis_units)

        # Label modules and tiles
        for ch, module in enumerate(self.modules):
            s = 'Q{Q}M{M}'.format(Q=(ch // 4) + 1, M=(ch % 4) + 1)
            cx, cy, _ = module[4].centre() * scale
            ax.text(cx, cy, s, fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='center')

            for t in [0, 7]:
                cx, cy, _ = module[t].centre() * scale
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
        corners = np.concatenate(corners)[:, :2] / self._pixel_shape

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

    @classmethod
    def _tile_slice(cls, tileno):
        # Which part of the array is this tile?
        # tileno = 0 to 7
        tile_offset = tileno * cls.frag_ss_pixels
        ss_slice = slice(tile_offset, tile_offset + cls.frag_ss_pixels)
        fs_slice = slice(0, cls.frag_fs_pixels)  # Every tile covers the full 128 pixels
        return ss_slice, fs_slice

    @classmethod
    def _module_coords_to_tile(cls, slow_scan, fast_scan):
        tileno, tile_ss = np.divmod(slow_scan, cls.frag_ss_pixels)
        return tileno.astype(np.int16), tile_ss, fast_scan

    def to_distortion_array(self, allow_negative_xy=False):
        """Return distortion matrix for AGIPD detector, suitable for pyFAI.

        Parameters
        ----------

        allow_negative_xy: bool
          If False (default), shift the origin so no x or y coordinates are
          negative. If True, the origin is the detector centre.

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
        # Overridden only for docstring
        return super().to_distortion_array(allow_negative_xy)


class SnappedGeometry:
    """Detector geometry approximated to align modules to a 2D grid

    The coordinates used in this class are (y, x) suitable for indexing a
    Numpy array; this does not match the (x, y, z) coordinates in the more
    precise geometry above.
    """
    def __init__(self, modules, geom: DetectorGeometryBase):
        self.modules = modules
        self.geom = geom
        self.size_yx, self.centre = self._get_dimensions()

    def make_output_array(self, extra_shape=(), dtype=np.float32):
        """Make an output array for self.position_modules()
        """
        shape = extra_shape + self.size_yx
        return np.full(shape, np.nan, dtype=dtype)

    def position_modules(self, data, out=None):
        """Implementation for position_modules_fast
        """
        assert data.shape[-3:] == self.geom.expected_data_shape
        if out is None:
            out = self.make_output_array(data.shape[:-3], data.dtype)
        else:
            assert out.shape == data.shape[:-3] + self.size_yx
            assert out.dtype == data.dtype

        for i, module in enumerate(self.modules):
            mod_data = data[..., i, :, :]
            tiles_data = self.geom.split_tiles(mod_data)
            for j, tile in enumerate(module):
                tile_data = tiles_data[j]
                # Offset by centre to make all coordinates positive
                y, x = tile.corner_idx + self.centre
                h, w = tile.pixel_dims
                out[..., y : y + h, x : x + w] = tile.transform(tile_data)

        return out, self.centre

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
        return ax


class LPD_1MGeometry(DetectorGeometryBase):
    """Detector layout for LPD-1M

    The coordinates used in this class are 3D (x, y, z), and represent metres.

    You won't normally instantiate this class directly:
    use one of the constructor class methods to create or load a geometry.
    """
    pixel_size = 5e-4  # 5e-4 metres == 0.5 mm
    frag_ss_pixels = 32
    frag_fs_pixels = 128
    n_modules = 16
    n_tiles_per_module = 16
    expected_data_shape = (16, 256, 256)
    _draw_first_px_on_tile = 8  # The first pixel in stored data is on tile 8

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
        panel_width = (256 + asic_gap_px + panel_gap_px) * cls.pixel_size
        # In y, we have 7 gaps between the 8 ASICs in each column.
        panel_height = (256 + (7 * asic_gap_px) + panel_gap_px) * cls.pixel_size

        # How much space does one tile take up, including gaps to its neighbours?
        tile_width = (cls.frag_fs_pixels + asic_gap_px) * cls.pixel_size
        tile_height = (cls.frag_ss_pixels + asic_gap_px) * cls.pixel_size

        # Size of a tile from corner to corner, excluding gaps
        tile_size = np.array([cls.frag_fs_pixels, cls.frag_ss_pixels, 0]) * cls.pixel_size

        panels_across = [-1, -1, 0, 0]
        panels_up = [0, -1, -1, 0]
        modules = []
        for p in range(cls.n_modules):
            quad = p // 4
            quad_corner_x = quad_pos[quad][0] * unit
            quad_corner_y = quad_pos[quad][1] * unit

            p_in_quad = p % 4
            # Top beam-left corner of panel
            panel_corner_x = (quad_corner_x +
                              (panels_across[p_in_quad] * panel_width))
            panel_corner_y = (quad_corner_y +
                              (panels_up[p_in_quad] * panel_height))

            tiles = []
            modules.append(tiles)

            for a in range(cls.n_tiles_per_module):
                if a < 8:
                    up = -a
                    across = -1
                else:
                    up = -(15 - a)
                    across = 0

                tile_last_corner = (
                    np.array([panel_corner_x, panel_corner_y, 0.0])
                    + np.array([across, 0, 0]) * tile_width
                    + np.array([0, up, 0]) * tile_height
                )
                tile_first_corner = tile_last_corner - tile_size

                tiles.append(GeometryFragment(
                    corner_pos=tile_first_corner,
                    ss_vec=np.array([0, 1, 0]) * cls.pixel_size,
                    fs_vec=np.array([1, 0, 0]) * cls.pixel_size,
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
                for T in range(1, cls.n_modules+1):
                    corner_pos = np.zeros(3)
                    tile_offset = mod_grp['T{:02}/Position'.format(T)][:2]
                    corner_pos[:2] = quad_pos + mod_offset + tile_offset

                    # Convert units (mm) to metres
                    corner_pos *= unit

                    # LPD geometry is measured to the last pixel of each tile.
                    # Subtract tile dimensions for the position of 1st pixel.
                    ss_vec = np.array([0, 1, 0]) * cls.pixel_size
                    fs_vec = np.array([1, 0, 0]) * cls.pixel_size
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

    def inspect(self, axis_units='px', frontview=True):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Axes object.

        Parameters
        ----------

        axis_units : str
          Show the detector scale in pixels ('px') or metres ('m').
        frontview : bool
          If True (the default), x increases to the left, as if you were looking
          along the beam. False gives a 'looking into the beam' view.
        """
        ax = super().inspect(axis_units=axis_units, frontview=frontview)
        scale = self._get_plot_scale_factor(axis_units)

        # Label modules and tiles
        for ch, module in enumerate(self.modules):
            s = 'Q{Q}M{M}'.format(Q=(ch // 4) + 1, M=(ch % 4) + 1)
            cx, cy, _ = module[0].centre() * scale
            ax.text(cx, cy, s, fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='center')

            for t in [7, 8, 15]:
                cx, cy, _ = module[t].centre() * scale
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

    @classmethod
    def _tile_slice(cls, tileno):
        # Which part of the array is this tile?
        if tileno < 8:  # First half of module (0 <= t <= 7)
            fs_slice = slice(0, 128)
            tiles_up = 7 - tileno
        else:  # Second half of module (8 <= t <= 15)
            fs_slice = slice(128, 256)
            tiles_up = tileno - 8
        tile_offset = tiles_up * 32
        ss_slice = slice(tile_offset, tile_offset + cls.frag_ss_pixels)
        return ss_slice, fs_slice

    @classmethod
    def _module_coords_to_tile(cls, slow_scan, fast_scan):
        tiles_across, tile_fs = np.divmod(fast_scan, cls.frag_fs_pixels)
        tiles_up, tile_ss = np.divmod(slow_scan, cls.frag_ss_pixels)

        # Each tiles_across is 0 or 1. To avoid iterating over the array with a
        # conditional, multiply the number we want by 1 and the other by 0.
        tileno = (
            (1 - tiles_across) * (7 - tiles_up)  # tileno 0-7
            + tiles_across * (tiles_up + 8)      # tileno 8-15
        )
        return tileno.astype(np.int16), tile_ss, tile_fs

    def to_distortion_array(self, allow_negative_xy=False):
        """Return distortion matrix for LPD detector, suitable for pyFAI.

        Parameters
        ----------

        allow_negative_xy: bool
          If False (default), shift the origin so no x or y coordinates are
          negative. If True, the origin is the detector centre.

        Returns
        -------
        out: ndarray
            Array of float 32 with shape (4096, 256, 4, 3).
            The dimensions mean:

            - 4096 = 16 modules * 256 pixels (slow scan axis)
            - 256 pixels (fast scan axis)
            - 4 corners of each pixel
            - 3 numbers for z, y, x
        """
        # Overridden only for docstring
        return super().to_distortion_array(allow_negative_xy)


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


class DSSC_1MGeometry(DetectorGeometryBase):
    """Detector layout for DSSC-1M

    The coordinates used in this class are 3D (x, y, z), and represent metres.

    You won't normally instantiate this class directly:
    use one of the constructor class methods to create or load a geometry.
    """
    # Hexagonal pixels, 236 μm step in fast-scan axis, 204 μm in slow-scan
    pixel_size = 236e-6
    frag_ss_pixels = 128
    frag_fs_pixels = 256
    n_modules = 16
    n_tiles_per_module = 2
    expected_data_shape = (16, 128, 512)
    # This stretches the dimensions for the 'snapped' geometry so that its pixel
    # grid matches the aspect ratio of the detector pixels.
    _pixel_shape = np.array([1., 1.5/np.sqrt(3)], dtype=np.float64) * pixel_size

    # Pixel corners described clockwise from the top, assuming the reference
    # point for a pixel is outside it, aligned with the top point & left edge.
    # The unit is the width of a pixel, 236 μm.
    # The 4/3 extends the hexagons into the next row to correctly tessellate.
    _pixel_corners = np.stack([
        (np.array([0, 0.25, 0.75, 1, 0.75, 0.25]) * 4 / 3),
        [0.5, 1, 1, 0.5, 0, 0]
    ])

    @classmethod
    def from_h5_file_and_quad_positions(cls, path, positions, unit=1e-3):
        """Load a DSSC geometry from an XFEL HDF5 format geometry file

        The quadrant positions are not stored in the file, and must be provided
        separately. The position given should refer to the bottom right (looking
        along the beam) corner of the quadrant.

        By default, both the quadrant positions and the positions
        in the file are measured in millimetres; the unit parameter controls
        this.

        The origin of the coordinates is in the centre of the detector.
        Coordinates increase upwards and to the left (looking along the beam).

        This version of the code only handles x and y translation,
        as this is all that is recorded in the initial LPD geometry file.

        Parameters
        ----------

        path : str
          Path of an EuXFEL format (HDF5) geometry file for DSSC.
        positions : list of 2-tuples
          (x, y) coordinates of the last corner (the one by module 4) of each
          quadrant.
        unit : float, optional
          The conversion factor to put the coordinates into metres.
          The default 1e-3 means the numbers are in millimetres.
        """
        assert len(positions) == 4
        modules = []

        quads_x_orientation = [-1, -1, 1, 1]
        quads_y_orientation = [1, 1, -1, -1]

        with h5py.File(path, 'r') as f:
            for Q, M in product(range(1, 5), range(1, 5)):
                quad_pos = np.array(positions[Q - 1])
                mod_grp = f['Q{}/M{}'.format(Q, M)]
                mod_offset = mod_grp['Position'][:2]

                # Which way round is this quadrant
                x_orient = quads_x_orientation[Q - 1]
                y_orient = quads_y_orientation[Q - 1]

                tiles = []
                for T in range(1, 3):
                    corner_pos = np.zeros(3)
                    tile_offset = mod_grp['T{:02}/Position'.format(T)][:2]
                    corner_pos[:2] = quad_pos + mod_offset + tile_offset

                    # Convert units (mm) to metres
                    corner_pos *= unit

                    # Measuring in terms of the step within a row, the
                    # step to the next row of hexagons is 1.5/sqrt(3).
                    ss_vec = np.array([0, y_orient, 0]) * cls.pixel_size * 1.5/np.sqrt(3)
                    fs_vec = np.array([x_orient, 0, 0]) * cls.pixel_size

                    # Corner position is measured at low-x, low-y corner (bottom
                    # right as plotted). We want the position of the corner
                    # with the first pixel, which is either high-x low-y or
                    # low-x high-y.
                    if x_orient == -1:
                        first_px_pos = corner_pos - (fs_vec * cls.frag_fs_pixels)
                    else:
                        first_px_pos = corner_pos - (ss_vec * cls.frag_ss_pixels)

                    tiles.append(GeometryFragment(
                        corner_pos=first_px_pos,
                        ss_vec=ss_vec,
                        fs_vec=fs_vec,
                        ss_pixels=cls.frag_ss_pixels,
                        fs_pixels=cls.frag_fs_pixels,
                    ))
                modules.append(tiles)

        return cls(modules, filename=path)

    def inspect(self, axis_units='px', frontview=True):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Axes object.

        Parameters
        ----------

        axis_units : str
          Show the detector scale in pixels ('px') or metres ('m').
        frontview : bool
          If True (the default), x increases to the left, as if you were looking
          along the beam. False gives a 'looking into the beam' view.
        """
        ax = super().inspect(axis_units=axis_units, frontview=frontview)
        scale = self._get_plot_scale_factor(axis_units)

        # Label modules and tiles
        for ch, module in enumerate(self.modules):
            s = 'Q{Q}M{M}'.format(Q=(ch // 4) + 1, M=(ch % 4) + 1)
            cx, cy, _ = module[0].centre() * scale
            ax.text(cx, cy, s, fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='center')

            for t in [1]:
                cx, cy, _ = module[t].centre() * scale
                ax.text(cx, cy, 'T{}'.format(t + 1),
                        verticalalignment='center',
                        horizontalalignment='center')

        ax.set_title('DSSC detector geometry ({})'.format(self.filename))
        return ax

    @staticmethod
    def split_tiles(module_data):
        # Split into 2 tiles along the fast-scan axis
        return np.split(module_data, 2, axis=-1)

    def plot_data_fast(self, data, axis_units='px', frontview=True):
        ax = super().plot_data_fast(data, axis_units, frontview)
        # Squash image to physically equal aspect ratio, so a circle projected
        # on the detector looks like a circle on screen.
        ax.set_aspect(204/236.)
        return ax

    @classmethod
    def _tile_slice(cls, tileno):
        tile_offset = tileno * cls.frag_fs_pixels
        fs_slice = slice(tile_offset, tile_offset + cls.frag_fs_pixels)
        ss_slice = slice(0, cls.frag_ss_pixels)  # Every tile covers the full pixel range
        return ss_slice, fs_slice

    def to_distortion_array(self, allow_negative_xy=False):
        """Return distortion matrix for DSSC detector, suitable for pyFAI.

        Parameters
        ----------

        allow_negative_xy: bool
          If False (default), shift the origin so no x or y coordinates are
          negative. If True, the origin is the detector centre.

        Returns
        -------
        out: ndarray
            Array of float 32 with shape (2048, 512, 6, 3).
            The dimensions mean:

            - 2048 = 16 modules * 128 pixels (slow scan axis)
            - 512 pixels (fast scan axis)
            - 6 corners of each pixel
            - 3 numbers for z, y, x
        """
        # Overridden only for docstring
        return super().to_distortion_array(allow_negative_xy=allow_negative_xy)

    @classmethod
    def _adjust_pixel_coords(cls, ss_coords, fs_coords, centre):
        # Shift odd-numbered rows by half a pixel.
        fs_coords[1::2] -= 0.5
        if centre:
            # Vertical (slow scan) centre is 2/3 of the way to the start of the
            # next row of hexagons, because the tessellating pixels extend
            # beyond the start of the next row.
            ss_coords += 2/3
            fs_coords += 0.5

class DSSC_Geometry(DSSC_1MGeometry):
    """DEPRECATED: Use DSSC_1MGeometry instead"""
    def __init__(self, modules, filename='No file'):
        super().__init__(modules, filename)
        warnings.warn(
            "DSSC_Geometry has been renamed to DSSC_1MGeometry.", stacklevel=2
        )
