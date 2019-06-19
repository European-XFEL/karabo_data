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

    def to_crystfel_geom(self, p, a, ss_dims, fs_dims, dims):
        tile_name = 'p{}a{}'.format(p, a)
        c = self.corner_pos
        dim_list = []
        for num, value in dims.items():
            if value == 'modno':
                key = p
            else:
                key = value
            dim_list.append('{}/dim{} = {}'.format(tile_name, num, key))

        return CRYSTFEL_PANEL_TEMPLATE.format(
            dims='\n'.join(dim_list),
            name=tile_name,
            min_ss=ss_dims[0],
            max_ss=ss_dims[1],
            min_fs=fs_dims[0],
            max_fs=fs_dims[1],
            ss_vec=_crystfel_format_vec(self.ss_vec),
            fs_vec=_crystfel_format_vec(self.fs_vec),
            corner_x=c[0],
            corner_y=c[1],
            coffset=c[2],
        )

    def snap(self, px_shape=np.array([1., 1.])):
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
    _pixel_shape = np.array([1., 1.])  # Overridden for DSSC
    _draw_first_px_on_tile = 1  # Tile num of 1st pixel - overridden for LPD

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
        from matplotlib.collections import PatchCollection, LineCollection
        from matplotlib.patches import Polygon

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        rects = []
        first_rows = []
        for module in self.modules:
            for t, fragment in enumerate(module, start=1):
                corners = fragment.corners()[:, :2]  # Drop the Z dimension
                rects.append(Polygon(corners))

                if t == self._draw_first_px_on_tile:
                    # Find the ends of the first row in reading order
                    c1 = fragment.corner_pos
                    c2 = c1 + (fragment.fs_vec * fragment.fs_pixels)
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

        # Draw cross in the centre.
        ax.hlines(0, -100, +100, colors='0.75', linewidths=2)
        ax.vlines(0, -100, +100, colors='0.75', linewidths=2)

        if frontview:
            ax.invert_xaxis()

        return ax

    def _tile_dims(self, tileno):
        """Implement in subclass: which part of module array each tile is.
        """
        raise NotImplementedError

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

    def _get_rigid_groups(self, nquads=4):
        """Create rigid stings for rigid groups definiton."""

        quads = ','.join(['q{}'.format(q) for q in range(nquads)])
        modules = ','.join(['p{}'.format(p) for p in range(self.n_modules)])

        prod = product(range(self.n_modules), range(self.n_tiles_per_module))
        rigid_group = ['p{}a{}'.format(p, a) for (p, a) in prod]
        rigid_string = '\n'

        for nn, rigid_group_q in enumerate(np.array_split(rigid_group, nquads)):
            rigid_string += 'rigid_group_q{} = {}\n'.format(nn, ','.join(rigid_group_q))
        rigid_string += '\n'
        for nn, rigid_group_p in enumerate(np.array_split(rigid_group, self.n_modules)):
            rigid_string += 'rigid_group_p{} = {}\n'.format(nn, ','.join(rigid_group_p))

        rigid_string += '\n'

        rigid_string += 'rigid_group_collection_quadrants = {}\n'.format(quads)
        rigid_string += 'rigid_group_collection_asics = {}\n\n'.format(modules)
        return rigid_string

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
            Note: the dimensions must contain frame, modno, ss, fs.
        adu_per_ev : float
            ADU (analog digital units) per electron volt for the considered
            detector.
        clen : float
            Distance between sample and detector in meters
        photon_energy : float
            Beam wave length in eV
        """
        from . import __version__

        if adu_per_ev is None:
            adu_per_ev_str = '; adu_per_eV = SET ME'
            # TODO: adu_per_ev should be fixed for each detector, we should
            #       find out the values and set them.
        else:
            adu_per_ev_str = 'adu_per_eV = {}'.format(adu_per_ev)

        if clen is None:
            clen_str = '; clen = SET ME'
        else:
            clen_str = 'clen = {}'.format(clen)

        if photon_energy is None:
            photon_energy_str = '; photon_energy = SET ME'
        else:
            photon_energy_str = 'photon_energy = {}'.format(photon_energy)

        # Get the frame dimension
        tile_dims = {}

        for nn, dim_name in enumerate(dims):
            if dim_name == 'frame':
                frame_dim = 'dim{} = %'.format(nn)
            else:
                tile_dims[nn] = dim_name
        if frame_dim is None:
            raise ValueError('No frame dimension given')

        panel_chunks = []
        for p, module in enumerate(self.modules):
            for a, fragment in enumerate(module):
                ss_dims, fs_dims = self._tile_dims(a)
                panel_chunks.append(fragment.to_crystfel_geom(p,
                                                              a,
                                                              ss_dims,
                                                              fs_dims,
                                                              tile_dims))
        resolution = 1.0 / self.pixel_size  # Pixels per metre
        paths = dict(data=data_path)
        if mask_path:
            paths['mask'] = mask_path
        path_str = '\n'.join('{} = {} ;'.format(i, j) for i, j in paths.items())
        with open(filename, 'w') as f:
            f.write(CRYSTFEL_HEADER_TEMPLATE.format(version=__version__,
                                                    paths=path_str,
                                                    frame_dim=frame_dim,
                                                    resolution=resolution,
                                                    adu_per_ev=adu_per_ev_str,
                                                    clen=clen_str,
                                                    photon_energy=photon_energy_str))
            rigid_groups = self._get_rigid_groups()
            f.write(rigid_groups)
            for chunk in panel_chunks:
                f.write(chunk)

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

    @classmethod
    def _distortion_array_slice(cls, m, t):
        """Implement in subclass: which part of distortion array each tile is.
        """
        raise NotImplementedError

    def to_distortion_array(self):
        """Generate a distortion array for pyFAI from this geometry.
        """
        nmods, mod_px_ss, mod_px_fs = self.expected_data_shape
        distortion = np.zeros((nmods * mod_px_ss, mod_px_fs, 4, 3),
                              dtype=np.float32)

        # Prepare some arrays to use inside the loop
        pixel_ss_index, pixel_fs_index = np.meshgrid(
            np.arange(0, self.frag_ss_pixels),
            np.arange(0, self.frag_fs_pixels),
            indexing='ij'
        )
        corner_ss_offsets = np.array([0, 1, 1, 0])
        corner_fs_offsets = np.array([0, 0, 1, 1])

        for m, mod in enumerate(self.modules, start=0):
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
                tile_ss_slice, tile_fs_slice = self._distortion_array_slice(m, t)

                # Insert the data into the array
                distortion[tile_ss_slice, tile_fs_slice, :, 0] = corners_z
                distortion[tile_ss_slice, tile_fs_slice, :, 1] = corners_y
                distortion[tile_ss_slice, tile_fs_slice, :, 2] = corners_x

        # Shift the x & y origin from the centre to the corner
        min_yx = distortion[..., 1:].min(axis=(0, 1, 2))
        distortion[..., 1:] -= min_yx

        return distortion


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

    @classmethod
    def _distortion_array_slice(cls, m, t):
        # Which part of the array is this tile?
        # m = 0 to 15,  t = 0 to 7
        module_offset = m * 512
        tile_offset = module_offset + (t * cls.frag_ss_pixels)
        ss_slice = slice(tile_offset, tile_offset + cls.frag_ss_pixels)
        fs_slice = slice(None, None)  # Every tile covers the full 128 pixels
        return ss_slice, fs_slice

    @classmethod
    def _tile_dims(cls, tileno):
        tile_offset = tileno * cls.frag_ss_pixels
        ss_dims = tile_offset, tile_offset + cls.frag_ss_pixels - 1
        fs_dims = 0, cls.frag_fs_pixels - 1  # Every tile covers the full pixel range
        return ss_dims, fs_dims

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
        return super().to_distortion_array()  # Overridden only for docstring


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
        return ax


CRYSTFEL_HEADER_TEMPLATE = """\
; AGIPD-1M geometry file written by karabo_data {version}
; You may need to edit this file to add:
; - data and mask locations in the file
; - mask_good & mask_bad values to interpret the mask
; - adu_per_eV & photon_energy
; - clen (detector distance)
;
; See: http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html

{paths}
{frame_dim}
res = {resolution} ; pixels per metre

; Beam energy in eV
{photon_energy}

; Camera length, aka detector distance
{clen}

; Analogue Digital Units per eV
{adu_per_ev}
"""


CRYSTFEL_PANEL_TEMPLATE = """
{dims}
{name}/min_fs = {min_fs}
{name}/min_ss = {min_ss}
{name}/max_fs = {max_fs}
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
        panel_width = 256 + asic_gap_px + panel_gap_px
        # In y, we have 7 gaps between the 8 ASICs in each column.
        panel_height = 256 + (7 * asic_gap_px) + panel_gap_px

        tile_size = np.array([cls.frag_fs_pixels, cls.frag_ss_pixels, 0])

        panels_across = [-1, -1, 0, 0]
        panels_up = [0, -1, -1, 0]
        modules = []
        for p in range(cls.n_modules):
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

            for a in range(cls.n_tiles_per_module):
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
                for T in range(1, cls.n_modules+1):
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

    @classmethod
    def _distortion_array_slice(cls, m, t):
        # Which part of the array is this tile?
        # The distortion array for LPD is 4096 x 256, with modules stacked
        # along their slow-scan axis, Q1M1 (m=0) to Q4M4 (m=15)
        module_offset = m * 256
        if t < 8:  # First half of module (0 <= t <= 7)
            fs_slice = slice(0, 128)
            tiles_up = 7 - t
        else:      # Second half of module (8 <= t <= 15)
            fs_slice = slice(128, 256)
            tiles_up = t - 8
        tile_offset = module_offset + (tiles_up * 32)
        ss_slice = slice(tile_offset, tile_offset + cls.frag_ss_pixels)
        return ss_slice, fs_slice

    def to_distortion_array(self):
        """Return distortion matrix for LPD detector, suitable for pyFAI.

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
        return super().to_distortion_array()  # Overridden only for docstring

    @classmethod
    def _tile_dims(cls, tileno):
        if tileno < 8:  # First half of module (0 <= t <=7)
            fs_dims = 0, 127
            tiles_up = 7 - tileno
        else:
            fs_dims = 128, 255
            tiles_up = tileno - 8

        tile_offset = tiles_up * 32
        ss_dims = tile_offset, tile_offset + cls.frag_ss_pixels - 1
        return ss_dims, fs_dims


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

    The coordinates used in this class are 3D (x, y, z), and represent multiples
    of the pixel size.

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
    _pixel_shape = np.array([1., 1.5/np.sqrt(3)])

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

                    # Convert units (mm) to pixels
                    corner_pos *= unit / cls.pixel_size

                    # Measuring in terms of the step within a row, the
                    # step to the next row of hexagons is 1.5/sqrt(3).
                    ss_vec = np.array([0, y_orient * 1.5/np.sqrt(3), 0])
                    fs_vec = np.array([x_orient, 0, 0])

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

            for t in [1]:
                cx, cy, _ = module[t].centre()
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
    def _distortion_array_slice(cls, m, t):
        # Which part of the array is this tile?
        # m = 0 to 15,  t = 0 to 1
        ss_slice = slice(m * cls.frag_ss_pixels, (m + 1) * cls.frag_ss_pixels)
        fs_slice = slice(t * cls.frag_fs_pixels, (t + 1) * cls.frag_fs_pixels)
        return ss_slice, fs_slice

    @classmethod
    def _tile_dims(cls, tileno):
        tile_offset = tileno * cls.frag_fs_pixels
        fs_dims = tile_offset, tile_offset + cls.frag_fs_pixels - 1
        ss_dims = 0, cls.frag_ss_pixels - 1  # Every tile covers the full pixel range
        return ss_dims, fs_dims

    def to_distortion_array(self):
        """Return distortion matrix for DSSC detector, suitable for pyFAI.

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
        nmods, mod_px_ss, mod_px_fs = self.expected_data_shape
        distortion = np.zeros((nmods * mod_px_ss, mod_px_fs, 6, 3),
                              dtype=np.float32)

        # Prepare some arrays to use inside the loop
        pixel_ss_index, pixel_fs_index = np.meshgrid(
            np.arange(0, self.frag_ss_pixels, dtype=np.float32),
            np.arange(0, self.frag_fs_pixels, dtype=np.float32),
            indexing='ij'
        )
        # Every second line of pixels across the slow-scan direction is shifted
        # half a pixel against the fast-scan direction so the hexagons tessalate.
        pixel_fs_index[1::2, :] -= 0.5

        # Corners described clockwise from the top, assuming the reference point
        # for a pixel is outside it, aligned with the top point & left edge.
        # The 4/3 extends the hexagons into the next row to correctly tessellate.
        corner_ss_offsets = np.array([0, 0.25, 0.75, 1, 0.75, 0.25]) * 4 / 3
        corner_fs_offsets = np.array([0.5, 1, 1, 0.5, 0, 0])

        for m, mod in enumerate(self.modules, start=0):
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
                tile_ss_slice, tile_fs_slice = self._distortion_array_slice(m, t)

                # Insert the data into the array
                distortion[tile_ss_slice, tile_fs_slice, :, 0] = corners_z
                distortion[tile_ss_slice, tile_fs_slice, :, 1] = corners_y
                distortion[tile_ss_slice, tile_fs_slice, :, 2] = corners_x

        # Shift the x & y origin from the centre to the corner
        min_yx = distortion[..., 1:].min(axis=(0, 1, 2))
        distortion[..., 1:] -= min_yx

        return distortion
