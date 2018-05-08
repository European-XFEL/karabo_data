from copy import copy
import h5py
from itertools import product
import numpy as np
import re
import sys
from textwrap import indent

def splitChannelDataIntoTiles(channelData, clockwiseOrder=False):
    """Splits the raw channel data into indiviual tiles

    Args
    ----

    channelData : ndarray
        Raw channel data. Must have shape (256, 256)

    clockwiseOrder : bool, optional
        If set to True, the sequence of tiles is given
        in the clockwise order starting with the top
        right tile (LPD standard). If set to false, tile
        data is returned in reading order

    Returns
    -------

    ndarray
        Same data, but reshaped into (12, 32, 128)
    """
    extra_dims = channelData.shape[:-2]
    a = np.asarray(np.split(channelData, 8, axis=-2))
    a = np.asarray(np.split(a, 2, axis=-1))
    a = np.reshape(a, (16,) + extra_dims + (32, 128))
    orderedTiles = a

    if clockwiseOrder:
        # The official LPD tile order is clockwise from the top right tile.
        # We need them in this order to apply the right offset to the right tile.
        readingOrderToClockwise = list(range(7, -1, -1)) + list(range(8, 16))
        orderedTiles = orderedTiles[readingOrderToClockwise]
    return orderedTiles


def getModulePosition(metrologyFile, moduleId):
    """Position (in mm) of a module relative to the top left
    corner of it's quadrant. In case of tile-level positions,
    the the position refers to the center of the top left
    pixel.

    Args
    ----

    metrologyFile : str
        Fully qualified path and filename of the metrology file
    moduleId : str
        Identifier of the module in question (e.g. 'Q1M2T03')

    Returns
    -------

    ndarray:
        (x, y)-Position of the module in it's quadrant

    Raises
    ------

    ValueError: In case the moduleId contains invalid module
        identifieres
    """
    # The moduleId does not directly appear as a dataset in the
    # metrology file. Instread, it follows the nomenclature of
    # the detector. LPD-style module identifier have the format
    #
    #   QXMYTZZ
    #
    # where X, Y, and Z are digits. Q denotes the quadrant
    # (X = 1, ..., 4), M the supermodule (Y = 1, ..., 4) and T
    # the tile (Z = 1, ..., 16; with leading zeros).
    modulePattern = re.compile(r'[QMT]\d+')
    # Give the module identifier Q1M1T01, the moduleList splits this
    # into the associated quadrant, supermodule, and tile identifiers:
    # >>> print(moduleList)
    # ['Q1', 'M1', 'T01']
    moduleList = modulePattern.findall(moduleId)
    # The metrology file is stored in hdf5 format. It stores positions
    # hierarchally, starting on the supermodule level. the h5Keys list
    # contains all path that will be accessed in the hdf5 file
    # >>> print(h5Keys)
    # ['Q1', 'Q1/M1', 'Q1/M1/T01']
    h5Keys = ['/'.join(moduleList[:idx + 1]) for idx in range(len(moduleList))]

    # Every module of the detector gives it's position relative to
    # the top left corner of its parent structure. Every position
    # is stored in the positions array
    positions = []
    # Access file
    with h5py.File(metrologyFile) as fh:
        # Check if the keys actually appear in the metrology file
        for key in h5Keys:
            if not key in fh:
                raise ValueError("Invalid key '{}'".format(key))
        # Extract the positions from the hdf5 groups corresponding
        # to a module, if the module has dataset 'Position'.
        positions = [
            np.asarray(fh[key]['Position']) for key in h5Keys if
            'Position' in fh[key]
        ]
    if len(positions) == 0:
        # This is the case when requesting a quadrant; e.g.
        # getModulePosition('Q1'). Key is valid, but quadrant
        # has no location (yet).
        positions = [[0.0, 0.0]]
    # Convert to numpy array
    positions = np.asarray(positions)
    # Return the sum of all positions retrieved
    return positions.sum(axis=0)



def returnPositioned(geometry_file, modules, dquads):
    smPositions = []
    smData = []

    tile_order = [1, 2, 3, 4]
    cells = 0
    for sm, mn in modules:
        position = np.asarray([getModulePosition(geometry_file,
                                                 'Q{}/M{:d}/T{:02d}'.format(
                                                     sm//4+1,
                                                     sm%4+1,
                                                     idx + 1))
                               for idx in range(16)])
        smPositions.append(position)
        mn_tile = splitChannelDataIntoTiles(mn[::-1, ::-1, :],
                                            clockwiseOrder=True)
        smData.append(mn_tile)
        cells = max(mn.shape[2], cells)

    quads = []
    quad_pos = []
    exceptions = []
    for q in range(4):
        try:
            qdata = np.concatenate(smData[4 * q:4 * (q+1)], axis=0)
            qpos = np.concatenate(smPositions[4 * q:4 * (q+1)], axis=0)
            quads.append(qdata)
            quad_pos.append(qpos)
        except Exception as e:
            print("len(smData):{}, q: {}".format(len(smData), q))
            exceptions.append(str(e))

    out = np.zeros([1500,1500,cells], np.float32)
    out[...] = 0.001
    cx, cy = out.shape[0]//2, out.shape[1]//2

    for q, tileData in enumerate(quads):
        metrologyPositions = quad_pos[q]
        numberOfTiles = tileData.shape[0]

        # The metrology file references module positions
        bottomRightCornerCoordinates = metrologyPositions

        # The offset here accounts for the fact that there
        # might be negative x,y values
        offset = np.asarray(
            [min(bottomRightCornerCoordinates[:, 0]),
             min(bottomRightCornerCoordinates[:, 1])]
        )

        for i in range(numberOfTiles):
            # This is the top left corner of the tile with
            # respect to the top left corner of the supermodule
            y0, x0 = bottomRightCornerCoordinates[i] + dquads[q]
            x0 *= 2
            y0 *= 2
            td = tileData[i][::-1,:,:]
            out[cx+x0:cx+x0+td.shape[0], cy+y0:cy+y0+td.shape[1], :] = td
    return out, len(smData), exceptions

class GeometryFragment:
    def __init__(self, offset, children):
        self.offset = offset  # in m
        self.children = children

    def _str_lines(self):
        r = []
        for name, child in sorted(self.children.items()):
            r.append("{}: {}".format(name, child.offset))
            r.extend(indent(str(child), '  ').splitlines())
        return r

    def __str__(self):
        return '\n'.join(self._str_lines())

    def find_offset(self, name_parts):
        if name_parts:
            child = self.children[name_parts[0]]
            return self.offset + child.find_offset(name_parts[1:])
        return self.offset

    def iter_leaf_paths(self):
        if not self.children:
            yield ()
            return
        for k, v in sorted(self.children.items()):
            for subpath in v.iter_leaf_paths():
                yield (k,) + subpath

    @classmethod
    def from_h5_group(cls, group, unit=1e-3):
        children = {key: GeometryFragment.from_h5_group(val)
                    for (key, val) in group.items()
                    if isinstance(val, h5py.Group)
                   }
        return cls(group['Position'][:] * unit, children)

class LPDGeometry(GeometryFragment):
    pixel_size = 0.5e-3  # Meter: 0.5 Millimeter

    @classmethod
    def from_h5_file_and_quad_positions(cls, file, positions, unit=1e-3):
        children = {}
        for n, position in enumerate(positions, start=1):
            quad = 'Q%d' % n
            modules = {name: GeometryFragment.from_h5_group(group, unit=unit)
                       for (name, group) in file[quad].items()}
            children[quad] = GeometryFragment(np.asarray(position) * unit, modules)

        return cls(np.asarray((0, 0)), children)

    def position_all_modules(self, data):
        """Assemble data from this detector according to where the pixels are.

        Parameters
        ----------

        data : dict (int: numpy array)
          Mapping of channel numbers (0-15 inclusive) to data.
          Each array must be at least two dimensional, with the last two
          dimensions being the pixel y and x.

        Returns
        -------
        out : ndarray
          Array with the same dimensionality as each module data passed in.
          The last two dimensions represent pixel y and x in the detector space.
        centre : ndarray
          (x, y) pixel location of the detector centre in this geometry.
        """
        data_1mod = list(data.values())[0]
        size_xy, centre = self._plotting_dimensions()
        size_yx = size_xy[::-1]
        out = np.empty(data_1mod.shape[:-2] + size_yx,
                       dtype=data_1mod.dtype)
        out[:] = np.nan

        for module_ix in range(16):
            try:
                module_data = data[module_ix]
            except KeyError:
                continue

            Q = 'Q{:d}'.format(module_ix // 4 + 1)
            M = 'M{:d}'.format(module_ix % 4 + 1)
            module_tiles_data = splitChannelDataIntoTiles(module_data, clockwiseOrder=True)
            for tileno, tile_data in enumerate(module_tiles_data, start=1):
                T = "T{:02d}".format(tileno)
                position = self.find_offset((Q, M, T))
                # Convert physical position (m) to pixel location:
                x0, y0 = (position // self.pixel_size) + centre
                x0, y0 = int(x0), int(y0)

                out[..., y0:y0 + tile_data.shape[-2],
                         x0:x0 + tile_data.shape[-1]] = tile_data[..., ::-1, ::-1]

        return out, centre

    def _plotting_dimensions(self):
        """Calculate appropriate dimensions for plotting assembled data

        Returns (size_x, size_y), (centre_x, centre_y)
        """
        min_x = max_x = min_y = max_y = 0
        for path in self.iter_leaf_paths():
            x, y = self.find_offset(path)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        # Convert physical distances to pixels
        # Add 20px margin, plus the 128*32 tile size
        min_x = int(min_x // self.pixel_size) - 20
        max_x = int(max_x // self.pixel_size) + 128 + 20
        min_y = int(min_y // self.pixel_size) - 20
        max_y = int(max_y // self.pixel_size) + 32 + 20

        size = (max_x - min_x), (max_y - min_y)
        centre = np.asarray([-min_x, -min_y])
        return size, centre

    def plot_data(self, modules_data, slice=()):
        """Plot data from the detector using this geometry.

        Returns a matplotlib figure.

        Parameters
        ----------

        modules_data : dict (int: ndarray)
          Mapping of channel numbers (0-15 inclusive) to data.
          Each array must be at least two dimensional, with the last two
          dimensions being the pixel y and x.
        slice
          If the arrays have more than two dimensions, this should specify which
          2D part is required. E.g. pulse number for train data.
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

        sliced_data = {k: v[slice] for (k, v) in modules_data.items()}
        res, centre = self.position_all_modules(sliced_data)
        ax.imshow(res, cmap=my_viridis)

        cx, cy = centre
        ax.hlines(cy, cx - 20, cx + 20, colors='w', linewidths=1)
        ax.vlines(cx, cy - 20, cy + 20, colors='w', linewidths=1)
        return fig

    def inspect(self):
        """Plot the 2D layout of this detector geometry.

        Returns a matplotlib Figure object.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure((10, 10))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)

        size_xy, centre = self._plotting_dimensions()
        size_yx = size_xy[::-1]  # Our array indexing is [y, x]
        bg = np.ones(size_yx + (3,), dtype='f4')

        # Mark the centre location
        cx, cy = centre
        ax.hlines(cy, cx - 100, cx + 100, colors='0.75', linewidths=2)
        ax.vlines(cx, cy - 100, cy + 100, colors='0.75', linewidths=2)

        # Show where detector elements (modules & tiles) are positioned.
        for (Q, M) in product(range(1, 5), range(1, 5)):
            position = self.find_offset(('Q%d' % Q, 'M%d' % M))
            xm, ym = (position // self.pixel_size) + centre
            ax.text(xm, ym, 'Q%dM%d' % (Q, M), color='k', ha='left', va='top',
                    bbox={'facecolor': 'w'})
            for T in range(1, 17):
                position = self.find_offset(('Q%d' % Q, 'M%d' % M, 'T%02d' % T))
                xt, yt = (position // self.pixel_size) + centre
                xt, yt = int(xt), int(yt)
                # Draw a light green block for each tile
                bg[yt:yt + 32, xt:xt + 128] = (0.75, 1.0, 0.75)
                # Label specific tiles to show the ordering
                if T in [1, 8, 9]:
                    ax.text(xt, yt, str(T), va='top', ha='left')

        ax.imshow(bg, origin='upper')
        ax.set_title('LPD detector geometry')
        return fig


if __name__ == '__main__':
    quadpos = [(11.4, -299), (11.5, -8), (-254.5, 16), (-278.5, -275)]  # MAR 18
    with h5py.File(sys.argv[1], 'r') as f:
        geom = LPDGeometry.from_h5_file_and_quad_positions(f, quadpos, unit=1e-3)

    print(geom)
    print('Q2/M1/T07:', geom.find_offset(('Q2', 'M1', 'T07')))
