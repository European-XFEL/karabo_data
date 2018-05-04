import numpy as np
import h5py, re
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
    orderedTiles = np.moveaxis(a, -1, -2)

    if clockwiseOrder:
        # Naturally, the tile data after splitting is in reading
        # order (i.e. top left tile is first, top right tile is second,
        # etc.). The official LPD tile order however is clockwise,
        # starting with the top right tile. The following array
        # contains indices of tiles in reading order as they would
        # be iterated in clockwise order (starting from the top right)
        readingOrderToClockwise = list(range(7, -1, -1)) + list(range(8, 16))
        # Return tiles in reading order
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
        data_1mod = list(data.values())[0]
        out = np.zeros(data_1mod.shape[:-2] + (1500, 1500),
                       dtype=data_1mod.dtype)
        centre = np.asarray(out.shape[:2]) // 2

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
                x0 -= 750  # Offset

                # ???
#                 if module_ix >= 8:
#                     tile_data = tile_data[..., ::-1]

                out[..., x0:x0 + tile_data.shape[-2],
                         y0:y0 + tile_data.shape[-1]] = tile_data[..., ::-1, :]

        return out


if __name__ == '__main__':
    quadpos = [(11.4, -299), (11.5, -8), (-254.5, 16), (-278.5, -275)]  # MAR 18
    with h5py.File(sys.argv[1], 'r') as f:
        geom = LPDGeometry.from_h5_file_and_quad_positions(f, quadpos, unit=1e-3)

    print(geom)
    print('Q2/M1/T07:', geom.find_offset(('Q2', 'M1', 'T07')))
