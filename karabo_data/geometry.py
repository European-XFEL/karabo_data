import numpy as np
import h5py, re

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
    tiles = np.asarray(np.split(channelData, 8, axis=1))
    tiles = np.asarray(np.split(tiles, 2, axis=1))
    orderedTiles = np.moveaxis(tiles.reshape(16, 128, 32, channelData.shape[2]),
                               2, 1)
    if clockwiseOrder:
        # Naturally, the tile data after splitting is in reading
        # order (i.e. top left tile is first, top right tile is second,
        # etc.). The official LPD tile order however is clockwise,
        # starting with the top right tile. The following array
        # contains indices of tiles in reading order as they would
        # be iterated in clockwise order (starting from the top right)
        readingOrderToClockwise = list(range(8, 16)) + list(range(7, -1, -1))
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
