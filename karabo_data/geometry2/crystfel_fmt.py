"""Read and write geometry in CrystFEL format.
"""

HEADER_TEMPLATE = """\
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


PANEL_TEMPLATE = """
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


def _crystfel_format_vec(vec):
    """Convert an array of 3 numbers to CrystFEL format like "+1.0x -0.1y"
    """
    s = '{:+}x {:+}y'.format(*vec[:2])
    if vec[2] != 0:
        s += ' {:+}z'.format(vec[2])
    return s


def frag_to_crystfel(fragment, p, a, ss_slice, fs_slice, dims):
    tile_name = 'p{}a{}'.format(p, a)
    c = fragment.corner_pos
    dim_list = []
    for num, value in dims.items():
        if value == 'modno':
            key = p
        else:
            key = value
        dim_list.append('{}/dim{} = {}'.format(tile_name, num, key))

    return PANEL_TEMPLATE.format(
        dims='\n'.join(dim_list),
        name=tile_name,
        min_ss=ss_slice.start,
        max_ss=ss_slice.stop - 1,
        min_fs=fs_slice.start,
        max_fs=fs_slice.stop - 1,
        ss_vec=_crystfel_format_vec(fragment.ss_vec),
        fs_vec=_crystfel_format_vec(fragment.fs_vec),
        corner_x=c[0],
        corner_y=c[1],
        coffset=c[2],
    )

def write_crystfel_geom(self, filename, *,
                        data_path='/entry_1/instrument_1/detector_1/data',
                        mask_path=None, dims=('frame', 'modno', 'ss', 'fs'),
                        adu_per_ev=None, clen=None, photon_energy=None):
    """Write this geometry to a CrystFEL format (.geom) geometry file.
    """
    from .. import __version__

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

    frame_dim = None
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
            ss_slice, fs_slice = self._tile_slice(a)
            panel_chunks.append(frag_to_crystfel(
                fragment, p, a, ss_slice, fs_slice, tile_dims
            ))

    resolution = 1.0 / self.pixel_size  # Pixels per metre

    paths = dict(data=data_path)
    if mask_path:
        paths['mask'] = mask_path
    path_str = '\n'.join('{} = {} ;'.format(i, j) for i, j in paths.items())

    with open(filename, 'w') as f:
        f.write(HEADER_TEMPLATE.format(
            version=__version__,
            paths=path_str,
            frame_dim=frame_dim,
            resolution=resolution,
            adu_per_ev=adu_per_ev_str,
            clen=clen_str,
            photon_energy=photon_energy_str
        ))
        rigid_groups = self._get_rigid_groups()
        f.write(rigid_groups)
        for chunk in panel_chunks:
            f.write(chunk)
