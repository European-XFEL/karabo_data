"""Script that creates a mock-run for the basler camera"""
from os.path import join as op_join

from .base import DeviceBase

class BaslerCamera(DeviceBase):
    """
    Basler Camera device
    Based on example /gpfs/exfel/exp/SPB/201930/p900061/raw/r0055/RAW-R0055-DA01-S00000.h5
    """

    output_channels = ('daqQutput/data',)

    instrument_keys = [
        ('image/bitsPerPixel', 'i4', ()),
        ('image/dimTypes', 'i4', (2,)),
        ('image/dims', 'u8', (2,)),
        ('image/encoding', 'i4', ()),
        ('image/pixels', 'u2', (1029, 1228)), # Odd shape
        ('image/roiOffsets', 'u8', (2,)),
        ('image/binning', 'u8', (2,)),
        ('image/flipX', 'u1', ()),
        ('image/flipY', 'u1', ())
    ]

    def write_instrument(self, f):
        super().write_instrument(f)

        # Add fixed metadata
        for channel in self.output_channels:
            image_grp = 'INSTRUMENT/{}:{}/image/'.format(self.device_id, channel)
            f[op_join(image_grp, 'bitsPerPixel')][:self.nsamples] = 16
            f[op_join(image_grp, 'dims')][:self.nsamples] = [1029, 1228]


