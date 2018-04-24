from .base import DeviceBase

class SidemicCamera(DeviceBase):
    # Based on example in /gpfs/exfel/d/raw/SPB/201701/p002012/r0309/RAW-R0309-DA01-S00000.h5

    # Technically, only the part before the / is the output channel.
    # But there is a structure associated with the part one level after that,
    # and we don't know what else to call it.
    output_channels = ('daqOutput/data',)

    instrument_keys = [
        ('image/bitsPerPixel', 'i4', ()),
        ('image/dimTypes', 'i4', (2,)),
        ('image/dims', 'u8', (2,)),
        ('image/encoding', 'i4', ()),
        ('image/pixels', 'u2', (2058, 2456)),
        ('image/roiOffsets', 'u8', (2,)),
    ]

    def write_instrument(self, f):
        super().write_instrument(f)

        # Fill in some fixed metadata about the image
        for channel in self.output_channels:
            image_grp = 'INSTRUMENT/%s:%s/image/' % (self.device_id, channel)
            f[image_grp + 'bitsPerPixel'][:self.nsamples] = 16
            f[image_grp + 'dims'][:self.nsamples] = [1000, 1000]
