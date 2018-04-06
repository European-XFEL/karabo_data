from .base import DeviceBase

class ADC(DeviceBase):
    def __init__(self, device_id, nsamples=None, channels=()):
        super().__init__(device_id, nsamples)
        self.output_channels = channels

    instrument_keys = [
        ('baseline', 'f8', ()),
        ('peakMean', 'f8', ()),
        ('peakStd', 'f8', ()),
        ('peaks', 'f4', (1000,)),
        ('rawBaseline', 'u4', ()),
        ('rawData', 'u2', (4096,)),
        ('rawPeaks', 'u4', (1000,)),
        ('samplesForBaseline', 'u4', ()),
        ('samplesPerPeak', 'u4', ()),
    ]

