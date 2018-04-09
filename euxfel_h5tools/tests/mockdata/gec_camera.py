from .base import DeviceBase

class GECCamera(DeviceBase):
    control_keys = [
        ('acquisitionTime', 'f4', ()),
        ('binningX', 'i4', ()),
        ('binningY', 'i4', ()),
        ('coolingLevel', 'i4', ()),
        ('cropLines', 'i4', ()),
        ('enableBiasCorrection', 'u1', ()),
        ('enableBurstMode', 'u1', ()),
        ('enableCooling', 'u1', ()),
        ('enableCropMode', 'u1', ()),
        ('enableExtTrigger', 'u1', ()),
        ('enableShutter', 'u1', ()),
        ('enableSync', 'u1', ()),
        ('exposureTime', 'i4', ()),
        ('firmwareVersion', 'i4', ()),
        ('modelId', 'i4', ()),
        ('numPixelInX', 'i4', ()),
        ('numPixelInY', 'i4', ()),
        ('numberOfCoolingLevels', 'i4', ()),
        ('numberOfMeasurements', 'i4', ()),
        ('pixelSize', 'f4', ()),
        ('readOutSpeed', 'i4', ()),
        ('shutterCloseTime', 'i4', ()),
        ('shutterOpenTime', 'i4', ()),
        ('shutterState', 'i4', ()),
        ('syncHigh', 'u1', ()),
        ('targetTemperature', 'i4', ()),
        ('temperatureBack', 'f4', ()),
        ('temperatureSensor', 'f4', ()),
        ('triggerTimeOut', 'i4', ()),
        ('updateInterval', 'i4', ()),
    ]

    # Technically, only the part before the / is the output channel.
    # But there is a structure associated with the part one level after that,
    # and we don't know what else to call it.
    output_channels = ('daqOutput/data',)

    instrument_keys = [
        ('image/bitsPerPixel', 'i4', ()),
        ('image/dimTypes', 'i4', (2,)),
        ('image/dims', 'u8', (2,)),
        ('image/encoding', 'i4', ()),
        ('image/pixels', 'u2', (255, 1024)),
        ('image/roiOffsets', 'u8', (2,)),
    ]

    def write_instrument(self, f):
        super().write_instrument(f)

        # Fill in some fixed metadata about the image
        for channel in self.output_channels:
            image_grp = 'INSTRUMENT/%s:%s/image/' % (self.device_id, channel)
            f[image_grp + 'bitsPerPixel'][:self.nsamples] = 16
            f[image_grp + 'dims'][:self.nsamples] = [1024, 255]
