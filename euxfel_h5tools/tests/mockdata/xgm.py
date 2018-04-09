from .base import DeviceBase

class XGM(DeviceBase):
    control_keys = [
        ('beamPosition/ixPos', 'f4', ()),
        ('beamPosition/iyPos', 'f4', ()),
        ('current/bottom/output', 'f4', ()),
        ('current/bottom/rangeCode', 'i4', ()),
        ('current/left/output', 'f4', ()),
        ('current/left/rangeCode', 'i4', ()),
        ('current/right/output', 'f4', ()),
        ('current/right/rangeCode', 'i4', ()),
        ('current/top/output', 'f4', ()),
        ('current/top/rangeCode', 'i4', ()),
        ('gasDosing/measuredPressure', 'f4', ()),
        ('gasDosing/pressureSetPoint', 'f4', ()),
        ('gasSupply/gasTypeId', 'i4', ()),
        ('gasSupply/gsdCompatId', 'i4', ()),
        ('pollingInterval', 'i4', ()),
        ('pressure/dcr', 'f4', ()),
        ('pressure/gasType', 'i4', ()),
        ('pressure/pressure1', 'f4', ()),
        ('pressure/pressureFiltered', 'f4', ()),
        ('pressure/rd', 'f4', ()),
        ('pressure/rsp', 'f4', ()),
        ('pulseEnergy/conversion', 'f8', ()),
        ('pulseEnergy/crossUsed', 'f4', ()),
        ('pulseEnergy/gammaUsed', 'f4', ()),
        ('pulseEnergy/gmdError', 'i4', ()),
        ('pulseEnergy/nummberOfBrunches', 'f4', ()),
        ('pulseEnergy/photonFlux', 'f4', ()),
        ('pulseEnergy/pressure', 'f4', ()),
        ('pulseEnergy/temperature', 'f4', ()),
        ('pulseEnergy/usedGasType', 'i4', ()),
        ('pulseEnergy/wavelengthUsed', 'f4', ()),
        ('signalAdaption/dig', 'i4', ()),
    ]

    # Technically, only the part before the / is the output channel.
    # But there is a structure associated with the part one level after that,
    # and we don't know what else to call it.
    output_channels = ('output/data',)

    instrument_keys = [
        ('intensityTD', 'f4', (1000,)),
        ('intensityAUXTD', 'f4', (1000,)),
        ('intensitySigma/x_data', 'f4', (1000,)),
        ('intensitySigma/y_data', 'f4', (1000,)),
        ('xTD', 'f4', (1000,)),
        ('yTD', 'f4', (1000,)),
    ]
