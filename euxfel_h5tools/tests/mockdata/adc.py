from .base import DeviceBase

class ADC(DeviceBase):
    def __init__(self, device_id, nsamples=None, channels=()):
        super().__init__(device_id, nsamples)
        self.output_channels = channels

    control_keys = [
        ('config/softTrigTime', 'u4', ()),
        ('dacNode/dacCyclesSamples', 'u4', ()),
        ('dacNode/dacData', 'i4', (1024,)),
        ('dacNode/dacSkipSamples', 'u4', ()),
        ('dacNode/dacTrigger', 'u1', ()),
        ('dacNode/dacTriggerPeriod', 'u4', ()),
        ('dacNode/dacVoltageData', 'f8', (1000,)),
        ('dacNode/enableDAC', 'u1', ()),
        ('dacNode/voltageIntercept', 'f8', ()),
        ('dacNode/voltageSlope', 'f8', ()),
        ('delay', 'u4', ()),
        ('numberRawSamples', 'u4', ()),
        ('skipSamples', 'u4', ()),
        ('trainId', 'u8', ()),
        ('triggerTime', 'i4', ()),
        ('triggerTimeStat', 'u2', (1000,)),
    ] + sum(([
        ('channel_%d/baseStart' % n, 'u4', ()),
        ('channel_%d/baseStop' % n, 'u4', ()),
        ('channel_%d/baseline' % n, 'f4', ()),
        ('channel_%d/calibrationFactor' % n, 'f8', ()),
        ('channel_%d/enablePeakComputation' % n, 'u1', ()),
        ('channel_%d/enableRawDataStreaming' % n, 'u1', ()),
        ('channel_%d/fixedBaseline' % n, 'f8', ()),
        ('channel_%d/fixedBaselineEna' % n, 'u1', ()),
        ('channel_%d/initialDelay' % n, 'u4', ()),
        ('channel_%d/numPulses' % n, 'u4', ()),
        ('channel_%d/peakMean' % n, 'f4', ()),
        ('channel_%d/peakSamples' % n, 'u4', ()),
        ('channel_%d/peakStd' % n, 'f4', ()),
        ('channel_%d/pulsePeriod' % n, 'u4', ()),
    ] for n in range(10)), [])

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
