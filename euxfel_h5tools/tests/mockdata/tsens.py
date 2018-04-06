from .base import DeviceBase
from .control_common import interlock_keys, triggers_keys


class TemperatureSensor(DeviceBase):
    control_keys = [
        ('AAlarmH', 'f4', ()),
        ('AAlarmL', 'f4', ()),
        ('AAverage', 'u1', ()),
        ('busy', 'u1', ()),
        ('calibration/expbase', 'f4', ()),
        ('calibration/formulaType', 'u1', ()),
        ('calibration/offset', 'f4', ()),
        ('calibration/rawValue', 'u4', ()),
        ('calibration/scale', 'f4', ()),
        ('calibration/terminalFactor', 'f4', ()),
        ('calibration/terminalOffset', 'f4', ()),
        ('epsSemiRaw', 'f4', ()),
        ('epsilon', 'f4', ()),
        ('force', 'u1', ()),
        ('hardwareErrorDescriptor', 'u4', ()),
        ('hardwareStatusBitField', 'u4', ()),
        ('maxUpdateFrequency', 'f4', ()),
        ('pollInterval', 'f4', ()),
        ('relativeEpsilon', 'u1', ()),
        ('semiRawValue', 'f4', ()),
        ('softDeviceId', 'u4', ()),
        ('terminal', 'u4', ()),
        ('value', 'f4', ()),
    ] + interlock_keys + triggers_keys
