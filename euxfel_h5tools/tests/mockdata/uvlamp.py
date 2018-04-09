from .base import DeviceBase
from .control_common import interlock_keys, triggers_keys

class UVLamp(DeviceBase):
    control_keys = [
        ('ASafeValue', 'u1', ()),
        ('busy', 'u1', ()),
        ('epsilon', 'f4', ()),
        ('force', 'u1', ()),
        ('hardwareErrorDescriptor', 'u4', ()),
        ('hardwareStatusBitField', 'u4', ()),
        ('maxUpdateFrequency', 'f4', ()),
        ('pollInterval', 'f4', ()),
        ('pwmCycleLimit', 'i2', ()),
        ('pwmDutyCycle', 'f4', ()),
        ('pwmFrequency', 'f4', ()),
        ('softDeviceId', 'u4', ()),
        ('terminal', 'u4', ()),
    ] + interlock_keys + triggers_keys

