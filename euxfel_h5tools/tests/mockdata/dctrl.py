from .base import DeviceBase
from .control_common import interlock_keys, triggers_keys


class DCtrl(DeviceBase):
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
    ] + [
        # TODO: is there a way to factor these out?
        ('interlock/AActionState', 'u4', ()),
        ('interlock/AConditionState', 'u4', ()),
        ('interlock/a1/AActCommand', 'u4', (1000,)),
        ('interlock/c1/ACndAriOp', 'i2', ()),
        ('interlock/c1/ACndComOp', 'i2', ()),
        ('interlock/c1/ACndEnable', 'u1', ()),
        ('interlock/c1/ACndFiltertime', 'i2', ()),
        ('interlock/c1/ACndHysteresis', 'u4', ()),
        ('interlock/c1/ACndSrc1Detail', 'i2', ()),
        ('interlock/c1/ACndSrc2Detail', 'i2', ()),
        ('interlock/c1/ACndThreshold', 'u4', ()),
        ('interlock/c1/ACndValue1', 'u4', ()),
        ('interlock/c1/ACndValue2', 'u1', ()),
        ('interlock/c2/ACndAriOp', 'i2', ()),
        ('interlock/c2/ACndComOp', 'i2', ()),
        ('interlock/c2/ACndEnable', 'u1', ()),
        ('interlock/c2/ACndFiltertime', 'i2', ()),
        ('interlock/c2/ACndHysteresis', 'u4', ()),
        ('interlock/c2/ACndSrc1Detail', 'i2', ()),
        ('interlock/c2/ACndSrc2Detail', 'i2', ()),
        ('interlock/c2/ACndThreshold', 'u4', ()),
        ('interlock/c2/ACndValue1', 'u4', ()),
        ('interlock/c2/ACndValue2', 'u1', ()),
        ('interlockOk', 'u1', ()),
        ('interlockOn', 'u1', ()),
    ] + triggers_keys
