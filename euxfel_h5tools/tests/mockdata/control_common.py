interlock_keys = [
    ('interlock/AActCommand', 'u4', ()),
    ('interlock/AActionState', 'u4', ()),
    ('interlock/ACndAriOp', 'i2', ()),
    ('interlock/ACndComOp', 'i2', ()),
    ('interlock/ACndEnable', 'u1', ()),
    ('interlock/ACndFiltertime', 'i2', ()),
    ('interlock/ACndHysteresis', 'u1', ()),
    ('interlock/ACndSrc1Detail', 'i2', ()),
    ('interlock/ACndSrc2Detail', 'i2', ()),
    ('interlock/ACndThreshold', 'u1', ()),
    ('interlock/ACndValue1', 'u1', ()),
    ('interlock/ACndValue2', 'u1', ()),
    ('interlock/AConditionState', 'u4', ()),
    ('interlockOk', 'u1', ()),
    ('interlockOn', 'u1', ()),
]

triggers_keys = [
    ('trigger', 'u4', (1000,)),
] + sum(([
    ('triggers/trig%d/enable' % n, 'u1', ()),
    ('triggers/trig%d/interval' % n, 'f8', ()),
] for n in range(1, 11)), [])
