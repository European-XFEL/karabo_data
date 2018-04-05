import h5py
import numpy as np

vlen_bytes = h5py.special_dtype(vlen=bytes)

def make_metadata(h5file, data_sources, chunksize=16):
    N = len(data_sources)
    if N % chunksize:
        N += chunksize - (N % chunksize)

    root = [ds.split('/', 1)[0] for ds in data_sources]
    devices = [ds.split('/', 1)[1] for ds in data_sources]

    sources_ds = h5file.create_dataset('METADATA/dataSourceId', (N,),
                                       dtype=vlen_bytes, maxshape=(None,))
    sources_ds[:len(data_sources)] = data_sources
    root_ds = h5file.create_dataset('METADATA/root', (N,),
                                    dtype=vlen_bytes, maxshape=(None,))
    root_ds[:len(data_sources)] = root
    devices_ds = h5file.create_dataset('METADATA/deviceId', (N,),
                                       dtype=vlen_bytes, maxshape=(None,))
    devices_ds[:len(data_sources)] = devices

def write_train_ids(f, path, N, chunksize=16):
    """Make a dataset of fake train IDs at the given path

    Real train IDs are much larger (~10^9), so hopefully these won't be mistaken
    for real ones.
    """
    if N % chunksize:
        Npad = N + chunksize - (N % chunksize)
    else:
        Npad = N
    ds = f.create_dataset(path, (Npad,), 'u8', maxshape=(None,))
    ds[:N] = np.arange(10000, 10000 + N)

def make_agipd_example_file(path):
    """Make the structure of a data file from the AGIPD detector

    Based on /gpfs/exfel/d/proc/XMPL/201750/p700000/r0803/CORR-R0803-AGIPD07-S00000.h5
    """
    f = h5py.File(path, 'w')

    slow_channels = ['header', 'detector', 'trailer']
    channels = slow_channels + ['image']
    train_ids = np.arange(10000, 10250)   # Real train IDs are ~10^9

    # RUN - empty in the example I'm working from
    f.create_group('RUN')

    # METADATA - lists the data sources in this file
    make_metadata(f, ['INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/'+ch
                      for ch in channels])

    def make_train_ids(path):
        d = f.create_dataset(path, (256,), 'u8', maxshape=(None,))
        d[:250] = train_ids

    # INDEX - matching up data to train IDs
    write_train_ids(f, 'INDEX/trainId', 250)
    for ch in channels:
        grp_name = 'INDEX/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/%s/' % ch
        first = f.create_dataset(grp_name + 'first', (256,), 'u8', maxshape=(None,))
        last = f.create_dataset(grp_name + 'last', (256,), 'u8', maxshape=(None,))
        status = f.create_dataset(grp_name + 'status', (256,), 'u4', maxshape=(None,))
        if ch in slow_channels:
            first[:250] = np.arange(250)
            last[:250] = np.arange(250)
        else:
            first[:250] = np.arange(0, 16000, 64)
            last[:250] = np.arange(63, 16000, 64)
        status[:250] = 1

    # INSTRUMENT - the data itself
    #   first, train IDs for each channel
    for ch in slow_channels:
        make_train_ids('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/%s/trainId' % ch)
    fast_tids = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/trainId',
                                    (16000, 1), 'u8')
    fast_tids[:,0] = np.repeat(train_ids, 64)

    # TODO: Not sure what this is, but it has quite a regular structure.
    # 5408 = 13 x 13 x 32
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/detector/data',
                        (256, 5408), 'u1', maxshape=(None, 5408))

    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/dataId',
                        (256,), 'u8', maxshape=(None,))  # Empty in example
    linkId = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/linkId',
                        (256,), 'u8', maxshape=(None,))
    linkId[:250] = 18446744069414584335  # Copied from example
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/magicNumberBegin',
                        (256, 8), 'i1', maxshape=(None, 8))  # TODO: fill in data
    vmaj = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/majorTrainFormatVersion',
                        (256,), 'u4', maxshape=(None,))
    vmaj[:250] = 1
    vmin = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/minorTrainFormatVersion',
                        (256,), 'u4', maxshape=(None,))
    vmin[:250] = 0
    pc = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/pulseCount',
                        (256,), 'u8', maxshape=(None,))
    pc[:250] = 64
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/reserved',
                        (256, 16), 'u1', maxshape=(None, 16))  # Empty in example

    cellId = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/cellId',
                        (16000,), 'u2')
    cellId[:] = np.tile(np.arange(64), 250)
    # The data itself
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/data',
                        (16000, 512, 128), 'f4')
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/gain',
                        (16000, 512, 128), 'u1')
    length = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/length',
                        (16000, 1), 'u4', maxshape=(None, 1))
    length[:] = 262144  # = 512*128*4(bytes) ?
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/mask',
                        (16000, 512, 128, 3), 'u1')  # TODO: values 128 or 0
    # TODO: Repeated 64 numbers, unevenly spaced 0-125
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/pulseId',
                        (16000, 1), 'u8')
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/status',
                        (16000, 1), 'u2')  # Empty in example

    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/checksum',
                        (256, 16), 'i1', maxshape=(None, 16))  # Empty in example
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/magicNumberEnd',
                        (256, 8), 'i1', maxshape=(None, 8))  # TODO: fill in data
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/status',
                        (256,), 'u8', maxshape=(None,))  # Empty in example


XGM_CONTROL_TOPICS = [
    ('beamPosition/ixPos', 'f4'),
    ('beamPosition/iyPos', 'f4'),
    ('current/bottom/output', 'f4'),
    ('current/bottom/rangeCode', 'i4'),
    ('current/left/output', 'f4'),
    ('current/left/rangeCode', 'i4'),
    ('current/right/output', 'f4'),
    ('current/right/rangeCode', 'i4'),
    ('current/top/output', 'f4'),
    ('current/top/rangeCode', 'i4'),
    ('gasDosing/measuredPressure', 'f4'),
    ('gasDosing/pressureSetPoint', 'f4'),
    ('gasSupply/gasTypeId', 'i4'),
    ('gasSupply/gsdCompatId', 'i4'),
    ('pollingInterval', 'i4'),
    ('pressure/dcr', 'f4'),
    ('pressure/gasType', 'i4'),
    ('pressure/pressure1', 'f4'),
    ('pressure/pressureFiltered', 'f4'),
    ('pressure/rd', 'f4'),
    ('pressure/rsp', 'f4'),
    ('pulseEnergy/conversion', 'f8'),
    ('pulseEnergy/crossUsed', 'f4'),
    ('pulseEnergy/gammaUsed', 'f4'),
    ('pulseEnergy/gmdError', 'i4'),
    ('pulseEnergy/nummberOfBrunches', 'f4'),
    ('pulseEnergy/photonFlux', 'f4'),
    ('pulseEnergy/pressure', 'f4'),
    ('pulseEnergy/temperature', 'f4'),
    ('pulseEnergy/usedGasType', 'i4'),
    ('pulseEnergy/wavelengthUsed', 'f4'),
    ('signalAdaption/dig', 'i4'),
]

XGM_INSTRUMENT_TOPICS = [
    ('intensityTD', 'f4'),
    ('intensityAUXTD', 'f4'),
    ('intensitySigma/x_data', 'f4'),
    ('intensitySigma/y_data', 'f4'),
    ('xTD', 'f4'),
    ('yTD', 'f4'),
]

def add_xgm_doocs(f, domain_id, N=400):
    deviceId = domain_id + "/DOOCS/MAIN"
    deviceId_output = domain_id + "/DOOCS/MAIN:output"

    # INDEX
    i_first = f.create_dataset('INDEX/%s/first' % deviceId,
                               (N,), 'u8', maxshape=(None,))
    i_count = f.create_dataset('INDEX/%s/count' % deviceId,
                               (N,), 'u8', maxshape=(None,))
    i2_first = f.create_dataset('INDEX/%s/data/first' % deviceId_output,
                                (N,), 'u8', maxshape=(None,))
    i2_count = f.create_dataset('INDEX/%s/data/count' % deviceId_output,
                                (N,), 'u8', maxshape=(None,))
    # This doesn't actually hold in all cases - a few trains don't have MAIN:output/data
    i_first[:] = i2_first[:] = np.arange(N)
    i_count[:] = i2_count[:] = 1

    # CONTROL & RUN
    # Creating empty datasets for now.
    for (topic, datatype) in XGM_CONTROL_TOPICS:
        f.create_dataset('CONTROL/%s/%s/timestamp' % (deviceId, topic),
                         (N,), 'u8', maxshape=(None,))
        f.create_dataset('CONTROL/%s/%s/value' % (deviceId, topic),
                         (N,), datatype, maxshape=(None,))

        # RUN is the value at the start of the run
        f.create_dataset('RUN/%s/%s/timestamp' % (deviceId, topic),
                         (1,), 'u8', maxshape=(None,))
        f.create_dataset('RUN/%s/%s/value' % (deviceId, topic),
                         (1,), datatype, maxshape=(None,))

    # INSTRUMENT
    write_train_ids(f, 'INSTRUMENT/%s/data/trainId' % deviceId_output,
                    N, chunksize=200)
    for (topic, datatype) in XGM_INSTRUMENT_TOPICS:
        # 1000 cols, but only the first 30 have data in the example
        f.create_dataset('INSTRUMENT/%s/data/%s' % (deviceId_output, topic),
                         (N, 1000), datatype, maxshape=(None, 1000))


GEC_CONTROL_TOPICS = [
    ('acquisitionTime', 'f4'),
    ('binningX', 'i4'),
    ('binningY', 'i4'),
    ('coolingLevel', 'i4'),
    ('cropLines', 'i4'),
    ('enableBiasCorrection', 'u1'),
    ('enableBurstMode', 'u1'),
    ('enableCooling', 'u1'),
    ('enableCropMode', 'u1'),
    ('enableExtTrigger', 'u1'),
    ('enableShutter', 'u1'),
    ('enableSync', 'u1'),
    ('exposureTime', 'i4'),
    ('firmwareVersion', 'i4'),
    ('modelId', 'i4'),
    ('numPixelInX', 'i4'),
    ('numPixelInY', 'i4'),
    ('numberOfCoolingLevels', 'i4'),
    ('numberOfMeasurements', 'i4'),
    ('pixelSize', 'f4'),
    ('readOutSpeed', 'i4'),
    ('shutterCloseTime', 'i4'),
    ('shutterOpenTime', 'i4'),
    ('shutterState', 'i4'),
    ('syncHigh', 'u1'),
    ('targetTemperature', 'i4'),
    ('temperatureBack', 'f4'),
    ('temperatureSensor', 'f4'),
    ('triggerTimeOut', 'i4'),
    ('updateInterval', 'i4'),
]

GEC_INSTRUMENT_TOPICS = [
    ('image/bitsPerPixel', 'i4', ()),
    ('image/dimTypes', 'i4', (2,)),
    ('image/dims', 'u8', (2,)),
    ('image/encoding', 'i4', ()),
    ('image/pixels', 'u2', (255, 1024)),
    ('image/roiOffsets', 'u8', (2,)),
]

def add_gec_camera(f, domain_id, N=400):
    deviceId = domain_id + "/CAM/CAMERA"
    deviceId_output = domain_id + "/CAM/CAMERA:daqOutput"

    # INDEX
    i_first = f.create_dataset('INDEX/%s/first' % deviceId,
                               (N,), 'u8', maxshape=(None,))
    i_count = f.create_dataset('INDEX/%s/count' % deviceId,
                               (N,), 'u8', maxshape=(None,))
    i_first[:] = np.arange(N)
    i_count[:] = 1

    i2_first = f.create_dataset('INDEX/%s/data/first' % deviceId_output,
                                (N,), 'u8', maxshape=(None,))
    i2_count = f.create_dataset('INDEX/%s/data/count' % deviceId_output,
                                (N,), 'u8', maxshape=(None,))
    # This pattern is an approximation of the real one
    i2_first[:] = np.repeat(np.arange(N // 50), 50)
    i2_count[::50] = 1

    # CONTROL & RUN
    # Creating empty datasets for now.
    for (topic, datatype) in GEC_CONTROL_TOPICS:
        f.create_dataset('CONTROL/%s/%s/timestamp' % (deviceId, topic),
                         (N,), 'u8', maxshape=(None,))
        f.create_dataset('CONTROL/%s/%s/value' % (deviceId, topic),
                         (N,), datatype, maxshape=(None,))

        # RUN is the value at the start of the run
        f.create_dataset('RUN/%s/%s/timestamp' % (deviceId, topic),
                         (1,), 'u8', maxshape=(None,))
        f.create_dataset('RUN/%s/%s/value' % (deviceId, topic),
                         (1,), datatype, maxshape=(None,))

    # INSTRUMENT
    tid = f.create_dataset('INSTRUMENT/%s/data/trainId' % deviceId_output,
                           (200,), 'u8', maxshape=(None,))
    tid[:N//50] = np.arange(10000, 10000 + N, 50, dtype='u8')
    for (topic, datatype, dims) in GEC_INSTRUMENT_TOPICS:
        f.create_dataset('INSTRUMENT/%s/data/%s' % (deviceId_output, topic),
                         (200,) + dims, datatype, maxshape=((None,) + dims))
    f['INSTRUMENT/%s/data/image/bitsPerPixel' % deviceId_output][:8] = 16
    f['INSTRUMENT/%s/data/image/dims' % deviceId_output][:8] = [1024, 255]

def make_fxe_da_file(path):
    """Make the structure of a file with non-detector data from the FXE experiment

    Based on .../FXE/201830/p900023/r0450/RAW-R0450-DA01-S00001.h5
    """
    f = h5py.File(path, 'w')

    data_sources = [
        'CONTROL/SA1_XTD2_XGM/DOOCS/MAIN',
        'INSTRUMENT/SA1_XTD2_XGM/DOOCS/MAIN:output/data',
        'CONTROL/SPB_XTD9_XGM/DOOCS/MAIN',
        'INSTRUMENT/SPB_XTD9_XGM/DOOCS/MAIN:output/data',
        'CONTROL/FXE_XAD_GEC/CAM/CAMERA',
        'INSTRUMENT/FXE_XAD_GEC/CAM/CAMERA:daqOutput/data',
    ]

    make_metadata(f, data_sources, chunksize=200)

    write_train_ids(f, 'INDEX/trainId', 400, chunksize=1)

    add_xgm_doocs(f, 'SA1_XTD2_XGM')
    add_xgm_doocs(f, 'SPB_XTD9_XGM')
    add_gec_camera(f, 'FXE_XAD_GEC')

if __name__ == '__main__':
    make_agipd_example_file('agipd_example.h5')
    make_fxe_da_file('fxe_control_example.h5')
    print("Written examples.")
