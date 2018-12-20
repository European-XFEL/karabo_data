import h5py
import os
import os.path as osp
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

    This has the old index format (first/last/status), whereas the other examples
    have the newer (first/count) format.
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
                        (16000, 1), 'u2')
    cellId[:, 0] = np.tile(np.arange(64), 250)
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
    pulseId = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/pulseId',
                        (16000, 1), 'u8')
    # In the real data, these are unevenly spaced, but this is close enough
    pulseId[:, 0] = np.tile(np.linspace(0, 125, 64, dtype='u8'), 250)
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/status',
                        (16000, 1), 'u2')  # Empty in example

    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/checksum',
                        (256, 16), 'i1', maxshape=(None, 16))  # Empty in example
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/magicNumberEnd',
                        (256, 8), 'i1', maxshape=(None, 8))  # TODO: fill in data
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/status',
                        (256,), 'u8', maxshape=(None,))  # Empty in example


from .mockdata import write_file
from .mockdata.xgm import XGM
from .mockdata.gec_camera import GECCamera
from .mockdata.sidemic_camera import SidemicCamera
from .mockdata.adc import ADC
from .mockdata.uvlamp import UVLamp
from .mockdata.motor import Motor
from .mockdata.tsens import TemperatureSensor
from .mockdata.imgfel import IMGFELCamera, IMGFELMotor
from .mockdata.gauge import Gauge
from .mockdata.dctrl import DCtrl
from .mockdata.mpod import MPOD
from .mockdata.detectors import AGIPDModule, LPDModule

def make_fxe_da_file(path):
    """Make the structure of a file with non-detector data from the FXE experiment

    Based on .../FXE/201830/p900023/r0450/RAW-R0450-DA01-S00001.h5
    """

    write_file(path, [
        XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA'),
    ], ntrains=400, chunksize=200)

def make_sa3_da_file(path):
    """Make the structure of a file with non-detector data from SASE3 tunnel

    Based on .../SA3/201830/p900026/r0317/RAW-R0317-DA01-S00000.h5
    """
    write_file(path, [
        ADC('SA3_XTD10_MCP/ADC/1', nsamples=0, channels=(
            'channel_3.output/data',
            'channel_5.output/data',
            'channel_9.output/data',
        )),
        UVLamp('SA3_XTD10_MCP/DCTRL/UVLAMP'),
        Motor('SA3_XTD10_MCP/MOTOR/X2'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30100K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30160K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30180K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30190K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30200K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30250K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30260K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30280K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30300K'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30470D_IN'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30480D_IN'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30490D_IN'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30500P'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30510C'),
        DCtrl('SA3_XTD10_VAC/DCTRL/D6_APERT_IN_OK'),
        DCtrl('SA3_XTD10_VAC/DCTRL/D12_APERT_IN_OK'),
        XGM('SA3_XTD10_XGM/XGM/DOOCS'),
        IMGFELCamera('SA3_XTD10_IMGFEL/CAM/BEAMVIEW', nsamples=0),
        IMGFELCamera('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2', nsamples=250),
        IMGFELMotor('SA3_XTD10_IMGFEL/MOTOR/FILTER'),
        IMGFELMotor('SA3_XTD10_IMGFEL/MOTOR/SCREEN'),
        MPOD('SA3_XTD10_MCP/MCPS/MPOD'),
    ], ntrains=500, chunksize=50)

def make_data_file_bad_device_name(path):
    """Not all devices have the Karabo standard A/B/C naming convention"""
    write_file(path, [
        SidemicCamera('SPB_IRU_SIDEMIC_CAM')
    ], ntrains=500, chunksize=50)

def make_agipd_file(path):
    write_file(path, [
        AGIPDModule('SPB_DET_AGIPD1M-1/DET/0CH0', frames_per_train=64)
    ], ntrains=486, chunksize=32)

def make_lpd_file(path):
    write_file(path, [
        LPDModule('FXE_DET_LPD1M-1/DET/0CH0', frames_per_train=128)
    ], ntrains=480, chunksize=32)

def make_fxe_run(dir_path):
    for modno in range(16):
        path = osp.join(dir_path, 'RAW-R0450-LPD{:0>2}-S00000.h5'.format(modno))
        write_file(path, [
            LPDModule('FXE_DET_LPD1M-1/DET/{}CH0'.format(modno), frames_per_train=128)
        ], ntrains=480, chunksize=32)

    write_file(osp.join(dir_path, 'RAW-R0450-DA01-S00000.h5'), [
        XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA_NODATA', nsamples=0),
    ], ntrains=400, chunksize=200)
    write_file(osp.join(dir_path, 'RAW-R0450-DA01-S00001.h5'), [
        XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA_NODATA', nsamples=0),
    ], ntrains=80, firsttrain=10400, chunksize=200)

if __name__ == '__main__':
    make_agipd_example_file('agipd_example.h5')
    make_fxe_da_file('fxe_control_example.h5')
    make_sa3_da_file('sa3_control_example.h5')
    make_agipd_file('agipd_example2.h5')
    make_lpd_file('lpd_example.h5')
    os.makedirs('fxe_example_run', exist_ok=True)
    make_fxe_run('fxe_example_run')
    print("Written examples.")
