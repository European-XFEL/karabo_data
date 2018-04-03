import h5py
import numpy as np

vlen_bytes = h5py.special_dtype(vlen=bytes)

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
    dsId = f.create_dataset('METADATA/dataSourceId', (16,), dtype=vlen_bytes, maxshape=(None,))
    devId = f.create_dataset('METADATA/deviceId', (16,), dtype=vlen_bytes, maxshape=(None,))
    root = f.create_dataset('METADATA/root', (16,), dtype=vlen_bytes, maxshape=(None,))
    for i, ch in enumerate(channels):
        ds = 'INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/'+ch
        dsId[i] = ds
        root[i], devId[i] = ds.split('/', 1)

    def make_train_ids(path):
        d = f.create_dataset(path, (256,), 'u8', maxshape=(None,))
        d[:250] = train_ids

    # INDEX - matching up data to train IDs
    make_train_ids('INDEX/trainId')
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


if __name__ == '__main__':
    make_agipd_example_file('test.h5')
    print("Written test.h5")
