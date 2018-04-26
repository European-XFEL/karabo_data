import numpy as np
import os.path as osp
import pytest
from time import sleep

from karabo_data import (H5File, RunDirectory, stack_data, stack_detector_data)

FILEPATH = '/home/tmichela/Downloads/data/R0019-lpd00-S00000.h5'
RUNPATH = '/home/tmichela/Downloads//data/r0185'
RUNPATH_SLOW = '/home/tmichela/Downloads/data/slow'
# RUNPATH_MIXED = '/media/tmichela/storage/p002013/r0385'

lpd_single_file = pytest.mark.skipif(not osp.exists(FILEPATH),
                                     reason="require data file.")
agipd_run = pytest.mark.skipif(not osp.exists(RUNPATH),
                               reason="require data files.")
xgm_run = pytest.mark.skipif(not osp.exists(RUNPATH_SLOW),
                             reason="require data files.")
# mixed_run = pytest.mark.skipif(not osp.exists(RUNPATH_MIXED),
#                              reason="require data files.")


@lpd_single_file
def test_H5File():
    f = H5File(FILEPATH)
    assert f.file.filename == FILEPATH
    assert len(f.train_ids) == 20
    f.close()


@lpd_single_file
def test_get_file_info():
    with H5File(FILEPATH) as f:

        trains = f.train_ids
        train_count = len(trains)
        devices = f.devices
        sources = f.sources

        print('train ids:', trains)
        print('count: ', train_count)
        print('devices: ', devices)
        print('data_sources: ', sources)

        assert 1455918683 in trains
        assert 1234 not in trains
        assert train_count == 20
        assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf/detector' in devices
        assert 'INSTRUMENT/FXE_DET_LPD1M-1/DET/0CH0:xtdf/detector' in sources


@lpd_single_file
def test_iterate_trains():
    with H5File(FILEPATH) as f:

        for train_id, data in f.trains():
            print(train_id)
            assert train_id in range(1455918683, 1455918703)
            assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in data.keys()
            assert len(data) == 1
            assert len(data['FXE_DET_LPD1M-1/DET/0CH0:xtdf']) == 21


@lpd_single_file
def test_get_train_per_id():
    with H5File(FILEPATH) as f:
        train, data = f.train_from_id(1455918700)
        print(data)
        assert data is not None
        assert train == 1455918700

        with pytest.raises(KeyError) as info:
            data = f.train_from_id(1234)  # train id not in file
        print(info)


@lpd_single_file
def test_get_train_per_index():
    with H5File(FILEPATH) as f:
        train, data = f.train_from_index(0)
        assert data is not None
        assert train == 1455918683

        with pytest.raises(ValueError) as info:
            data, _, _ = f.train_from_index(20)  # index out of range
        print(info)


@lpd_single_file
def test_read_metadata():
    with H5File(FILEPATH) as f:

        _, data1 = f.train_from_index(0)
        sleep(2)
        _, data2 = f.train_from_index(0)
        assert 'metadata' in data1['FXE_DET_LPD1M-1/DET/0CH0:xtdf']
        assert data1['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['metadata']['timestamp']['tid'] == 1455918683
        assert data1['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['metadata']['timestamp']['sec'] != data2['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['metadata']['timestamp']['sec']
        assert data1['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['metadata']['source'] == 'FXE_DET_LPD1M-1/DET/0CH0:xtdf'


@agipd_run
def test_run():
    test_run = RunDirectory(RUNPATH)

    train = test_run.trains()

    first_train = next(train)
    next(train)
    third_train = next(train)

    train_id = first_train[0]
    train_data = first_train[1]

    assert train_id == 1472806005
    for i in range(11):
        key = 'SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf'.format(i)
        assert key in train_data.keys()

    train_id = third_train[0]
    train_data = third_train[1]

    assert train_id == 1472810838
    assert len(train_data) == 11


@agipd_run
def test_run_single_train_from_id():
    test_run = RunDirectory(RUNPATH)

    tid, data = test_run.train_from_id(1472810853)
    assert len(data) == 10

    tid, data = test_run.train_from_id(1472810854)
    assert len(data) == 11

    img = data['SPB_DET_AGIPD1M-1/DET/8CH0:xtdf']['image.data']
    assert img.shape == (60, 512, 128)

    with pytest.raises(KeyError) as info:
        tid, data = test_run.train_from_id(123456789)
    print(info)


@agipd_run
def test_run_single_train_from_index():
    test_run = RunDirectory(RUNPATH)

    tid, data = test_run.train_from_index(541)
    assert len(data) == 2

    tid, data = test_run.train_from_index(123)
    assert len(data) == 11

    img = data['SPB_DET_AGIPD1M-1/DET/9CH0:xtdf']['image.data']
    assert img.shape == (60, 512, 128)

    with pytest.raises(IndexError) as info:
        tid, data = test_run.train_from_index(542)
    print(info)


@agipd_run
def test_stack_data():
    test_run = RunDirectory(RUNPATH)
    tid, data = test_run.train_from_id(1472810853)

    comb = stack_data(data, 'image.data')
    print(comb.shape)
    assert comb.shape == (60, 10, 512, 128)
    np.testing.assert_equal(comb[:, 0, ...], data['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']['image.data'])


@agipd_run
def test_stack_data_2():
    test_run = RunDirectory(RUNPATH)
    tid, data = test_run.train_from_id(1472810853)

    skip = ['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'SPB_DET_AGIPD1M-1/DET/9CH0:xtdf']
    comb = stack_data(data, 'image.data', axis=0, xcept=skip)
    print(comb.shape)
    assert comb.shape == (8, 60, 512, 128)
    np.testing.assert_equal(comb[0, ...], data['SPB_DET_AGIPD1M-1/DET/1CH0:xtdf']['image.data'])


@agipd_run
def test_stack_detector_data():
    test_run = RunDirectory(RUNPATH)
    tid, data = test_run.train_from_id(1472810853)

    comb = stack_detector_data(data, 'image.data', only='AGIPD1M-1')
    print(comb.shape)
    assert comb.shape == (60, 16, 512, 128)
    np.testing.assert_equal(comb[:, 0, ...], data['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']['image.data'])


@agipd_run
def test_stack_detector_data_2():
    test_run = RunDirectory(RUNPATH)
    tid, data = test_run.train_from_id(1472810853)

    skip = ['SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf'.format(i) for i in range(3,8)]
    comb = stack_detector_data(data, 'image.data', axis=0, only='AGIPD1M-1', xcept=skip)
    print(comb.shape)
    assert comb.shape == (16, 60, 512, 128)
    for i in (0, 1, 2, 8, 9, 10):
        np.testing.assert_equal(comb[i, ...], data['SPB_DET_AGIPD1M-1/DET/{}CH0:xtdf'.format(i)]['image.data'])
    for i in (3, 4, 5, 6, 7, 11, 12, 13, 14, 15):
        np.testing.assert_equal(comb[i, ...], np.full((60, 512, 128), np.nan, dtype=np.float32))


@xgm_run
def test_filter_device():

    dev_filter_1 = {'SPB_XTD9_XGM/XGM/DOOCS': {'pulseEnergy.pulseEnergy.value',
                                               'current.right.output.value'}}
    dev_filter_2 = {''}
    dev_filter_3 = {'SPB_XTD9_XGM/XGM/DOOCS': {},
                    'SA1_XTD2_XGM/XGM/DOOCS': {'pulseEnergy.pulseEnergy.value'}}

    with H5File(RUNPATH_SLOW + '/RAW-R0115-DA01-S00000.h5') as f:
        _, data = f.train_from_index(500, devices=dev_filter_1)

        assert len(data) == 1
        assert 'SPB_XTD9_XGM/XGM/DOOCS' in data
        xgm = data['SPB_XTD9_XGM/XGM/DOOCS']
        assert len(xgm) == 3  # metadata, pulseEnergy, current
        assert 'pulseEnergy.pulseEnergy.value' in xgm
        # assert xgm['pulseEnergy.pulseEnergy.value'] == approx(0.06392462, rel=1e-7)
        # assert xgm['current.right.output.value'] == approx(-7.968561e-15, rel=1e-6)

        _, data = f.train_from_index(0, devices=dev_filter_2)
        assert len(data) == 0

        _, data = f.train_from_index(0, devices=dev_filter_3)
        assert len(data) == 2
        assert len(data['SPB_XTD9_XGM/XGM/DOOCS']) == 77
        assert len(data['SA1_XTD2_XGM/XGM/DOOCS']) == 2


@agipd_run
def test_run_info(capsys):
    test_run = RunDirectory(RUNPATH)
    test_run.info()
    captured = capsys.readouterr()
    print(captured)
    assert captured[0] == (
        'Run information\n'
        '\tDuration:       0:08:57.200000\n'
        '\tFirst train ID: 1472806005\n'
        '\tLast train ID:  1472811377\n'
        '\t# of trains:    542\n'
        '\n'
        'Devices\n'
        '\tInstruments\n'
        '\t- SPB_DET_AGIPD1M-1/DET/0CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/10CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/1CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/2CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/3CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/4CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/5CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/6CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/7CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/8CH0:xtdf\n'
        '\t- SPB_DET_AGIPD1M-1/DET/9CH0:xtdf\n'
        '\tControls\n'
        '\t-\n')


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
