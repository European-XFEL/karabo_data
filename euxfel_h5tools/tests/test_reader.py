import os.path as osp
import pytest
from time import sleep

from euxfel_h5tools import H5File, RunHandler

FILEPATH = './R0019-lpd00-S00000.h5'
RUNPATH = './r0185'
require_data = pytest.mark.skipif(not osp.exists(FILEPATH),
                                  reason="require data file.")
require_data2 = pytest.mark.skipif(not osp.exists(RUNPATH),
                                   reason="require data files.")


@pytest.yield_fixture
@pytest.fixture(scope="session")
def file_opened():
    """open and close hdf5 file"""

    print("opening file: ", FILEPATH)
    h5f = H5File(FILEPATH)

    yield h5f

    print("closing file...")
    h5f.close()


@require_data
def test_get_file_infos(file_opened):
    f = file_opened

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


@require_data
def test_iterate_trains(file_opened):
    f = file_opened

    for data, train_id, index in f.trains():
        print(index, train_id)
        assert index in range(0,20)
        assert train_id in range(1455918683, 1455918703)
        assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in data.keys()
        assert len(data) == 1
        assert len(data['FXE_DET_LPD1M-1/DET/0CH0:xtdf']) == 21


@require_data
def test_get_train_per_id(file_opened):
    f = file_opened

    data, train, idx = f.train_from_id(1455918700)
    print(data)
    assert data is not None
    assert train == 1455918700
    assert idx == 17

    with pytest.raises(ValueError) as info:
        data = f.train_from_id(1234)  # train id not in file


@require_data
def test_get_train_per_index(file_opened):
    f = file_opened

    data, train, idx = f.train_from_index(0)
    assert data is not None
    assert train == 1455918683
    assert idx == 0

    with pytest.raises(ValueError) as info:
        data, _, _ = f.train_from_index(20)  # index out of range


@require_data
def test_read_metadata(file_opened):
    f = file_opened

    data1, _, _ = f.train_from_index(0)
    sleep(2)
    data2, _, _ = f.train_from_index(0)
    assert 'metadata' in data1['FXE_DET_LPD1M-1/DET/0CH0:xtdf']
    assert data1['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['metadata']['tid'] == 1455918683
    assert data1['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['metadata']['sec'] != data2['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['metadata']['sec']
    assert data1['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['metadata']['source'] == 'FXE_DET_LPD1M-1/DET/0CH0:xtdf'


@require_data2
def test_run():
    test_run = RunHandler(RUNPATH)

    train = test_run.trains()

    first_train = next(train)
    second_train = next(train)
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


@require_data2
def test_run_single():
    test_run = RunHandler(RUNPATH)

    tid, data = test_run.train_from_id(1472810853)
    assert len(data) == 10

    tid, data = test_run.train_from_id(1472810854)
    assert len(data) == 11

    img = data['SPB_DET_AGIPD1M-1/DET/8CH0:xtdf']['image.data']
    assert img.shape == (60, 512, 128)


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
