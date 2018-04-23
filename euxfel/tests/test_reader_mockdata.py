from itertools import islice
import os.path as osp
import pytest
from tempfile import TemporaryDirectory

from euxfel import (H5File, RunDirectory, stack_data, stack_detector_data)
from . import make_examples

@pytest.fixture(scope='module')
def mock_agipd_data():
    # This one uses the older index format
    # (first/last/status instead of first/count)
    with TemporaryDirectory() as td:
        path = osp.join(td, 'CORR-R9999-AGIPD07-S00000.h5')
        make_examples.make_agipd_example_file(path)
        yield path

@pytest.fixture(scope='module')
def mock_lpd_data():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R9999-AGIPD00-S00000.h5')
        make_examples.make_lpd_file(path)
        yield path

@pytest.fixture(scope='module')
def mock_fxe_control_data():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00001.h5')
        make_examples.make_fxe_da_file(path)
        yield path

@pytest.fixture(scope='module')
def mock_fxe_run():
    with TemporaryDirectory() as td:
        make_examples.make_fxe_run(td)
        yield td

def test_iterate_trains(mock_agipd_data):
    with H5File(mock_agipd_data) as f:
        for train_id, data in islice(f.trains(), 10):
            assert train_id in range(10000, 10250)
            assert 'SPB_DET_AGIPD1M-1/DET/7CH0:xtdf' in data.keys()
            assert len(data) == 1
            assert 'image.data' in data['SPB_DET_AGIPD1M-1/DET/7CH0:xtdf']

def test_detector_info_oldfmt(mock_agipd_data):
    with H5File(mock_agipd_data) as f:
        di = f.detector_info()
        assert di['dims'] == (512, 128)
        assert di['frames_per_train'] == 64
        assert di['total_frames'] == 16000

def test_detector_info(mock_lpd_data):
    with H5File(mock_lpd_data) as f:
        di = f.detector_info()
        assert di['dims'] == (256, 256)
        assert di['frames_per_train'] == 128
        assert di['total_frames'] == 128 * 480

def test_iterate_trains_fxe(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        for train_id, data in islice(f.trains(), 10):
            assert train_id in range(10000, 10400)
            assert 'SA1_XTD2_XGM/DOOCS/MAIN' in data.keys()
            assert 'beamPosition.ixPos.value' in data['SA1_XTD2_XGM/DOOCS/MAIN']

def test_read_fxe_run(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    assert len(run.files) == 17  # 16 detector modules + 1 control data file
    assert [tid for tid, _ in run.ordered_trains] == list(range(10000, 10480))
    run.info()  # Smoke test

def test_iterate_fxe_run(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)

    trains_iter = run.trains()
    tid, data = next(trains_iter)
    assert tid == 10000
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' in data
    assert 'firmwareVersion.value' in data['FXE_XAD_GEC/CAM/CAMERA']

def test_train_by_id_fxe_run(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    _, data = run.train_from_id(10024)
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' in data
    assert 'firmwareVersion.value' in data['FXE_XAD_GEC/CAM/CAMERA']

def test_train_from_index_fxe_run(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    _, data = run.train_from_index(479)
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' in data
    assert 'firmwareVersion.value' in data['FXE_XAD_GEC/CAM/CAMERA']
