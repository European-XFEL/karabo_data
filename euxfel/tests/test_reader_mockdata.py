import os.path as osp
import pytest
from tempfile import TemporaryDirectory

from euxfel import (H5File, RunHandler, stack_data, stack_detector_data)
from . import make_examples

@pytest.fixture(scope='module')
def mock_agipd_data():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'CORR-R9999-AGIPD07-S00000.h5')
        make_examples.make_agipd_example_file(path)
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
        for data, train_id, index in f.trains():
            assert index in range(0, 250)
            assert train_id in range(10000, 10250)
            assert 'SPB_DET_AGIPD1M-1/DET/7CH0:xtdf' in data.keys()
            assert len(data) == 1
            assert 'image.data' in data['SPB_DET_AGIPD1M-1/DET/7CH0:xtdf']

def test_iterate_trains_fxe(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        for data, train_id, index in f.trains():
            assert index in range(0, 400)
            assert train_id in range(10000, 10400)
            assert 'SA1_XTD2_XGM/DOOCS/MAIN' in data.keys()
            assert 'beamPosition.ixPos.value' in data['SA1_XTD2_XGM/DOOCS/MAIN']

def test_read_fxe_run(mock_fxe_run):
    run = RunHandler(mock_fxe_run)
    assert len(run.files) == 17  # 16 detector modules + 1 control data file
    assert [tid for tid, _ in run.ordered_trains] == list(range(10000, 10480))
    run.info()  # Smoke test
