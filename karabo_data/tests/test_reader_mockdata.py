from itertools import islice
import os.path as osp
import pandas as pd
import pytest
from tempfile import TemporaryDirectory
from xarray import DataArray

from karabo_data import (H5File, RunDirectory, stack_data, stack_detector_data)
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
def mock_spb_control_data_badname():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0309-DA01-S00000.h5')
        make_examples.make_data_file_bad_device_name(path)
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

def test_get_train_bad_device_name(mock_spb_control_data_badname):
    # Check that we can handle devices which don't have the standard Karabo
    # name structure A/B/C.
    with H5File(mock_spb_control_data_badname) as f:
        train_id, data = f.train_from_id(10004)
        assert train_id == 10004
        assert 'SPB_IRU_SIDEMIC_CAM:daqOutput' in data
        assert 'data.image.dims' in data['SPB_IRU_SIDEMIC_CAM:daqOutput']
        dims = data['SPB_IRU_SIDEMIC_CAM:daqOutput']['data.image.dims']
        assert list(dims) == [1000, 1000]

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


def test_iterate_trains_select_keys(mock_fxe_control_data):
    sel = {
        'SA1_XTD2_XGM/DOOCS/MAIN': {
            'beamPosition.ixPos.value',
            'beamPosition.ixPos.timestamp',
        }
    }

    with H5File(mock_fxe_control_data) as f:
        for train_id, data in islice(f.trains(devices=sel), 10):
            assert train_id in range(10000, 10400)
            assert 'SA1_XTD2_XGM/DOOCS/MAIN' in data.keys()
            assert 'beamPosition.ixPos.value' in data['SA1_XTD2_XGM/DOOCS/MAIN']
            assert 'beamPosition.ixPos.timestamp' in data['SA1_XTD2_XGM/DOOCS/MAIN']
            assert 'beamPosition.iyPos.value' not in data['SA1_XTD2_XGM/DOOCS/MAIN']
            assert 'SA3_XTD10_VAC/TSENS/S30160K' not in data

def test_read_fxe_run(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    assert len(run.files) == 18  # 16 detector modules + 2 control data files
    assert [tid for tid, _ in run.ordered_trains] == list(range(10000, 10480))
    run.info()  # Smoke test

def test_properties_fxe_run(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)

    assert run.train_ids == list(range(10000, 10480))
    assert 'SPB_XTD9_XGM/DOOCS/MAIN' in run.control_sources
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in run.instrument_sources

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

def test_file_get_series_control(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        s = f.get_series('SA1_XTD2_XGM/DOOCS/MAIN', "beamPosition.iyPos.value")
        assert isinstance(s, pd.Series)
        assert len(s) == 400
        assert s.index[0] == 10000

def test_file_get_series_instrument(mock_agipd_data):
    with H5File(mock_agipd_data) as f:
        s = f.get_series('SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'header.linkId')
        assert isinstance(s, pd.Series)
        assert len(s) == 250
        assert s.index[0] == 10000

        # Multiple readings per train
        s2 = f.get_series('SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'image.status')
        assert isinstance(s2, pd.Series)
        assert isinstance(s2.index, pd.MultiIndex)
        assert len(s2) == 16000
        assert len(s2.loc[10000:10004]) == 5 * 64

def test_run_get_series_control(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    s = run.get_series('SA1_XTD2_XGM/DOOCS/MAIN', "beamPosition.iyPos.value")
    assert isinstance(s, pd.Series)
    assert len(s) == 480
    assert list(s.index) == list(range(10000, 10480))

def test_run_get_dataframe(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    df = run.get_dataframe(fields=[("*_XGM/*", "*.i[xy]Pos*")])
    assert len(df.columns) == 4
    assert "SA1_XTD2_XGM/DOOCS/MAIN/beamPosition.ixPos" in df.columns

    df2 = run.get_dataframe(fields=[("*_XGM/*", "*.i[xy]Pos*")], timestamps=True)
    assert len(df2.columns) == 8
    assert "SA1_XTD2_XGM/DOOCS/MAIN/beamPosition.ixPos" in df2.columns
    assert "SA1_XTD2_XGM/DOOCS/MAIN/beamPosition.ixPos.timestamp" in df2.columns

def test_file_get_array(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        arr = f.get_array('FXE_XAD_GEC/CAM/CAMERA:daqOutput', 'data.image.pixels')

    assert isinstance(arr, DataArray)
    assert arr.dims == ('trainId', 'dim_0', 'dim_1')
    assert arr.shape == (400, 255, 1024)
    assert arr.coords['trainId'][0] == 10000

def test_run_get_array(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    arr = run.get_array('SA1_XTD2_XGM/DOOCS/MAIN:output', 'data.intensityTD',
                        extra_dims=['pulse'])

    assert isinstance(arr, DataArray)
    assert arr.dims == ('trainId', 'pulse')
    assert arr.shape == (480, 1000)
    assert arr.coords['trainId'][0] == 10000
