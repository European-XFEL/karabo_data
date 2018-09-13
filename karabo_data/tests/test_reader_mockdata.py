from itertools import islice
import pandas as pd
import pytest
from xarray import DataArray

from karabo_data import (
    H5File, RunDirectory, stack_data, stack_detector_data, by_index, by_id,
    SourceNameError, PropertyNameError, DataCollection,
)


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
        di = f.detector_info('SPB_DET_AGIPD1M-1/DET/7CH0:xtdf')
        assert di['dims'] == (512, 128)
        assert di['frames_per_train'] == 64
        assert di['total_frames'] == 16000

def test_detector_info(mock_lpd_data):
    with H5File(mock_lpd_data) as f:
        di = f.detector_info('FXE_DET_LPD1M-1/DET/0CH0:xtdf')
        assert di['dims'] == (256, 256)
        assert di['frames_per_train'] == 128
        assert di['total_frames'] == 128 * 480

def test_iterate_trains_fxe(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        for train_id, data in islice(f.trains(), 10):
            assert train_id in range(10000, 10400)
            assert 'SA1_XTD2_XGM/DOOCS/MAIN' in data.keys()
            assert 'beamPosition.ixPos.value' in data['SA1_XTD2_XGM/DOOCS/MAIN']


def test_iterate_file_select_trains(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        tids = [tid for (tid, _) in f.trains(train_range=by_id[:10003])]
        assert tids == [10000, 10001, 10002]

        tids = [tid for (tid, _) in f.trains(train_range=by_index[-2:])]
        assert tids == [10398, 10399]

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

def test_iterate_trains_require_all(mock_sa3_control_data):
    with H5File(mock_sa3_control_data) as f:
        trains_iter = f.trains(devices=[('*/CAM/BEAMVIEW:daqOutput', 'data.image.dims')], require_all=True)
        tids = [t for (t, _) in trains_iter]
        assert tids == []
        trains_iter = f.trains(devices=[('*/CAM/BEAMVIEW:daqOutput', 'data.image.dims')], require_all=False)
        tids = [t for (t, _) in trains_iter]
        assert tids != []

def test_read_fxe_run(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    assert len(run.files) == 18  # 16 detector modules + 2 control data files
    assert run.train_ids == list(range(10000, 10480))
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

def test_iterate_select_trains(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)

    tids = [tid for (tid, _) in run.trains(train_range=by_id[10004:10006])]
    assert tids == [10004, 10005]

    tids = [tid for (tid, _) in run.trains(train_range=by_id[:10003])]
    assert tids == [10000, 10001, 10002]

    # Overlap with start of run
    tids = [tid for (tid, _) in run.trains(train_range=by_id[9000:10003])]
    assert tids == [10000, 10001, 10002]

    # Overlap with end of run
    tids = [tid for (tid, _) in run.trains(train_range=by_id[10478: 10500])]
    assert tids == [10478, 10479]

    # Not overlapping
    with pytest.raises(ValueError) as excinfo:
        list(run.trains(train_range=by_id[9000:9050]))
    assert 'before' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        list(run.trains(train_range=by_id[10500:10550]))
    assert 'after' in str(excinfo.value)

    tids = [tid for (tid, _) in run.trains(train_range=by_index[4:6])]
    assert tids == [10004, 10005]

def test_iterate_run_glob_devices(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    trains_iter = run.trains([("*/DET/*", "image.data")])
    tid, data = next(trains_iter)
    assert tid == 10000
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'detector.data' not in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' not in data

def test_train_by_id_fxe_run(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    _, data = run.train_from_id(10024)
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' in data
    assert 'firmwareVersion.value' in data['FXE_XAD_GEC/CAM/CAMERA']

def test_train_by_id_fxe_run_selection(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    _, data = run.train_from_id(10024, [('*/DET/*', 'image.data')])
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' not in data

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

def test_run_get_array_empty(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    arr = run.get_array('FXE_XAD_GEC/CAM/CAMERA_NODATA:daqOutput', 'data.image.pixels')

    assert isinstance(arr, DataArray)
    assert arr.dims[0] == 'trainId'
    assert arr.shape == (0, 255, 1024)

def test_run_get_array_error(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)

    with pytest.raises(SourceNameError):
        run.get_array('bad_name', 'data.intensityTD')

    with pytest.raises(PropertyNameError):
        run.get_array('SA1_XTD2_XGM/DOOCS/MAIN:output', 'bad_name')

def test_select(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)

    assert 'SPB_XTD9_XGM/DOOCS/MAIN' in run.control_sources

    sel = run.select('*/DET/*', 'image.pulseId')
    assert 'SPB_XTD9_XGM/DOOCS/MAIN' not in sel.control_sources
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in sel.instrument_sources
    _, data = sel.train_from_id(10000)
    for source, source_data in data.items():
        print(source)
        assert set(source_data.keys()) == {'image.pulseId', 'metadata'}

def test_deselect(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)

    xtd9_xgm = 'SPB_XTD9_XGM/DOOCS/MAIN'
    assert xtd9_xgm in run.control_sources

    sel = run.deselect('*_XGM/DOOCS*')
    assert xtd9_xgm not in sel.control_sources
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in sel.instrument_sources

    sel = run.deselect('*_XGM/DOOCS*', '*.ixPos')
    assert xtd9_xgm in sel.control_sources
    assert 'beamPosition.ixPos.value' not in sel.selection[xtd9_xgm]
    assert 'beamPosition.iyPos.value' in sel.selection[xtd9_xgm]

def test_select_trains(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)

    assert len(run.train_ids) == 480

    sel = run.select_trains(by_id[10200:10220])
    assert sel.train_ids == list(range(10200, 10220))

    sel = run.select_trains(by_index[:10])
    assert sel.train_ids == list(range(10000, 10010))

    with pytest.raises(ValueError):
        run.select_trains(by_id[9000:9100])  # Before data

    with pytest.raises(ValueError):
        run.select_trains(by_id[12000:12500])  # After data

def test_union(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)

    sel1 = run.select('SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos')
    sel2 = run.select('SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.iyPos')
    joined = sel1.union(sel2)
    assert joined.control_sources == {'SPB_XTD9_XGM/DOOCS/MAIN'}
    assert joined.selection == \
       {'SPB_XTD9_XGM/DOOCS/MAIN': {'beamPosition.ixPos.value', 'beamPosition.iyPos.value'}}

    sel1 = run.select_trains(by_id[10200:10220])
    sel2 = run.select_trains(by_index[:10])
    joined = sel1.union(sel2)
    assert joined.train_ids == list(range(10000, 10010)) + list(range(10200, 10220))

def test_read_skip_invalid(mock_lpd_data, empty_h5_file, capsys):
    d = DataCollection.from_paths([mock_lpd_data, empty_h5_file])
    assert d.instrument_sources == {'FXE_DET_LPD1M-1/DET/0CH0:xtdf'}
    out, err = capsys.readouterr()
    assert "Skipping file" in err

def test_stack_data(mock_fxe_run):
    test_run = RunDirectory(mock_fxe_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    comb = stack_data(data, 'image.data')
    assert comb.shape == (128, 1, 16, 256, 256)

def test_stack_detector_data(mock_fxe_run):
    test_run = RunDirectory(mock_fxe_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    comb = stack_detector_data(data, 'image.data')
    assert comb.shape == (128, 1, 16, 256, 256)
