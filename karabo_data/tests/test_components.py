import h5py
import numpy as np
import os.path as osp
import pytest
from testpath import assert_isfile

from karabo_data.reader import RunDirectory, by_id, by_index
from karabo_data.components import AGIPD1M, LPD1M


def test_get_array(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    assert det.detector_name == 'FXE_DET_LPD1M-1'

    arr = det.get_array('image.data')
    assert arr.shape == (16, 3, 128, 256, 256)
    assert arr.dims == ('module', 'train', 'pulse', 'slow_scan', 'fast_scan')


def test_get_array_pulse_id(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    arr = det.get_array('image.data', pulses=by_id[0])
    assert arr.shape == (16, 3, 1, 256, 256)
    assert (arr.coords['pulse'] == 0).all()

    arr = det.get_array('image.data', pulses=by_id[:5])
    assert arr.shape == (16, 3, 5, 256, 256)

    # Empty selection
    arr = det.get_array('image.data', pulses=by_id[:0])
    assert arr.shape == (16, 0, 0, 256, 256)

    arr = det.get_array('image.data', pulses=by_id[122:])
    assert arr.shape == (16, 3, 6, 256, 256)

    arr = det.get_array('image.data', pulses=by_id[[1, 7, 22, 23]])
    assert arr.shape == (16, 3, 4, 256, 256)
    assert list(arr.coords['pulse']) == [1, 7, 22, 23]


def test_get_array_pulse_indexes(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    arr = det.get_array('image.data', pulses=by_index[0])
    assert arr.shape == (16, 3, 1, 256, 256)
    assert (arr.coords['pulse'] == 0).all()

    arr = det.get_array('image.data', pulses=by_index[:5])
    assert arr.shape == (16, 3, 5, 256, 256)

    # Empty selection
    arr = det.get_array('image.data', pulses=by_index[:0])
    assert arr.shape == (16, 0, 0, 256, 256)

    arr = det.get_array('image.data', pulses=by_index[122:])
    assert arr.shape == (16, 3, 6, 256, 256)

    arr = det.get_array('image.data', pulses=by_index[[1, 7, 22, 23]])
    assert arr.shape == (16, 3, 4, 256, 256)


def test_iterate(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:2]))
    it = iter(det.trains())
    tid, d = next(it)
    assert d['image.data'].shape == (16, 1, 128, 256, 256)
    assert d['image.data'].dims == ('module', 'train', 'pulse', 'slow_scan', 'fast_scan')

    tid, d = next(it)
    assert d['image.data'].shape == (16, 1, 128, 256, 256)

    with pytest.raises(StopIteration):
        next(it)


def test_iterate_pulse_id(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    tid, d = next(iter(det.trains(pulses=by_id[0])))
    assert d['image.data'].shape == (16, 1, 1, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_id[:5])))
    assert d['image.data'].shape == (16, 1, 5, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_id[122:])))
    assert d['image.data'].shape == (16, 1, 6, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_id[[1, 7, 22, 23]])))
    assert d['image.data'].shape == (16, 1, 4, 256, 256)
    assert list(d['image.data'].coords['pulse']) == [1, 7, 22, 23]


def test_iterate_pulse_index(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    tid, d = next(iter(det.trains(pulses=by_index[0])))
    assert d['image.data'].shape == (16, 1, 1, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_index[:5])))
    assert d['image.data'].shape == (16, 1, 5, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_index[122:])))
    assert d['image.data'].shape == (16, 1, 6, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_index[[1, 7, 22, 23]])))
    assert d['image.data'].shape == (16, 1, 4, 256, 256)
    assert list(d['image.data'].coords['pulse']) == [1, 7, 22, 23]

def test_write_virtual_cxi(mock_spb_proc_run, tmpdir):
    run = RunDirectory(mock_spb_proc_run)
    det = AGIPD1M(run)

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)

    with h5py.File(test_file) as f:
        det_grp = f['entry_1/instrument_1/detector_1']
        ds = det_grp['data']
        assert isinstance(ds, h5py.Dataset)
        assert ds.is_virtual
        assert ds.shape[1:] == (16, 512, 128)
        assert 'axes' in ds.attrs

        assert len(ds.virtual_sources()) == 16

        # Check position of each source file in the modules dimension
        for src in ds.virtual_sources():
            start, _, block, count = src.vspace.get_regular_hyperslab()
            assert block[1] == 1
            assert count[1] == 1

            expected_file = 'CORR-R0238-AGIPD{:0>2}-S00000.h5'.format(start[1])
            assert osp.basename(src.file_name) == expected_file

        # Check presence of other datasets
        assert 'gain' in det_grp
        assert 'mask' in det_grp
        assert 'experiment_identifier' in det_grp

def test_write_virtual_cxi_some_modules(mock_spb_proc_run, tmpdir):
    run = RunDirectory(mock_spb_proc_run)
    det = AGIPD1M(run, modules=[3, 4, 8, 15])

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)

    with h5py.File(test_file) as f:
        det_grp = f['entry_1/instrument_1/detector_1']
        ds = det_grp['data']
        assert ds.shape[1:] == (16, 512, 128)

def test_write_virtual_cxi_raw_data(mock_fxe_raw_run, tmpdir, caplog):
    import logging
    caplog.set_level(logging.INFO)
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run)

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)
