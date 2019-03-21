import pytest

from karabo_data.reader import RunDirectory, by_id, by_index
from karabo_data.components import LPD1M


def test_get_array(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    assert det.detector_name == 'FXE_DET_LPD1M-1'

    arr = det.get_array('image.data')
    assert arr.shape == (16, 3, 128, 256, 256)
    assert arr.dims == ('module', 'train', 'pulse', 'slow_scan', 'fast_scan')


def test_get_array_pulse_id(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
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


def test_get_array_pulse_indexes(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
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


def test_iterate(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    det = LPD1M(run.select_trains(by_index[:2]))
    it = iter(det.trains())
    tid, d = next(it)
    assert d['image.data'].shape == (16, 1, 128, 256, 256)
    assert d['image.data'].dims == ('module', 'train', 'pulse', 'slow_scan', 'fast_scan')

    tid, d = next(it)
    assert d['image.data'].shape == (16, 1, 128, 256, 256)

    with pytest.raises(StopIteration):
        next(it)


def test_iterate_pulse_id(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
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


def test_iterate_pulse_index(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
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
