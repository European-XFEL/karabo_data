from karabo_data.reader import RunDirectory, by_id, by_index

def test_get_array(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    det = run.select_trains(by_index[:3]).detector()
    arr = det.get_array('image.data')
    assert arr.shape == (16, 3, 128, 256, 256)

def test_get_array_pulse_id(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    det = run.select_trains(by_index[:3]).detector()
    arr = det.get_array('image.data', pulses=by_id[0])
    assert arr.shape == (16, 3, 1, 256, 256)
    assert (arr.coords['pulseId'] == 0).all()

    arr = det.get_array('image.data', pulses=by_id[:5])
    assert arr.shape == (16, 3, 5, 256, 256)

    arr = det.get_array('image.data', pulses=by_id[[1, 7, 22, 23]])
    assert arr.shape == (16, 3, 4, 256, 256)
    assert list(arr.coords['pulseId'][:4]) == [1, 7, 22, 23]


def test_get_array_pulse_indexes(mock_fxe_run):
    run = RunDirectory(mock_fxe_run)
    det = run.select_trains(by_index[:3]).detector()
    arr = det.get_array('image.data', pulses=by_index[0])
    assert arr.shape == (16, 3, 1, 256, 256)
    assert (arr.coords['pulseId'] == 0).all()

    arr = det.get_array('image.data', pulses=by_index[:5])
    assert arr.shape == (16, 3, 5, 256, 256)

    arr = det.get_array('image.data', pulses=by_index[[1, 7, 22, 23]])
    assert arr.shape == (16, 3, 4, 256, 256)
