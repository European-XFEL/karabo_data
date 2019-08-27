import numpy as np
import pytest

from karabo_data import RunDirectory, stack_data, stack_detector_data

def test_stack_data(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    comb = stack_data(data, 'image.data')
    assert comb.shape == (128, 1, 16, 256, 256)


def test_stack_detector_data(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    comb = stack_detector_data(data, 'image.data')
    assert comb.shape == (128, 1, 16, 256, 256)


def test_stack_detector_data_missing(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    # Three variants of missing data:
    # 1. Source missing
    del data['FXE_DET_LPD1M-1/DET/3CH0:xtdf']
    # 2. Key missing
    del data['FXE_DET_LPD1M-1/DET/7CH0:xtdf']['image.data']
    # 3. Empty array
    missing = ['FXE_DET_LPD1M-1/DET/{}CH0:xtdf'.format(m) for m in (1, 5, 9, 15)]
    for module in missing:
        data[module]['image.data'] = np.zeros((0, 1, 256, 256), dtype=np.uint16)

    comb = stack_detector_data(data, 'image.data', fillvalue=22)
    assert comb.shape == (128, 1, 16, 256, 256)

    assert not (comb[:, :, 0] == 22).any()  # Control
    assert (comb[:, :, 3] == 22).all()  # Source missing
    assert (comb[:, :, 7] == 22).all()  # Key missing
    assert (comb[:, :, 5] == 22).all()  # Empty array


def test_stack_detector_data_stackview(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    # Three variants of missing data:
    # 1. Source missing
    del data['FXE_DET_LPD1M-1/DET/3CH0:xtdf']
    # 2. Key missing
    del data['FXE_DET_LPD1M-1/DET/7CH0:xtdf']['image.data']
    # 3. Empty array
    missing = ['FXE_DET_LPD1M-1/DET/{}CH0:xtdf'.format(m) for m in (1, 5, 9, 15)]
    for module in missing:
        data[module]['image.data'] = np.zeros((0, 1, 256, 256), dtype=np.uint16)

    comb = stack_detector_data(data, 'image.data', fillvalue=22, real_array=False)
    assert comb.shape == (128, 1, 16, 256, 256)

    assert not (comb[:, :, 0] == 22).any()  # Control
    assert (comb[:, :, 3] == 22).all()  # Source missing
    assert (comb[:, :, 7] == 22).all()  # Key missing
    assert (comb[:, :, 5] == 22).all()  # Empty array

    # Slice across all modules
    pulse = comb[0, 0]
    assert pulse.shape == (16, 256, 256)
    assert not (pulse[0] == 22).any()
    assert (pulse[3] == 22).all()
    assert (pulse[7] == 22).all()
    assert (pulse[5] == 22).all()

    pulse_arr = pulse.asarray()
    assert pulse_arr.shape == (16, 256, 256)
    assert pulse_arr.max() == 22
    assert pulse_arr.min() == 0


def test_stack_detector_data_wrong_pulses(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    misshaped = ['FXE_DET_LPD1M-1/DET/{}CH0:xtdf'.format(m) for m in (12, 13)]
    for module in misshaped:
        data[module]['image.data'] = np.zeros((64, 1, 256, 256), dtype=np.uint16)

    with pytest.raises(ValueError) as excinfo:
        comb = stack_detector_data(data, 'image.data')
    assert '(64, 1, 256, 256)' in str(excinfo.value)


def test_stack_detector_data_wrong_shape(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    misshaped = ['FXE_DET_LPD1M-1/DET/{}CH0:xtdf'.format(m) for m in (0, 15)]
    for module in misshaped:
        data[module]['image.data'] = np.zeros((128, 1, 512, 128), dtype=np.uint16)

    with pytest.raises(ValueError) as excinfo:
        comb = stack_detector_data(data, 'image.data')
    assert '(128, 1, 512, 128)' in str(excinfo.value)


def test_stack_detector_data_type_error(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    module = 'FXE_DET_LPD1M-1/DET/3CH0:xtdf'
    data[module]['image.data'] = data[module]['image.data'].astype(np.float32)

    with pytest.raises(ValueError) as excinfo:
        comb = stack_detector_data(data, 'image.data')
    assert "dtype('float32')" in str(excinfo.value)


def test_stack_detector_data_extra_mods(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    tid, data = test_run.train_from_id(10000, devices=[('*/DET/*', 'image.data')])

    data.setdefault(
        'FXE_DET_LPD1M-1/DET/16CH0:xtdf',
        {'image.data': np.zeros((128, 1, 256, 256), dtype=np.uint16)},
    )

    with pytest.raises(IndexError) as excinfo:
        comb = stack_detector_data(data, 'image.data')
    assert "16" in str(excinfo.value)
