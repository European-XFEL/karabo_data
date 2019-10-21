import h5py
import mock
import os
import numpy as np
import pytest

from .mockdata import write_file
from .mockdata.xgm import XGM
from karabo_data import run_files_map, RunDirectory

def test_candidate_paths(mock_fxe_raw_run, tmp_path):
    prop_path = tmp_path / 'FXE' / '201901' / 'p001234'
    run_path = prop_path / 'raw' / 'r0450'
    run_path.parent.mkdir(parents=True)
    run_path.symlink_to(mock_fxe_raw_run)

    with mock.patch.object(run_files_map, 'DATA_ROOT_DIR', str(tmp_path)):
        rfm = run_files_map.RunFilesMap(str(run_path))

    assert rfm.candidate_paths == [
        str(run_path / 'karabo_data_map.json'),
        str(prop_path / 'scratch' / '.karabo_data_maps' / 'raw_r0450.json'),
    ]


@pytest.fixture()
def run_with_extra_file(mock_fxe_raw_run):
    extra_file = os.path.join(mock_fxe_raw_run, 'RAW-R0450-DA02-S00000.h5')
    write_file(extra_file, [
        XGM('FXE_TEST_XGM/DOOCS/MAIN'),
    ], ntrains=480)
    try:
        yield mock_fxe_raw_run, extra_file
    finally:
        os.unlink(extra_file)


def test_save_load_map(run_with_extra_file, tmp_path):
    run_dir, extra_file = run_with_extra_file
    run_map_path = str(tmp_path / 'kd_test_run_map.json')

    class TestRunFilesMap(run_files_map.RunFilesMap):
        def map_paths_for_run(self, directory):
            return directory, [run_map_path]

    rfm = TestRunFilesMap(run_dir)
    assert rfm.files_data == {}

    with RunDirectory(run_dir) as run:
        rfm.save(run.files)

    rfm2 = TestRunFilesMap(run_dir)
    assert rfm2.cache_file == run_map_path
    file_info = rfm2.get(extra_file)

    assert isinstance(file_info['train_ids'], np.ndarray)
    assert isinstance(file_info['control_sources'], frozenset)
    assert isinstance(file_info['instrument_sources'], frozenset)

    # Modify a file; this should make the cache invalid
    with h5py.File(extra_file, 'r+') as f:
        f.attrs['test_save_load_map'] = 1

    rfm3 = TestRunFilesMap(run_dir)
    assert rfm3.cache_file == run_map_path
    assert rfm3.get(extra_file) is None
