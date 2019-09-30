import h5py
import mock
import numpy as np
import pytest

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


def test_save_load_map(mock_fxe_raw_run, tmp_path):
    run_map_path = str(tmp_path / 'kd_test_run_map.json')

    class TestRunFilesMap(run_files_map.RunFilesMap):
        def map_paths_for_run(self, directory):
            return [run_map_path]

    rfm = TestRunFilesMap(mock_fxe_raw_run)
    assert rfm.files_data == {}

    with RunDirectory(mock_fxe_raw_run) as run:
        rfm.save(run.files)
        filename = run.files[0].filename

    rfm2 = TestRunFilesMap(mock_fxe_raw_run)
    assert rfm2.cache_file == run_map_path
    file_info = rfm2.get(filename)

    assert isinstance(file_info['train_ids'], np.ndarray)
    assert isinstance(file_info['control_sources'], frozenset)
    assert isinstance(file_info['instrument_sources'], frozenset)

    # Modify a file; this should make the cache invalid
    with h5py.File(filename, 'r+') as f:
        f.attrs['test_save_load_map'] = 1

    rfm3 = TestRunFilesMap(mock_fxe_raw_run)
    assert rfm3.cache_file == run_map_path
    assert rfm3.get(filename) is None
