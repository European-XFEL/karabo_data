import h5py
import mock
import os
import numpy as np
import pytest

from .mockdata import write_file
from .mockdata.xgm import XGM
from karabo_data import run_files_map, RunDirectory

def test_candidate_paths(tmp_path):
    # 'real' paths (like /gpfs/exfel/d)
    prop_raw_path = tmp_path / 'raw' / 'FXE' / '201901' / 'p001234'
    run_dir = prop_raw_path / 'r0450'
    run_dir.mkdir(parents=True)

    # stable paths (like /gpfs/exfel/exp)
    exp = tmp_path / 'exp'
    prop_dir = exp / 'FXE' / '201901' / 'p001234'
    prop_scratch = exp / 'FXE' / '201901' / 'p001234' / 'scratch'
    prop_scratch.mkdir(parents=True)
    (prop_dir / 'raw').symlink_to(prop_raw_path)
    run_in_exp = prop_dir / 'raw' / 'r0450'

    with mock.patch.object(run_files_map, 'SCRATCH_ROOT_DIR', str(exp)):
        rfm = run_files_map.RunFilesMap(str(run_dir))
        rfm_exp = run_files_map.RunFilesMap(str(run_in_exp))

    assert rfm.candidate_paths == [
        str(run_dir / 'karabo_data_map.json'),
        str(prop_scratch / '.karabo_data_maps' / 'raw_r0450.json'),
    ]
    assert rfm_exp.candidate_paths == [
        str(run_in_exp / 'karabo_data_map.json'),
        str(prop_scratch / '.karabo_data_maps' / 'raw_r0450.json'),
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
            return [run_map_path]

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
