import h5py
import os.path as osp
import numpy as np
from tempfile import TemporaryDirectory
from testpath import assert_isfile

from karabo_data import RunDirectory, H5File


def test_write_selected(mock_fxe_raw_run):
    with TemporaryDirectory() as td:
        new_file = osp.join(td, 'test.h5')

        with RunDirectory(mock_fxe_raw_run) as run:
            run.select('SPB_XTD9_XGM/*').write(new_file)

        assert_isfile(new_file)

        with H5File(new_file) as f:
            assert f.control_sources == {'SPB_XTD9_XGM/DOOCS/MAIN'}
            assert f.instrument_sources == {'SPB_XTD9_XGM/DOOCS/MAIN:output'}

            s = f.get_series('SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value')
            # This should have concatenated the two sequence files (400 + 80)
            assert len(s) == 480

            a = f.get_array('SPB_XTD9_XGM/DOOCS/MAIN:output', 'data.intensityTD')
            assert a.shape == (480, 1000)

def test_write_virtual(mock_fxe_raw_run):
    with TemporaryDirectory() as td:
        new_file = osp.join(td, 'test.h5')

        with RunDirectory(mock_fxe_raw_run) as run:
            run.write_virtual(new_file)

        assert_isfile(new_file)

        with h5py.File(new_file) as f:
            ds = f['CONTROL/SPB_XTD9_XGM/DOOCS/MAIN/beamPosition/ixPos/value']
            assert ds.is_virtual

        with H5File(new_file) as f:
            np.testing.assert_array_equal(f.train_ids,
                                      np.arange(10000, 10480, dtype=np.uint64))

            assert 'SPB_XTD9_XGM/DOOCS/MAIN' in f.control_sources
            assert 'SPB_XTD9_XGM/DOOCS/MAIN:output' in f.instrument_sources

            s = f.get_series('SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value')
            # This should have concatenated the two sequence files (400 + 80)
            assert len(s) == 480

            a = f.get_array('SPB_XTD9_XGM/DOOCS/MAIN:output', 'data.intensityTD')
            assert a.shape == (480, 1000)
