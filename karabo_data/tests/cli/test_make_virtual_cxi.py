import os
import os.path as osp
from testpath import assert_isfile

from karabo_data.cli.make_virtual_cxi import main

def test_make_virtual_cxi(mock_spb_proc_run, tmpdir):
    output = osp.join(str(tmpdir), 'test.cxi')
    main([mock_spb_proc_run, '-o', output])
    assert_isfile(output)

def test_make_virtual_cxi_runno(mock_spb_proc_run, tmpdir):
    proc = osp.join(str(tmpdir), 'proc')
    os.mkdir(proc)
    os.symlink(mock_spb_proc_run, osp.join(proc, 'r0238'))
    output = osp.join(str(tmpdir), 'test.cxi')

    # Pass proposal directory and run number
    main([str(tmpdir), '238', '-o', output])
    assert_isfile(output)
