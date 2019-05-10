import os
import os.path as osp
from unittest import mock

from karabo_data import read_machinery

def test_find_proposal(tmpdir):
    prop_dir = osp.join(str(tmpdir), 'SPB', '201701', 'p002012')
    os.makedirs(prop_dir)

    with mock.patch.object(read_machinery, 'DATA_ROOT_DIR', str(tmpdir)):
        assert read_machinery.find_proposal('p002012') == prop_dir

        assert read_machinery.find_proposal(prop_dir) == prop_dir
