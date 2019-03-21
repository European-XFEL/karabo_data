import numpy as np
import os
import pytest
import re
import tempfile
from testpath import assert_isfile

from karabo_data import utils
from karabo_data.utils import QuickView


def test_cbf_conversion(mock_agipd_data, capsys):
    with tempfile.TemporaryDirectory() as td:
        out_file = os.path.join(td, 'out.cbf')
        utils.hdf5_to_cbf(mock_agipd_data, out_file, index=0)
        assert_isfile(out_file)

    captured = capsys.readouterr()
    assert re.match("Convert .* to .*/out.cbf", captured.out)


def test_init_quick_view():
    qv = QuickView()

    assert qv.data is None
    qv.data = np.empty((1, 1, 1), dtype=np.int8)
    assert len(qv) == 1
    assert qv.pos == 0

    with pytest.raises(TypeError) as info:
        qv.data = 4

    with pytest.raises(TypeError) as info:
        qv.data = np.empty((1, 1, 1, 1), dtype=np.int8)


if __name__ == "__main__":
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
