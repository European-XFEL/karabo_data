import numpy as np
import os
import pytest
import subprocess

from karabo_data import utils
from karabo_data.utils import QuickView


def test_summary_on_data_file():
    testdatapath = "data/example_data/R0126-AGG01-S00002.h5"
    if os.path.exists(testdatapath):
        # next line assumes that the have installed the package
        output = str(subprocess.check_output("euxfel_h5tool.py {}".format(
            testdatapath), shell=True))
        print(output)
        assert "Size: 665.596177 MB" in output
        assert "Entries: 10" in output
        assert "First Train: 1362168960" in output
    else:
        pytest.skip("test data file not available ()".format(testdatapath))


def test_cbf_conversion():
    testdatapath = "data/example_data/R0126-AGG01-S00002.h5"
    if os.path.exists(testdatapath):
        # Test that the help message pops up when the command is malformed
        command = "euxfel_h5tool.py convert-cbf".format(testdatapath)
        output = str(subprocess.check_output(command), shell=True)
        print(output)
        assert "Usage:" in output

        # Test that the cbf file is correctly created for index 0
        command = ("euxfel_h5tool.py convert-cbf {}"
                   "0 out.cbf".format(testdatapath))
        expected_output = "Convert {} index 0 to out.cbf".format(testdatapath)
        output = str(subprocess.check_output(command), shell=True)
        print(output)
        assert expected_output == output

        # Test that the cbf file is correctly created for an arbitrary index
        command = ("euxfel_h5tool.py convert-cbf {}"
                   "42 out.cbf".format(testdatapath))
        expected_output = "Convert {} index 42 to out.cbf".format(testdatapath)
        output = str(subprocess.check_output(command), shell=True)
        print(output)
        assert expected_output == output

        # Test graceful fail for inexisting file
        command = "euxfel_h5tool.py convert-cbf non_existing_data.h5 0 out.cbf"
        expected_output = "non_exisiting_data.h5: Could not be opened."
        output = str(subprocess.check_output(command), shell=True)
        print(output)
        assert expected_output == output

        # Test graceful fail for index out of range
        maxint_64 = 9223372036854775808
        command = ("euxfel_h5tool.py convert-cbf non_existing_data.h5"
                   "{} out.cbf".format(maxint_64))
        expected_output = "Index ({}) out of range".format(maxint_64)
        output = str(subprocess.check_output(command), shell=True)
        print(output)
        assert expected_output in output

        # Clean up
        os.remove("out.cbf")

    else:
        pytest.skip("test data file not available ()".format(testdatapath))


def test_init_quick_view():
    qv = QuickView()

    assert qv.data is None
    qv.data = np.empty((1,1,1), dtype=np.int8)
    assert len(qv) == 1
    assert qv.pos == 0

    with pytest.raises(TypeError) as info:
        qv.data = 4

    with pytest.raises(TypeError) as info:
        qv.data = np.empty((1,1,1,1), dtype=np.int8)


def test_hdf5_file_info(mock_lpd_data, capsys):
    utils.hdf5_file_info([mock_lpd_data])

    out, err = capsys.readouterr()
    assert "Entries: 480" in out

if __name__ == "__main__":
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
