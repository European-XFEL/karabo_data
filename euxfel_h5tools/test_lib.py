import os
import subprocess

import pytest

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
