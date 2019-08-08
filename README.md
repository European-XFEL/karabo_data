[![Build Status](https://travis-ci.org/European-XFEL/karabo_data.svg?branch=master)](https://travis-ci.org/European-XFEL/karabo_data)
[![codecov](https://codecov.io/gh/European-XFEL/karabo_data/branch/master/graph/badge.svg)](https://codecov.io/gh/European-XFEL/karabo_data)

Python 3 tools for reading European XFEL's HDF5 files.

[Documentation](https://karabo-data.readthedocs.io/en/latest/)

Installing
==========

`karabo_data` is available on our Anaconda installation on the Maxwell cluster:

    module load exfel exfel_anaconda3

You can also install it [from PyPI](https://pypi.org/project/karabo-data/)
to use in other environments with Python 3.5 or later:

    pip install karabo_data

If you get a permissions error, add the `--user` flag to that command.


Contributing
===========

Tests
-----

Tests can be run as follows:

    python3 -m pytest -v
