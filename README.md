[![Build Status](https://travis-ci.org/European-XFEL/karabo_data.svg?branch=master)](https://travis-ci.org/European-XFEL/karabo_data)
[![codecov](https://codecov.io/gh/European-XFEL/karabo_data/branch/master/graph/badge.svg)](https://codecov.io/gh/European-XFEL/karabo_data)

Python 3 tools for reading European XFEL's HDF5 files.

[Documentation](https://karabo-data.readthedocs.io/en/latest/)

Installing
==========

To install the package on the Maxwell cluster, run:

    module load anaconda/3
    pip install --user karabo_data

If this causes problems for Jupyter, you may need to upgrade ``ipykernel``
to fix them::

    pip install --user --upgrade ipykernel


Contributing
===========

Tests
-----

Tests can be run as follows:

    python3 -m pytest -v
