[![Build Status](https://travis-ci.org/European-XFEL/karabo_data.svg?branch=master)](https://travis-ci.org/European-XFEL/karabo_data)
[![codecov](https://codecov.io/gh/European-XFEL/karabo_data/branch/master/graph/badge.svg)](https://codecov.io/gh/European-XFEL/karabo_data)

Python 3 tools for reading European XFEL's HDF5 files.


Installing
==========
This package is dependent on fabio, h5py, matplotlib, numpy, pandas.

Installing h5py on EuXFEL computer may require installing the *libhdf5-dev*
package from the ubuntu repositories.

to install package:

    pip3 install .


Contributing
===========

Tests
-----

Tests can be run as follows:

    python3 -m pytest -v
