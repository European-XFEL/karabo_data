[![Build Status](https://travis-ci.org/European-XFEL/h5tools-py.svg?branch=master)](https://travis-ci.org/European-XFEL/h5tools-py)

# h5tools-py
Python 3 tools for reading European XFEL's HDF5 files.


Installing
==========
This package is dependent on fabio, h5py, matplotlib, numpy.

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
