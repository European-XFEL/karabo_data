#!/usr/bin/python
"""
European XFEL HDF5 files stat
"""

import sys

import h5py
import numpy as np

import euxfel_h5tools

import matplotlib.pyplot as plt




if __name__ == "__main__":
    euxfel_h5tools.stat(sys.argv[1:])

    filename = sys.argv[1]
    # need to split the following in functions that can be activated
    # with switches. Only works with one file at the moment
    print("Overview of structure")
    f = h5py.File(filename, 'r')
    euxfel_h5tools.rec_print_h5_level(f, maxlen=3)
