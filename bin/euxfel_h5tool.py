#!/usr/bin/python
"""
European XFEL HDF5 files stat
"""

import sys

import h5py
import numpy as np

import euxfel_h5tools



if __name__ == "__main__":
    euxfel_h5tools.stat(sys.argv[1:])
