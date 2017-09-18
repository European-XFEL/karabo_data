#!/usr/bin/env python
""""
European XFEL HDF5 files exploration tool

Usage: euxfel_h5tool.py [-shvi] PATH

-h, --help         Show this screen.
-v, --version      Show version.
-s, --structure    Display structure of file
-i, --info         Display summary info of file


"""

import logging
import sys

import docopt
import h5py
import matplotlib.pyplot as plt
import numpy as np

import euxfel_h5tools

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



if __name__ == "__main__":

    arguments = docopt.docopt(__doc__, version=0.9)
    logging.debug("arguments = {}".format(arguments))

    filename = arguments['PATH']

    if arguments['--structure'] == True:
            # Only works with one file at the moment
            print("Overview of structure")
            f = h5py.File(filename, 'r')
            euxfel_h5tools.rec_print_h5_level(f, maxlen=3)

    if arguments['--info'] == True:
        euxfel_h5tools.stat([filename])
