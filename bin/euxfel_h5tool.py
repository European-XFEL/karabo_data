#!/usr/bin/env python
"""European XFEL HDF5 files exploration tool

Usage:
  euxfel_h5tool.py [options] PATH ...
  euxfel_h5tool.py --help      # use to display options
  euxfel_h5tool.py --h         # use to display options

Options:

  -h, --help         Show this screen.
  -v, --version      Show version.
  -s, --structure    Display structure of file
  -i, --info         Display summary info of file

Details:

  PATH is meant to point to a directory of files, or a list of files,
  or a single FILE.

  Some options will only work for a single file. Some will iterate
  over a list of files if given multiple files.

  Others will make sense of a directory full of files belonging to the
  same run.

"""

import logging
import os

import docopt
import h5py

import euxfel_h5tools

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# get version from external file?
version = 0.1


if __name__ == "__main__":

    arguments = docopt.docopt(__doc__, version=version)
    logging.debug("arguments = {}".format(arguments))

    path = arguments['PATH']
    if len(path) == 1:   # could be file or directory

        if os.path.isdir(path):
            # if PATH is directory, read list of files, TODO
            filenames = os.path.listdir(path)
    else:
        # assume that all arguments are files
        filenames = arguments['PATH']
        for filename in filenames:
            assert os.path.isfile(filename)

    # Iterate over files for options that process single files
    if arguments['--structure'] == True:
            # Only works with one file at the moment

            for filename in filenames:
                print("-"*70)
                print("Structure for file '{}'".format(filename))
                with h5py.File(filename, 'r') as f:
                    euxfel_h5tools.rec_print_h5_level(f, maxlen=3)

    if arguments['--info'] == True:
        euxfel_h5tools.stat(filenames)
