#!/usr/bin/env python
"""European XFEL HDF5 files exploration tool

Usage:
  euxfel_h5tool.py structure FILE [-l n] [--maxnode=n] [--size]
  euxfel_h5tool.py info FILES ... [-l n]
  euxfel_h5tool.py convert-cbf FILE INDEX OUTPUTFILENAME [-l n]
  euxfel_h5tool.py --help      # use to display options

Options:

  -h, --help           Show this screen.
  -v, --version        Show version.
Generic options
  -l n, --loglevel=n   Set log level level [default: 20].
                       DEBUG is 10. INFO is 20.
For commmand structure
  --maxnode=n          Maximum number of nodes [default: 10]
  --size               Display size of each node

Details:

  PATH is meant to point to a directory of files.
  FILE is a single FILE.
  FILES is a sequence of one or more files.

Commands:
  structure: display hdf5 structure

  info: display overview of saved trainIDs and pulses

"""

import logging
import os

import docopt
import h5py

import euxfel_h5tools

# get version from external file?
version = 0.1


def check_filenames_are_files(filenames):
    # check filenames are files
    logging.debug("Filenames={}".format(filenames))
    for filename in filenames:
        assert os.path.isfile(filename)
    return filenames


def convert_path_to_list_of_files(path):
    # convert path into list of filenames where required

    if len(path) == 1:   # could be file or directory
        if os.path.isdir(path[0]):
            # if PATH is directory, read list of files, TODO
            filenames = os.path.listdir(path)
        else:
            filenames = path
    else:
        # assume that all arguments are files
        filenames = arguments['PATH']


if __name__ == "__main__":

    arguments = docopt.docopt(__doc__, version=version)
    loglevel = int(arguments['--loglevel'])
    logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel)
    logging.debug("arguments = {}".format(arguments))

    # Iterate over files for options that process single files

    # structure
    if arguments['structure'] is True:
        # Only works with one file
        filename = arguments["FILE"]
        print("Structure for file '{}'".format(filename))
        max_len = int(arguments['--maxnode'])
        with h5py.File(filename, 'r') as f:
            euxfel_h5tools.rec_print_h5_level(f, maxlen=max_len)

    # info
    if arguments['info'] is True:
        filenames = check_filenames_are_files(arguments['FILES'])
        euxfel_h5tools.stat(filenames)

    # converting
    if arguments['convert-cbf'] is True:
        euxfel_h5tools.h5_to_cbf(arguments['FILE'],
                                 arguments['OUTPUTFILENAME'],
                                 int(arguments['INDEX']))
