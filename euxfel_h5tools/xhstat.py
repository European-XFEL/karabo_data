#!/usr/bin/python
"""
European XFEL HDF5 files stat
"""

import sys

import h5py
import numpy as np

import euxfel_h5tools

def stat(args):
    first_train = 0
    last_train = 0
    total_size = 0
    total_entries = 0
    invalid_files = []

    for filename in args:
        out = ""
        try:
            xfel_file = h5py.File(filename, 'r')
            out = " File: '{}'\n".format(xfel_file.filename)
            size_mb = xfel_file.fid.get_filesize() / 1000000
            out += " Size: {} MB".format(size_mb)
            entries = len(xfel_file["INDEX/trainId"])
            out += "\t Entries: {}".format(entries)
            f_train = xfel_file["INDEX/trainId"][0]
            out += "\t First Train: {}".format(f_train)
            l_train = xfel_file["INDEX/trainId"][-1]
            out += "\t Last Train: {}".format(l_train)
            out += "\n"

            total_size += size_mb
            total_entries += entries

            if f_train <= first_train:
                first_train = f_train
            if l_train >= last_train:
                last_train = l_train
        except:
            # The errors could be:
            #  - OSError: not an HDF5 file
            #  - IOError: truncated file
            #  - IndexError: one of the keys used was not found,
            #                therefore not EuXFEL specific
            out = "{}: not an EuXFEL HDF5 file".format(filename)
            invalid_files.append(filename)

        print(out)

    if len(args) > 1:
        total = "Total Files: {}\t".format(len(args) - len(invalid_files))
        total += "First Train: {}\n".format(first_train)
        total += "Total File Size: {}\t".format(total_size)
        total += "Last Train: {}\n".format(last_train)
        print(total)

    if invalid_files:
        print("These are not valid files: {}".format(", ".join(invalid_files)))


if __name__ == "__main__":
    stat(sys.argv[1:])
