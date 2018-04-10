"""Summarise XFEL data in files or folders
"""
import argparse
import os
import re

from .reader import H5File, RunHandler

rawcorr_descr = {'RAW': 'Raw', 'CORR': 'Corrected'}
detector_names = {'AGIPD', 'LPD'}

def describe_file(path):
    """Describe a single HDF5 data file"""
    basename = os.path.basename(path)
    nameparts = basename[:-3].split('-')
    assert len(nameparts) == 4, basename
    rawcorr, runno, datasrc, segment = nameparts
    m = re.match(r'([A-Z]+)(\d+)', datasrc)
    is_detector = False
    if m and m.group(1) == 'DA':
        file_descr = "Aggregated data"
    elif m and m.group(1) in detector_names:
        is_detector = True
        name, moduleno = m.groups()
        file_descr = "{} detector data from {} module {}".format(
            rawcorr_descr.get(rawcorr, '?'), name, moduleno
        )
    else:
        file_descr = "Unknown data source ({})", datasrc

    print(basename, ":", file_descr)

    h5file = H5File(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsxfel',
        description="Summarise XFEL data in files or folders")
    ap.add_argument('paths', nargs='*', help="Files/folders to look at")
    args = ap.parse_args(argv)
    paths = args.paths or [os.path.abspath(os.getcwd())]

    if len(paths) == 1:
        path = paths[0]
        if os.path.isdir(path):
            contents = os.listdir(path)
        elif path.endswith('.h5'):
            describe_file(path)
        else:
            print(os.path.basename(path), ": Unrecognised file")
            return 1
    else:
        print("TODO: Multiple files/folders")

if __name__ == '__main__':
    main()
