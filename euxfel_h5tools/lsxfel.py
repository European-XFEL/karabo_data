"""Summarise XFEL data in files or folders
"""
import argparse
import os
import re
import sys

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
    print(len(h5file.train_ids), "trains")
    print()

    if is_detector:
        img_source = [src for src in h5file.sources
                      if re.match(r'INSTRUMENT/.+/image', src)][0]
        img_ds = h5file.file[img_source + '/data']
        img_index_name = 'INDEX/' + img_source.split('/', 1)[1]
        img_index_count = h5file.file[img_index_name + '/count']
        # Some trains have 0 frames; max is the interesting value
        frames_per_train = img_index_count[:].max()

        print("{} × {}".format(*img_ds.shape[-2:]), "pixels")
        print("{} frames per train, {} total".format(
            frames_per_train, img_ds.shape[0]
        ))
    else:
        ctrl, inst = set(), set()
        for src in h5file.sources:
            srcparts = src.split('/')
            if srcparts[0] == 'CONTROL':
                ctrl.add('/'.join(srcparts[1:4]))
            elif srcparts[0] == 'INSTRUMENT':
                inst.add('/'.join(srcparts[1:4]))

        print(len(ctrl), "control devices")
        for dev in sorted(ctrl):
            print("  - ", dev)
        print()

        print(len(inst), "instrument devices")
        for dev in sorted(inst):
            print("  - ", dev)
        print()

def describe_run(path):
    basename = os.path.basename(path)
    print(basename, ": Run directory")
    print()

    run = RunHandler(path)
    run.info()

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
            if any(f.endswith('.h5') for f in contents):
                # Run directory
                describe_run(path)
        elif os.path.isfile(path):
            if path.endswith('.h5'):
                describe_file(path)
            else:
                print(os.path.basename(path), ": Unrecognised file")
                return 2
        else:
            print(path, ': File/folder not found')
            return 2
    else:
        print("TODO: Multiple files/folders")

if __name__ == '__main__':
    sys.exit(main())
