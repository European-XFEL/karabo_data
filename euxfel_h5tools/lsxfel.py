"""Summarise XFEL data in files or folders
"""
import argparse
from enum import Enum
import os
import re
import sys

from .reader import H5File, RunHandler

rawcorr_descr = {'RAW': 'Raw', 'CORR': 'Corrected'}
detector_names = {'AGIPD', 'LPD'}


class FileInfo:
    is_detector = False

    def __init__(self, basename):
        self.basename = basename
        nameparts = basename[:-3].split('-')
        assert len(nameparts) == 4, basename
        rawcorr, runno, datasrc, segment = nameparts
        m = re.match(r'([A-Z]+)(\d+)', datasrc)

        if m and m.group(1) == 'DA':
            self.description = "Aggregated data"
        elif m and m.group(1) in detector_names:
            self.is_detector = True
            name, moduleno = m.groups()
            self.description = "{} detector data from {} module {}".format(
                rawcorr_descr.get(rawcorr, '?'), name, moduleno
            )
        else:
            self.description = "Unknown data source ({})", datasrc

def find_image(h5file):
    """Find the image data in a detector file

    Returns (img_data, index). img_data is a HDF5 dataset, index is a group
    """
    img_source = [src for src in h5file.sources
                 if re.match(r'INSTRUMENT/.+/image', src)][0]
    img_ds = h5file.file[img_source + '/data']
    img_index_name = 'INDEX/' + img_source.split('/', 1)[1]
    return img_ds, h5file.file[img_index_name]


def describe_file(path):
    """Describe a single HDF5 data file"""
    basename = os.path.basename(path)
    info = FileInfo(basename)
    print(basename, ":", info.description)

    h5file = H5File(path)
    print(len(h5file.train_ids), "trains")
    print()

    if info.is_detector:
        img_data, img_index = find_image(h5file)
        # Some trains have 0 frames; max is the interesting value
        frames_per_train = img_index['count'][:].max()
        total_frames = img_index['count'][:].sum()

        print("{} × {}".format(*img_data.shape[-2:]), "pixels")
        print("{} frames per train, {} total".format(
            frames_per_train, total_frames,
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

def summarise_file(path):
    basename = os.path.basename(path)
    info = FileInfo(basename)
    print(basename, ":", info.description)

    h5file = H5File(path)
    ntrains = len(h5file.train_ids)

    if info.is_detector:
        img_data, img_index = find_image(h5file)
        # Some trains have 0 frames; max is the interesting value
        frames_per_train = img_index['count'][:].max()
        total_frames = img_index['count'][:].sum()

        print("  {} trains, {} frames/train, {} total frames".format(
            len(h5file.train_ids), frames_per_train, total_frames,
        ))
    else:
        print("  {} trains, {} devices".format(
            ntrains, len(h5file.sources)
        ))

def describe_run(path):
    basename = os.path.basename(path)
    print(basename, ": Run directory")
    print()

    run = RunHandler(path)
    run.info()

def summarise_run(path, indent=''):
    basename = os.path.basename(path)
    run = RunHandler(path)
    print("{}{} : Run of {} trains, with {} files".format(
        indent, basename, len(run.ordered_trains), len(run.files)
    ))

def main(argv=None):
    ap = argparse.ArgumentParser(prog='lsxfel',
        description="Summarise XFEL data in files or folders")
    ap.add_argument('paths', nargs='*', help="Files/folders to look at")
    args = ap.parse_args(argv)
    paths = args.paths or [os.path.abspath(os.getcwd())]

    if len(paths) == 1:
        path = paths[0]
        basename = os.path.basename(os.path.abspath(path))

        if os.path.isdir(path):
            contents = os.listdir(path)
            if any(f.endswith('.h5') for f in contents):
                # Run directory
                describe_run(path)
            elif any(re.match(r'r\d+', f) for f in contents):
                # Proposal directory, containing runs
                print(basename, ": Proposal directory")
                print()
                for f in contents:
                    child_path = os.path.join(path, f)
                    if re.match(r'r\d+', f) and os.path.isdir(child_path):
                        summarise_run(child_path, indent='  ')
            else:
                print(basename, ": Unrecognised directory")
        elif os.path.isfile(path):
            if path.endswith('.h5'):
                describe_file(path)
            else:
                print(basename, ": Unrecognised file")
                return 2
        else:
            print(path, ': File/folder not found')
            return 2
    else:
        exit_code = 0
        for path in paths:
            basename = os.path.basename(path)

            if os.path.isdir(path):
                contents = os.listdir(path)
                if any(f.endswith('.h5') for f in contents):
                    # Run directory
                    summarise_run(path)
                elif any(re.match(r'r\d+', f) for f in contents):
                    # Proposal directory, containing runs
                    print(basename, ": Proposal directory")
                    print()
                    for f in contents:
                        child_path = os.path.join(path, f)
                        if re.match(r'r\d+', f) and os.path.isdir(child_path):
                            summarise_run(child_path, indent='  ')
                else:
                    print(basename, ": Unrecognised directory")
                    exit_code = 2
            elif os.path.isfile(path):
                if path.endswith('.h5'):
                    summarise_file(path)
                else:
                    print(basename, ": Unrecognised file")
                    exit_code = 2
            else:
                print(path, ': File/folder not found')
                exit_code = 2

        return exit_code

if __name__ == '__main__':
    sys.exit(main())
