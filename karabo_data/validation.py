from argparse import ArgumentParser
from functools import partial
from glob import glob
import h5py
import numpy as np
import os
import sys

from .reader import DataCollection, RunDirectory, H5File, FileAccess


class ValidationError(Exception):
    def __init__(self, problems):
        self.problems = problems

    def __str__(self):
        lines = []
        for prob in self.problems:
            lines.extend(['', prob['msg']])
            for k, v in sorted(prob.items()):
                if k != 'msg':
                    lines.append("  {}: {}".format(k, v))

        return '\n'.join(lines)


class FileValidator:
    def __init__(self, file: FileAccess):
        self.file = file
        self.filename = file.filename
        self.problems = []

    def validate(self):
        problems = self.run_checks()
        if problems:
            raise ValidationError(problems)

    def run_checks(self):
        self.problems = []
        self.check_indices()
        self.check_trainids()
        return self.problems

    def record(self, msg, **kwargs):
        self.problems.append(dict(msg=msg, file=self.filename, **kwargs))

    def check_trainids(self):
        ds_path = 'INDEX/trainId'
        train_ids = self.file.file[ds_path][:]

        if (train_ids == 0).any():
            first0 = train_ids.tolist().index(0)
            if not (train_ids[first0:] == 0).all():
                self.record(
                    'Zeroes in trainId index before last train ID', dataset=ds_path
                )
            nonzero_tids = train_ids[train_ids != 0]
        else:
            nonzero_tids = train_ids

        if len(nonzero_tids) > 1:
            non_incr = (nonzero_tids[1:] <= nonzero_tids[:-1]).nonzero()[0]
            if non_incr.size > 0:
                pos = non_incr[0]
                self.record(
                    'Train IDs are not strictly increasing, e.g. at {} ({} >= {})'.format(
                        pos, nonzero_tids[pos], nonzero_tids[pos + 1]
                    ),
                    dataset=ds_path,
                )

    def check_indices(self):
        for src in self.file.instrument_sources:
            src_groups = set()
            for key in self.file.get_keys(src):
                ds_path = 'INSTRUMENT/{}/{}'.format(src, key.replace('.', '/'))
                group = key.split('.', 1)[0]
                src_groups.add((src, group))
                first, count = self.file.get_index(src, group)
                data_dim0 = self.file.file[ds_path].shape[0]
                if np.any((first + count) > data_dim0):
                    max_end = (first + count).max()
                    self.record(
                        'Index referring to data ({}) outside dataset ({})'.format(
                            max_end, data_dim0
                        ),
                        dataset=ds_path,
                    )

            for src, group in src_groups:
                record = partial(self.record, dataset='INDEX/{}/{}'.format(src, group))
                first, count = self.file._read_index(src, group)
                check_index_contiguous(first, count, record)


def check_index_contiguous(firsts, counts, record):
    probs = []

    if firsts[0] != 0:
        record("Index doesn't start at 0")

    gaps = firsts[1:].astype(np.int64) - (firsts + counts)[:-1]

    gap_ixs = (gaps > 0).nonzero()[0]
    if gap_ixs.size > 0:
        pos = gap_ixs[0]
        record("Gaps ({}) in index, e.g. at {} ({} + {} < {})".format(
            gap_ixs.size, pos, firsts[pos], counts[pos], firsts[pos+1]
        ))

    overlap_ixs = (gaps < 0).nonzero()[0]
    if overlap_ixs.size > 0:
        pos = overlap_ixs[0]
        record("Overlaps ({}) in index, e.g. at {} ({} + {} > {})".format(
            overlap_ixs.size, pos, firsts[pos], counts[pos], firsts[pos + 1]
        ))

    return probs


class RunValidator:
    def __init__(self, run_dir: str):
        files = []
        self.files_excluded = []
        self.run_dir = run_dir

        for path in glob(os.path.join(run_dir, '*.h5')):
            try:
                fa = FileAccess(h5py.File(path, 'r'))
            except Exception as e:
                self.files_excluded.append((path, e))
            else:
                files.append(fa)

        self.run = DataCollection(files)
        self.problems = []

    def validate(self):
        problems = self.run_checks()
        if problems:
            raise ValidationError(problems)

    def run_checks(self):
        self.problems = []
        self.check_files_openable()
        self.check_files()
        return self.problems

    def check_files_openable(self):
        for path, err in self.files_excluded:
            self.problems.append(dict(msg="Could not open file", file=path, error=err))

        if not self.run.files:
            self.problems.append(
                dict(msg="No usable files found", directory=self.run_dir)
            )

    def check_files(self):
        for f in self.run.files:
            fv = FileValidator(f)
            self.problems.extend(fv.run_checks())


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = ArgumentParser(prog='karabo-data-validate')
    ap.add_argument('path', help="HDF5 file or run directory of HDF5 files.")
    args = ap.parse_args(argv)

    path = args.path
    if os.path.isdir(path):
        print("Checking run directory:", path)
        validator = RunValidator(path)
    else:
        print("Checking file:", path)
        validator = FileValidator(H5File(path).files[0])

    try:
        validator.validate()
        print("No problems found")
    except ValidationError as ve:
        print("Validation failed!")
        print(str(ve))
        return 1


if __name__ == '__main__':
    sys.exit(main())
