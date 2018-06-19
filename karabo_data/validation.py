from functools import partial
import numpy as np
import os
import sys

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
    def __init__(self, file):
        self.file = file
        self.filename = file.file.filename
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
        self.problems.append(dict(
            msg=msg, file=self.filename, **kwargs
        ))

    def check_trainids(self):
        ds_path = 'INDEX/trainId'
        train_ids = self.file.file[ds_path][:]

        if (train_ids == 0).any():
            first0 = train_ids.tolist().index(0)
            if not (train_ids[first0:] == 0).all():
                self.record(
                    'Zeroes in trainId index before last train ID',
                    dataset=ds_path,
                )
            nonzero_tids = train_ids[train_ids != 0]
        else:
            nonzero_tids = train_ids

        if len(nonzero_tids) > 1:
            if not (nonzero_tids[1:] > nonzero_tids[:-1]).all():
                self.record(
                    'Train IDs are not strictly increasing',
                    dataset=ds_path,
                )

    def check_indices(self):
        for src in self.file.instrument_sources:
            h5_sources = set()
            for key in self.file._keys_for_source(src):
                ds_path = 'INSTRUMENT/{}/{}'.format(src, key.replace('.', '/'))
                h5_source = src + '/' + key.split('.', 1)[0]
                first, count = self.file._read_index(h5_source)
                data_dim0 = self.file.file[ds_path].shape[0]
                if np.any((first + count) > data_dim0):
                    max_end = (first + count).max()
                    self.record(
                        'Index referring to data ({}) outside dataset ({})'
                        .format(max_end, data_dim0),
                        dataset=ds_path,
                    )

            for h5_source in h5_sources:
                record = partial(self.record, dataset='INDEX/'+h5_source)
                first, count = self.file._read_index(h5_source)
                check_index_contiguous(first, count, record)

def check_index_contiguous(firsts, counts, record):
    probs = []

    record("Index doesn't start at 0")

    if np.all((firsts + counts)[:-1] == firsts[1:]):
        return probs

    for ix, (first, count, first_next) in enumerate(zip(firsts, counts, firsts[1:])):
        if (first + count) < first_next:
            record("Gap in index at {} ({} + {} < {})".format(
                    ix, first, count, first_next))
        elif (first + count) > first_next:
            record("Overlap in index at {} ({} + {} > {})".format(
                    ix, first, count, first_next))

    return probs

class RunValidator:
    def __init__(self, run):
        self.run = run
        self.problems = []

    def validate(self):
        problems = self.run_checks()
        if problems:
            raise ValidationError(problems)

    def run_checks(self):
        self.problems = []
        self.check_files()
        return self.problems

    def check_files(self):
        for f in self.run.files:
            fv = FileValidator(f)
            fv.validate()
            self.problems.extend(fv.problems)

def main(argv=None):
    from .reader import RunDirectory, H5File
    if argv is None:
        argv = sys.argv[1:]

    path = argv[0]
    if os.path.isdir(path):
        print("Checking run directory:", path)
        validator = RunValidator(RunDirectory(path))
    else:
        print("Checking file:", path)
        validator = FileValidator(H5File(path))

    try:
        validator.validate()
        print("No problems found")
    except ValidationError as ve:
        print("Validation failed!")
        print(str(ve))
        sys.exit(1)

if __name__ == '__main__':
    main()
