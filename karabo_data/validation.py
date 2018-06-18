import numpy as np

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
        return self.problems

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
                    self.problems.append({
                        'msg': 'Index referring to data ({}) outside dataset ({})'
                               .format(max_end, data_dim0),
                        'file': self.filename,
                        'dataset': ds_path,
                    })

            for h5_source in h5_sources:
                context = {
                    'file': self.filename,
                    'dataset': 'INDEX/' + h5_source,
                }
                first, count = self.file._read_index(h5_source)
                self.problems.extend(check_index_contiguous(first, count, context))

def check_index_contiguous(firsts, counts, context):
    probs = []

    probs.append(dict(
        msg="Index doesn't start at 0",
        **context
    ))

    if np.all((firsts + counts)[:-1] == firsts[1:]):
        return probs

    for ix, (first, count, first_next) in enumerate(zip(firsts, counts, firsts[1:])):
        if (first + count) < first_next:
            probs.append(dict(
                msg="Gap in index at {} ({} + {} < {})".format(
                    ix, first, count, first_next
                ),
                **context
            ))
        elif (first + count) > first_next:
            probs.append(dict(
                msg="Overlap in index at {} ({} + {} > {})".format(
                    ix, first, count, first_next
                ),
                **context
            ))

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
