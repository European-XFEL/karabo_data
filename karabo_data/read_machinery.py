"""Machinery for reading Karabo HDF5 files

The public API is in karabo_data.reader; this is internal code.
"""
from collections import defaultdict
import numpy as np
import os.path as osp
import re

from .exceptions import SourceNameError

DETECTOR_NAMES = {'AGIPD', 'LPD'}
DETECTOR_SOURCE_RE = re.compile(r'(.+)/DET/(\d+)CH')


class _SliceConstructor(type):
    """Allows instantiation like subclass[1:5]
    """

    def __getitem__(self, item):
        return self(item)


class _SliceConstructable(metaclass=_SliceConstructor):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        indices = self.value
        if not isinstance(indices, tuple):
            indices = (indices,)

        return "{}[{}]".format(
            type(self).__name__, ', '.join(self._indexing_repr(v) for v in indices)
        )

    @staticmethod
    def _indexing_repr(value):
        """Represent values as used in canonical slicing syntax"""
        if value is Ellipsis:
            return '...'
        elif isinstance(value, slice):
            start = value.start if (value.start is not None) else ''
            stop = value.stop if (value.stop is not None) else ''
            step = ':{}'.format(value.step) if (value.step is not None) else ''
            return '{}:{}{}'.format(start, stop, step)

        return repr(value)


def _tid_to_slice_ix(tid, train_ids, stop=False):
    """Convert a train ID to an integer index for slicing the dataset

    Throws ValueError if the slice won't overlap the trains in the data.
    The *stop* parameter tells it which end of the slice it is making.
    """
    if tid is None:
        return None

    try:
        return train_ids.index(tid)
    except ValueError:
        pass

    if tid < train_ids[0]:
        if stop:
            raise ValueError("Train ID {} is before this run (starts at {})"
                             .format(tid, train_ids[0]))
        else:
            return None
    elif tid > train_ids[-1]:
        if stop:
            return None
        else:
            raise ValueError("Train ID {} is after this run (ends at {})"
                             .format(tid, train_ids[-1]))
    else:
        # This train ID is within the run, but doesn't have an entry.
        # Find the first ID in the run greater than the one given.
        return (train_ids > tid).nonzero()[0][0]


class DataChunk:
    """Reference to a contiguous chunk of data for one or more trains."""

    def __init__(self, file, source, key, first, train_ids, counts):
        self.file = file
        self.source = source
        self.key = key
        self.first = first
        self.train_ids = train_ids
        self.counts = counts

    @property
    def slice(self):
        return slice(self.first, self.first + np.sum(self.counts))

    @property
    def dataset(self):
        if self.source in self.file.instrument_sources:
            group = 'INSTRUMENT'
        elif self.source in self.file.control_sources:
            group = 'CONTROL'
        else:
            raise SourceNameError(self.source)

        return self.file.file[
            '/{}/{}/{}'.format(group, self.source, self.key.replace('.', '/'))
        ]


# contiguous_regions() by Joe Kington on Stackoverflow
# https://stackoverflow.com/a/4495197/434217
# Used here under Stackoverflow's default CC-BY-SA 3.0 license.
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx


def union_selections(selections):
    """Merge together different selections

    A selection is a dict of {source: set(keys)}, or {source: None}
    to include all keys for a given source.
    """
    selection_multi = defaultdict(list)

    for seln in selections:
        for source, keys in seln.items():
            selection_multi[source].append(keys)

    # Merge selected keys; None -> all keys selected
    return {
        source: None if (None in keygroups) else set().union(*keygroups)
        for (source, keygroups) in selection_multi.items()
    }


class FilenameInfo:
    is_detector = False
    detector_name = None
    detector_moduleno = -1

    _rawcorr_descr = {'RAW': 'Raw', 'CORR': 'Corrected'}

    def __init__(self, path):
        self.basename = osp.basename(path)
        nameparts = self.basename[:-3].split('-')
        assert len(nameparts) == 4, self.basename
        rawcorr, runno, datasrc, segment = nameparts
        m = re.match(r'([A-Z]+)(\d+)', datasrc)

        if m and m.group(1) == 'DA':
            self.description = "Aggregated data"
        elif m and m.group(1) in DETECTOR_NAMES:
            self.is_detector = True
            name, moduleno = m.groups()
            self.detector_name = name
            self.detector_moduleno = moduleno
            self.description = "{} detector data from {} module {}".format(
                self._rawcorr_descr.get(rawcorr, '?'), name, moduleno
            )
        else:
            self.description = "Unknown data source ({})", datasrc
