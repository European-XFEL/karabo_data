# coding: utf-8
"""
Collection of classes and functions to help reading HDF5 file generated at
The European XFEL.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

from collections import defaultdict
import datetime
import fnmatch
from glob import glob
import h5py
import numpy as np
import os.path as osp
import pandas as pd
import re
import sys
import xarray

from .exceptions import SourceNameError, PropertyNameError
from .read_machinery import (
    DETECTOR_SOURCE_RE,
    DataChunk,
    FilenameInfo,
    _SliceConstructable,
    _tid_to_slice_ix,
    union_selections,
    contiguous_regions,
)

__all__ = [
    'H5File',
    'RunDirectory',
    'FileAccess',
    'DataCollection',
    'stack_data',
    'stack_detector_data',
    'by_id',
    'by_index',
    'SourceNameError',
    'PropertyNameError',
]


RUN_DATA = 'RUN'
INDEX_DATA = 'INDEX'
METADATA = 'METADATA'


class by_id(_SliceConstructable):
    pass


class by_index(_SliceConstructable):
    pass


class FileAccess:
    """Accesses a single Karabo HDF5 file

    Parameters
    ----------
    file: h5py.File
        Open h5py file object
    """

    def __init__(self, file):
        self.file = file
        self.filename = file.filename
        tid_data = file['INDEX/trainId'][:]
        self.train_ids = tid_data[tid_data != 0]

        self.control_sources = set()
        self.instrument_sources = set()

        for source in file['METADATA/dataSourceId'][:]:
            if not source:
                continue
            source = source.decode()
            category, _, h5_source = source.partition('/')
            if category == 'INSTRUMENT':
                device, _, chan_grp = h5_source.partition(':')
                chan, _, group = chan_grp.partition('/')
                source = device + ':' + chan
                self.instrument_sources.add(source)
                # TODO: Do something with groups?
            elif category == 'CONTROL':
                self.control_sources.add(h5_source)
            else:
                raise ValueError("Unknown data category %r" % category)

        self.control_sources = frozenset(self.control_sources)
        self.instrument_sources = frozenset(self.instrument_sources)

        # {(file, source, group): (firsts, counts)}
        self._index_cache = {}
        # {source: set(keys)}
        self._keys_cache = {}

    def __hash__(self):
        return hash(self.filename)

    def __eq__(self, other):
        return isinstance(other, FileAccess) and (other.filename == self.filename)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(self.file))

    def get_index(self, source, group):
        """Get first index & count for a source and for a specific train ID.

        Indices are cached; this appears to provide some performance benefit.
        """
        try:
            return self._index_cache[(source, group)]
        except KeyError:
            ix = self._read_index(source, group)
            self._index_cache[(source, group)] = ix
            return ix

    def _read_index(self, source, group):
        """Get first index & count for a source.

        This is 'real' reading when the requested index is not in the cache.
        """
        ntrains = len(self.train_ids)
        ix_group = self.file['/INDEX/{}/{}'.format(source, group)]
        firsts = ix_group['first'][:ntrains]
        if 'count' in ix_group:
            counts = ix_group['count'][:ntrains]
        else:
            status = ix_group['status'][:ntrains]
            counts = np.uint64((ix_group['last'][:ntrains] - firsts + 1) * status)
        return firsts, counts

    def get_keys(self, source):
        """Get keys for a given source name

        Keys are found by walking the HDF5 file, and cached for reuse.
        """
        try:
            return self._keys_cache[source]
        except KeyError:
            pass

        if source in self.control_sources:
            group = '/CONTROL/' + source
        elif source in self.instrument_sources:
            group = '/INSTRUMENT/' + source
        else:
            raise SourceNameError(source)

        res = set()

        def add_key(key, value):
            if isinstance(value, h5py.Dataset):
                res.add(key.replace('/', '.'))

        self.file[group].visititems(add_key)
        self._keys_cache[source] = res
        return res


class DataCollection:
    """An assemblage of data generated at European XFEL

    Data consists of *sources* which each have *keys*. It is further
    organised by *trains*, which are identified by train IDs.

    You normally get an instance of this class by calling :func:`H5File`
    for a single file or :func:`RunDirectory` for a directory.
    """
    def __init__(self, files, selection=None, train_ids=None, ctx_closes=False):
        self.files = list(files)
        self.ctx_closes = ctx_closes

        # selection: {source: set(keys)}
        # None as value -> all keys for this source
        if selection is None:
            selection = {}
            for f in self.files:
                selection.update(dict.fromkeys(f.control_sources))
                selection.update(dict.fromkeys(f.instrument_sources))
        self.selection = selection

        self.control_sources = set()
        self.instrument_sources = set()
        self._source_index = defaultdict(list)
        for f in self.files:
            self.control_sources.update(f.control_sources.intersection(selection))
            self.instrument_sources.update(f.instrument_sources.intersection(selection))
            for source in (f.control_sources | f.instrument_sources):
                self._source_index[source].append(f)

        # Throw an error if we have conflicting data for the same source
        self._check_source_conflicts()

        self.control_sources = frozenset(self.control_sources)
        self.instrument_sources = frozenset(self.instrument_sources)

        if train_ids is None:
            train_ids = sorted(set().union(*(f.train_ids for f in files)))
        self.train_ids = train_ids

    @classmethod
    def from_paths(cls, paths):
        files = []
        for path in paths:
            try:
                fa = FileAccess(h5py.File(path, 'r'))
            except Exception as e:
                print("Skipping file", path, file=sys.stderr)
                print("  (error was: {})".format(e), file=sys.stderr)
            else:
                files.append(fa)

        if not files:
            raise Exception("All HDF5 files specified are unusable")

        return cls(files, ctx_closes=True)

    @classmethod
    def from_path(cls, path):
        files = [FileAccess(h5py.File(path, 'r'))]
        return cls(files, ctx_closes=True)

    def __enter__(self):
        if not self.ctx_closes:
            raise Exception(
                "Only DataCollection objects created by opening "
                "files directly can be used in a 'with' statement, "
                "not those created by selecting from or merging "
                "others."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the files if this collection was created by opening them.
        if self.ctx_closes:
            for file in self.files:
                file.file.close()

    @property
    def all_sources(self):
        return self.control_sources | self.instrument_sources

    @property
    def detector_sources(self):
        return set(filter(DETECTOR_SOURCE_RE.match, self.instrument_sources))

    def _check_field(self, source, key):
        if source not in self.all_sources:
            raise SourceNameError(source)
        if key not in self.keys_for_source(source):
            raise PropertyNameError(key, source)

    def keys_for_source(self, source):
        selected_keys = self.selection[source]
        if selected_keys is not None:
            return selected_keys

        # The same source may be in multiple files, but this assumes it has
        # the same keys in all files that it appears in.
        for f in self._source_index[source]:
            return f.get_keys(source)

    # Leave old name in case anything external was using it:
    _keys_for_source = keys_for_source

    def _check_data_missing(self, tid) -> bool:
        """Return True if a train does not have data for all sources"""
        for source in self.control_sources:
            file, _ = self._find_data(source, tid)
            if file is None:
                return True

        for source in self.instrument_sources:
            file, pos = self._find_data(source, tid)
            if file is None:
                return True

            groups = {k.partition('.')[0] for k in self.keys_for_source(source)}
            for group in groups:
                _, counts = file.get_index(source, group)
                if counts[pos] == 0:
                    return True

        return False

    def trains(self, devices=None, train_range=None, *, require_all=False):
        """Iterate over all trains in the data and gather all sources.

        ::

            run = Run('/path/to/my/run/r0123')
            for train_id, data in run.trains():
                value = data['device']['parameter']

        Parameters
        ----------

        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.

        train_range: by_id or by_index object, optional
            Iterate over only selected trains, by train ID or by index.
            Refer to :meth:`select_trains` for how to use this.

        require_all: bool
            False (default) returns any data available for the requested trains.
            True skips trains which don't have all the selected data;
            this only makes sense if you make a selection with *devices*
            or :meth:`select`.

        Yields
        ------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name
        """
        dc = self
        if devices is not None:
            dc = dc.select(devices)
        if train_range is not None:
            dc = dc.select_trains(train_range)
        return iter(TrainIterator(dc, require_all=require_all))

    def train_from_id(self, train_id, devices=None):
        """Get Train data for specified train ID.

        Parameters
        ----------

        train_id: int
            The train ID
        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name

        Raises
        ------
        KeyError
            if `train_id` is not found in the run.
        """
        if train_id not in self.train_ids:
            raise KeyError(train_id)

        if devices is not None:
            return self.select(devices).train_from_id(train_id)

        res = {}
        for source in self.control_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': train_id}
            }
            file, pos = self._find_data(source, train_id)
            if file is None:
                continue

            for key in self.keys_for_source(source):
                path = '/CONTROL/{}/{}'.format(source, key.replace('.', '/'))
                source_data[key] = file.file[path][pos]

        for source in self.instrument_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': train_id}
            }
            file, pos = self._find_data(source, train_id)
            if file is None:
                continue

            for key in self.keys_for_source(source):
                group = key.partition('.')[0]
                firsts, counts = file.get_index(source, group)
                first, count = firsts[pos], counts[pos]
                if not count:
                    continue

                path = '/INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
                if count == 1:
                    source_data[key] = file.file[path][first]
                else:
                    source_data[key] = file.file[path][first : first + count]

        return train_id, res

    def train_from_index(self, train_index, devices=None):
        """Get train data of the nth train in this data.

        Parameters
        ----------
        train_index: int
            Index of the train in the file.
        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name
        """
        train_id = self.train_ids[train_index]
        return self.train_from_id(int(train_id), devices=devices)

    def get_series(self, source, key):
        """Return a pandas Series for a particular data field.

        Parameters
        ----------

        source: str
            Device name with optional output channel, e.g.
            "SA1_XTD2_XGM/DOOCS/MAIN" or "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "beamPosition.iyPos.value"
            or "header.linkId". The data must be 1D in the file.
        """
        self._check_field(source, key)
        name = source + '/' + key
        if name.endswith('.value'):
            name = name[:-6]

        seq_series = []

        if source in self.control_sources:
            data_path = "/CONTROL/{}/{}".format(source, key.replace('.', '/'))
            for f in self._source_index[source]:
                data = f.file[data_path][: len(f.train_ids), ...]
                index = pd.Index(f.train_ids, name='trainId')

                seq_series.append(pd.Series(data, name=name, index=index))

        elif source in self.instrument_sources:
            data_path = "/INSTRUMENT/{}/{}".format(source, key.replace('.', '/'))
            for f in self._source_index[source]:
                group = key.partition('.')[0]
                firsts, counts = f.get_index(source, group)
                trainids = self._expand_trainids(counts, f.train_ids)

                index = pd.Index(trainids, name='trainId')
                data = f.file[data_path][:]
                if not index.is_unique:
                    pulse_id = f.file['/INSTRUMENT/{}/{}/pulseId'.format(source, group)]
                    pulse_id = pulse_id[: len(index), 0]
                    index = pd.MultiIndex.from_arrays(
                        [trainids, pulse_id], names=['trainId', 'pulseId']
                    )
                    # Does pulse-oriented data always have an extra dimension?
                    assert data.shape[1] == 1
                    data = data[:, 0]
                data = data[: len(index)]

                seq_series.append(pd.Series(data, name=name, index=index))
        else:
            raise Exception("Unknown source category")

        ser = pd.concat(sorted(seq_series, key=lambda s: s.index[0]))

        # Select out only the train IDs of interest
        if isinstance(ser.index, pd.MultiIndex):
            train_ids = ser.index.levels[0].intersection(self.train_ids)
            # A numpy array works for selecting, but a pandas index doesn't
            train_ids = np.asarray(train_ids)
        else:
            train_ids = ser.index.intersection(self.train_ids)

        return ser.loc[train_ids]

    def get_dataframe(self, fields=None, *, timestamps=False):
        """Return a pandas dataframe for given data fields.

        Parameters
        ----------
        fields : dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.
        timestamps : bool
            If false (the default), exclude the timestamps associated with each
            control data field.
        """
        if fields is not None:
            return self.select(fields).get_dataframe(timestamps=timestamps)

        series = []
        for source in self.all_sources:
            for key in self.keys_for_source(source):
                if (not timestamps) and key.endswith('.timestamp'):
                    continue
                series.append(self.get_series(source, key))

        return pd.concat(series, axis=1)

    def get_array(self, source, key, extra_dims=None, roi=by_index[...]):
        """Return a labelled array for a particular data field.

        The first axis of the returned data will be the train IDs.

        Parameters
        ----------

        source: str
            Device name with optional output channel, e.g.
            "SA1_XTD2_XGM/DOOCS/MAIN" or "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "beamPosition.iyPos.value"
            or "header.linkId".
        extra_dims: list of str
            Name extra dimensions in the array. The first dimension is
            automatically called 'train'. The default for extra dimensions
            is dim_0, dim_1, ...
        roi: by_index
            The region of interest. This expression selects data in all
            dimensions apart from the first (trains) dimension. If the data
            holds a 1D array for each entry, roi=by_index[:8] would get the
            first 8 values from every train. If the data is 2D or more at
            each entry, selection looks like roi=by_index[:8, 5:10] .
        """
        self._check_field(source, key)

        if not isinstance(roi, by_index):
            raise TypeError("roi parameter must be instance of by_index")
        else:
            roi = roi.value
            if not isinstance(roi, tuple):
                roi = (roi,)

        seq_arrays = []

        for chunk in self._find_data_chunks(source, key):
            trainids = self._expand_trainids(chunk.counts, chunk.train_ids)

            slices = (chunk.slice,) + roi
            data = chunk.dataset[slices]

            if extra_dims is None:
                extra_dims = ['dim_%d' % i for i in range(data.ndim - 1)]
            dims = ['trainId'] + extra_dims

            seq_arrays.append(
                xarray.DataArray(data, dims=dims, coords={'trainId': trainids})
            )

        non_empty = [a for a in seq_arrays if (a.size > 0)]
        if not non_empty:
            if seq_arrays:
                # All per-file arrays are empty, so just return the first one.
                return seq_arrays[0]

            raise Exception(("Unable to get data for source {!r}, key {!r}. "
                             "Please report an issue so we can investigate")
                            .format(source, key))

        return xarray.concat(
            sorted(non_empty, key=lambda a: a.coords['trainId'][0]), dim='trainId'
        )

    def union(self, *others):
        """Join the data in this collection with one or more others.

        This can be used to join multiple sources for the same trains,
        or to extend the same sources with data for further trains.
        The order of the datasets doesn't matter.

        Returns a new :class:`DataCollection` object.
        """
        files = set(self.files)
        train_ids = set(self.train_ids)

        for other in others:
            files.update(other.files)
            train_ids.update(other.train_ids)

        train_ids = sorted(train_ids)
        selection = union_selections([self.selection] +
                                     [o.selection for o in others])

        return DataCollection(files, selection=selection, train_ids=train_ids)

    def _expand_selection(self, selection):
        res = defaultdict(set)
        if isinstance(selection, dict):
            # {source: {key1, key2}}
            # {source: {}} -> all keys for this source
            for source, keys in selection.items():  #
                if source not in self.all_sources:
                    raise SourceNameError(source)

                res[source].update(keys or None)

        elif isinstance(selection, list):
            # selection = [('src_glob', 'key_glob'), ...]
            res = union_selections(
                self._select_glob(src_glob, key_glob)
                for (src_glob, key_glob) in selection
            )
        else:
            TypeError("Unknown selection type: {}".format(type(selection)))

        return dict(res)

    def _select_glob(self, source_glob, key_glob):
        source_re = re.compile(fnmatch.translate(source_glob))
        key_re = re.compile(fnmatch.translate(key_glob))
        if key_glob.endswith(('.value', '*')):
            ctrl_key_re = key_re
        else:
            # The translated pattern ends with "\Z" - insert before this
            p = key_re.pattern
            end_ix = p.rindex(r'\Z')
            ctrl_key_re = re.compile(p[:end_ix] + r'(\.value)?' + p[end_ix:])

        matched = {}
        for source in self.all_sources:
            if not source_re.match(source):
                continue

            if key_glob == '*':
                matched[source] = None
            else:
                r = ctrl_key_re if source in self.control_sources else key_re
                keys = set(filter(r.match, self.keys_for_source(source)))
                if keys:
                    matched[source] = keys

        if not matched:
            raise ValueError("No matches for pattern {}"
                             .format((source_glob, key_glob)))
        return matched

    def select(self, seln_or_source_glob, key_glob='*'):
        """Select a subset of sources and keys from this data.

        There are three possible ways to select data:

        1. With glob patterns (* is a wildcard) for source and key::

            sel = run.select('*/DET/*, 'image.*')

        2. With a list of (source, key) glob patterns::

            sel = run.select([('*/DET/*, 'image.data'), ('*/DET/*, 'image.mask')])

        3. With a dict of source names mapped to sets of key names
           (or empty sets to get all keys)::

            sel = run.select({'SPB_DET_AGIPD1M-1/DET/ALLCH:xtdf': {'image.data'},
                              'SA1_XTD2_XGM/XGM/DOOCS': set()})

        Returns a new :class:`DataCollection` object for the selected data.
        """
        if isinstance(seln_or_source_glob, str):
            seln_or_source_glob = [(seln_or_source_glob, key_glob)]
        selection = self._expand_selection(seln_or_source_glob)

        return DataCollection(self.files, selection=selection, train_ids=self.train_ids)

    def deselect(self, seln_or_source_glob, key_glob='*'):
        """Select everything except the specified sources and keys.

        This takes the same arguments as :meth:`select`, but the sources and
        keys you specify are dropped from the selection.

        Returns a new :class:`DataCollection` object for the remaining data.
        """

        if isinstance(seln_or_source_glob, str):
            seln_or_source_glob = [(seln_or_source_glob, key_glob)]
        deselection = self._expand_selection(seln_or_source_glob)

        # Subtract deselection from self.selection
        selection = {}
        for source, keys in self.selection.items():
            if source not in deselection:
                selection[source] = keys
                continue

            desel_keys = deselection[source]
            if desel_keys is None:
                continue  # Drop the entire source

            if keys is None:
                keys = self.keys_for_source(source)

            selection[source] = keys - desel_keys

        return DataCollection(self.files, selection=selection, train_ids=self.train_ids)

    def select_trains(self, train_range):
        """Select a subset of trains from this data.

        Choose a slice of trains by train ID::

            from karabo_data import by_id
            sel = run.select_trains(by_id[142844490:142844495])

        Or select a list of trains::

            sel = run.select_trains(by_id[[142844490, 142844493, 142844494]])

        Or select trains by index within this collection::

            from karabo_data import by_index
            sel = run.select_trains(by_index[:5])

        Returns a new :class:`DataCollection` object for the selected trains.

        Raises
        ------
        ValueError
            If given train IDs do not overlap with the trains in this data.
        """
        tr = train_range
        if isinstance(tr, by_id) and isinstance(tr.value, slice):
            # Slice by train IDs
            start_ix = _tid_to_slice_ix(tr.value.start, self.train_ids, stop=False)
            stop_ix = _tid_to_slice_ix(tr.value.stop, self.train_ids, stop=True)
            new_train_ids = self.train_ids[start_ix : stop_ix : tr.value.step]
        elif isinstance(tr, by_index) and isinstance(tr.value, slice):
            # Slice by indexes in this collection
            new_train_ids = self.train_ids[tr.value]
        elif isinstance(tr, by_id) and isinstance(tr.value, (list, np.ndarray)):
            # Select a list of trains by train ID
            new_train_ids = sorted(set(self.train_ids).intersection(tr.value))
            if not new_train_ids:
                raise ValueError(
                    "Given train IDs not found among {} trains in "
                    "collection".format(len(self.train_ids))
                )
        elif isinstance(tr, by_index) and isinstance(tr.value, (list, np.ndarray)):
            # Select a list of trains by index in this collection
            new_train_ids = sorted([self.train_ids[i] for i in tr.value])
        else:
            raise TypeError(type(train_range))

        files = [f for f in self.files
                 if np.intersect1d(f.train_ids, new_train_ids).size > 0]

        return DataCollection(files, selection=self.selection, train_ids=new_train_ids)

    def _check_source_conflicts(self):
        """Check for data with the same source and train ID in different files.
        """
        sources_with_conflicts = set()
        for source, files in self._source_index.items():
            got_tids = np.array([], dtype=np.uint64)
            for file in files:
                if np.intersect1d(got_tids, file.train_ids).size > 0:
                    sources_with_conflicts.add(source)
                    break
                got_tids = np.union1d(got_tids, file.train_ids)

        if sources_with_conflicts:
            raise ValueError("{} sources have conflicting data "
                             "(same train ID in different files): {}".format(
                len(sources_with_conflicts), ", ".join(sources_with_conflicts)
            ))

    def _expand_trainids(self, counts, trainIds):
        n = min(len(counts), len(trainIds))
        return np.repeat(trainIds[:n], counts.astype(np.intp)[:n])

    def _find_data_chunks(self, source, key):
        """Find contiguous chunks of data for the given source & key

        Yields DataChunk objects.
        """
        if source in self.instrument_sources:
            key_group = key.partition('.')[0]
        else:
            key_group = ''

        for file in self._source_index[source]:
            file_has_data = False
            firsts, counts = file.get_index(source, key_group)

            # Of trains in this file, which are in selection
            selected = np.isin(file.train_ids, self.train_ids)

            # Assemble contiguous chunks of data from this file
            for _from, _to in contiguous_regions(selected):
                file_has_data = True
                yield DataChunk(file, source, key,
                                first=firsts[_from],
                                train_ids=file.train_ids[_from:_to],
                                counts=counts[_from:_to],
                                )

            if not file_has_data:
                # Make an empty chunk to let e.g. get_array find the shape
                yield DataChunk(file, source, key,
                                first=np.uint64(0),
                                train_ids=file.train_ids[:0],
                                counts=counts[:0],
                                )


    def _find_data(self, source, train_id) -> (FileAccess, int):
        for f in self._source_index[source]:
            ixs = (f.train_ids == train_id).nonzero()[0]
            if ixs.size > 0:
                return f, ixs[0]

        return None, None

    def info(self):
        """Show information about the run.
        """
        # time info
        first_train = self.train_ids[0]
        last_train = self.train_ids[-1]
        train_count = len(self.train_ids)
        span_sec = (last_train - first_train) / 10
        span_txt = str(datetime.timedelta(seconds=span_sec))

        detector_modules = {}
        for source in self.detector_sources:
            name, modno = DETECTOR_SOURCE_RE.match(source).groups((1, 2))
            detector_modules[(name, modno)] = source

        # A run should only have one detector, but if that changes, don't hide it
        detector_name = ','.join(sorted(set(k[0] for k in detector_modules)))

        # disp
        print('# of trains:   ', train_count)
        print('Duration:      ', span_txt)
        print('First train ID:', first_train)
        print('Last train ID: ', last_train)
        print()

        print("{} detector modules ({})".format(
            len(self.detector_sources), detector_name
        ))
        if len(detector_modules) > 0:
            # Show detail on the first module (the others should be similar)
            mod_key = sorted(detector_modules)[0]
            mod_source = detector_modules[mod_key]
            dinfo = self.detector_info(mod_source)
            module = ' '.join(mod_key)
            dims = ' x '.join(str(d) for d in dinfo['dims'])
            print("  e.g. module {} : {} pixels".format(module, dims))
            print("  {} frames per train, {} total frames".format(
                dinfo['frames_per_train'], dinfo['total_frames']
            ))
        print()

        non_detector_inst_srcs = self.instrument_sources - self.detector_sources
        print(len(non_detector_inst_srcs), 'instrument sources (excluding detectors):')
        for d in sorted(non_detector_inst_srcs):
            print('  -', d)
        print()
        print(len(self.control_sources), 'control sources:')
        for d in sorted(self.control_sources):
            print('  -', d)
        print()

    def detector_info(self, source):
        """Get statistics about the detector data.

        Returns a dictionary with keys:
        - 'dims' (pixel dimensions)
        - 'frames_per_train'
        - 'total_frames'
        """
        all_counts = []
        for file in self._source_index[source]:
            _, counts = file.get_index(source, 'image')
            all_counts.append(counts)

        all_counts = np.concatenate(all_counts)
        dims = file.file['/INSTRUMENT/{}/image/data'.format(source)].shape[-2:]

        return {
            'dims': dims,
            # Some trains have 0 frames; max is the interesting value
            'frames_per_train': all_counts.max(),
            'total_frames': all_counts.sum(),
        }

    def train_info(self, train_id):
        """Show information about a specific train in the run.

        Parameters
        ----------
        train_id: int
            The specific train ID you get details information.

        Raises
        ------
        ValueError
            if `train_id` is not found in the run.
        """
        if train_id not in self.train_ids:
            raise ValueError("train {} not found in run.".format(train_id))
        files = [f for f in self.files if train_id in f.train_ids]
        ctrl = set().union(*[f.control_sources for f in files])
        inst = set().union(*[f.instrument_sources for f in files])

        # disp
        print('Train [{}] information'.format(train_id))
        print('Devices')
        print('\tInstruments')
        [print('\t-', d) for d in sorted(inst)] or print('\t-')
        print('\tControls')
        [print('\t-', d) for d in sorted(ctrl)] or print('\t-')

    def write(self, filename):
        """Write the selected data to a new HDF5 file

        You can choose a subset of the data using methods
        like :meth:`select` and :meth:`select_trains`,
        then use this write it to a new, smaller file.

        The target filename will be overwritten if it already exists.
        """
        from .writer import FileWriter
        FileWriter(filename, self).write()


class TrainIterator:
    """Iterate over trains in a collection of data

    Created by :meth:`DataCollection.trains`.
    """
    def __init__(self, data, require_all=True):
        self.data = data
        self.require_all = require_all
        # {(source, key): (f, dataset)}
        self._datasets_cache = {}

    def _find_data(self, source, key, tid):
        try:
            file, ds = self._datasets_cache[(source, key)]
        except KeyError:
            pass
        else:
            ixs = (file.train_ids == tid).nonzero()[0]
            if ixs.size > 0:
                return file, ixs[0], ds

        data = self.data
        section = 'CONTROL' if source in data.control_sources else 'INSTRUMENT'
        path = '/{}/{}/{}'.format(section, source, key.replace('.', '/'))
        f, pos = data._find_data(source, tid)
        if f is not None:
            ds = f.file[path]
            self._datasets_cache[(source, key)] = (f, ds)
            return f, pos, ds

        return None, None, None

    def _assemble_data(self, tid):
        res = {}
        for source in self.data.control_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': tid}
            }
            for key in self.data.keys_for_source(source):
                _, pos, ds = self._find_data(source, key, tid)
                if ds is None:
                    continue
                source_data[key] = ds[pos]

        for source in self.data.instrument_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': tid}
            }
            for key in self.data.keys_for_source(source):
                file, pos, ds = self._find_data(source, key, tid)
                if ds is None:
                    continue
                group = key.partition('.')[0]
                firsts, counts = file.get_index(source, group)
                first, count = firsts[pos], counts[pos]
                if count == 1:
                    source_data[key] = ds[first]
                else:
                    source_data[key] = ds[first : first + count]

        return res

    def __iter__(self):
        for tid in self.data.train_ids:
            tid = int(tid)  # Convert numpy int to regular Python int
            if self.require_all and self.data._check_data_missing(tid):
                continue
            yield tid, self._assemble_data(tid)


def H5File(path):
    """Open a single HDF5 file generated at European XFEL.

    Returns a :class:`DataCollection` object.

    Parameters
    ----------
    path: str
        Path to the HDF5 file
    """
    return DataCollection.from_path(path)


def RunDirectory(path):
    """Open data files from a 'run' at European XFEL.

    A 'run' is a directory containing a number of HDF5 files with data from the
    same time period.

    Returns a :class:`DataCollection` object.

    Parameters
    ----------
    path: str
        Path to the run directory containing HDF5 files.
    """
    files = list(filter(h5py.is_hdf5, glob(osp.join(path, '*.h5'))))
    if not files:
        raise Exception("No HDF5 files found in {}".format(path))
    return DataCollection.from_paths(files)


# RunDirectory was previously RunHandler; we'll leave it accessible in case
# any code was already using it.
RunHandler = RunDirectory


def stack_data(train, data, axis=-3, xcept=()):
    """Stack data from devices in a train.

    For detector data, use stack_detector_data instead: it can handle missing
    modules, which this function cannot.

    The returned array will have an extra dimension. The data will be ordered
    according to any groups of digits in the source name, interpreted as
    integers. Other characters do not affect sorting. So:

        "B_7_0" < "A_12_0" < "A_12_1"

    Parameters
    ----------
    train: dict
        Train data.
    data: str
        The path to the device parameter of the data you want to stack.
    axis: int, optional
        Array axis on which you wish to stack.
    xcept: list
        List of devices to ignore (useful if you have reccored slow data with
        detector data in the same run).

    Returns
    -------
    combined: numpy.array
        Stacked data for requested data path.
    """
    devices = [dev for dev in train.keys() if dev not in xcept]

    if not devices:
        raise ValueError("No data after filtering by 'xcept' argument.")

    dtypes, shapes = set(), set()
    ordered_arrays = []
    for device in sorted(devices, key=lambda d: list(map(int, re.findall(r'\d+', d)))):
        array = train[device][data]
        dtypes.add(array.dtype)
        ordered_arrays.append(array)

    if len(dtypes) > 1:
        raise ValueError("Arrays have mismatched dtypes: {}".format(dtypes))

    return np.stack(ordered_arrays, axis=axis)


def stack_detector_data(train, data, axis=-3, modules=16, only='', xcept=()):
    """Stack data from detector modules in a train.

    Parameters
    ----------
    train: dict
        Train data.
    data: str
        The path to the device parameter of the data you want to stack, e.g. 'image.data'.
    axis: int
        Array axis on which you wish to stack (default is -3).
    modules: int
        Number of modules composing a detector (default is 16).
    only: str
        Only use devices in train containing this substring.
    xcept: list
        List of devices to ignore (useful if you have reccored slow data with
        detector data in the same run).

    Returns
    -------
    combined: numpy.array
        Stacked data for requested data path.
    """
    devices = [dev for dev in train.keys() if only in dev and dev not in xcept]

    if not devices:
        raise ValueError("No data after filtering by 'only' and 'xcept' arguments.")

    dtypes, shapes, skip = set(), set(), set()
    modno_arrays = {}
    for device in devices:
        det_mod_match = re.search(r'/DET/(\d+)CH', device)
        if not det_mod_match:
            raise ValueError("Non-detector source: {}".format(device))
        modno = int(det_mod_match.group(1))

        array = train[device][data]
        dtypes.add(array.dtype)
        shapes.add(array.shape)
        modno_arrays[modno] = array

    if len(dtypes) > 1:
        raise ValueError("Arrays have mismatched dtypes: {}".format(dtypes))
    if len(shapes) > 1:
        s1, s2, *_ = sorted(shapes)
        if len(shapes) > 2 or (s1[0] != 0) or (s1[1:] != s2[1:]):
            raise ValueError("Arrays have mismatched shapes: {}".format(shapes))
        skip = {n for n, a in modno_arrays.items() if a.shape == s1}
        shapes.remove(s1)
    if max(modno_arrays) >= modules:
        raise IndexError("Module {} is out of range for a detector with {} modules"
                         .format(max(modno_arrays), modules))

    dtype = dtypes.pop()
    shape = shapes.pop()
    combined = np.full((modules,) + shape, np.nan, dtype=dtype)
    for modno, array in modno_arrays.items():
        if modno in skip:
            continue
        combined[modno] = array

    return np.moveaxis(combined, 0, axis)
