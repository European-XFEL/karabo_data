from collections import defaultdict
import datetime
from enum import Enum
import fnmatch
from glob import glob
import h5py
import numpy as np
import os.path as osp
import pandas as pd
import re
import xarray

__all__ = ['DataCollection', 'RunDirectory', 'H5File',
           'SourceNameError', 'PropertyNameError',
           'stack_data', 'stack_detector_data', 'by_index', 'by_id',
           ]

from .reader import (SourceNameError, PropertyNameError,
                     stack_data, stack_detector_data,
                     by_id, by_index, FilenameInfo
                    )

class SourceCategory(Enum):
    CONTROL = 1
    INSTRUMENT = 2

class _SourceInfo:
    def __init__(self, category, keys=None):
        self.files = []  # [(trainids, file)]
        self.category = category
        self.keys = keys

    def copy(self):
        new = type(self)(self.category, keys=self.keys)
        new.files = self.files[:]
        return new

    def with_keys(self, keys):
        new = type(self)(self.category, keys=keys)
        new.files = self.files
        return new

    def with_train_ids(self, train_ids):
        new = type(self)(self.category, keys=self.keys)
        for f_train_ids, f in self.files:
            if np.intersect1d(train_ids, f_train_ids).size > 0:
                new.files.append((f_train_ids, f))

        return new


class DataCollection:
    def __init__(self, _sources=None, train_ids=None):
        # {source: [(train_id_range, file)]}
        self._sources = _sources or {}
        # {(file, source, group): (firsts, counts)}
        self._index_cache = {}
        # {source: set(keys)}
        self._source_keys = {}

        collect_train_ids = set()
        for data in self._sources.values():
            for f_train_ids, _ in data.files:
                collect_train_ids.update(f_train_ids)
        if train_ids is not None:
            collect_train_ids.intersection_update(train_ids)
        self.train_ids = sorted(collect_train_ids)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def control_sources(self):
        return {s for (s, i) in self._sources.items()
                if i.category == SourceCategory.CONTROL}

    @property
    def instrument_sources(self):
        return {s for (s, i) in self._sources.items()
                if i.category == SourceCategory.INSTRUMENT}

    @property
    def all_sources(self):
        return set(self._sources)

    @property
    def files(self):
        res = set()
        for data in self._sources.values():
            for _, file in data.files:
                res.add(file)
        return res

    @classmethod
    def from_file(cls, path):
        f = h5py.File(path)

        tid_data = f['INDEX/trainId'].value
        train_ids = tid_data[tid_data != 0]

        sources = {}

        for source in f['METADATA/dataSourceId'].value:
            if not source:
                continue
            source = source.decode()
            category, _, h5_source = source.partition('/')
            if category == 'INSTRUMENT':
                device, _, chan_grp = h5_source.partition(':')
                chan, _, group = chan_grp.partition('/')
                source = device + ':' + chan
                sources[source] = info = _SourceInfo(SourceCategory.INSTRUMENT)
                # TODO: Do something with groups?
            elif category == 'CONTROL':
                sources[h5_source] = info = _SourceInfo(SourceCategory.CONTROL)
            else:
                raise ValueError("Unknown data category %r" % category)

            info.files.append((train_ids, f))

        return cls(sources)

    def union(self, *others):
        sources = {s: i.copy() for (s, i) in self._sources.items()}
        for other in others:
            for source, other_info in other._sources.items():
                if source in sources:
                    info = sources[source]
                    # Unify keys; if either is None, include all keys
                    if other_info.keys is None:
                        info.keys = None
                    elif info.keys is not None:
                        info.keys.update(other_info.keys)

                    for trains, file in other_info.files:
                        if not any(np.array_equal(trains, t) for (t, f) in info.files):
                            info.files.append((trains, file))
                else:
                    sources[source] = other_info.copy()

        return DataCollection(sources)

    def _expand_selection(self, selection):
        res = defaultdict(set)
        if isinstance(selection, dict):
            # {source: {key1, key2}}
            # {source: {}} -> all keys for this source
            for source, keys in selection.items():  #
                if source not in self.all_sources:
                    raise SourceNameError(source)

                res[source].update(keys or ['*'])

        elif isinstance(selection, list):
            # [('src_glob', 'key_glob'), ...]
            for src_glob, key_glob in selection:
                matched = self._select_glob(src_glob, key_glob)

                if not matched:
                    raise ValueError("No matches for pattern {}"
                                     .format((src_glob, key_glob)))

                for s, k in matched.items():
                    res[s].update(k)
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
            end_ix = p.rindex('\Z')
            ctrl_key_re = re.compile(p[:end_ix] + r'(\.value)?' + p[end_ix:])

        matched = {}
        for source in self.all_sources:
            if not source_re.match(source):
                continue

            if key_glob == '*':
                matched[source] = {'*'}
            else:
                r = ctrl_key_re if source in self.control_sources else key_re
                keys = set(filter(r.match, self._keys_for_source(source)))
                if keys:
                    matched[source] = keys

        if not matched:
            raise ValueError("No matches for pattern {}"
                             .format((source_glob, key_glob)))
        return matched

    def select(self, seln_or_source_glob, key_glob='*'):
        """Return a new DataCollection with selected sources & keys
        """
        if isinstance(seln_or_source_glob, str):
            selection = self._select_glob(seln_or_source_glob, key_glob)
        else:
            selection = self._expand_selection(seln_or_source_glob)

        selection = self._expand_selection(selection)

        new_sources = {}
        for source, keys in selection.items():
            if '*' in keys:
                new_sources[source] = self._sources[source]
            else:
                new_sources[source] = self._sources[source].with_keys(keys)
        res = DataCollection(new_sources, self.train_ids)
        res._index_cache = {(f, s, g): v for ((f, s, g), v) in self._index_cache.items()
                            if s in selection}

        return res

    def select_trains(self, train_range):
        if isinstance(train_range, by_id):
            start_ix = _tid_to_slice_ix(train_range.value.start, self.train_ids, stop=False)
            stop_ix = _tid_to_slice_ix(train_range.value.stop, self.train_ids, stop=True)
            ix_slice = slice(start_ix, stop_ix, train_range.value.step)
        elif isinstance(train_range, by_index):
            ix_slice = train_range.value
        else:
            raise TypeError(type(train_range))

        new_train_ids = self.train_ids[ix_slice]

        new_sources = {}
        for source, info in self._sources.items():
            new_info = info.with_train_ids(new_train_ids)
            if new_info.files:
                new_sources[source] = new_info

        res = DataCollection(new_sources, new_train_ids)

        res._index_cache = {(f, s, g): v for ((f, s, g), v) in self._index_cache.items()
                            if s in new_sources}
        return res

    def _check_field(self, source, key):
        if source not in self.all_sources:
            raise SourceNameError(source)
        if key not in self._keys_for_source(source):
            raise PropertyNameError(key, source)

    def get_array(self, source, key, extra_dims=None):
        self._check_field(source, key)
        seq_arrays = []

        if source in self.control_sources:
            data_path = "/CONTROL/{}/{}".format(source, key.replace('.', '/'))
            for trainids, f in self._sources[source].files:
                data = f[data_path][:len(trainids), ...]
                if extra_dims is None:
                    extra_dims = ['dim_%d' % i for i in range(data.ndim - 1)]
                dims = ['trainId'] + extra_dims

                seq_arrays.append(
                    xarray.DataArray(data, dims=dims, coords={'trainId': trainids}))

        elif source in self.instrument_sources:
            data_path = "/INSTRUMENT/{}/{}".format(source, key.replace('.', '/'))
            for trainids, f in self._sources[source].files:
                group = key.partition('.')[0]
                firsts, counts = self._get_index(f, source, group)
                if (counts > 1).any():
                    raise ValueError("{}/{} data has more than one data point per train"
                                     .format(source, group))
                trainids = self._expand_trainids(firsts, counts, trainids)

                data = f[data_path][:len(trainids), ...]

                if extra_dims is None:
                    extra_dims = ['dim_%d' % i for i in range(data.ndim - 1)]
                dims = ['trainId'] + extra_dims

                seq_arrays.append(
                    xarray.DataArray(data, dims=dims, coords={'trainId': trainids}))
        else:
            raise SourceNameError(source)

        non_empty = [a for a in seq_arrays if (a.size > 0)]
        if not non_empty:
            if seq_arrays:
                # All per-file arrays are empty, so just return the first one.
                return seq_arrays[0]

            raise Exception(("Unable to get data for source {!r}, key {!r}. "
                             "Please report an issue so we can investigate")
                            .format(source, key))

        return xarray.concat(sorted(non_empty,
                                    key=lambda a: a.coords['trainId'][0]),
                             dim='trainId')

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
            for trainids, f in self._sources[source].files:
                data = f[data_path][:len(trainids), ...]
                index = pd.Index(trainids, name='trainId')

                seq_series.append(pd.Series(data, name=name, index=index))

        elif source in self.instrument_sources:
            data_path = "/INSTRUMENT/{}/{}".format(source, key.replace('.', '/'))
            for trainids, f in self._sources[source].files:
                group = key.partition('.')[0]
                firsts, counts = self._get_index(f, source, group)
                trainids = self._expand_trainids(firsts, counts, trainids)

                index = pd.Index(trainids, name='trainId')
                data = f[data_path][:]
                if not index.is_unique:
                    pulse_id = f['/INSTRUMENT/{}/{}/pulseId'
                                 .format(source, group)]
                    pulse_id = pulse_id[:len(index), 0]
                    index = pd.MultiIndex.from_arrays([trainids, pulse_id],
                                                      names=['trainId', 'pulseId'])
                    # Does pulse-oriented data always have an extra dimension?
                    assert data.shape[1] == 1
                    data = data[:, 0]
                data = data[:len(index)]

                seq_series.append(pd.Series(data, name=name, index=index))
        else:
            raise Exception("Unknown source category")

        return pd.concat(sorted(seq_series, key=lambda s: s.index[0]))

    def get_dataframe(self, fields=None, *, timestamps=False):
        if fields is not None:
            return self.select(fields).get_dataframe(timestamps=timestamps)

        series = []
        for source in self.all_sources:
            for key in self._keys_for_source(source):
                if (not timestamps) and key.endswith('.timestamp'):
                    continue
                series.append(self.get_series(source, key))

        return pd.concat(series, axis=1)

    def _get_index(self, file, source, group):
        """Get first index & count for a source and for a specific train ID.

        Indices are cached; this appears to provide some performance benefit.
        """
        try:
            return self._index_cache[(file, source, group)]
        except KeyError:
            ix = self._read_index(file, source, group)
            self._index_cache[(file, source, group)] = ix
            return ix

    def _read_index(self, file, source, group):
        """Get first index & count for a source.

        This is 'real' reading when the requested index is not in the cache.
        """
        ix_group = file['/INDEX/{}/{}'.format(source, group)]
        firsts = ix_group['first'][:]
        if 'count' in ix_group:
            counts = ix_group['count'][:]
        else:
            status = ix_group['status'][:]
            counts = np.uint64((ix_group['last'][:] - firsts + 1) * status)
        return firsts, counts

    def _expand_trainids(self, first, counts, trainIds):
        n = min(len(counts), len(trainIds))
        return np.repeat(trainIds[:n], counts.astype(np.intp)[:n])

    def _keys_for_source(self, source):
        try:
            info = self._sources[source]
        except KeyError:
            raise SourceNameError(source)

        if info.keys is not None:
            return info.keys

        if info.category is SourceCategory.CONTROL:
            group = '/CONTROL/' + source
        else:
            group = '/INSTRUMENT/' + source

        # The same source may be in multiple files, but this assumes it has
        # the same keys in all files that it appears in.
        for trainids, f in info.files:
            res = set()

            def add_key(key, value):
                if isinstance(value, h5py.Dataset):
                    res.add(key.replace('/', '.'))

            f[group].visititems(add_key)
            info.keys = res
            return res

    def _find_data(self, source, train_id):
        for trainids, f in self._sources[source].files:
            ixs = (trainids == train_id).nonzero()[0]
            if ixs.size > 0:
                return f, ixs[0]

        return None, None

    def train_from_id(self, train_id, devices=None):
        if devices is not None:
            return self.select(devices).train_from_id(train_id)

        res = {}
        for source in self.control_sources:
            source_data = res[source] = {}
            file, pos = self._find_data(source, train_id)
            if file is None:
                continue

            for key in self._keys_for_source(source):
                path = '/CONTROL/{}/{}'.format(source, key.replace('.', '/'))
                source_data[key] = file[path][pos]

        for source in self.instrument_sources:
            source_data = res[source] = {}
            file, pos = self._find_data(source, train_id)
            if file is None:
                continue

            for key in self._keys_for_source(source):
                group = key.partition('.')[0]
                firsts, counts = self._get_index(file, source, group)
                first, count = firsts[pos], counts[pos]
                if not count:
                    continue

                path = '/INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
                if count == 1:
                    source_data[key] = file[path][first]
                else:
                    source_data[key] = file[path][first:first+count]

        return train_id, res

    def train_from_index(self, train_index, devices=None):
        train_id = self.train_ids[train_index]
        return self.train_from_id(train_id, devices=devices)

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

            groups = {k.partition('.')[0] for k in self._keys_for_source(source)}
            for group in groups:
                _, counts = self._get_index(file, source, group)
                if counts[pos] == 0:
                    return True

        return False

    def info(self):
        """Show information about the run.
        """
        # time info
        first_train = self.train_ids[0]
        last_train = self.train_ids[-1]
        train_count = len(self.train_ids)
        span_sec = (last_train - first_train) / 10
        span_txt = str(datetime.timedelta(seconds=span_sec))

        detector_srcs, non_detector_inst_srcs = [], []
        detector_modules = {}
        for source in self.instrument_sources:
            m = re.match(r'(.+)/DET/(\d+)CH', source)
            if m:
                detector_srcs.append(source)
                name, modno = m.groups((1, 2))
                detector_modules[(name, modno)] = source
            else:
                non_detector_inst_srcs.append(source)

        # A run should only have one detector, but if that changes, don't hide it
        detector_name = ','.join(sorted(set(k[0] for k in detector_modules)))

        # disp
        print('# of trains:   ', train_count)
        print('Duration:      ', span_txt)
        print('First train ID:', first_train)
        print('Last train ID: ', last_train)
        print()

        print("{} detector modules ({})".format(
            len(detector_srcs), detector_name
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
        for _, file in self._sources[source].files:
            _, counts = self._get_index(file, source, 'image')
            all_counts.append(counts)

        all_counts = np.concatenate(all_counts)
        dims = file['/INSTRUMENT/{}/image/data'.format(source)].shape[-2:]

        return {
            'dims': dims,
            # Some trains have 0 frames; max is the interesting value
            'frames_per_train': all_counts.max(),
            'total_frames': all_counts.sum(),
        }

    def trains(self, devices=None, train_range=None, *, require_all=False):
        dc = self
        if devices is not None:
            dc = dc.select(devices)
        if train_range is not None:
            dc = dc.select_trains(train_range)
        return iter(TrainIterator(dc, require_all=require_all))


class TrainIterator:
    def __init__(self, data, require_all=True):
        self.data = data
        self.require_all = require_all
        # {(source, key): (trainids, f, dataset)}
        self._datasets_cache = {}

    def _find_data(self, source, key, tid):
        try:
            cache_tids, file, ds = self._datasets_cache[(source, key)]
        except KeyError:
            pass
        else:
            ixs = (cache_tids == tid).nonzero()[0]
            if ixs.size > 0:
                return file, ixs[0], ds

        source_info = self.data._sources[source]
        section = 'CONTROL' if source_info.category is SourceCategory.CONTROL else 'INSTRUMENT'
        path = '/{}/{}/{}'.format(section, source, key.replace('.', '/'))
        for trainids, f in source_info.files:
            if tid in trainids:
                ds = f[path]
                self._datasets_cache[(source, key)] = (trainids, f, ds)
                ix = (trainids == tid).nonzero()[0][0]
                return f, ix, ds

        return None, None, None

    def _assemble_data(self, tid):
        res = {}
        for source in self.data.control_sources:
            source_data = res[source] = {}
            for key in self.data._keys_for_source(source):
                _, pos, ds = self._find_data(source, key, tid)
                if ds is None:
                    continue
                source_data[key] = ds[pos]

        for source in self.data.instrument_sources:
            source_data = res[source] = {}
            for key in self.data._keys_for_source(source):
                file, pos, ds = self._find_data(source, key, tid)
                if ds is None:
                    continue
                group = key.partition('.')[0]
                firsts, counts = self.data._get_index(file, source, group)
                first, count = firsts[pos], counts[pos]
                if count == 1:
                    source_data[key] = ds[first]
                else:
                    source_data[key] = ds[first:first+count]

        return res

    def __iter__(self):
        for tid in self.data.train_ids:
            if self.require_all and self.data._check_data_missing(tid):
                continue
            yield tid, self._assemble_data(tid)

def H5File(path):
    return DataCollection.from_file(path)

def RunDirectory(path):
    ds = [DataCollection.from_file(file)
          for file in glob(osp.join(path, '*.h5'))
          if h5py.is_hdf5(file)]
    if not ds:
        raise Exception("No HDF5 files found in {}".format(path))
    return DataCollection.union(*ds)


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
