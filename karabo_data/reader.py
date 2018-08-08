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
from fnmatch import fnmatchcase
from glob import glob
import h5py
import numpy as np
import os.path as osp
import pandas as pd
import re
import xarray as xr


__all__ = ['H5File', 'RunDirectory', 'RunHandler', 'stack_data',
           'stack_detector_data', 'by_id', 'by_index', 'SourceNameError',
           'PropertyNameError',
          ]


RUN_DATA = 'RUN'
INDEX_DATA = 'INDEX'
METADATA = 'METADATA'

DETECTOR_NAMES = {'AGIPD', 'LPD'}


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

class _SliceConstructor(type):
    """Allows instantiation like subclass[1:5]
    """
    def __getitem__(self, item):
        return self(item)

class by_id(metaclass=_SliceConstructor):
    def __init__(self, value):
        self.value = value

class by_index(metaclass=_SliceConstructor):
    def __init__(self, value):
        self.value = value

def _tid_to_slice_ix(tid, dataset, stop=False):
    """Convert a train ID to an integer index for slicing the dataset

    Throws ValueError if the slice won't overlap the trains in the data.
    The *stop* parameter tells it which end of the slice it is making.
    """
    if tid is None:
        return None

    try:
        return dataset.train_indices[tid]
    except KeyError:
        if tid < dataset.train_ids[0]:
            if stop:
                raise ValueError("Train ID {} is before this run (starts at {})"
                                 .format(tid, dataset.train_ids[0]))
            else:
                return None
        elif tid > dataset.train_ids[-1]:
            if stop:
                return None
            else:
                raise ValueError("Train ID {} is after this run (ends at {})"
                                 .format(tid, dataset.train_ids[-1]))
        else:
            # This train ID is within the run, but doesn't have an entry
            for ix, tid_cmp in enumerate(dataset.train_ids):
                if tid_cmp > tid:
                    return ix

    raise Exception("Shouldn't reach here. tid={}".format(tid))


def _normalize_data_selection(selection, dataset):
    """Normalize selected data fields

    We offer different ways to select sources and keys from the data.
    This function normalises the different inputs to a set containing
    (source, key) tuples.

    dataset is meant to be a H5File or RunDirectory object
    """
    if isinstance(selection, set):
        return selection

    res = set()
    if isinstance(selection, dict):
        # {source: {key1, key2}}
        # {source: {}} -> all keys for this source
        for source, keys in selection.items():#
            if source not in (dataset.control_sources | dataset.instrument_sources):
                raise ValueError("Source {} not in this run".format(source))

            for k in (keys or dataset._keys_for_source(source)):
                res.add((source, k))

    elif isinstance(selection, list):
        # [('src_glob', 'key_glob'), ...]
        for src_glob, key_glob in selection:
            matched = set()
            for source in (dataset.control_sources | dataset.instrument_sources):
                if not fnmatchcase(source, src_glob):
                    continue

                for key in dataset._keys_for_source(source):
                    if fnmatchcase(key, key_glob):
                        matched.add((source, key))

            if not matched:
                raise ValueError("No matches for pattern {}"
                                 .format((src_glob, key_glob)))

            res.update(matched)
    else:
        TypeError("Unknown selection type: {}".format(type(selection)))

    return res


class SourceNameError(KeyError):
    def __init__(self, source, run=True):
        self.source = source
        self.run = run

    def __str__(self):
        run_file = 'run' if self.run else 'file'
        return "This {0} has no source named {1!r}.\n" \
               "See {0}.all_sources for available sources.".format(
            run_file, self.source
        )

class PropertyNameError(KeyError):
    def __init__(self, prop, source):
        self.prop = prop
        self.source = source

    def __str__(self):
        return "No property {!r} for source {!r}".format(self.prop, self.source)


class H5File:
    """Access an HDF5 file generated at European XFEL.

    This class helps select data by train and by device ID, following the
    file layout defined for European XFEL data::

        h5f = H5File('/path/to/my/file.h5')
        for data, train_id, index in h5f.trains():
            value = data['device']['parameter']

    Parameters
    ----------
    path: str
        Path to the HDF5 file
    driver: str, optional
        Driver option for h5py. You should usually not set this.
        http://docs.h5py.org/en/latest/high/file.html#file-drivers

    Raises
    ------
    FileNotFoundError
        If the provided path is not a file
    ValueError
        If the path exists but is not an HDF5 file
    """
    def __init__(self, path, driver=None):
        self.path = path
        if not osp.isfile(path):
            raise FileNotFoundError(path)
        if not h5py.is_hdf5(path):
            raise ValueError('%s is not a valid HDF5 file.' % path)
        self.file = h5py.File(path, 'r', driver=driver)

        self.metadata = self.file[METADATA]
        self.index = self.file[INDEX_DATA]
        self.run = self.file[RUN_DATA]

        self.sources = [source.decode() for source in
                        self.metadata['dataSourceId'].value if source]
        self.control_sources = set()
        self.instrument_sources = set()
        for src in self.sources:
            category, device, _ = self._parse_data_src(src)
            if category == 'CONTROL':
                self.control_sources.add(device)
            elif category == 'INSTRUMENT':
                self.instrument_sources.add(device)
            else:
                raise ValueError("Unknown data category %r" % category)

        self.train_ids = [tid for tid in self.index['trainId'][()].tolist()
                          if tid != 0]
        self.train_indices = {tid: idx for idx, tid in enumerate(self.train_ids)}

        self._index_cache = {}

    @property
    def all_sources(self):
        return self.control_sources | self.instrument_sources

    @staticmethod
    def _parse_data_src(source):
        # Device names for INSTRUMENT data in the file include the top level
        # group (e.g. "FXE_DET_LPD1M-1/DET/15CH0:xtdf/image"). We want to
        # separate this out ("FXE_DET_LPD1M-1/DET/15CH0:xtdf", "image")
        category, _, h5_device = source.partition('/')
        if category == 'INSTRUMENT':
            device, _, chan_grp = h5_device.partition(':')
            chan, _, group = chan_grp.partition('/')
            return category, device + ':' + chan, group
        else:
            return category, h5_device, ''

    def _keys_for_source(self, source):
        res = set()
        def add_key(key, value):
            if isinstance(value, h5py.Dataset):
                res.add(key.replace('/', '.'))

        if source in self.control_sources:
            self.file['/CONTROL/' + source].visititems(add_key)
        elif source in self.instrument_sources:
            self.file['/INSTRUMENT/' + source].visititems(add_key)
        else:
            raise KeyError("Source {} not in file".format(source))
        return res

    def _check_data_missing(self, selection, tid):
        missing = set()
        for source, key in selection:
            if source in self.instrument_sources:
                key_head, _, h5_key = key.partition('.')
                h5_source = source + '/' + key_head
            elif source in self.control_sources:
                h5_source = source
            else:
                missing.add((source, key))
                continue

            _, count = self._read_index(h5_source, self.train_indices[tid])
            if count < 1:
               missing.add((source, key))

        return missing

    def _filter_selection(self, selection=None):
        """Filter sources in this file from selected data for a run.
        """
        if selection is None:
            return None

        return {(src, key) for (src, key) in selection
                if src in (self.instrument_sources | self.control_sources)}

    def _gen_train_data(self, train_index, only_this=None):
        """Get data for the specified index in file.
        """
        train_data = defaultdict(dict)

        train_id = self.train_ids[train_index]

        if only_this is not None:
            for source, key in only_this:
                h5_source, h5_key = source, key
                if source in self.instrument_sources:
                    key_head, _, h5_key = key.partition('.')
                    h5_source = source + '/' + key_head
                    path = '/INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
                else:
                    path = '/CONTROL/{}/{}'.format(source, key.replace('.', '/'))

                # Which parts of the data to get for this train:
                first, count = self._read_index(h5_source, train_index)

                if not count:
                    # No data here
                    continue

                ds = self.file[path]
                if count == 1:
                    data = ds[first]
                else:
                    data = ds[first:first + count, ]
                train_data[source][key] = data

                train_data[source]['metadata'] = {
                    'source': source,
                    'timestamp.tid': train_id,
                }
            return train_id, train_data

        for source in self.sources:
            h5_source = source.split('/', 1)[1]
            index = self.index[h5_source]
            table = self.file[source]

            # Which parts of the data to get for this train:
            first, count = self._read_index(h5_source, train_index)

            category, device, path_base = self._parse_data_src(source)

            if not count:
                continue

            data = train_data[device]

            def append_data(key, value):
                if isinstance(value, h5py.Dataset):
                    path = '.'.join(filter(None,
                                    (path_base,) + tuple(key.split('/'))))

                    if count == 1:
                        data[path] = value[first]
                    else:
                        data[path] = value[first:first + count, ]

            table.visititems(append_data)


            train_data[device]['metadata'] = {
                'source': source,
                'timestamp.tid': train_id,
            }

        return train_id, train_data

    def trains(self, devices=None, train_range=None, *, require_all=False):
        """Iterate over all trains in the file.

        Parameters
        ----------
        devices: dict or list, optional
            Filter data by sources and by parameters.
            There are two ways to do this:

            1. With a list of (source, key) glob patterns::

                f.trains([('*/DET/*', 'image.*')])

            2. With a dict of source names mapped to sets of key names
               (or empty sets to get all keys)::

                dev = {
                    'device1': {'param_m', 'param_n.subparam'},
                    'device2': set(),
                }
                for tid, data in handler.trains(devices=dev):
                    ...

        train_range: by_id or by_index object, optional
            Iterate over only selected trains, by train ID or by index::

                f.trains(train_range=by_index[20:])

        require_all: bool
            False (default) returns any data available for the requested trains.
            True skips trains which don't have all the requested data;
            this requires that you specify required data using *devices*.

        Examples
        --------

        >>> h5file = H5File('r0450/RAW-R0450-DA01-S00000.h5')

        Iterate over all trains

        >>> for id, data in h5file.trains():
                pos = data['device_x']['param_n']

        Filter devices and parameters

        >>> dev = {'xray_monitor': {'pulseEnergy', 'beamPosition'},
        ...        'sample_x': {}, 'sample_y': {}}
        >>> trains = h5file.trains(devices=dev)
        >>> traind_id, train_1 = next(trains)
        >>> train_1.keys()
        dict_keys(['xray_monitor', 'sample_x', 'sample_y'])

        The returned data will contains the devices 'xray_monitor' and 2 of
        it's parameters (pulseEnergy and beamPosition), sample_x and
        sample_y (with all of their parameters). All other devices are ignored.
        """
        if isinstance(train_range, by_id):
            start_ix = _tid_to_slice_ix(train_range.value.start, self, stop=False)
            stop_ix = _tid_to_slice_ix(train_range.value.stop, self, stop=True)
            ix_slice = slice(start_ix, stop_ix, train_range.value.step)
        elif isinstance(train_range, by_index):
            ix_slice = train_range.value
        elif train_range is None:
            ix_slice = slice(None, None)
        else:
            raise TypeError(train_range)

        if devices:
            devices = _normalize_data_selection(devices, self)
        elif require_all:
            raise ValueError("Cannot skip partial data without devices= parameter")

        for tid in self.train_ids[ix_slice]:
            if require_all and self._check_data_missing(devices, tid):
                continue

            index = self.train_indices[tid]
            yield self._gen_train_data(index, only_this=devices)

    def train_from_id(self, train_id, devices=None):
        """Get Train data for specified train ID.

        Parameters
        ----------
        train_id: int
            The train ID
        devices: dict or list, optional
            Filter data by devices and by parameters.

            Refer to :meth:`~.H5File.trains` for how to use this.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name

        Raises
        ------
        KeyError
            if `train_id` is not found in the file.
        """
        if devices is not None:
            devices = _normalize_data_selection(devices, self)

        try:
            index = self.train_indices[train_id]
        except KeyError:
            raise KeyError("train {} not found in {}.".format(
                            train_id, self.file.filename))
        else:
            return self._gen_train_data(index, only_this=devices)

    def train_from_index(self, index, devices=None):
        """Get train data of the nth train in file.

        Parameters
        ----------
        index: int
            Index of the train in the file.
        devices: dict or list, optional
            Filter data by devices and by parameters.

            Refer to :meth:`~.H5File.trains` for how to use this.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name
        """
        if devices is not None:
            devices = _normalize_data_selection(devices, self)

        return self._gen_train_data(index, only_this=devices)

    @staticmethod
    def _make_field_name(device, key):
        name = device + '/' + key
        if name.endswith('.value'):
            name = name[:-6]
        return name

    @staticmethod
    def _field_match(device, key, patterns):
        if key.endswith('.value'):
            key = key[:-6]
        return any(fnmatchcase(device, p[0]) and fnmatchcase(key, p[1])
                   for p in patterns)

    def _check_field(self, source, key):
        if source not in self.all_sources:
            raise SourceNameError(source, run=False)
        if key not in self._keys_for_source(source):
            raise PropertyNameError(key, source)

    def _read_index(self, h5_source, train_ix=slice(None, None)):
        """Get first index & count for a source and for a specific train ID.

        Indices are cached; this appears to provide some performance benefit.
        """
        try:
            first, count = self._index_cache[h5_source]
        except KeyError:
            first, count = self._read_index_real(h5_source)
            self._index_cache[h5_source] = (first, count)
        return first[train_ix], count[train_ix]

    def _read_index_real(self, h5_source):
        """Get first index & count for a source.

        This is 'real' reading when the requested index is not in the cache.
        """
        ix_group = self.index[h5_source]
        first = ix_group['first'][:]
        if 'count' in ix_group:
            count = ix_group['count'][:]
        else:
            status = ix_group['status'][:]
            count = np.uint64((ix_group['last'][:] - first + 1) * status)
        return first, count

    def _index_to_trainids(self, ix_group, check_unique=True):
        if 'count' in ix_group:
            count = ix_group['count'][:]
        else:
            first = ix_group['first'][:]
            count = (ix_group['last'][:] - first + 1) * ix_group['status'][:]

        if check_unique and (count > 1).any():
            data_src = ix_group.name.split('INDEX/', 1)[1]
            raise ValueError("%s data has more than one data point per train" % data_src)

        trainId = self.index['trainId'][:]
        n = min(len(count), len(trainId))

        res = np.repeat(trainId[:n], count.astype(np.intp)[:n])

        # The output should contain valid train IDs, without zeroes.
        # If not, something has gone wrong, and it needs to be debugged.
        if (res == 0).any():
            data_src = ix_group.name.split('INDEX/', 1)[1]
            raise ValueError("Error calculating train IDs for %s: 0 in index"
                             % data_src)

        return res

    def get_series(self, device, key):
        """Return a pandas Series for a particular data field.

        Parameters
        ----------

        device: str
            Device name with optional output channel, e.g.
            "SA1_XTD2_XGM/DOOCS/MAIN" or "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "beamPosition.iyPos.value"
            or "header.linkId". The data must be 1D in the file.
        """
        self._check_field(device, key)
        name = self._make_field_name(device, key)

        # Find the data
        if ':' in device:  # INSTRUMENT data
            keyhead, _, key = key.partition('.')
            device += '/' + keyhead
            data_src = 'INSTRUMENT/' + device
        else:
            data_src = 'CONTROL/' + device
        data_path = "/{}/{}".format(data_src, key.replace('.', '/'))
        ds = self.file[data_path]

        # Get the index
        if data_src.startswith('CONTROL'):
            index_ds = self.index['trainId']
            index = pd.Index(index_ds[index_ds[:] != 0], name='trainId')
            data = ds[:len(index)]
        elif data_src.startswith('INSTRUMENT'):
            trainids = self._index_to_trainids(self.index[device],
                                               check_unique=False)
            index = pd.Index(trainids, name='trainId')
            data = ds[:]
            if not index.is_unique:
                pulse_id = self.file['/{}/pulseId'.format(data_src)]
                pulse_id = pulse_id[:len(index), 0]
                index = pd.MultiIndex.from_arrays([trainids, pulse_id],
                                                  names=['trainId', 'pulseId'])
                # Does pulse-oriented data always have an extra dimension?
                assert data.shape[1] == 1
                data = data[:, 0]
            data = data[:len(index)]
        else:
            raise ValueError("Unknown data source %r" % data_src)

        return pd.Series(data, name=name, index=index)

    def get_dataframe(self, fields=(('*', '*'),), *, timestamps=False):
        """Return a pandas dataframe for given data fields.

        Parameters
        ----------
        fields : list of 2-tuples
            Glob patterns to match device and field names, e.g.
            ``("*_XGM/*", "*.i[xy]Pos")`` matches ixPos and iyPos from any XGM devices.
            By default, all fields from all control devices are matched.
        timestamps : bool
            If false (the default), exclude the timestamps associated with each
            control data field.
        """
        if isinstance(fields, str):
            fields = [fields]

        control_series = []
        for dev in self.control_sources:
            def append_ctrl_data(key, value):
                if (not timestamps) and key.endswith('/timestamp'):
                    return
                if isinstance(value, h5py.Dataset):
                    key = key.replace('/', '.')
                    if self._field_match(dev, key, fields):
                        control_series.append(self.get_series(dev, key))
            self.file['/CONTROL/' + dev].visititems(append_ctrl_data)

        if not control_series:
            return None
        return pd.concat(control_series, axis=1)

    def get_array(self, device, key, extra_dims=None):
        """Return a labelled array for a particular data field.

        The first axis of the returned data will be the train IDs.
        Datasets which are per-pulse in the first dimension are not supported.

        Parameters
        ----------

        device: str
            Device name with optional output channel, e.g.
            "SA1_XTD2_XGM/DOOCS/MAIN" or "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "beamPosition.iyPos.value"
            or "header.linkId". The data must be 1D in the file.
        extra_dims: list of str
            Name extra dimensions in the array. The first dimension is
            automatically called 'train'. The default for extra dimensions
            is dim_0, dim_1, ...
        """
        self._check_field(device, key)
        name = self._make_field_name(device, key)

        # Find the data
        if ':' in device:  # INSTRUMENT data
            keyhead, _, key = key.partition('.')
            device += '/' + keyhead
            data_src = 'INSTRUMENT/' + device
        else:
            data_src = 'CONTROL/' + device
        data_path = "/{}/{}".format(data_src, key.replace('.', '/'))
        ds = self.file[data_path]

        # Get the index
        if data_src.startswith('CONTROL'):
            index_ds = self.index['trainId']
            trainids = index_ds[index_ds[:] != 0]
            data = ds[:len(trainids), ...]
        elif data_src.startswith('INSTRUMENT'):
            trainids = self._index_to_trainids(self.index[device])
            data = ds[:len(trainids), ...]
        else:
            raise ValueError("Unknown data source %r" % data_src)

        if extra_dims is None:
            extra_dims = ['dim_%d' % i for i in range(data.ndim - 1)]
        dims = ['trainId'] + extra_dims
        return xr.DataArray(data, dims=dims, coords={'trainId': trainids})

    def close(self):
        self.file.close()

    # Context manager protocol - enables "with H5File(...):"
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def detector_info(self):
        """Get statistics about the detector data.

        Returns a dictionary with keys:
        - 'dims' (pixel dimensions)
        - 'frames_per_train'
        - 'total_frames'
        """

        img_source = [src for src in self.sources
                      if re.match(r'INSTRUMENT/.+/image', src)][0]
        img_ds = self.file[img_source + '/data']
        img_index = self.index[img_source.split('/', 1)[1]]

        if 'last' in img_index:
            # Older (?) format: status (0/1), first, last
            count = img_index['last'][:] + 1 - img_index['first'][:]
            count[img_index['status'][:] == 0] = 0
        else:
            # Newer (?) format: first, count
            count = img_index['count'][:]

        return {
            'dims': img_ds.shape[-2:],
            # Some trains have 0 frames; max is the interesting value
            'frames_per_train': count.max(),
            'total_frames': count.sum(),
        }


class RunDirectory:
    """Access data from a 'run' generated at European XFEL.

    A 'run' is a directory containing a number of HDF5 files with data from the
    same time period. This class can read data from the collection of files,
    selected by train and by device.

    Parameters
    ----------
    path: str
        Path to the run directory.
    """
    def __init__(self, path):
        self.files = [H5File(f) for f in glob(osp.join(path, '*.h5'))
                      if h5py.is_hdf5(f)]

        self._trains = {}
        for fhandler in self.files:
            for train in fhandler.train_ids:
                if train not in self._trains:
                    self._trains[train] = []
                self._trains[train].append(fhandler)

        self.ordered_trains = list(sorted(self._trains.items()))
        self.train_indices = {tid: idx for idx, tid in enumerate(self.train_ids)}

    @property
    def train_ids(self):
        return [x[0] for x in self.ordered_trains]

    @property
    def control_sources(self):
        r = set()
        for f in self.files:
            r.update(f.control_sources)
        return r

    @property
    def instrument_sources(self):
        r = set()
        for f in self.files:
            r.update(f.instrument_sources)
        return r

    @property
    def all_sources(self):
        return self.control_sources | self.instrument_sources

    def _check_field(self, source, key):
        if source not in self.all_sources:
            raise SourceNameError(source, run=True)
        if key not in self._keys_for_source(source):
            raise PropertyNameError(key, source)

    def _keys_for_source(self, source):
        # The same source may be in multiple files, but this assumes it has
        # the same keys in all files that it appears in.
        for f in self.files:
            if source in (f.control_sources | f.instrument_sources):
                return f._keys_for_source(source)

        raise ValueError("No keys found for source {}".format(source))

    def _check_data_missing(self, selection, tid, fhs):
        missing = selection.copy()
        for file in fhs:
            missing = file._check_data_missing(missing, tid)
        return missing

    def trains(self, devices=None, train_range=None, *, require_all=False):
        """Iterate over all trains in the run and gather all sources.

        ::

            run = Run('/path/to/my/run/r0123')
            for train_id, data in run.trains():
                value = data['device']['parameter']

        Parameters
        ----------
        devices: dict or list, optional
            Filter data by devices and by parameters.

            Refer to :meth:`H5File.trains` for how to use this.

        train_range: by_id or by_index object, optional
            Iterate over only selected trains, by train ID or by index::

                f.trains(train_range=by_index[20:])

        require_all: bool
            False (default) returns any data available for the requested trains.
            True skips trains which don't have all the requested data;
            this requires that you specify required data using *devices*.

        Yields
        ------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name
        """
        if isinstance(train_range, by_id):
            start_ix = _tid_to_slice_ix(train_range.value.start, self, stop=False)
            stop_ix = _tid_to_slice_ix(train_range.value.stop, self, stop=True)
            ix_slice = slice(start_ix, stop_ix, train_range.value.step)
        elif isinstance(train_range, by_index):
            ix_slice = train_range.value
        elif train_range is None:
            ix_slice = slice(None, None)
        else:
            raise TypeError(train_range)

        if devices:
            devices = _normalize_data_selection(devices, self)
        elif require_all:
            raise ValueError("Cannot skip partial data without devices= parameter")

        for tid, fhs in self.ordered_trains[ix_slice]:
            if require_all and self._check_data_missing(devices, tid, fhs):
                continue

            train_data = {}
            for fh in fhs:
                file_selection = fh._filter_selection(devices)
                _, data = fh.train_from_id(tid, devices=file_selection)
                train_data.update(data)

            yield (tid, train_data)

    def train_from_id(self, train_id, devices=None):
        """Get Train data for specified train ID.

        Parameters
        ----------
        train_id: int
            The train ID
        devices: dict or list, optional
            Filter data by devices and by parameters.

            Refer to :meth:`H5File.trains` for how to use this.

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
        try:
            files = self._trains[train_id]
        except KeyError:
            raise KeyError("train {} not found in run.".format(train_id))

        if devices is not None:
            devices = _normalize_data_selection(devices, self)

        data = {}
        for fh in files:
            file_selection = fh._filter_selection(devices)
            _, d = fh.train_from_id(train_id, devices=file_selection)
            data.update(d)
        return (train_id, data)

    def train_from_index(self, index, devices=None):
        """Get the nth train in the run.

        Parameters
        ----------
        index: int
            The train index within this run
        devices: dict or list, optional
            Filter data by devices and by parameters.

            Refer to :meth:`H5File.trains` for how to use this.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name

        Raises
        ------
        IndexError
            if train `index` is out of range.
        """
        try:
            train_id, files = self.ordered_trains[index]
        except IndexError:
            raise IndexError("Train index {} out of range.".format(index))

        if devices is not None:
            devices = _normalize_data_selection(devices, self)

        data = {}
        for fh in files:
            file_selection = fh._filter_selection(devices)
            _, d = fh.train_from_id(train_id, devices=file_selection)
            data.update(d)
        return (train_id, data)

    def get_series(self, device, key):
        """Return a pandas Series for a particular data field.

        Parameters
        ----------

        device: str
            Device name with optional output channel, e.g.
            "SA1_XTD2_XGM/DOOCS/MAIN" or "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "beamPosition.iyPos.value"
            or "header.linkId". The data must be 1D in the file.
        """
        self._check_field(device, key)
        seq_series = [f.get_series(device, key) for f in self.files
                      if device in (f.control_sources | f.instrument_sources)]

        return pd.concat(sorted(seq_series, key=lambda s: s.index[0]))

    def get_dataframe(self, fields=(('*', '*'),), *, timestamps=False):
        """Return a pandas Dataframe for the 1D, train-oriented data in this run

        Parameters
        ----------
        fields : list of 2-tuples
            Glob patterns to match device and field names, e.g.
            ``("*_XGM/*", "*.i[xy]Pos")`` matches ixPos and iyPos from any XGM devices.
            By default, all fields from all control devices are matched.
        timestamps : bool
            If false (the default), exclude the timestamps associated with each
            control data field.
        """
        group_dfs = []
        for _, files in self._assemble_sequences().items():
            file_dfs = []
            for f in files:
                df = f.get_dataframe(fields=fields, timestamps=timestamps)
                if df is not None:
                    file_dfs.append(df)
            if file_dfs:
                group_dfs.append(pd.concat(file_dfs))
        return pd.concat(group_dfs, axis=1)

    def get_array(self, device, key, extra_dims=None):
        """Return a labelled array for a particular data field.

        The first axis of the returned data will be the train IDs.
        Datasets which are per-pulse in the first dimension are not supported.

        Parameters
        ----------

        device: str
            Device name with optional output channel, e.g.
            "SA1_XTD2_XGM/DOOCS/MAIN" or "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "beamPosition.iyPos.value"
            or "header.linkId". The data must be 1D in the file.
        extra_dims: list of str
            Name extra dimensions in the array. The first dimension is
            automatically called 'train'. The default for extra dimensions
            is dim_0, dim_1, ...
        """
        self._check_field(device, key)
        seq_arrays = [f.get_array(device, key, extra_dims=extra_dims)
                      for f in self.files
                      if device in (f.control_sources | f.instrument_sources)]

        non_empty = [a for a in seq_arrays if (a.size > 0)]
        if not non_empty:
            if seq_arrays:
                # All per-file arrays are empty, so just return the first one.
                return seq_arrays[0]

            raise Exception(("Unable to get data for source {!r}, key {!r}. "
                             "Please report an issue so we can investigate")
                            .format(device, key))

        return xr.concat(sorted(non_empty, key=lambda a: a.coords['trainId'][0]),
                         dim='trainId')

    def _assemble_sequences(self):
        """Assemble the sequences for each data recorder.

        Returns a dict keyed by filename prefix, with ordered lists of the
        H5File objects (...-S00000.h5, ...-S00001.h5, etc.)
        """
        segment_sequences = defaultdict(list)
        for f in sorted(self.files, key=lambda f: osp.basename(f.file.filename)):
            m = re.match(r'(.+)-S\d+\.h5', osp.basename(f.file.filename))
            if not m:
                raise ValueError("Unrecognised filename: %s" % f.file.filename)
            segment_sequences[m.group(1)].append(f)
        return dict(segment_sequences)

    def _get_sources(self, src):
        """Return sets of control and instrument source names.
        control: train data
        instrument: pulse data
        """
        ctrl, inst = set(), set()
        for file in src:
            ctrl.update(file.control_sources)
            inst.update(file.instrument_sources)
        return ctrl, inst

    def info(self):
        """Show information about the run.
        """
        # time info
        first_train, _ = self.ordered_trains[0]
        last_train, _ = self.ordered_trains[-1]
        train_count = len(self.ordered_trains)
        span_sec = (last_train - first_train) / 10
        span_txt = str(datetime.timedelta(seconds=span_sec))

        detector_files, non_detector_files = [], []
        detector_modules = defaultdict(list)
        for f in self.files:
            fni = FilenameInfo(f.path)
            if fni.is_detector:
                detector_files.append(f)
                detector_modules[(fni.detector_name, fni.detector_moduleno)].append(f)
            else:
                non_detector_files.append(f)

        # A run should only have one detector, but if that changes, don't hide it
        detector_name = ','.join(sorted(set(k[0] for k in detector_modules)))

        # devices info
        ctrl, inst = self._get_sources(non_detector_files)

        # disp
        print('# of trains:   ', train_count)
        print('Duration:      ', span_txt)
        print('First train ID:', first_train)
        print('Last train ID: ', last_train)
        print()

        print("{} detector modules ({})".format(
            len(detector_modules), detector_name
        ))
        if len(detector_modules) > 0:
            # Show detail on the first module (the others should be similar)
            mod_key = sorted(detector_modules)[0]
            mod_files = detector_modules[mod_key]
            dinfo = [f.detector_info() for f in mod_files]
            module = ''.join(mod_key)
            dims = ' x '.join(str(d) for d in dinfo[0]['dims'])
            print("  e.g. module {} : {} pixels".format(module, dims))
            print("  {} frames per train, {} total frames".format(
                max(i['frames_per_train'] for i in dinfo),
                sum(i['total_frames'] for i in dinfo),
            ))
        print()

        print(len(inst), 'instrument sources (excluding detectors):')
        for d in sorted(inst):
            print('  -', d)
        print()
        print(len(ctrl), 'control sources:')
        for d in sorted(ctrl):
            print('  -', d)
        print()

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
        tid, files = next((t for t in self.ordered_trains
                          if t[0] == train_id), (None, None))
        if tid is None:
            raise ValueError("train {} not found in run.".format(train_id))
        ctrl, inst = self._get_sources(files)

        # disp
        print('Train [{}] information'.format(train_id))
        print('Devices')
        print('\tInstruments')
        [print('\t-', d) for d in sorted(inst)] or print('\t-')
        print('\tControls')
        [print('\t-', d) for d in sorted(ctrl)] or print('\t-')

# RunDirectory was previously RunHandler; we'll leave it accessible in case
# any code was already using it.
RunHandler = RunDirectory

def stack_data(train, data, axis=-3, xcept=()):
    """Stack data from devices in a train.

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
    devs = [(list(map(int, re.findall(r'\d+', dev))), dev)
            for dev in train.keys() if dev not in xcept]
    devices = [dev for _, dev in sorted(devs)]

    dtype, shape = next(((d[data].dtype, d[data].shape) for d in train.values()
                        if data in d and 0 not in d[data].shape), (None, None))
    if dtype is None or shape is None:
        return np.empty(0)

    combined = np.zeros((len(devices),) + shape, dtype=dtype)
    for index, device in enumerate(devices):
        try:
            if 0 in train[device][data].shape:
                continue
            combined[index, ] = train[device][data]
        except KeyError:
            print('stack_data(): missing {} in {}'.format(data, device))
    return np.moveaxis(combined, 0, axis)


def stack_detector_data(train, data, axis=-3, modules=16, only='', xcept=()):
    """Stack data from detector modules in a train.

    Parameters
    ----------
    train: dict
        Train data.
    data: str
        The path to the device parameter of the data you want to stack.
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

    dtype, shape = next(((d[data].dtype, d[data].shape) for d in train.values()
                        if data in d and 0 not in d[data].shape), (None, None))
    if dtype is None or shape is None:
        return np.array([])

    combined = np.full((modules, ) + shape, np.nan, dtype=dtype)
    for device in devices:
        index = None
        try:
            if 0 in train[device][data].shape:
                continue
            index = int(re.findall(r'\d+', device)[-2])
            combined[index, ] = train[device][data]
        except KeyError:
            print('stack_detector_data(): missing {} in {}'.format(data, device))
        except IndexError:
            print('stack_detector_Data(): module {} is out or range for a'
                  'detector of {} modules'.format(index, modules))
        except ValueError:
            print('stack_detector_Data(): inconsistent data shape of {} in {}: '
                  'found both {} and {}'.format(data,
                                                device,
                                                combined.shape[1:],
                                                train[device][data].shape))
    return np.moveaxis(combined, 0, axis)


if __name__ == '__main__':
    r = RunDirectory('./data/r0185')
    for tid, d in r.trains():
        print(tid)
