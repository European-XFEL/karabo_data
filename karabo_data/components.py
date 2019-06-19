"""Interfaces to data from specific instruments
"""
import logging
import numpy as np
import pandas as pd
import re
import xarray

from .exceptions import SourceNameError
from .reader import DataCollection, by_id, by_index

log = logging.getLogger(__name__)

MAX_PULSES = 2700


def _guess_axes(data, train_pulse_ids):
    # Raw files have a spurious extra dimension
    if data.ndim >= 2 and data.shape[1] == 1:
        data = data[:, 0]

    # TODO: this assumes we can tell what the axes are just from the
    # number of dimensions. Works for the data we've seen, but we
    # should look for a more reliable way.
    if data.ndim == 4:
        # image.data in raw data
        dims = ['train_pulse', 'data_gain', 'slow_scan', 'fast_scan']
    elif data.ndim == 3:
        # image.data, image.gain, image.mask in calibrated data
        dims = ['train_pulse', 'slow_scan', 'fast_scan']
    else:
        # Everything else seems to be 1D
        dims = ['train_pulse']

    arr = xarray.DataArray(data, {'train_pulse': train_pulse_ids}, dims=dims)

    # Separate train & pulse dimensions, and arrange dimensions
    # so that the data is contiguous in memory.
    dim_order = ['train', 'pulse'] + dims[1:]
    return arr.unstack('train_pulse').transpose(*dim_order)


def _check_pulse_selection(pulses):
    """Check and normalise a pulse selection"""
    if not isinstance(pulses, (by_id, by_index)):
        raise TypeError("pulses selection should be by_id or by_index object")

    val = pulses.value

    if isinstance(pulses.value, slice):
        # Ensure start/stop/step are all real numbers
        start = val.start if (val.start is not None) else 0
        stop = val.stop if (val.stop is not None) else MAX_PULSES
        step = val.step if (val.step is not None) else 1

        if not all(isinstance(s, int) for s in (start, stop, step)):
            raise TypeError("Pulse selection slice must use integers or None")
        if step < 1:
            raise ValueError("Pulse selection slice must have positive step")
        if (start < 0) or (stop < 0):
            raise NotImplementedError("Negative pulse indices not supported")

        return type(pulses)(slice(start, stop, step))

    # Convert everything except slices to numpy arrays
    elif isinstance(pulses.value, int):
        val = np.array([val], dtype=np.uint64)
    else:
        val = np.asarray(val, dtype=np.uint64)

    if (val < 0).any():
        if isinstance(pulses, by_id):
            raise ValueError("Pulse IDs cannot be negative")
        else:
            raise NotImplementedError("Negative pulse indices not supported")

    return type(pulses)(val)


class MPxDetectorBase:
    """Base class for megapixel detectors (AGIPD, LPD)
    """

    _source_re = re.compile(r'(?P<detname>.+)/DET/(\d+)CH')
    n_modules = 16
    # Override in subclass
    module_shape = (0, 0)

    def __init__(self, data: DataCollection, detector_name=None, modules=None,
                 *, min_modules=1):
        if detector_name is None:
            detector_name = self._find_detector_name(data)

        source_to_modno = self._identify_sources(data, detector_name, modules)

        data = data.select([(src, '*') for src in source_to_modno])
        self.detector_name = detector_name
        self.source_to_modno = source_to_modno

        # pandas' missing-data handling converts the data to floats if there
        # are any gaps - so fill them with 0s and convert back to uint64.
        mod_data_counts = pd.DataFrame({
            src: data.get_data_counts(src, 'image.data')
            for src in source_to_modno
        }).fillna(0).astype(np.uint64)

        # Sanity check: all non-zero counts should be equal
        data_count_values = set(mod_data_counts.values.flatten()) - {0}
        if len(data_count_values) > 1:
            raise ValueError("Inconsistent frame counts: {}"
                             .format(data_count_values))
        elif not data_count_values:
            raise ValueError("No data found for selected sources")

        self.frames_per_train = data_count_values.pop()
        self.data = self._select_trains(data, mod_data_counts, min_modules)

        # This should be a reversible 1-to-1 mapping
        self.modno_to_source = {m: s for (s, m) in source_to_modno.items()}
        assert len(self.modno_to_source) == len(self.source_to_modno)

        train_id_arr = np.asarray(self.data.train_ids)
        split_indices = np.where(np.diff(train_id_arr) != 1)[0] + 1
        self.train_id_chunks = np.split(train_id_arr, split_indices)

    @classmethod
    def _find_detector_name(cls, data):
        detector_names = set()
        for source in data.instrument_sources:
            m = cls._source_re.match(source)
            if m:
                detector_names.add(m.group(1))
        if not detector_names:
            raise SourceNameError(cls._source_re.pattern)
        elif len(detector_names) > 1:
            raise ValueError(
                "Multiple detectors found in the data: {}. "
                "Pass a name to data.detector() to pick one.".format(
                    ', '.join(repr(n) for n in detector_names)
                )
            )
        return detector_names.pop()

    @staticmethod
    def _identify_sources(data, detector_name=None, modules=None):
        detector_re = re.compile(re.escape(detector_name) + r'/DET/(\d+)CH')
        source_to_modno = {}
        for source in data.instrument_sources:
            m = detector_re.match(source)
            if m:
                source_to_modno[source] = int(m.group(1))

        if modules is not None:
            source_to_modno = {s: n for (s, n) in source_to_modno.items()
                               if n in modules}

        if not source_to_modno:
            raise SourceNameError(detector_re.pattern)

        return source_to_modno

    @classmethod
    def _select_trains(cls, data, mod_data_counts, min_modules):
        modules_present = (mod_data_counts > 0).sum(axis=1)
        mod_data_counts = mod_data_counts[modules_present >= min_modules]
        ntrains = len(mod_data_counts)
        if not ntrains:
            raise ValueError("No data found with >= {} modules present"
                             .format(min_modules))
        log.info("Found %d trains with data for at least %d modules",
                 ntrains, min_modules)
        train_ids = mod_data_counts.index.values
        return data.select_trains(by_id[train_ids])

    @property
    def train_ids(self):
        return self.data.train_ids

    def __repr__(self):
        return "<{}: Data interface for detector {!r} with {} modules>".format(
            type(self).__name__, self.detector_name, len(self.source_to_modno),
        )

    @staticmethod
    def _select_pulse_ids(pulses, data_pulse_ids):
        """Select pulses by ID across a chunk of trains

        Returns an array or slice of the indexes to include.
        """
        if isinstance(pulses.value, slice):
            if pulses.value == slice(0, MAX_PULSES, 1):
                # All pulses included
                return slice(0, len(data_pulse_ids))
            else:
                s = pulses.value
                desired = np.arange(s.start, s.stop, step=s.step, dtype=np.uint64)
        else:
            desired = pulses.value

        return np.nonzero(np.isin(data_pulse_ids, desired))[0]

    @staticmethod
    def _select_pulse_indices(pulses, firsts, counts):
        """Select pulses by index across a chunk of trains

        Returns an array or slice of the indexes to include.
        """
        if isinstance(pulses.value, slice):
            if pulses.value == slice(0, MAX_PULSES, 1):
                # All pulses included
                return slice(0, counts.sum())
            else:
                s = pulses.value
                desired = np.arange(s.start, s.stop, step=s.step, dtype=np.uint64)
        else:
            desired = pulses.value

        positions = []
        for first, count in zip(firsts, counts):
            train_desired = desired[desired < count]
            positions.append(first + train_desired)

        return np.concatenate(positions)

    def _get_module_pulse_data(self, source, key, pulses):
        seq_arrays = []
        data_path = "/INSTRUMENT/{}/{}".format(source, key.replace('.', '/'))
        for f in self.data._source_index[source]:
            group = key.partition('.')[0]
            firsts, counts = f.get_index(source, group)

            for chunk_tids in self.train_id_chunks:
                if chunk_tids[-1] < f.train_ids[0] or chunk_tids[0] > f.train_ids[-1]:
                    # No overlap
                    continue
                first_tid = max(chunk_tids[0], f.train_ids[0])
                first_train_idx = np.nonzero(f.train_ids == first_tid)[0][0]
                last_tid = min(chunk_tids[-1], f.train_ids[-1])
                last_train_idx = np.nonzero(f.train_ids == last_tid)[0][0]
                chunk_firsts = firsts[first_train_idx : last_train_idx + 1]
                chunk_counts = counts[first_train_idx : last_train_idx + 1]
                data_slice = slice(
                    chunk_firsts[0], int(chunk_firsts[-1] + chunk_counts[-1])
                )
                trainids = np.repeat(
                    np.arange(first_tid, last_tid + 1, dtype=np.uint64),
                    chunk_counts.astype(np.intp),
                )
                pulse_id = f.file['/INSTRUMENT/{}/{}/pulseId'.format(source, group)][
                    data_slice
                ]
                # Raw files have a spurious extra dimension
                if pulse_id.ndim >= 2 and pulse_id.shape[1] == 1:
                    pulse_id = pulse_id[:, 0]

                if isinstance(pulses, by_id):
                    positions = self._select_pulse_ids(pulses, pulse_id)
                else:  # by_index
                    positions = self._select_pulse_indices(
                        pulses, chunk_firsts - data_slice.start, chunk_counts
                    )

                trainids = trainids[positions]
                pulse_id = pulse_id[positions]
                index = pd.MultiIndex.from_arrays(
                    [trainids, pulse_id], names=['train', 'pulse']
                )

                if isinstance(positions, slice):
                    data_positions = slice(
                        int(data_slice.start + positions.start),
                        int(data_slice.start + positions.stop),
                        positions.step
                    )
                else:  # ndarray
                    # h5py fancy indexing needs a list, not an ndarray
                    data_positions = list(data_slice.start + positions)
                    if data_positions == []:
                        # Work around a limitation of h5py
                        # https://github.com/h5py/h5py/issues/1169
                        data_positions = slice(0, 0)

                data = f.file[data_path][data_positions]

                arr = _guess_axes(data, index)

                seq_arrays.append(arr)

        non_empty = [a for a in seq_arrays if (a.size > 0)]
        if not non_empty:
            if seq_arrays:
                # All per-file arrays are empty, so just return the first one.
                return seq_arrays[0]

            raise Exception(
                "Unable to get data for source {!r}, key {!r}. "
                "Please report an issue so we can investigate"
                    .format(source, key)
            )

        return xarray.concat(
            sorted(non_empty, key=lambda a: a.coords['train'][0]), dim='train'
        )

    def get_array(self, key, pulses=by_index[:]):
        """Get a labelled array of detector data

        Parameters
        ----------

        key: str
          The data to get, e.g. 'image.data' for pixel values.
        pulses: by_id or by_index
          Select the pulses to include from each train. by_id selects by pulse
          ID, by_index by index within the data being read. The default includes
          all pulses. Only used for per-train data.
        """
        pulses = _check_pulse_selection(pulses)

        arrays = []
        modnos = []
        for modno, source in sorted(self.modno_to_source.items()):
            # At present, all the per-pulse data is stored in the 'image' key.
            # If that changes, this check will need to change as well.
            if key.startswith('image.'):
                arrays.append(self._get_module_pulse_data(source, key, pulses))
            else:
                arrays.append(self.data.get_array(source, key))
            modnos.append(modno)

        return xarray.concat(arrays, pd.Index(modnos, name='module'))

    def trains(self, pulses=by_index[:]):
        """Iterate over trains for detector data.

        Parameters
        ----------

        pulses: by_index or by_id
          Select which pulses to include for each train.
          The default is to include all pulses.

        Yields
        ------

        train_data: dict
          A dictionary mapping key names (e.g. ``image.data``) to labelled
          arrays.
        """
        pulses = _check_pulse_selection(pulses)
        return MPxDetectorTrainIterator(self, pulses)

    def write_virtual_cxi(self, filename):
        """Write a virtual CXI file to access the detector data.

        The virtual datasets in the file provide a view of the detector
        data as if it was a single huge array, but without copying the data.
        Creating and using virtual datasets requires HDF5 1.10.

        Parameters
        ----------
        filename: str
          The file to be written. Will be overwritten if it already exists.
        """
        from .write_cxi import VirtualCXIWriter
        VirtualCXIWriter(self).write(filename)


class MPxDetectorTrainIterator:
    """Iterate over trains in detector data, assembling arrays.

    Created by :meth:`DetectorData.trains`.
    """
    def __init__(self, data, pulses=by_index[:], require_all=True):
        self.data = data
        self.pulses = pulses
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

        data = self.data.data
        path = '/INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
        f, pos = data._find_data(source, tid)
        if f is not None:
            ds = f.file[path]
            self._datasets_cache[(source, key)] = (f, ds)
            return f, pos, ds

        return None, None, None

    def _get_slow_data(self, source, key, tid):
        file, pos, ds = self._find_data(source, key, tid)
        if file is None:
            return None

        group = key.partition('.')[0]
        firsts, counts = file.get_index(source, group)
        first, count = firsts[pos], counts[pos]
        if count == 1:
            return xarray.DataArray(ds[first])
        else:
            return xarray.DataArray(ds[first : first + count])

    def _get_pulse_data(self, source, key, tid):
        file, pos, ds = self._find_data(source, key, tid)
        if file is None:
            return None

        group = key.partition('.')[0]
        firsts, counts = file.get_index(source, group)
        first, count = firsts[pos], counts[pos]

        pulse_ids = file.file['/INSTRUMENT/{}/{}/pulseId'.format(source, group)][
            first : first + count
        ]
        # Raw files have a spurious extra dimension
        if pulse_ids.ndim >= 2 and pulse_ids.shape[1] == 1:
            pulse_ids = pulse_ids[:, 0]

        if isinstance(self.pulses, by_id):
            positions = self._select_pulse_ids(pulse_ids)
        else:  # by_index
            positions = self._select_pulse_indices(count)
        pulse_ids = pulse_ids[positions]
        train_ids = np.array([tid] * len(pulse_ids), dtype=np.uint64)
        train_pulse_ids = pd.MultiIndex.from_arrays(
            [train_ids, pulse_ids], names=['train', 'pulse']
        )

        if isinstance(positions, slice):
            data_positions = slice(
                int(first + positions.start),
                int(first + positions.stop),
                positions.step
            )
        else:  # ndarray
            # h5py fancy indexing needs a list, not an ndarray
            data_positions = list(first + positions)

        return _guess_axes(ds[data_positions], train_pulse_ids)

    def _select_pulse_ids(self, pulse_ids):
        """Select pulses by ID

        Returns an array or slice of the indexes to include.
        """
        val = self.pulses.value
        N = len(pulse_ids)
        if isinstance(val, slice):
            if val.step == 1:
                after_start = np.nonzero(pulse_ids >= val.start)[0]
                after_stop = np.nonzero(pulse_ids >= val.stop)[0]
                start_ix = after_start[0] if (after_start.size > 0) else N
                stop_ix = after_stop[0] if (after_stop.size > 0) else N
                return slice(start_ix, stop_ix)

            # step != 1
            desired = np.arange(val.start, val.stop, step=val.step, dtype=np.uint64)

        else:
            desired = val

        return np.nonzero(np.isin(pulse_ids, desired))[0]

    def _select_pulse_indices(self, count):
        """Select pulses by index

        Returns an array or slice of the indexes to include.
        """
        val = self.pulses.value
        if isinstance(val, slice):
            return slice(val.start, min(val.stop, count), val.step)

        # ndarray
        return val[val < count]

    def _assemble_data(self, tid):
        key_module_arrays = {}

        for modno, source in sorted(self.data.modno_to_source.items()):

            for key in self.data.data._keys_for_source(source):
                # At present, all the per-pulse data is stored in the 'image' key.
                # If that changes, this check will need to change as well.

                if key.startswith('image.'):
                    mod_data = self._get_pulse_data(source, key, tid)
                else:
                    mod_data = self._get_slow_data(source, key, tid)

                if mod_data is None:
                    continue

                if key not in key_module_arrays:
                    key_module_arrays[key] = [], []
                modnos, data = key_module_arrays[key]
                modnos.append(modno)
                data.append(mod_data)

        # Assemble the data for each key into one xarray
        return {
            k: xarray.concat(data, pd.Index(modnos, name='module'))
            for (k, (modnos, data)) in key_module_arrays.items()
        }

    def __iter__(self):
        for tid in self.data.train_ids:
            tid = int(tid)  # Convert numpy int to regular Python int
            if self.require_all and self.data.data._check_data_missing(tid):
                continue
            yield tid, self._assemble_data(tid)


class AGIPD1M(MPxDetectorBase):
    """An interface to AGIPD-1M data.

    Parameters
    ----------

    data: DataCollection
      A data collection, e.g. from RunDirectory.
    modules: set of ints, optional
      Detector module numbers to use. By default, all available modules
      are used.
    detector_name: str, optional
      Name of a detector, e.g. 'SPB_DET_AGIPD1M-1'. This is only needed
      if the dataset includes more than one AGIPD detector.
    min_modules: int
      Include trains where at least n modules have data. Default is 1.
    """
    _source_re = re.compile(r'(?P<detname>(.+)_AGIPD1M(.*))/DET/(\d+)CH')
    module_shape = (512, 128)


class DSSC1M(MPxDetectorBase):
    """An interface to DSSC-1M data.

    Parameters
    ----------

    data: DataCollection
      A data collection, e.g. from RunDirectory.
    modules: set of ints, optional
      Detector module numbers to use. By default, all available modules
      are used.
    detector_name: str, optional
      Name of a detector, e.g. 'SCS_DET_DSSC1M-1'. This is only needed
      if the dataset includes more than one DSSC detector.
    min_modules: int
      Include trains where at least n modules have data. Default is 1.
    """
    _source_re = re.compile(r'(?P<detname>(.+)_DSSC1M(.*))/DET/(\d+)CH')
    module_shape = (128, 512)


class LPD1M(MPxDetectorBase):
    """An interface to LPD-1M data.

    Parameters
    ----------

    data: DataCollection
      A data collection, e.g. from RunDirectory.
    modules: set of ints, optional
      Detector module numbers to use. By default, all available modules
      are used.
    detector_name: str, optional
      Name of a detector, e.g. 'FXE_DET_LPD1M-1'. This is only needed
      if the dataset includes more than one LPD detector.
    min_modules: int
      Include trains where at least n modules have data. Default is 1.
    """
    _source_re = re.compile(r'(?P<detname>(.+)_LPD1M(.*))/DET/(\d+)CH')
    module_shape = (256, 256)
