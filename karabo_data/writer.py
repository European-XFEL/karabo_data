import h5py
import numpy as np


class FileWriter:
    """Write data in European XFEL HDF5 format

    This is intended to allow copying a subset of data into a smaller,
    more portable file.
    """

    def __init__(self, path, data):
        self.file = h5py.File(path, 'w')
        self.data = data
        self.indexes = {}  # {path: (first, count)}
        self.data_sources = set()

    def add_control_dataset(self, source, key):
        a = self.data.get_array(source, key)
        assert a.shape[0] == len(self.data.train_ids)
        path = 'CONTROL/{}/{}'.format(source, key.replace('.', '/'))
        self.file[path] = a.values
        self._make_control_index(source, a.coords['trainId'].values)

    def _make_control_index(self, source, data_tids):
        # Original files contain exactly 1 entry per train for control data,
        # but if one file starts before another, there can be some values
        # missing when we collect several files together. We don't try to
        # extrapolate to fill missing data, so some counts may be 0.
        if source not in self.indexes:
            assert len(np.unique(data_tids)) == len(data_tids),\
                "Duplicate train IDs in control data!"
            counts = np.isin(self.data.train_ids, data_tids).astype(np.uint64)
            firsts = np.zeros_like(counts)
            firsts[1:] = np.cumsum(counts)[:-1]  # firsts[0] is always 0
            self.indexes[source] = (firsts, counts)
            self.data_sources.add('CONTROL/' + source)

    def add_instrument_dataset(self, source, key):
        a = self.data.get_array(source, key)
        path = 'INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
        self.file[path] = a.values
        self._make_instrument_index(source, key, a.coords['trainId'].values)

    def _make_instrument_index(self, source, key, data_tids):
        index_path = source + '/' + key.partition('.')[0]
        if index_path not in self.indexes:
            self.indexes[index_path] = self._generate_index(data_tids)
            self.data_sources.add('INSTRUMENT/' + index_path)

    def _generate_index(self, data_tids):
        """Convert an array of train IDs to first/count for each train"""
        assert (np.diff(data_tids) >= 0).all(), "Out-of-order train IDs"
        counts = np.array([np.count_nonzero(t == data_tids)
                          for t in self.data.train_ids], dtype=np.uint64)
        firsts = np.zeros_like(counts)
        firsts[1:] = np.cumsum(counts)[:-1]  # firsts[0] is always 0

        return firsts, counts

    def write_train_ids(self):
        d = self.data
        self.file.create_dataset('INDEX/trainId', data=d.train_ids, dtype='u8')

    def write_indexes(self):
        for groupname, (first, count) in self.indexes.items():
            self.file['INDEX/{}/first'.format(groupname)] = first
            self.file['INDEX/{}/count'.format(groupname)] = count

    def write_metadata(self):
        vlen_bytes = h5py.special_dtype(vlen=bytes)
        data_sources = sorted(self.data_sources)
        N = len(data_sources)

        sources_ds = self.file.create_dataset(
            'METADATA/dataSourceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        sources_ds[:] = data_sources

        root_ds = self.file.create_dataset(
            'METADATA/root', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        root_ds[:] = [ds.split('/', 1)[0] for ds in data_sources]

        devices_ds = self.file.create_dataset(
            'METADATA/deviceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        devices_ds[:] = [ds.split('/', 1)[1] for ds in data_sources]

    def set_writer(self):
        from . import __version__

        self.file.attrs['writer'] = 'karabo_data {}'.format(__version__)

    def write(self):
        d = self.data
        self.write_train_ids()

        for source in d.control_sources:
            for key in d.keys_for_source(source):
                self.add_control_dataset(source, key)

        for source in d.instrument_sources:
            for key in d.keys_for_source(source):
                self.add_instrument_dataset(source, key)

        self.write_indexes()
        self.write_metadata()
        self.set_writer()


class VirtualFileWriter(FileWriter):
    """Write virtual datasets in European XFEL format

    The new files refer to the original data files, so they aren't portable,
    but they provide more convenient access by reassembling data spread over
    several sequence files.
    """
    def __init__(self, path, data):
        if not hasattr(h5py, 'VirtualLayout'):
            raise Exception("Creating virtual datasets requires HDF5 1.10 "
                            "and h5py 2.9")

        super().__init__(path,  data)

    def _assemble_data(self, source, key):
        """Assemble chunks of data into a virtual layout"""
        # First, get a list of all non-empty data chunks
        chunks = [c for c in self.data._find_data_chunks(source, key)
                  if (c.counts > 0).any()]
        chunks.sort(key = lambda c: c.train_ids[0])
        if not chunks:
            return None, None

        # Create the layout, which will describe what data is where
        n_total = np.sum([c.counts.sum() for c in chunks])
        ds0 = chunks[0].dataset
        layout = h5py.VirtualLayout(shape=(n_total,) + ds0.shape[1:],
                                    dtype=ds0.dtype)

        # Map each chunk into the relevant part of the layout
        output_cursor = np.uint64(0)
        for chunk in chunks:
            n = chunk.counts.sum()
            src = h5py.VirtualSource(chunk.dataset)
            src = src[chunk.slice]
            layout[output_cursor : output_cursor + n] = src
            output_cursor += n

        assert output_cursor == n_total

        # Make an array of which train ID each data entry is for:
        train_ids = np.concatenate([
            np.repeat(c.train_ids, c.counts.astype(np.intp)) for c in chunks
        ])
        return layout, train_ids

    def add_control_dataset(self, source, key):
        layout, train_ids = self._assemble_data(source, key)
        if not layout:
            return  # No data

        path = 'CONTROL/{}/{}'.format(source, key.replace('.', '/'))
        self.file.create_virtual_dataset(path, layout)

        self._make_control_index(source, train_ids)
        return path

    def add_instrument_dataset(self, source, key):
        layout, train_ids = self._assemble_data(source, key)
        if not layout:
            return  # No data

        path = 'INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
        self.file.create_virtual_dataset(path, layout)

        self._make_instrument_index(source, key, train_ids)
        return path
