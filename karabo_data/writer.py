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
        self.indexes = {} # {path: (first, count)}
        self.data_sources = set()

    def add_control_dataset(self, source, key):
        a = self.data.get_array(source, key)
        path = 'CONTROL/{}/{}'.format(source, key.replace('.', '/'))
        self.file[path] = a.values

        if source not in self.indexes:
            n = a.shape[0]
            self.indexes[source] = \
                (np.arange(n, dtype='u8'), np.ones(n, dtype='u8'))
            self.data_sources.add('CONTROL/' + source)

    def add_instrument_dataset(self, source, key):
        a = self.data.get_array(source, key)
        path = 'INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
        self.file[path] = a.values

        index_path = source + '/' + key.partition('.')[0]
        if index_path not in self.indexes:
            data_tids = a.coords['trainId'].values
            self.indexes[index_path] = self._generate_index(data_tids)
            self.data_sources.add('INSTRUMENT/' + index_path)

    def _generate_index(self, data_tids):
        """Convert an array of train IDs to first/count for each train"""
        first = np.zeros_like(self.data.train_ids, dtype='u8')
        count = np.zeros_like(self.data.train_ids, dtype='u8')

        for ix, tid in enumerate(self.data.train_ids):
            matches = (data_tids == tid)
            if matches.any():
                first[ix] = matches.nonzero()[0][0]
                count[ix] = matches.sum()

        return first, count

    def write_indexes(self):
        for groupname, (first, count) in self.indexes.items():
            self.file['INDEX/{}/first'.format(groupname)] = first
            self.file['INDEX/{}/count'.format(groupname)] = count

    def write_metadata(self):
        vlen_bytes = h5py.special_dtype(vlen=bytes)
        data_sources = sorted(self.data_sources)
        N = len(data_sources)

        sources_ds = self.file.create_dataset('METADATA/dataSourceId', (N,),
                                              dtype=vlen_bytes, maxshape=(None,))
        sources_ds[:] = data_sources

        root_ds = self.file.create_dataset('METADATA/root', (N,),
                                           dtype=vlen_bytes, maxshape=(None,))
        root_ds[:] = [ds.split('/', 1)[0] for ds in data_sources]

        devices_ds = self.file.create_dataset('METADATA/deviceId', (N,),
                                              dtype=vlen_bytes, maxshape=(None,))
        devices_ds[:] = [ds.split('/', 1)[1] for ds in data_sources]

    def set_writer(self):
        from . import __version__
        self.file.attrs['writer'] = 'karabo_data {}'.format(__version__)

    def write(self):
        d = self.data
        self.file.create_dataset('INDEX/trainId', data=d.train_ids, dtype='u8')

        for source in d.control_sources:
            for key in d._keys_for_source(source):
                self.add_control_dataset(source, key)

        for source in d.instrument_sources:
            for key in d._keys_for_source(source):
                self.add_instrument_dataset(source, key)

        self.write_indexes()
        self.write_metadata()
        self.set_writer()
