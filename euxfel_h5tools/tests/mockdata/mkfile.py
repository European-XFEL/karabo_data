import h5py
from .base import write_train_ids, write_metadata

class FileBuilder:
    def __init__(self, ntrains, chunksize=200):
        self.ntrains = ntrains
        self.chunksize = chunksize
        self.devices = []

    def add_device(self, device_cls, device_id, nsamples=None):
        self.devices.append(device_cls(device_id, ntrains=self.ntrains,
                                   nsamples=nsamples, chunksize=self.chunksize))

    def write(self, filename):
        f = h5py.File(filename, 'w')

        write_train_ids(f, 'INDEX/trainId', self.ntrains, chunksize=self.chunksize)

        data_sources = []
        for dev in self.devices:
            dev.write_control(f)
            dev.write_instrument(f)
            data_sources.extend(dev.datasource_ids())
        write_metadata(f, data_sources, chunksize=self.chunksize)
