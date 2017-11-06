#############################################################################
# Author: <thomas.michelat@xfel.eu>
# Created on October 26, 2017
# Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
# All rights reserved.
#############################################################################

from contextlib import contextmanager
import h5py
from time import time


RUN_DATA = 'RUN'
INDEX_DATA = 'INDEX'
METADATA = 'METADATA'


class H5File:
    """Handles a HDF5 file generated at the European XFEL.
    
    An HDF5 file contains in general several record of data. This helps
    handling such file and extract back instrument data per XRAY train.

    Usage::

        h5f = H5File('/path/to/my/file.h5')
        for data, train_id, index in h5f.trains():
            image = data['detector']['image.data']
            ...
    """
    def __init__(self, path, driver=None):
        assert h5py.is_hdf5(path)
        self.file = h5py.File(path, 'r', driver=driver)

        self.metadata = self.file[METADATA]
        self.index = self.file[INDEX_DATA]
        self.run = self.file[RUN_DATA]

        self.sources = [source.decode() for source in
                        self.metadata['dataSourceId'].value]
        self.devices = [device.decode() for device in
                        self.metadata['deviceId'].value]
        self.train_ids = self.index['trainId']

    def _gen_train_data(self, train_index):
        """Get data for the specified index in file."""
        train_data = {}
        for device, source in zip(self.devices, self.sources):
            index = self.index[device]
            table = self.file[source]

            first = index['first']
            last = index['last']
            status = index['status']

            dev = device.split('/')
            src = '/'.join((dev[:3]))
            path_base = '.'.join((dev[3:]))
            if not src in train_data:
                train_data[src] = {}
            data = train_data[src]

            if status[train_index]:
                def append_data(key, value):
                    if isinstance(value, h5py.Dataset):
                        path = '.'.join(filter(None,
                                        (path_base,) + tuple(key.split('/'))))
                        data[path] = value[int(first[train_index]):
                                           int(last[train_index]+1),]

                table.visititems(append_data)

            sec, frac = str(time()).split('.')
            data.update({'metadata': {'source': src,
                                      'tid': int(self.train_ids[train_index]),
                                      'sec': int(sec), 'frac': int(frac)}})

        return (train_data, self.train_ids[(train_index,)], train_index)

    def trains(self):
        """Iterate over all trains in the file"""
        for index, train in enumerate(self.train_ids):
            yield self._gen_train_data(index)

    def train_from_id(self, train_id):
        """Get Train data for specified train ID."""
        try:
            index = self.train_ids[()].tolist().index(train_id)            
        except ValueError:
            raise ValueError("train {} not found in {}.".format(
                             train_id, self.file.filename))
        else:
            return self._gen_train_data(index)

    def train_from_index(self, index):
        """Get train data of the nth train in file."""
        return self._gen_train_data(index)

    def close(self):
        self.file.close()


@contextmanager
def open_H5File(path, driver=None):
    """factory function for with statement context managers, for H5File.
    
    Best use this function than the class directly to ensure proper closing::
    
       with open_H5File('/path/to/my/file.h5') as xfel_data:
           first_train = xfel_data.train_from_index(0)
           ...
    """
    h5f = H5File(path, driver=driver)
    try:
        yield h5f
    finally:
        h5f.close()
