#############################################################################
# Author: <thomas.michelat@xfel.eu>
# Created on October 26, 2017
# Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
# All rights reserved.
#############################################################################

from contextlib import contextmanager
from glob import glob
import h5py
import os.path as osp
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
        self.train_ids = self.index['trainId'][()].tolist()

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

        return (train_data, self.train_ids[train_index], train_index)

    def trains(self):
        """Iterate over all trains in the file"""
        for index, train in enumerate(self.train_ids):
            yield self._gen_train_data(index)

    def train_from_id(self, train_id):
        """Get Train data for specified train ID."""
        try:
            index = self.train_ids.index(train_id)
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


class RunHandler:
    """Handles a 'run' generated at the European XFEL.

    A 'run' is a directory containing a various amount of HDF5 file recorded 
    in the European XFEL format. This class can iterate through the data
    contained in the run and extract instrument data per XRAY train.
    """
    def __init__(self, path):
        self.files = [H5File(f) for f in glob(osp.join(path, '*.h5'))
                      if h5py.is_hdf5(f)]

        trains = {}
        for fhandler in self.files:
            for train in fhandler.train_ids:
                if train not in trains:
                    trains[train] = []
                trains[train].append(fhandler)

        self.ordered_trains = list(sorted(trains.items()))

    def trains(self):
        """Iterate over all trains in the run and gather all sources.

        returns a tuple: ([int]train_id, [dict]train_data)

        Usage::

            run = Run('/path/to/my/run/r0123')
            for data, train_id in run.trains():
                image = data['detector']['image.data']
                ...
        """
        for tid, fhs in self.ordered_trains:
            train_data = {}
            for fh in fhs:
                data, _, _ = fh.train_from_id(tid)
                train_data.update(data)

            yield (tid, train_data)

    def train_from_id(self, train_id):
        """Get Train data for specified train ID.

        returns a tuple: ([int]train_id, [dict]train_data)
        """
        try:
            tid, files = next((t for t in self.ordered_trains
                               if t[0] == train_id))
        except StopIteration:
            raise ValueError("train {} not found in run.".format(train_id))
        data = {}
        for fh in files:
            d, _, _ = fh.train_from_id(tid)
            data.update(d)
        return (tid, data)
