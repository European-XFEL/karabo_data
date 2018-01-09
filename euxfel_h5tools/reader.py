#############################################################################
# Author: <thomas.michelat@xfel.eu>
# Created on October 26, 2017
# Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
# All rights reserved.
#############################################################################

from contextlib import contextmanager
import datetime
from glob import glob
import h5py
import numpy as np
import os.path as osp
import re
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
                        self.metadata['dataSourceId'].value if source]
        self.devices = [device.decode() for device in
                        self.metadata['deviceId'].value if device]
        self.train_ids = [tid for tid in self.index['trainId'][()].tolist()
                          if tid != 0]

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
            if src not in train_data:
                train_data[src] = {}
            data = train_data[src]

            if status[train_index]:
                def append_data(key, value):
                    if isinstance(value, h5py.Dataset):
                        path = '.'.join(filter(None,
                                        (path_base,) + tuple(key.split('/'))))
                        data[path] = value[int(first[train_index]):
                                           int(last[train_index]+1), ]

                table.visititems(append_data)

            sec, frac = str(time()).split('.')
            timestamp = {'tid': int(self.train_ids[train_index]),
                         'sec': int(sec), 'frac': int(frac)}
            data.update({'metadata': {'source': src, 'timestamp': timestamp}})

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
        tid, files = next((t for t in self.ordered_trains
                          if t[0] == train_id), (None, None))
        if tid is None:
            raise ValueError("train {} not found in run.".format(train_id))
        data = {}
        for fh in files:
            d, _, _ = fh.train_from_id(tid)
            data.update(d)
        return (tid, data)

    def _get_devices(self, src):
        src = [s.split('/') for f in src for s in f.sources]
        ctrl = set(['/'.join(c[1:4]) for c in src if c[0] == 'CONTROL'])
        inst = set(['/'.join(i[1:4]) for i in src if i[0] == 'INSTRUMENT'])
        return ctrl, inst

    def infos(self):
        # time info
        first_train, _ = self.ordered_trains[0]
        last_train, _ = self.ordered_trains[-1]
        train_count = len(self.ordered_trains)
        span_sec = (last_train - first_train) / 10
        span_txt = str(datetime.timedelta(seconds=span_sec))

        # devices infos
        ctrl, inst = self._get_devices(self.files)

        # disp
        print('Run information')
        print('\tDuration:      ', span_txt)
        print('\tFirst train ID:', first_train)
        print('\tLast train ID: ', last_train)
        print('\t# of trains:   ', train_count)
        print()
        print('Devices')
        print('\tInstruments')
        [print('\t-', d) for d in sorted(inst)] or print('\t-')
        print('\tControls')
        [print('\t-', d) for d in sorted(ctrl)] or print('\t-')

    def train_info(self, train_id):
        tid, files = next((t for t in self.ordered_trains
                          if t[0] == train_id), (None, None))
        if tid is None:
            raise ValueError("train {} not found in run.".format(train_id))
        ctrl, inst = self._get_devices(files)

        # disp
        print('Train [{}] information'.format(train_id))
        print('Devices')
        print('\tInstruments')
        [print('\t-', d) for d in sorted(inst)] or print('\t-')
        print('\tControls')
        [print('\t-', d) for d in sorted(ctrl)] or print('\t-')


def stack_data(train, data, axis=-3, xcept=[]):
    """Stack data from devices in a train.

    :train:    train data
    :data:     the data path in the devices
    :axis:     axis on which you wish to stack
    :xcept:    list of devices to ignore (useful if you have reccored slow
               data with detector data in the same run)

    returns ndarray stacked data for requested data path.
    """
    devs = [(list(map(int, re.findall(r'\d+', dev))), dev)
            for dev in train.keys() if dev not in xcept]
    devices = [dev for _, dev in sorted(devs)]

    dtype, shape = next(((d[data].dtype, d[data].shape) 
                        for d in train.values() if data in d), (None, None))
    if dtype is None or shape is None:
        return np.empty(0)

    combined = np.zeros((len(devices),) + shape, dtype=dtype)
    for index, device in enumerate(devices):
        try:
            combined[index, ] = train[device][data]
        except KeyError:
            print('stack_data(): missing {} in {}'.format(data, device))
    return np.moveaxis(combined, 0, axis)


def stack_detector_data(train, data, axis=-3, modules=16, only='', xcept=[]):
    """Stack data from detector modules in a train.

    :train:    train data
    :data:     the data path in the devices
    :axis:     axis on which you wish to stack (default is -3)
    :modules:  number of modules composing a detector (default is 16)
    :only:     Only use devices in train containing this substring.
    :xcept:    list of devices to ignore (useful if you have reccored slow
               data with detector data in the same run)

    returns ndarray stacked data for requested data path.
    """
    devices = [dev for dev in train.keys() if only in dev and dev not in xcept]

    dtype, shape = next(((d[data].dtype, d[data].shape) 
                        for d in train.values() if data in d), (None, None))
    if dtype is None or shape is None:
        return np.empty(0)

    combined = np.zeros((modules,) + shape, dtype=dtype)
    for device in devices:
        try:
            index = int(re.findall(r'\d+', device)[-2])
            combined[index, ] = train[device][data]
        except KeyError:
            print('stack_detector_data(): missing {} in {}'.format(data, device))
        except IndexError:
            print('stack_detector_Data(): module {} is out or range for a'
                  'detector of {} modules'.format(index, modules))
    return np.moveaxis(combined, 0, axis)


if __name__ == '__main__':
    r = RunHandler('./data/r0185')
    for tid, d in r.trains():
        print(tid)
