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
    handling such file and extract back instrument data per XRAY train::

        h5f = H5File('/path/to/my/file.h5')
        for data, train_id, index in h5f.trains():
            value = data['device']['parameter']

    Parameters
    ----------
    path: str
        Path to the HDF5 file.
    driver: str, optional
        HDF5 driver.

    Raises
    ------
    FileNotFoundError
        If the provided path is not a valid HDF5 file.
    """
    def __init__(self, path, driver=None):
        if not h5py.is_hdf5(path):
            raise FileNotFoundError(path, 'is not a valid HDF5 file.')
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
        self._trains = {tid: idx for idx, tid in enumerate(self.train_ids)}

    def _gen_train_data(self, train_index, only_this=None):
        """Get data for the specified index in file.
        """
        train_data = {}
        for device, source in zip(self.devices, self.sources):
            index = self.index[device]
            table = self.file[source]
            status = index['status'][train_index]

            dev = device.split('/')
            src = '/'.join((dev[:3]))
            path_base = '.'.join((dev[3:]))

            if only_this and src not in only_this:
                continue

            if src not in train_data:
                train_data[src] = {}
            data = train_data[src]

            if status:
                first = int(index['first'][train_index])
                last = int(index['last'][train_index])

                def append_data(key, value):
                    if isinstance(value, h5py.Dataset):
                        path = '.'.join(filter(None,
                                        (path_base,) + tuple(key.split('/'))))
                        if only_this and path not in only_this[src]:
                            return

                        if first == last:
                            data[path] = value[first]
                        else:
                            data[path] = value[first:last+1, ]

                table.visititems(append_data)

            sec, frac = str(time()).split('.')
            timestamp = {'tid': int(self.train_ids[train_index]),
                         'sec': int(sec), 'frac': int(frac)}
            data.update({'metadata': {'source': src, 'timestamp': timestamp}})

        return (train_data, self.train_ids[train_index], train_index)

    def trains(self, devices=None):
        """Iterate over all trains in the file.

        Parameters
        ----------
        devices: dict, optional
            Use to filter data by devices and by parameters, i.e., for::

                dev = {'xray_monitor': {'pulseEnergy', 'beamPosition'}}
                for id, data in file_handler.trains(devices=dev):
                    pos = data['xray_monitor']['beamPosition']

            the returned data will only contains the device 'xray_monitor'
            and 2 of it's parameters (pulseEnergy and beamPosition).
        """
        for index in range(len(self.train_ids)):
            yield self._gen_train_data(index, only_this=devices)

    def train_from_id(self, train_id, devices=None):
        """Get Train data for specified train ID.

        Parameters
        ----------
        train_id: int
            the train ID you want to return
        devices: dict, optional
            Use to filter data by devices and by parameters, i.e., for::

                dev = {'xray_monitor': {'pulseEnergy', 'beamPosition'}}
                for id, data in run,trains(devices=dev)

            the returned data will only contains the device 'xray_monitor'
            and 2 of it's parameters (pulseEnergy and beamPosition).

        returns
        -------
        data, tid, index: tuple(dict, int, int)
            data contains the train data.
            tid is the train ID of the returned train.
            index is the index of the train in the file.

        Raises
        ------
        KeyError
            if `train_id` is not found in the file.
        """
        try:
            index = self._trains[train_id]
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
        devices: dict, optional
            Use to filter data by devices and by parameters, i.e., for::

                dev = {'xray_monitor': {'pulseEnergy', 'beamPosition'}}
                for id, data in run,trains(devices=dev)

            the returned data will only contains the device 'xray_monitor'
            and 2 of it's parameters (pulseEnergy and beamPosition).

        returns
        -------
        data, tid, index: tuple(dict, int, int)
            data contains the train data.
            tid is the train ID of the returned train.
            index is the index of the train in the file.
        """
        return self._gen_train_data(index, only_this=devices)

    def close(self):
        self.file.close()


@contextmanager
def open_H5File(path, driver=None):
    """factory function for with statement context managers, for H5File.

        with open_H5File('/path/to/my/file.h5') as xfel_data:
            first_train = xfel_data.train_from_index(0)

    Parameters
    ----------
    path: str
        Path to the HDF5 file.
    driver: str, optional
        HDF5 driver.
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

    def trains(self, devices=None):
        """Iterate over all trains in the run and gather all sources.

            run = Run('/path/to/my/run/r0123')
            for data, train_id in run.trains():
                value = data['device']['parameter']

        Parameters
        ----------
        devices: dict, optional
            Use to filter data by devices and by parameters, i.e., for::

                dev = {'xray_monitor': {'pulseEnergy', 'beamPosition'}}
                for id, data in run.trains(devices=dev)

            the returned data will only contains the device 'xray_monitor'
            and 2 of it's parameters (pulseEnergy and beamPosition).

        Returns
        -------
        (tid, data): tuple(int, dict)
            tid is the train ID of the returned train
            data contains data of the returned train for the selected run.
        """
        for tid, fhs in self.ordered_trains:
            train_data = {}
            for fh in fhs:
                data, _, _ = fh.train_from_id(tid, devices=devices)
                train_data.update(data)

            yield (tid, train_data)

    def train_from_id(self, train_id, devices=None):
        """Get Train data for specified train ID.

        Parameters
        ----------
        train_id: int
            the train ID you want to return
        devices: dict, optional
            Use to filter data by devices and by parameters, i.e., for::

                dev = {'xray_monitor': {'pulseEnergy', 'beamPosition'}}
                for id, data in run,trains(devices=dev)

            the returned data will only contains the device 'xray_monitor'
            and 2 of it's parameters (pulseEnergy and beamPosition).

        returns
        -------
        train_id, data: tuple(int, dict)
            train_id is the train ID of the returned train
            data contains the train data.

        Raises
        ------
        KeyError
            if `train_id` is not found in the run.
        """
        try:
            files = self._trains[train_id]
        except KeyError:
            raise KeyError("train {} not found in run.".format(train_id))
        data = {}
        for fh in files:
            d, _, _ = fh.train_from_id(train_id, devices=devices)
            data.update(d)
        return (train_id, data)

    def train_from_index(self, index, devices=None):
        """Get the nth train in the run.

        Parameters
        ----------
        index: int
            the train ID you want to return
        devices: dict, optional
            Use to filter data by devices and by parameters, i.e., for::

                dev = {'xray_monitor': {'pulseEnergy', 'beamPosition'}}
                for id, data in run,trains(devices=dev)

            the returned data will only contains the device 'xray_monitor'
            and 2 of it's parameters (pulseEnergy and beamPosition).

        returns
        -------
        train_id, data: tuple(int, dict)
            train_id is the train ID of the returned train
            data contains the train data.

        Raises
        ------
        IndexError
            if train `index` is out of range.
        """
        try:
            train_id, files = self.ordered_trains[index]
        except IndexError:
            raise IndexError("Train index {} out of range.".format(index))
        data = {}
        for fh in files:
            d, _, _ = fh.train_from_id(train_id, devices=devices)
            data.update(d)
        return (train_id, data)

    def _get_devices(self, src):
        """Return sets of control and instrument device names.
        control: train data
        instrument: pulse data
        """
        src = [s.split('/') for f in src for s in f.sources]
        ctrl = set(['/'.join(c[1:4]) for c in src if c[0] == 'CONTROL'])
        inst = set(['/'.join(i[1:4]) for i in src if i[0] == 'INSTRUMENT'])
        return ctrl, inst

    def infos(self):
        """Show information about the run.
        """
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


def stack_detector_data(train, data, axis=-3, modules=16, only='', xcept=[]):
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
        return np.empty(0)

    combined = np.zeros((modules,) + shape, dtype=dtype)
    for device in devices:
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
    return np.moveaxis(combined, 0, axis)


def extract_param(run_path, train_id, param):
    run_handler = RunHandler(run_path)

    depth_param = len(param.split('.'))
    slashed_param = param.replace('.', '/')

    devspaths = {}
    def add_dev_and_path(key):
        if (key.startswith(('CONTROL', 'INSTRUMENT'))
            and key.endswith(slashed_param)):

            key = key.split('/')
            dev = '/'.join(key[1:-depth_param])
            devspaths[dev] = {param} 

    devspaths = {}
    for fh in run_handler._trains[train_id]:
            fh.file.visit(add_dev_and_path)
        
    _, data = run_handler.train_from_id(train_id, devices=devspaths)
    stack = stack_detector_data(data, param)
    return stack


if __name__ == '__main__':
    r = RunHandler('./data/r0185')
    for tid, d in r.trains():
        print(tid)
