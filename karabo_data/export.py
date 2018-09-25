# coding: utf-8
"""Expose data to different interface

ZMQStream explose to a ZeroMQ socket in a REQ/REP pattern.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

from argparse import ArgumentParser
import os.path as osp
from queue import Queue
from threading import Event, Thread
from time import time

import msgpack
import numpy as np
import zmq

from .reader import RunDirectory, H5File

__all__ = ['ZMQStreamer', 'serve_files']


class REPInterface(Thread):
    def __init__(self, context, port, buffer):
        super(REPInterface, self).__init__()
        self.context = context
        self.port = port
        self.buffer = buffer
        self._stop_event = Event()

    def run(self):
        interface = self.context.socket(zmq.REP)
        try:
            interface.bind('tcp://*:{}'.format(self.port))

            while not self.stopped():
                req = interface.recv()
                if req != b'next':
                    raise RuntimeError('Unknown request:', req)
                interface.send_multipart(self.buffer.get())
        finally:
            interface.setsockopt(zmq.LINGER, 0)
            interface.close()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class ZMQStreamer:
    """ZeroMQ inteface sending data over a TCP socket.

    ::

        # Server:
        serve = ZMQStreamer(1234)
        serve.start()

        for tid, data in run.trains():
            result = important_processing(data)
            serve.feed(result)

        # Client:
        from karabo_bridge import Client
        client = Client('tcp://server.hostname:1234')
        data = client.next()

    Parameters
    ----------
    port: int
        Local TCP port to bind socket to
    maxlen: int, optional
        How many trains to cache before sending (default: 10)
    protocol_version: ('1.0' | '2.1')
        Which version of the bridge protocol to use. Defaults to the latest
        version implemented.
    dummy_timestamps: bool
        Some tools (such as OnDA) expect the timestamp information to be in the
        messages. We can't give accurate timestamps where these are not in the
        file, so this option generates fake timestamps from the time the data
        is fed in.
    """
    def __init__(self, port, maxlen=10, protocol_version='2.2', dummy_timestamps=False):
        self._context = zmq.Context()
        self.port = port
        if protocol_version not in {'1.0', '2.2'}:
            raise ValueError("Unknown protocol version %r" % protocol_version)
        elif protocol_version == '1.0':
            import msgpack_numpy
            self.pack = msgpack.Packer(use_bin_type=True,
                                       default=msgpack_numpy.encode).pack
        else:
            self.pack = msgpack.Packer(use_bin_type=True).pack
        self.protocol_version = protocol_version
        self.dummy_timestamps = dummy_timestamps
        self._buffer = Queue(maxsize=maxlen)
        self._interface = None

    def start(self):
        """Start a zmq.REP socket.
        """
        self._interface = REPInterface(self._context, self.port, self._buffer)
        self._interface.daemon = True
        self._interface.start()

    def stop(self):
        if self._interface:
            self._interface.stop()
            self._interface.join()
            self._interface = None

    def _serialize(self, data, metadata=None):
        if not metadata:
            metadata = {src: v.get('metadata', {}) for src, v in data.items()}

        if self.dummy_timestamps:
            ts = time()
            sec, frac = str(ts).split('.')
            frac = frac.ljust(18, '0')
            update_dummy = {'timestamp': ts, 'timestamp.sec': sec, 'timestamp.frac': frac}
            for src in data.keys():
                if 'timestamp' not in metadata[src]:
                    metadata[src].update(update_dummy)

        if self.protocol_version == '1.0':
            return [self.pack(data)]

        msg = []
        for src, props in sorted(data.items()):
            main_data = {}
            arrays = []
            for key, value in props.items():
                if isinstance(value, np.ndarray):
                    arrays.append((key, value))
                elif isinstance(value, np.number):
                    # Convert numpy type to native Python type
                    main_data[key] = value.item()
                else:
                    main_data[key] = value

            msg.extend([
                self.pack({
                    'source': src, 'content': 'msgpack',
                    'metadata': metadata[src]
                }),
                self.pack(main_data)
            ])

            for key, array in arrays:
                if not array.flags['C_CONTIGUOUS']:
                    array = np.ascontiguousarray(array)
                msg.extend([
                    self.pack({
                        'source': src, 'content': 'array', 'path': key,
                        'dtype': str(array.dtype), 'shape': array.shape
                    }),
                    array.data,
                ])

        return msg

    def feed(self, data, metadata=None):
        """Push data to the sending queue.

        This blocks if the queue already has *maxlen* items waiting to be sent.

        Parameters
        ----------
        data : dict
            Contains train data. The dictionary has to follow the karabo_bridge
            protocol structure:

            - keys are source names
            - values are dict, where the keys are the parameter names and
              values must be python built-in types or numpy.ndarray.

        metadata : dict, optional
            Contains train metadata. The dictionary has to follow the
            karabo_bridge protocol structure:

            - keys are (str) source names
            - values (dict) should contain the following items:

              - 'timestamp' Unix time with subsecond resolution
              - 'timestamp.sec' Unix time with second resolution
              - 'timestamp.frac' fractional part with attosecond resolution
              - 'timestamp.tid' is European XFEL train unique ID

            ::

              {
                  'source': 'sourceName'  # str
                  'timestamp': 1234.567890  # float
                  'timestamp.sec': '1234'  # str
                  'timestamp.frac': '567890000000000000'  # str
                  'timestamp.tid': 1234567890  # int
              }

            If the metadata dict is not provided it will be extracted from
            'data' or an empty dict if 'metadata' key is missing from a data
            source.
        """
        self._buffer.put(self._serialize(data, metadata))


def serve_files(path, port, **kwargs):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    path: str
        Path to the HDF5 file or file folder.
    port: int
        Local TCP port to bind socket to.
    """
    if osp.isdir(path):
        data = RunDirectory(path)
    else:
        data = H5File(path)

    streamer = ZMQStreamer(port, **kwargs)
    streamer.start()
    for tid, train_data in data.trains():
        if train_data:
            streamer.feed(train_data)
    streamer.stop()


def main(argv=None):
    ap = ArgumentParser(prog="karabo-bridge-serve-files")
    ap.add_argument("path", help="Path of a file or run directory to serve")
    ap.add_argument("port", help="TCP port to run server on")
    args = ap.parse_args(argv)

    serve_files(args.path, args.port)
