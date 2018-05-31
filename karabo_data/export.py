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
from functools import partial

import msgpack
import numpy as np
import msgpack_numpy as numpack
from queue import Queue
from threading import Event, Thread
import zmq

from .reader import RunDirectory, H5File

__all__ = ['ZMQStreamer']


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
    protocol_version: ('1.0' | '2.1' | 'latest')
        Which version of the bridge protocol to use. Defaults to latest.
    """
    def __init__(self, port, maxlen=10, protocol_version='latest'):
        self._context = zmq.Context()
        self.port = port
        if protocol_version == 'latest':
            protocol_version = '2.1'
        self.protocol_version = protocol_version
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

    def _serialize(self, data):
        if self.protocol_version == '1.0':
            return [msgpack.dumps(data, use_bin_type=True, default=numpack.encode)]

        pack = partial(msgpack.dumps, use_bin_type=True)
        msg = []
        for src, props in sorted(data.items()):
            main_data = {}
            arrays = []
            for key, value in props.items():
                if isinstance(value, np.ndarray):
                    arrays.append((key, value))
                else:
                    main_data[key] = value

            msg.extend([
                pack({'source': src, 'content': 'msgpack'}),
                pack(main_data)
            ])
            for key, array in arrays:
                if not array.flags['C_CONTIGUOUS']:
                    array = np.ascontiguousarray(array)
                msg.extend([
                    pack({'source': src, 'content': 'array', 'path': key,
                          'dtype': str(array.dtype), 'shape': array.shape}),
                    memoryview(array),
                ])

        return msg

    def feed(self, data):
        """Push data to the sending queue.

        This blocks if the queue already has *maxlen* items waiting to be sent.
        """
        self._buffer.put(self._serialize(data))


def main(argv=None):
    ap = ArgumentParser(prog="karabo-bridge-serve-files")
    ap.add_argument("path", help="Path of a file or run directory to serve")
    ap.add_argument("port", help="TCP port to run server on")
    args = ap.parse_args(argv)


    if osp.isdir(args.path):
        data = RunDirectory(args.path)
    else:
        data = H5File(args.path)

    streamer = ZMQStreamer(args.port)
    streamer.start()
    for tid, train_data in data.trains():
        streamer.feed(train_data)

    streamer.stop()
