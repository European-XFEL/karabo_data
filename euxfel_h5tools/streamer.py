"""Expose data to different interface

ZMQStream explose to a ZeroMQ socket in a REQ/REP pattern.
"""


__all__ = ('ZMQStreamer',)


import msgpack
import msgpack_numpy as numpack
from queue import Queue
from threading import Event, Thread
import zmq


class REPInterface(Thread):
    def __init__(self, context, port, buffer):
        super(REPInterface, self).__init__()
        self.context = context
        self.port = port
        self.buffer = buffer
        self._stop_event = Event()

    def run(self):
        try:
            interface = self.context.socket(zmq.REP)
            interface.bind('tcp://*:{}'.format(self.port))

            while not self.stopped():
                req = interface.recv()
                if req != b'next':
                    raise RuntimeError('Unknown request:', req)
                interface.send_multipart(self.buffer.get())
        finally:
            if interface:
                interface.setsockopt(zmq.LINGER, 0)
                interface.close()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class ZMQStreamer:
    """ZeroMQ inteface exposing data to a tcp socket.

        # server
        serve = ZMQStreamer(1234)
        serve.start()

        for tid, data in run.trains():
            result = important_processing(data)
            serve.feed(result)
            
        # client
        from karabo_bridge import KaraboBridge
        client = KaraboBridge('tcp://server.hostname:1234')
        data = client.next()

    Parameters
    ----------
    port: int
        Port to bind socket to.
    maxlen: int, optional
        size of the buffer - default: 10.
    """
    def __init__(self, port, maxlen=10):
        self._context = zmq.Context()
        self.port = port
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
        # TODO: optimize this...
        return [msgpack.dumps(data, use_bin_type=True, default=numpack.encode)]

    def feed(self, data):
        """Push data to the sending queue.

        This call is blocking if the Queue is full.
        """
        self._buffer.put(self._serialize(data))
