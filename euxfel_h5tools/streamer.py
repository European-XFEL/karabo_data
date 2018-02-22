from collections import deque
import msgpack
import msgpack_numpy
from threading import Thread
from time import sleep
import zmq


msgpack_numpy.patch()


class ZMQStreamer:
    """ZeroMQ inteface exposing data to a tcp socket.

        # server
        serve = ZMQStreamer(1234)
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
        size of the buffer (old data is discarded if max len is reached)
        default: infinite.
    """
    def __init__(self, port, maxlen=None):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind("tcp://*:{}".format(port))

        self._buffer = deque(maxlen=maxlen)
        self._rep = Thread(target=self._interface, args=())
        self._rep.daemon = True
        self._rep.start()

    def _interface(self):
        while True:
            if len(self._buffer):
                req = self._socket.recv()
                if req != b'next':
                    raise RuntimeError('Unknown request:', req)                
                self._socket.send_multipart(self._buffer.popleft())
            else:
                sleep(0.1)

    def _serialize(self, data):
        # TODO: optimize this...
        return [msgpack.dumps(data, use_bin_type=True)]

    def feed(self, data):
        """Push data to the sending queue.
        """
        self._buffer.append(self._serialize(data))
