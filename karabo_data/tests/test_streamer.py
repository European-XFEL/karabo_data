"""Test streaming data with ZMQ interface."""

import msgpack
import msgpack_numpy as numpack
import numpy as np
import pytest
from queue import Full
from struct import pack
from time import sleep

from karabo_bridge import Client

from karabo_data import ZMQStreamer

DATA = {
    'source1': {
        'parameter.1.value': 123,
        'list.of.int': [1, 2, 3],
        'string.param': 'True',
        'boolean': False,
        'metadata': {'timestamp.tid': 9876543210},
    },
    'XMPL/DET/MOD0': {
        'image.data': np.random.randint(255, size=(2, 3, 4), dtype=np.uint8),
        'something.else': ['a', 'bc', 'd'],
    },
}


def compare_nested_dict(d1, d2, path=''):
    for key in d1.keys():
        if key not in d2:
            print(d1.keys())
            print(d2.keys())
            raise KeyError('key is missing in d2: {}{}'.format(path, key))

        if isinstance(d1[key], dict):
            path += key + '.'
            compare_nested_dict(d1[key], d2[key], path)
        else:
            v1 = d1[key]
            v2 = d2[key]

            try:
                if isinstance(v1, np.ndarray):
                    assert (v1 == v2).all()
                elif isinstance(v1, tuple) or isinstance(v2, tuple):
                    # msgpack doesn't know about complex types, everything is
                    # an array. So tuples are packed as array and then
                    # unpacked as list by default.
                    assert list(v1) == list(v2)
                else:
                    assert v1 == v2
            except AssertionError:
                raise ValueError('diff: {}{}'.format(path, key), v1, v2)


@pytest.fixture(scope="session")
def server_1():
    server = ZMQStreamer(4444, maxlen=10, protocol_version='1.0')
    yield server


@pytest.fixture(scope="session")
def server_2_2():
    server = ZMQStreamer(5555, maxlen=10, protocol_version='2.2')
    yield server


@pytest.fixture(scope="session")
def client():
    client = Client('tcp://localhost:5555')
    yield client


class DummyFrame:
    """Client._deserialize() now expects the message in ZMQ Frame objects.

    TODO: avoid using a private method from karabo_data for tests.
    """

    def __init__(self, data):
        self.bytes = data
        self.buffer = data


def test_serialize_1(server_1, client):
    msg = server_1._serialize(DATA)

    assert isinstance(msg, list)
    assert len(msg) == 1
    assert msg[-1] == msgpack.dumps(DATA, use_bin_type=True, default=numpack.encode)

    msg_framed = [DummyFrame(b) for b in msg]
    data, meta = client._deserialize(msg_framed)
    compare_nested_dict(data, DATA)


def test_serialize_2_2(server_2_2, client):
    msg = server_2_2._serialize(DATA)
    assert isinstance(msg, list)
    assert len(msg) == 6

    m0 = msgpack.loads(msg[0], raw=False)
    assert m0['source'] == 'XMPL/DET/MOD0'
    assert m0['content'] == 'msgpack'
    m2 = msgpack.loads(msg[2], raw=False)
    assert m2['source'] == 'XMPL/DET/MOD0'
    assert m2['path'] == 'image.data'
    assert m2['content'] == 'array'

    m2 = msgpack.loads(msg[4], raw=False)
    print(m2)
    assert m2['source'] == 'source1'
    assert m2['content'] == 'msgpack'

    msg_framed = [DummyFrame(b) for b in msg]
    data, meta = client._deserialize(msg_framed)
    compare_nested_dict(data, DATA)

    assert meta['source1']['timestamp.tid'] == 9876543210


def test_fill_queue(server_2_2):
    for i in range(10):
        server_2_2.feed({str(i): {str(i): i}})

    assert server_2_2._buffer.full()
    with pytest.raises(Full):
        server_2_2._buffer.put_nowait({'too much': {'prop': 0}})

    for i in range(10):
        assert server_2_2._buffer.get()[1] == msgpack.dumps({str(i): i})


def test_req_rep(server_2_2, client):
    server_2_2.start()

    for i in range(3):
        server_2_2.feed(DATA)

    for i in range(3):
        data, metadata = client.next()
        compare_nested_dict(data, DATA)


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
