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

DATA = {'source1': {'parameter.1.value': 123,
                    'list.of.int': [1, 2, 3],
                    'string.param': 'True',
                    'boolean': False},
        'XMPL/DET/MOD0': {'image.data': np.random.randint(255, size=(2, 3, 4), 
                                                          dtype=np.uint8),
                          'something.else': ['a', 'bc', 'd']}
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
def server():
    serve = ZMQStreamer(1234, maxlen=10, protocol_version='1.0')
    yield serve


def test_serialize(server):
    serve = server
    msg = serve._serialize(DATA)
    
    assert isinstance(msg, list)
    assert len(msg) == 1
    assert msg[-1] == msgpack.dumps(DATA, use_bin_type=True,
                                    default=numpack.encode)

    proto_2_server = ZMQStreamer(1235, protocol_version='2.1')
    msg = proto_2_server._serialize(DATA)
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


def test_fill_queue(server):
    serve = server

    for i in range(10):
        serve.feed(i)

    assert serve._buffer.full()
    with pytest.raises(Full):
        serve._buffer.put_nowait(b'too much')

    for i in range(10):
        assert serve._buffer.get() == [msgpack.dumps(i)]


def test_req_rep(server):
    serve = server
    serve.start()

    for i in range(3):
        serve.feed(DATA)

    client = Client('tcp://localhost:1234')
    for i in range(3):
        data = client.next()
        compare_nested_dict(data, DATA)


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
