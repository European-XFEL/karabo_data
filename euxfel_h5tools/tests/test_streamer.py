"""Test streaming data with ZMQ interface."""

import numpy as np
import pytest
from struct import pack
from time import sleep

from karabo_bridge import KaraboBridge

from euxfel_h5tools import ZMQStreamer

DATA = {'source1': {'parameter.1.value': 123,
                    'list.of.int': [1, 2, 3],
                    'string.param': 'True',
                    'boolean': False},
        'XMPL/DET/MOD0': {'image.data': np.random.randint(255, size=(2, 3, 4), 
                                                          dtype=np.uint8),
                          'something.else': ['a', 'bc', 'd']}
        }
BDATA = b'\x82\xa7source1\x84\xb1parameter.1.value{\xablist.of.int\x93\x01\x02\x03\xacstring.param\xa4True\xa7boolean\xc2\xadXMPL/DET/MOD0\x82\xaesomething.else\x93\xa1a\xa2bc\xa1d\xaaimage.data\x85\xc4\x02nd\xc3\xc4\x05shape\x93\x02\x03\x04\xc4\x04kind\xc4\x00\xc4\x04type\xa3|u1\xc4\x04data\xc4\x18\xf2F\x14O\xc4\xd9\x8f\x82+\x89\x98\x0f3i,2j\x91k\xa2\x19|\x84\x8c'


@pytest.yield_fixture
@pytest.fixture(scope="session")
def server():
    serve = ZMQStreamer(1234, maxlen=10)
    yield serve
    serve.stop()


def test_serialize(server):
    serve = server
    msg = serve._serialize(DATA)
    
    assert isinstance(msg, list)
    assert len(msg) == 1
    assert msg[-1] == BDATA


def test_fill_queue(server):
    serve = server

    for i in range(20):
        serve.feed(i)

    assert len(serve._buffer) == 10
    assert serve._buffer[9] == [pack('b', 19)]
    assert serve._buffer[0] == [pack('b', 10)]


def test_req_rep(server):
    serve = server
    client = KaraboBridge('tcp://localhost:1234')

    for i in range(3):
        serve.feed(DATA)

    for i in range(3):
        data = client.next()
        assert data == DATA


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")