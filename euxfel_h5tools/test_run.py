import pytest

from .run import FakeRun, Run


@pytest.fixture
def fakerun1():
    trains = range(0, 11)
    r = FakeRun(trains=trains, pulses=30, name="r0007")
    yield r
    # could destroy r here



def test_run_class_name(fakerun1):
    r = fakerun1
    assert r.name == "r0007"


@pytest.mark.skip("not implemented")
def test_set_default_instrument(fakerun1):
    # choose which data to return by default
    r.set_default_instrument('instrument1')


@pytest.mark.skip("not implemented")
def test_index(fakerun1):
    r = FakeRun(trains=trains, pulses=30, name="r0007")



@pytest.mark.skip("not implemented")
def test_run_load():
    r = Run()
    r.from_files('example_data')
    # have tests here
    assert r.name == "example_data"
