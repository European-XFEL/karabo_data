class Run():
    def __init__(self):
        pass

    def load(self, path):
        raise NotImplementedError("Loading run from file missing")

class FakeRun(Run):
    def __init__(self, trains, pulses, name):
        super(FakeRun, self).__init__()
        self.name = name
