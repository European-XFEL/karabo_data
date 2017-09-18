import os

class Run():
    hdf5_files = []
    instruments = []
    self.instrument = None

    def __init__(self, path):
        self.load(path)

    @property
    def instrument(self):
        return self.instrument

    @instrument.setter
    def name(self, inst):
        if inst in self.instruments:
            self.instrument = inst
        else:
            raise Exception("{} not available".format(inst))

    def load(self, path):
        if not os.path.isdir(path):
            raise Exception("Could not open {}".format(path))
        for root, _, fnames in os.walk(path):
            file_list = [os.path.join(root, fname) for fname in fnames
                         if fname.endswith(".hdf5")]
            self.hdf5_files.extend(file_list)

        # The walk may happen in neither alphabetical nor numerical order.
        # The data, however, is created in numerical order using the runId.
        # Therefore, to ease data manipulation, let's sort the files.
        self.hdf5_files.sort()


class FakeRun(Run):
    def __init__(self, trains, pulses, name):
        super(FakeRun, self).__init__()
        self.name = name
