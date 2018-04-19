import h5py
import numpy as np

class DeviceBase:
    # Override these in subclasses
    control_keys = []
    output_channels = ()
    instrument_keys = []

    # These are set by write_file
    ntrains = 400
    firsttrain = 10000
    chunksize = 200

    def __init__(self, device_id, nsamples=None):
        """Create a dummy device

        :param str device_id: e.g. "SA1_XTD2_XGM/DOOCS/MAIN"
        :param int ntrains: e.g. 256
        :param int nsamples: For INSTRUMENT data only. Default is ntrains.
            If more, should be a multiple of ntrains. If fewer, samples will be
            spread evenly across the trains.
        :param int chunksize: The sample dimension will be padded to a multiple
            of this.
        """
        self.device_id = device_id
        self.nsamples = nsamples

    def write_control(self, f):
        """Write the CONTROL and RUN data, and the relevant parts of INDEX"""
        N = self.ntrains

        # INDEX
        i_first = f.create_dataset('INDEX/%s/first' % self.device_id,
                                   (N,), 'u8', maxshape=(None,))
        i_count = f.create_dataset('INDEX/%s/count' % self.device_id,
                                   (N,), 'u8', maxshape=(None,))
        i_first[:] = np.arange(N)
        i_count[:] = 1

        # CONTROL & RUN
        # Creating empty datasets for now.
        for (topic, datatype, dims) in self.control_keys:
            f.create_dataset('CONTROL/%s/%s/timestamp' % (self.device_id, topic),
                             (N,), 'u8', maxshape=(None,))
            f.create_dataset('CONTROL/%s/%s/value' % (self.device_id, topic),
                             (N,)+dims, datatype, maxshape=((None,)+dims))

            # RUN is the value at the start of the run
            f.create_dataset('RUN/%s/%s/timestamp' % (self.device_id, topic),
                             (1,), 'u8', maxshape=(None,))
            f.create_dataset('RUN/%s/%s/value' % (self.device_id, topic),
                             (1,)+dims, datatype, maxshape=((None,)+dims))

    def write_instrument(self, f):
        """Write the INSTRUMENT data, and the relevants parts of INDEX"""
        train0 = self.firsttrain

        if self.nsamples is None:
            self.nsamples = self.ntrains

        if self.nsamples == 0:
            first = count = 0
            trainids = []
        elif self.nsamples < self.ntrains:
            first = np.linspace(0, self.nsamples, endpoint=False,
                                num=self.ntrains, dtype='u8')
            count = np.zeros((self.ntrains,), dtype='u8')
            count[:-1] = first[1:] - first[:-1]
            if count.sum() < self.nsamples:
                count[-1] = 1
            assert count.sum() == self.nsamples
            trainids = np.linspace(train0, train0 + self.ntrains, endpoint=False,
                                   num=self.nsamples, dtype='u8')
        elif self.nsamples == self.ntrains:
            first = np.arange(self.ntrains)
            count = 1
            trainids = np.arange(train0, train0 + self.ntrains)
        else:  # nsamples > ntrains
            count = self.nsamples // self.ntrains
            first = np.arange(0, self.nsamples, step=count)
            trainids = np.repeat(np.arange(train0, train0 + self.ntrains), count)

        Npad = self.nsamples
        if Npad % self.chunksize:
            Npad += + self.chunksize - (Npad % self.chunksize)

        for channel in self.output_channels:
            dev_chan = '%s:%s' % (self.device_id, channel)

            # INDEX
            i_first = f.create_dataset('INDEX/%s/first' % dev_chan,
                                       (self.ntrains,), 'u8', maxshape=(None,))
            i_count = f.create_dataset('INDEX/%s/count' % dev_chan,
                                       (self.ntrains,), 'u8', maxshape=(None,))
            i_first[:] = first
            i_count[:] = count

            # INSTRUMENT
            tid = f.create_dataset('INSTRUMENT/%s/trainId' % dev_chan,
                                   (Npad,), 'u8', maxshape=(None,))
            tid[:self.nsamples] = trainids
            for (topic, datatype, dims) in self.instrument_keys:
                f.create_dataset('INSTRUMENT/%s/%s' % (dev_chan, topic),
                                 (Npad,) + dims, datatype, maxshape=((None,) + dims))

    def datasource_ids(self):
        if self.control_keys:
            yield 'CONTROL/' + self.device_id
        if self.instrument_keys:
            for channel in self.output_channels:
                yield 'INSTRUMENT/%s:%s' % (self.device_id, channel)


vlen_bytes = h5py.special_dtype(vlen=bytes)

def write_metadata(h5file, data_sources, chunksize=16):
    N = len(data_sources)
    if N % chunksize:
        N += chunksize - (N % chunksize)

    root = [ds.split('/', 1)[0] for ds in data_sources]
    devices = [ds.split('/', 1)[1] for ds in data_sources]

    sources_ds = h5file.create_dataset('METADATA/dataSourceId', (N,),
                                       dtype=vlen_bytes, maxshape=(None,))
    sources_ds[:len(data_sources)] = data_sources
    root_ds = h5file.create_dataset('METADATA/root', (N,),
                                    dtype=vlen_bytes, maxshape=(None,))
    root_ds[:len(data_sources)] = root
    devices_ds = h5file.create_dataset('METADATA/deviceId', (N,),
                                       dtype=vlen_bytes, maxshape=(None,))
    devices_ds[:len(data_sources)] = devices

def write_train_ids(f, path, N, first=10000, chunksize=16):
    """Make a dataset of fake train IDs at the given path

    Real train IDs are much larger (~10^9), so hopefully these won't be mistaken
    for real ones.
    """
    if N % chunksize:
        Npad = N + chunksize - (N % chunksize)
    else:
        Npad = N
    ds = f.create_dataset(path, (Npad,), 'u8', maxshape=(None,))
    ds[:N] = np.arange(first, first + N)
