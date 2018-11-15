import numpy as np

class DetectorModule:
    # Overridden in subclasses:
    image_dims = ()
    detector_data_size = 0

    # Set by write_file:
    ntrains = 100
    firsttrain = 10000
    chunksize = 32

    output_parts = [
        'detector',
        'header',
        'image',
        'trailer',
    ]

    def __init__(self, device_id, frames_per_train=64):
        self.device_id = device_id
        self.frames_per_train = frames_per_train

    def write_control(self, f):
        """Write the CONTROL and RUN data, and the relevant parts of INDEX"""
        pass

    @property
    def image_keys(self):
        return [
            ('cellId', 'u2', (1,)),
            ('data', 'u2', self.image_dims),
            ('length', 'u4', (1,)),
            ('status', 'u2', (1,)),
        ]

    @property
    def other_keys(self):
        return [
            ('detector/data', 'u1', (self.detector_data_size,)),
            ('header/dataId', 'u8', ()),
            ('header/linkId', 'u8', ()),
            ('header/magicNumberBegin', 'i1', (8,)),
            ('header/majorTrainFormatVersion', 'u4', ()),
            ('header/minorTrainFormatVersion', 'u4', ()),
            ('header/pulseCount', 'u8', ()),
            ('header/reserved', 'u1', (16,)),
            ('trailer/checksum', 'i1', (16,)),
            ('trailer/magicNumberEnd', 'i1', (8,)),
            ('trailer/status', 'u8', ()),
        ]

    def write_instrument(self, f):
        """Write the INSTRUMENT data, and the relevants parts of INDEX"""
        trainids = np.arange(self.firsttrain, self.firsttrain + self.ntrains)

        ntrains_pad = self.ntrains
        if ntrains_pad % self.chunksize:
            ntrains_pad += + self.chunksize - (ntrains_pad % self.chunksize)

        # INDEX
        for part in self.output_parts:
            dev_chan = '%s:xtdf/%s' % (self.device_id, part)

            i_first = f.create_dataset('INDEX/%s/first' % dev_chan,
                                       (self.ntrains,), 'u8', maxshape=(None,))
            i_count = f.create_dataset('INDEX/%s/count' % dev_chan,
                                       (self.ntrains,), 'u8', maxshape=(None,))
            if part == 'image':
                i_first[:] = np.arange(self.ntrains) * self.frames_per_train
                i_count[:] = self.frames_per_train
            else:
                i_first[:] = np.arange(self.ntrains)
                i_count[:] = 1


        # INSTRUMENT (image)
        nframes = self.ntrains * self.frames_per_train
        ds = f.create_dataset('INSTRUMENT/%s:xtdf/image/trainId' % self.device_id,
                              (nframes, 1), 'u8', maxshape=(None, 1))
        ds[:, 0] = np.repeat(trainids, self.frames_per_train)

        pid = f.create_dataset('INSTRUMENT/%s:xtdf/image/pulseId' % self.device_id,
                               (nframes, 1), 'u8', maxshape=(None, 1))
        pid[:, 0] = np.tile(np.arange(0, self.frames_per_train, dtype='u8'),
                                self.ntrains)

        for (key, datatype, dims) in self.image_keys:
            f.create_dataset('INSTRUMENT/%s:xtdf/image/%s' % (self.device_id, key),
                             (nframes,) + dims, datatype, maxshape=((None,) + dims))


        # INSTRUMENT (other parts)
        for part in ['detector', 'header', 'trailer']:
            ds = f.create_dataset('INSTRUMENT/%s:xtdf/%s/trainId' % (self.device_id, part),
                                  (ntrains_pad,), 'u8', maxshape=(None,))
            ds[:self.ntrains] = trainids

        for (key, datatype, dims) in self.other_keys:
            f.create_dataset('INSTRUMENT/%s:xtdf/%s' % (self.device_id, key),
                     (ntrains_pad,) + dims, datatype, maxshape=((None,) + dims))

    def datasource_ids(self):
        for part in self.output_parts:
            yield 'INSTRUMENT/%s:xtdf/%s' % (self.device_id, part)


class AGIPDModule(DetectorModule):
    image_dims = (2, 512, 128)
    detector_data_size = 5408

class LPDModule(DetectorModule):
    image_dims = (1, 256, 256)
    detector_data_size = 416
