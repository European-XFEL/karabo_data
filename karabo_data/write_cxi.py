"""Writing CXI files from AGIPD/LPD data"""
import h5py
import logging
import numpy as np

log = logging.getLogger(__name__)

class VirtualCXIWriter:
    def __init__(self, detdata):
        self.detdata = detdata

        self.nmodules = len(detdata.modno_to_source)
        self.modulenos = sorted(detdata.modno_to_source)

        train_ids = detdata.train_ids
        frames_per_train = detdata.frames_per_train
        ntrains = np.uint64(len(train_ids))
        self.nframes = ntrains * frames_per_train
        log.info("%d frames per train, %d frames in total",
                     frames_per_train, self.nframes)
        self.shape = (self.nframes, self.nmodules) + detdata.module_shape
        log.info("Virtual data shape: %r", self.shape)

        self.train_ids_perframe = np.repeat(train_ids, frames_per_train)

        positions = np.arange(0, ntrains, dtype=np.uint64) * frames_per_train
        self.train_id_to_ix = dict(zip(train_ids, positions))

    @property
    def data(self):
        return self.detdata.data

    def collect_pulse_ids(self):
        # Gather pulse IDs
        NO_PULSE_ID = 9999
        pulse_ids = np.full((self.nmodules, self.nframes), NO_PULSE_ID,
                            dtype=np.uint64)

        for i, source in enumerate(self.detdata.source_to_modno):
            for chunk in self.data._find_data_chunks(source, 'image.pulseId'):
                ix = self.train_id_to_ix[chunk.train_ids[0]]
                n = chunk.counts.sum()
                chunk_tids = np.repeat(chunk.train_ids, chunk.counts.astype(np.intp))
                if (chunk_tids != self.train_ids_perframe[ix: ix + n]).any():
                    raise Exception(
                        "Train IDs mismatch in chunk from source {} "
                        "({} frames from train ID {})"
                            .format(source, n, chunk_tids[0])
                    )
                # In some cases, there's an extra dimension of length 1
                if chunk.dataset.ndim > 1:
                    chunk_pulse_ids = chunk.dataset[chunk.slice, 0]
                else:
                    chunk_pulse_ids = chunk.dataset[chunk.slice]
                pulse_ids[i, ix: ix + n] = chunk_pulse_ids

        # Sanity checks on pulse IDs
        pulse_ids_min = pulse_ids.min(axis=0)
        if (pulse_ids_min == NO_PULSE_ID).any():
            raise Exception("Failed to find pulse IDs for some data")
        pulse_ids[pulse_ids == NO_PULSE_ID] = 0
        if (pulse_ids_min != pulse_ids.max(axis=0)).any():
            raise Exception("Inconsistent pulse IDs for different modules")

        # Pulse IDs make sense. Drop the modules dimension, giving one pulse ID
        # for each frame.
        return pulse_ids_min

    def collect_data(self):
        src = next(iter(self.detdata.source_to_modno))
        h5file = self.data._source_index[src][0].file
        image_grp = h5file['INSTRUMENT'][src]['image']

        layouts = {
            'data': h5py.VirtualLayout(self.shape, dtype=image_grp['data'].dtype),
            'gain': h5py.VirtualLayout(self.shape, dtype=image_grp['gain'].dtype),
            'mask': h5py.VirtualLayout(self.shape, dtype=image_grp['mask'].dtype),
            'cellId': h5py.VirtualLayout((self.nframes, self.nmodules),
                                         dtype=image_grp['data'].dtype),
        }

        for name, layout in layouts.items():
            key = 'image.{}'.format(name)
            nchunks = 0
            have_data = np.zeros((self.nframes, self.nmodules), dtype=bool)
            for source, modno in self.detdata.source_to_modno.items():
                mod_ix = self.modulenos.index(modno)
                for chunk in self.data._find_data_chunks(source, key):
                    ix = self.train_id_to_ix[chunk.train_ids[0]]
                    n = chunk.counts.sum()
                    vsrc = h5py.VirtualSource(chunk.dataset)
                    layout[ix:ix+n, mod_ix] = vsrc[chunk.slice]

                    have_data[ix:ix+n, mod_ix] = True
                    nchunks += 1

            filled_pct = 100 * have_data.sum() / have_data.size
            log.info("Assembled %d chunks for %s, filling %.2f%% of the hyperslab",
                     nchunks, key, filled_pct)

        return layouts

    def write(self, filename):
        pulse_ids = self.collect_pulse_ids()
        experiment_ids = np.core.defchararray.add(np.core.defchararray.add(
            self.train_ids_perframe.astype(str), ':'), pulse_ids.astype(str))

        layouts = self.collect_data()

        log.info("Writing to %s", filename)

        with h5py.File(filename, 'w', libver='latest') as f:
            f.create_dataset('cxi_version', data=[150])
            d = f.create_dataset('entry_1/experiment_identifier',
                                 shape=experiment_ids.shape,
                                 dtype=h5py.special_dtype(vlen=str))
            d[:] = experiment_ids

            # pulseId, trainId, cellId are not part of the CXI standard, but
            # it allows extra data.
            f.create_dataset('entry_1/pulseId', data=pulse_ids)
            f.create_dataset('entry_1/trainId', data=self.train_ids_perframe)
            cellids = f.create_virtual_dataset('entry_1/cellId',
                                               layouts['cellId'])
            cellids.attrs['axes'] = 'experiment_identifier:module_identifier'


            dgrp = f.create_group('entry_1/instrument_1/detector_1')
            data = dgrp.create_virtual_dataset(
                'data', layouts['data'], fillvalue=np.nan
            )
            data.attrs['axes'] = 'experiment_identifier:module_identifier:y:x'
            gain = dgrp.create_virtual_dataset(
                'gain', layouts['gain'], fillvalue=np.nan
            )
            gain.attrs['axes'] = 'experiment_identifier:module_identifier:y:x'
            mask = dgrp.create_virtual_dataset(
                'mask', layouts['mask'], fillvalue=np.nan
            )
            mask.attrs['axes'] = 'experiment_identifier:module_identifier:y:x'
            dgrp['experiment_identifier'] = h5py.SoftLink('/entry_1/experiment_identifier')

            f['entry_1/data_1'] = h5py.SoftLink('/entry_1/instrument_1/detector_1')

            dgrp.create_dataset('module_identifier', data=self.modulenos)

        log.info("Finished writing virtual CXI file")
