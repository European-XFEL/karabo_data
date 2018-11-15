import numpy as np
import pandas as pd
import xarray

class DetectorData:
    """Interface to access X-ray detector data, from e.g. AGIPD or LPD
    """
    def __init__(self, data, source_to_modno, det_name):
        self.data = data
        self.source_to_modno = source_to_modno
        self.det_name = det_name

        # This should be a reversible 1-to-1 mapping
        self.modno_to_source = {m:s for (s, m) in source_to_modno.items()}
        assert len(self.modno_to_source) == len(self.source_to_modno)

        train_id_arr = np.asarray(self.data.train_ids)
        split_indices = np.where(np.diff(train_id_arr) != 1)[0]+1
        self.train_id_chunks = np.split(train_id_arr, split_indices)


    def __repr__(self):
        return "<DetectorData for detector {!r} with {} modules>".format(
            self.det_name, len(self.source_to_modno),
        )

    def _get_module_pulse_data(self, source, key):
        seq_arrays = []
        data_path = "/INSTRUMENT/{}/{}".format(source, key.replace('.', '/'))
        for f in self.data._source_index[source]:
            group = key.partition('.')[0]
            firsts, counts = f.get_index(source, group)

            for chunk_tids in self.train_id_chunks:
                if chunk_tids[-1] < f.train_ids[0] or chunk_tids[0] > f.train_ids[-1]:
                    # No overlap
                    continue
                first_tid = max(chunk_tids[0], f.train_ids[0])
                first_train_idx = np.nonzero(f.train_ids == first_tid)[0][0]
                last_tid = min(chunk_tids[-1], f.train_ids[-1])
                last_train_idx = np.nonzero(f.train_ids == last_tid)[0][0]
                chunk_firsts = firsts[first_train_idx:last_train_idx+1]
                chunk_counts = counts[first_train_idx:last_train_idx+1]
                data_slice = slice(
                    chunk_firsts[0],
                    int(chunk_firsts[-1] + chunk_counts[-1]),
                )
                trainids = np.repeat(np.arange(first_tid, last_tid+1, dtype=np.uint64),
                                     chunk_counts.astype(np.intp))
                pulse_id = f.file['/INSTRUMENT/{}/{}/pulseId'.format(source, group)][data_slice]
                # Raw files have a spurious extra dimension
                if pulse_id.ndim >= 2 and pulse_id.shape[1] == 1:
                    pulse_id = pulse_id[:, 0]

                index = pd.MultiIndex.from_arrays([trainids, pulse_id],
                                                  names=['trainId', 'pulseId'])

                data = f.file[data_path][data_slice]
                # Raw files have a spurious extra dimension
                if data.ndim >= 2 and data.shape[1] == 1:
                    data = data[:, 0]

                # TODO: this assumes we can tell what the axes are just from the
                # number of dimensions. Works for the data we've seen, but we
                # should look for a more reliable way.
                if data.ndim == 4:
                    # image.data in raw data
                    dims = ['train_pulse', 'data_gain', 'ss', 'fs']
                elif data.ndim == 3:
                    # image.data, image.gain, image.mask in calibrated data
                    dims = ['train_pulse', 'ss', 'fs']
                else:
                    # Everything else seems to be 1D
                    dims = ['train_pulse']

                arr = xarray.DataArray(data, {'train_pulse': index}, dims=dims)

                # Separate train & pulse dimensions, and arrange dimensions
                # so that the data is contiguous in memory.
                arr = arr.unstack('train_pulse').transpose(
                    'trainId',  'pulseId', 'ss', 'fs')
                seq_arrays.append(arr)

        non_empty = [a for a in seq_arrays if (a.size > 0)]
        if not non_empty:
            if seq_arrays:
                # All per-file arrays are empty, so just return the first one.
                return seq_arrays[0]

            raise Exception(("Unable to get data for source {!r}, key {!r}. "
                             "Please report an issue so we can investigate")
                            .format(source, key))

        return xarray.concat(sorted(non_empty,
                                   key=lambda a: a.coords['trainId'][0]),
                            dim='trainId')

    def get_array(self, key):
        """Get a labelled array of detector data

        Parameters
        ----------

        key: str
          The data to get, e.g. 'image.data' for pixel values.
        """
        arrays = []
        modnos = []
        for modno, source in sorted(self.modno_to_source.items()):
            # At present, all the per-pulse data is stored in the 'image' key.
            # If that changes, this check will need to change as well.
            if key.startswith('image.'):
                arrays.append(self._get_module_pulse_data(source, key))
            else:
                arrays.append(self.data.get_array(source, key))
            modnos.append(modno)

        return xarray.concat(arrays, pd.Index(modnos, name='module'))
