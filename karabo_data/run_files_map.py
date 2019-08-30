import json
import logging
import os
import os.path as osp
import re

from .read_machinery import DATA_ROOT_DIR

log = logging.getLogger(__name__)

class RunFilesMap:
    """Cached data about HDF5 files in a run directory

    Stores the train IDs and source names in each file, along with some
    metadata to check that the cache is still valid. The cached information
    can be stored in:

    - (run dir)/karabo_data_map.json
    - (proposal dir)/scratch/.karabo_data_maps/raw_r0032.json
    """
    cache_file = None

    def __init__(self, directory):
        self.directory = osp.abspath(directory)
        self.files_data = {}

        self.candidate_paths = [osp.join(directory, 'karabo_data_map.json')]
        m = re.match(
            r'(%s/\w+/\w+/\w+)/(raw|proc)/(r\d+)/?$' % DATA_ROOT_DIR, directory
        )
        if m:
            prop_dir, raw_proc, run_nr = m.groups()
            fname = '%s_%s.json' % (raw_proc, run_nr)
            self.candidate_paths.append(
                osp.join(prop_dir, 'scratch', '.karabo_data_maps', fname)
            )

        self.load()

    def load(self):
        """Load the cached data

        This skips over invalid cache entries(based on the file's size & mtime).
        """
        loaded_data = []

        for path in self.candidate_paths:
            if osp.isfile(path):
                with open(path) as f:
                    loaded_data = json.load(f)

                self.cache_file = path
                log.debug("Loaded cached files map from %s", path)
                break

        for info in loaded_data:
            filename = info['filename']
            st = os.stat(osp.join(self.directory, filename))
            if (st.st_mtime == info['mtime']) and (st.st_size == info['size']):
                self.files_data[filename] = info

    def get(self, path):
        """Get cache entry for a file path

        Returns a dict or None
        """
        dirname, fname = osp.split(osp.abspath(path))
        if (dirname == self.directory) and (fname in self.files_data):
            return self.files_data[fname]

        return None

    def save(self, files):
        """Save the cache if needed

        This skips writing the cache out if all the data files already have
        valid cache entries. It also silences permission errors from writing
        the cache file.
        """
        need_save = False

        for file_access in files:
            dirname, fname = osp.split(osp.abspath(file_access.filename))
            if (dirname == self.directory) and (fname not in self.files_data):
                log.debug("Will save cached data for %s", fname)
                need_save = True

                st = os.stat(file_access.filename)
                self.files_data[fname] = {
                    'filename': fname,
                    'mtime': st.st_mtime,
                    'size': st.st_size,
                    'train_ids': [int(t) for t in file_access.train_ids],
                    'control_sources': sorted(file_access.control_sources),
                    'instrument_sources': sorted(file_access.instrument_sources),
                }

        if need_save:
            save_data = [info for (_, info) in sorted(self.files_data.items())]
            for path in self.candidate_paths:
                try:
                    os.makedirs(osp.dirname(path), exist_ok=True)
                    f = open(path, 'w')
                except PermissionError:
                    continue

                with f:
                    json.dump(save_data, f, indent=2)
