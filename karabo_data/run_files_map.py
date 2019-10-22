import json
import logging
import os
import os.path as osp
import numpy as np
import re
from tempfile import mkstemp
import time

SCRATCH_ROOT_DIR = "/gpfs/exfel/exp/"

log = logging.getLogger(__name__)

def atomic_dump(obj, path, **kwargs):
    """Write JSON to a file atomically

    This aims to avoid garbled files from multiple processes writing the same
    cache. It doesn't try to protect against e.g. sudden power failures, as
    forcing the OS to flush changes to disk may hurt performance.
    """
    dirname, basename = osp.split(path)
    fd, tmp_filename = mkstemp(dir=dirname, prefix=basename)
    try:
        with open(fd, 'w') as f:
            json.dump(obj, f, **kwargs)
    except:
        os.unlink(tmp_filename)
        raise

    os.replace(tmp_filename, path)


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
        self.dir_stat = os.stat(self.directory)
        self.files_data = {}

        self.candidate_paths = self.map_paths_for_run(directory)

        self.load()

    def map_paths_for_run(self, directory):
        paths = [osp.join(directory, 'karabo_data_map.json')]
        # After resolving symlinks, data on Maxwell is stored in either
        # GPFS, e.g. /gpfs/exfel/d/proc/SCS/201901/p002212  or
        # dCache, e.g. /pnfs/xfel.eu/exfel/archive/XFEL/raw/SCS/201901/p002212
        m = re.match(
            #     raw/proc  instr  cycle prop   run
            r'.+/(raw|proc)/(\w+)/(\w+)/(p\d+)/(r\d+)/?$',
            os.path.realpath(directory)
        )
        if m:
            raw_proc, instr, cycle, prop, run_nr = m.groups()
            fname = '%s_%s.json' % (raw_proc, run_nr)
            prop_scratch = osp.join(
                SCRATCH_ROOT_DIR, instr, cycle, prop, 'scratch'
            )
            if osp.isdir(prop_scratch):
                paths.append(
                    osp.join(prop_scratch, '.karabo_data_maps', fname)
                )
        return paths

    def load(self):
        """Load the cached data

        This skips over invalid cache entries(based on the file's size & mtime).
        """
        loaded_data = []
        t0 = time.monotonic()

        for path in self.candidate_paths:
            try:
                with open(path) as f:
                    loaded_data = json.load(f)

                self.cache_file = path
                log.debug("Loaded cached files map from %s", path)
                break
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        for info in loaded_data:
            filename = info['filename']
            try:
                st = os.stat(osp.join(self.directory, filename))
            except OSError:
                continue
            if (st.st_mtime == info['mtime']) and (st.st_size == info['size']):
                self.files_data[filename] = info

        if loaded_data:
            dt = time.monotonic() - t0
            log.debug("Loaded cached files map in %.2g s", dt)

    def is_my_directory(self, dir_path):
        return osp.samestat(os.stat(dir_path), self.dir_stat)

    def get(self, path):
        """Get cache entry for a file path

        Returns a dict or None
        """
        dirname, fname = osp.split(osp.abspath(path))
        if self.is_my_directory(dirname) and (fname in self.files_data):
            d = self.files_data[fname]
            return {
                'train_ids': np.array(d['train_ids'], dtype=np.uint64),
                'control_sources': frozenset(d['control_sources']),
                'instrument_sources': frozenset(d['instrument_sources'])
            }

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
            if self.is_my_directory(dirname) and (fname not in self.files_data):
                log.debug("Will save cached data for %s", fname)
                need_save = True

                # It's possible that the file we opened has been replaced by a
                # new one before this runs. If possible, get the stat from the
                # file descriptor, which will always be accurate. Stat-ing the
                # filename will almost always work as a fallback.
                try:
                    fd = file_access.file.id.get_vfd_handle()
                except Exception:
                    log.warning("Can't get fd for %r, will stat name instead",
                                fname, exc_info=True)
                    st = os.stat(file_access.filename)
                else:
                    st = os.stat(fd)

                self.files_data[fname] = {
                    'filename': fname,
                    'mtime': st.st_mtime,
                    'size': st.st_size,
                    'train_ids': [int(t) for t in file_access.train_ids],
                    'control_sources': sorted(file_access.control_sources),
                    'instrument_sources': sorted(file_access.instrument_sources),
                }

        if need_save:
            t0 = time.monotonic()
            save_data = [info for (_, info) in sorted(self.files_data.items())]
            for path in self.candidate_paths:
                try:
                    os.makedirs(osp.dirname(path), exist_ok=True)
                    atomic_dump(save_data, path, indent=2)
                except PermissionError:
                    continue
                else:
                    dt = time.monotonic() - t0
                    log.debug("Saved run files map to %s in %.2g s", path, dt)
                    return

            log.debug("Unable to save run files map")
