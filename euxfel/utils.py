"""
Helpers functions for the euxfel_h5tools package.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

import fabio
import h5py
import numpy as np


__all__ = ['hdf5_file_info', 'hdf5_paths', 'hdf5_to_cbf', 'numpy_to_cbf',
           'QuickView']


class QuickView:
    """Pun intended
        This object displays a 3D array as provided by calibrated
        hdf5 files.
        Given a 3D numpy array, it will provide you with a way to
        easily iterate over the pulses to display their respective
        images.

        First, instantiate and give it the data (a 3D numpy array):

            quick_v = QuickView(data)
            # or
            quick_v = QuickView()
            quick_v.data = data

        You can now iterate over it in three different ways:

            next(quick_v)

            quick_v.next()
            quick_v.previous()

            quick_v.pos = len(quick_v-1)

        You can also display a specific image without changing
        the position:

            quick_v.display(int)

    """
    _image = None
    _data = None
    _current_index = 0

    def __init__(self, data=None):
        if data:
            self.data = data

    @property
    def pos(self):
        return self._current_index

    @pos.setter
    def pos(self, pos):
        if self._data is not None:
            if 0 <= pos < len(self):
                self._current_index = pos
                self.show()
            else:
                err = ("value should be 0 < value < "
                       "{}".format(self._data.shape[0]))
                raise ValueError(err)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 3:
            raise TypeError("Expected a 3D numpy array")

        self._data = data
        self._current_index = 0
        self.display()

    def next(self):
        if self._current_index < len(self):
            self._current_index += 1
            self.display()

    def prev(self):
        if self._current_index > 0:
            self._current_index -= 1
            self.display()

    def display(self, index=None):
        import matplotlib.pyplot as plot
        if index is None:
            index = self._current_index

        image_frame = self.data[index, :, :]

        if self._image is None:
            self._image = plot.imshow(image_frame)
        else:
            self._image.set_data(image_frame)

        self._image.axes.set_title("pulseId: {}".format(index))
        plot.draw()

    def __next__(self):
        self.next()

    def __len__(self):
        return self._data.shape[0]


def hdf5_file_info(files, multiline=False):
    """Gather basic info about HDF5 file

    :param files: a list of filenames to check
    :param multiline: print each information on a new line (default False)
    """
    first_train = float('inf')
    last_train = 0
    total_size = 0
    total_entries = 0
    instruments = set()
    invalid = []

    for filename in files:
        out = ""
        try:
            xfel_file = h5py.File(filename, 'r')
            sfile = "File: '{}'".format(xfel_file.filename)

            size_mb = xfel_file.fid.get_filesize() / 1000000
            ssize = "Size: {} MB".format(size_mb)

            inst = xfel_file["INSTRUMENT"]
            sinst = "Instruments: {}".format(", ".join(inst))

            f_train = xfel_file["INDEX/trainId"][0]
            sf_train = "First Train: {}".format(f_train)

            l_train = xfel_file["INDEX/trainId"][-1]
            sl_train = "Last Train: {}".format(l_train)

            entries = len(xfel_file["INDEX/trainId"])
            sentries = "Entries: {}".format(entries)

            paths = list(xfel_file["METADATA/dataSourceId"])
            header_path = [p for p in paths if p.endswith(b"header")][0]
            pulses = xfel_file[header_path + b"/pulseCount"][0]
            spulses = "Pulses per Train: {}".format(pulses)

            instruments.update(inst)
            total_size += size_mb
            total_entries += entries

            if f_train <= first_train:
                first_train = f_train
            if l_train >= last_train:
                last_train = l_train

            if multiline:
                out = ("{f}\n{size}\n{entries}\n"
                       "{pulses}\n{first}\n{last}\n"
                       "{instruments}".format(f=sfile,
                                              size=ssize,
                                              entries=sentries,
                                              pulses=spulses,
                                              first=sf_train,
                                              last=sl_train,
                                              instruments=sinst))
            else:
                out = ("{f}\n\t{size}\t{entries}\t"
                       "{pulses}\n\t{first}\t{last}\n\t"
                       "{instruments}\n".format(f=sfile,
                                                size=ssize,
                                                entries=sentries,
                                                pulses=spulses,
                                                first=sf_train,
                                                last=sl_train,
                                                instruments=sinst))

        except (OSError, IOError, KeyError):
            # The errors could be:
            #  - OSError: not an HDF5 file
            #  - IOError: truncated file
            #  - KeyError: one of the keys used was not found,
            #                therefore not a EuXFEL specific file
            out = "{}: not an EuXFEL HDF5 file\n".format(filename)
            invalid.append(filename)

        print(out)

    if len(files) > 1:
        if multiline:
            total = "Total Files: {}\n".format(len(files) - len(invalid))
            total += "Total File Size: {} MB\n".format(total_size)
            total += "First Train: {}\n".format(first_train)
            total += "Last Train: {}\n".format(last_train)
            total += "Instruments: {}\n".format(", ".join(instruments))
        else:
            total = "---\n"
            total += "Total Files: {}\t".format(len(files) - len(invalid))
            total += "Total File Size: {} MB\n".format(total_size)
            total += "First Train: {}\t".format(first_train)
            total += "Last Train: {}\n".format(last_train)
            total += "Instruments: {}\n".format(", ".join(instruments))
            total += "-" * 3

        print(total)

    if invalid:
        print("These are not valid files: {}".format(", ".join(invalid)))


def hdf5_paths(ds, indent=0, maxlen=100):
    """Visit and print name of all element in HDF5 file (from S Hauf)"""

    for k in list(ds.keys())[:maxlen]:
        print(" " * indent + k)
        if isinstance(ds[k], h5py.Group):
            hdf5_paths(ds[k], indent + 4, maxlen)
        else:
            print(" " * indent + k)


def numpy_to_cbf(np_array, index=0, header=None):
    """Given a 3D numpy array, convert it to a CBF data object"""
    img_reduced = np_array[index, ...]
    return fabio.cbfimage.cbfimage(header=header or {}, data=img_reduced)


def hdf5_to_cbf(in_h5file, cbf_filename, index, header=None):
    """Conversion from HDF5 file to cbf binary image file"""
    try:
        tmpf = h5py.File(in_h5file, 'r')
        paths = list(tmpf["METADATA/dataSourceId"])
        image_path = [p for p in paths if p.endswith(b"image")][0]
        images = tmpf[image_path + b"/data"][index]
        cbf_out = numpy_to_cbf(images)
        cbf_out.write(cbf_filename)
        print("Convert {} index {} to {}".format(in_h5file,
                                                 index,
                                                 cbf_filename))
    except IOError:
        print("{}: Could not be opened.".format(in_h5file))
        return
    except ValueError as ve:
        print(str(ve))
