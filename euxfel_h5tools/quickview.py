import numpy as np
import matplotlib.pyplot as plot


class QuickView:
    """Pun intended
        This object displays a 3D array as provided by calibrated
        hdf5 files.
        Given a 3D numpy array, it will provide you with a way to
        easily iterate over the pulses to display their respective
        images.

        First, instantiate:

            quick_v = QuickView()
            # or
            quick_v = QuickView(data)

        Set the data to your ndArray:

            shower.data = data

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
