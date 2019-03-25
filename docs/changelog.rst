Release Notes
=============

0.4
---

- Python 3.5 is now the minimum required version.
- Fix compatibility with numpy 1.14 (the version installed in Anaconda on the
  Maxwell cluster).
- Better error message from :func:`~.stack_detector_data` when passed
  non-detector data.

0.3
---

New features:

- New interfaces for working with :doc:`geometry`.
- New interfaces for accessing :doc:`agipd_lpd_data`.
- :meth:`~.DataCollection.select_trains` can now select arbitrary specified
  trains, not just a slice.
- :meth:`~.DataCollection.get_array` can take a region of interest (``roi``)
  parameter to select a slice of data from each train.
- A newly public :meth:`~.DataCollection.keys_for_source` method to list keys
  for a given source.

Fixes:

- :func:`~.stack_detector_data` can handle missing detector modules.
- Source sets have been changed to frozen sets. Use
  :meth:`~.DataCollection.select` to choose a subset of sources.
- :meth:`~.DataCollection.get_array` now only loads the data for selected
  trains.
- :meth:`~.DataCollection.get_array` works with data recorded more than once per
  train.

0.2
---

- New command ``karabo-data-validate`` to check the integrity of data files.
- New methods to select a subset of data: :meth:`~.DataCollection.select`,
  :meth:`~.DataCollection.deselect`, :meth:`~.DataCollection.select_trains`,
  :meth:`~.DataCollection.union`,
- Selected data can be written back to a new HDF5 file with
  :meth:`~.DataCollection.write`.
- :func:`~.RunDirectory` and :func:`~.H5File` are now functions which return a
  :class:`DataCollection` object, rather than separate classes. Most code using
  these should still work, but checking the type with e.g. ``isinstance()``
  may break.
