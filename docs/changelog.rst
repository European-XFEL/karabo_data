Release Notes
=============

0.5
---

Data access
~~~~~~~~~~~

- New method :meth:`~.get_data_counts` to find how many data points were
  recorded in each train for a given source and key.
- Create a virtual dataset for any single dataset with
  :meth:`~.get_virtual_dataset` (:ghpull:`162`).
  See :doc:`parallel_example` for how this can be useful.
- Write a file with virtual datasets for all selected data with
  :meth:`~.write_virtual` (:ghpull:`132`).
- Data from the supported multi-module detectors (AGIPD, LPD & DSSC) can be
  exposed in CXI format using a virtual dataset - see
  :meth:`~.write_virtual_cxi` (:ghpull:`150`, :ghpull:`166`, :ghpull:`173`).
- New class :class:`~.DSSC` for accessing DSSC data (:ghpull:`171`).
- New function :func:`~.open_run` to access a run by proposal and run number
  rather than path (:ghpull:`147`).
- :func:`~.stack_detector_data` now allows input data where some sources don't
  have the specified key (:ghpull:`141`).

Detector geometry
~~~~~~~~~~~~~~~~~

- New class :class:`~.DSSC_Geometry` for handling DSSC detector geometry (:ghpull:`155`).
- :class:`~.LPD_1MGeometry` can now read and write CrystFEL format
  geometry files, and produce PyFAI distortion arrays (:ghpull:`168`, :ghpull:`129`).
- :meth:`~.AGIPD_1MGeometry.write_crystfel_geom` (for AGIPD and LPD geometry)
  now accepts various optional parameters for other details to be written into
  the geometry file, such as the detector distance (``clen``) and the photon
  energy (:ghpull:`168`).
- New method :meth:`~.AGIPD_1MGeometry.get_pixel_positions` to get the physical
  position of every pixel in a detector, for all of AGIPD, LPD and DSSC
  (:ghpull:`142`).
- New method :meth:`~.AGIPD_1MGeometry.data_coords_to_positions` to convert data
  array coordinates to physical positions, for AGIPD and LPD (:ghpull:`142`).

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
