Release Notes
=============

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
