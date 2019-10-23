Reading data files
==================

.. _opening-files:

Opening files
-------------

You will normally access data from a run, which is stored as a directory
containing HDF5 files. You can open a run using :func:`RunDirectory` with the
path of the directory, or using :func:`open_run` with the proposal number and
run number to look up the standard data paths on the Maxwell cluster.

.. module:: karabo_data

.. autofunction:: RunDirectory

.. autofunction:: open_run

   .. versionadded:: 0.5

You can also open a single file. The methods described below all work for either
a run or a single file.

.. autofunction:: H5File

Data structure
--------------

A run (or file) contains data from various *sources*, each of which has *keys*.
For instance, ``SA1_XTD2_XGM/XGM/DOOCS`` is one source, for an 'XGM' device
which monitors the beam, and its keys include ``beamPosition.ixPos`` and
``beamPosition.iyPos``.

European XFEL produces ten *pulse trains* per second, each of which can contain
up to 2700 X-ray pulses. Each pulse train has a unique train ID, which is used
to refer to all data associated with that 0.1 second window.

.. class:: DataCollection

   .. attribute:: train_ids

      A list of the train IDs included in this data.
      The data recorded may not be the same for each train.

   .. attribute:: control_sources

      A set of the control source names in this data, in the format
      ``"SA3_XTD10_VAC/TSENS/S30100K"``. Control data is always recorded
      exactly once per train.

   .. attribute:: instrument_sources

      A set of the instrument source names in this data,
      in the format ``"FXE_DET_LPD1M-1/DET/15CH0:xtdf"``.
      Instrument data may be recorded zero to many times per train.

   .. attribute:: all_sources

      A set of names for both instrument and control sources.
      This is the union of the two sets above.

   .. automethod:: keys_for_source

   .. automethod:: get_data_counts

   .. automethod:: info

.. _data-by-source-and-key:

Getting data by source & key
----------------------------

Where data will fit into memory, it's usually quickest and most convenient
to load it like this.

.. seealso::

   :doc:`xpd_examples`
      Examples of accessing data like this

   `xarray documentation <http://xarray.pydata.org/en/stable/indexing.html>`__
     How to use the arrays returned by :meth:`~.DataCollection.get_array`

   `pandas documentation <https://pandas.pydata.org/pandas-docs/stable/>`__
     How to use the objects returned by :meth:`~.DataCollection.get_series` and
     :meth:`~.DataCollection.get_dataframe`

.. class:: DataCollection

   .. automethod:: get_array

   .. automethod:: get_series

   .. automethod:: get_dataframe

   .. automethod:: get_virtual_dataset

      .. versionadded:: 0.5

.. _data-by-train:

Getting data by train
---------------------

Some kinds of data, e.g. from AGIPD, are too big to load a whole run into
memory at once. In these cases, it's convenient to load one train at a time.

When accessing data like this, it's worth selecting which sources you're
interested in, either using :meth:`~.DataCollection.select`, or the ``devices=``
parameter. This avoids reading all the other data.

.. class:: DataCollection

   .. automethod:: trains

   .. automethod:: train_from_id

   .. automethod:: train_from_index

Selecting & combining data
--------------------------

These methods all return a new :class:`DataCollection` object with the selected
data, so you use them like this::

    sel = run.select("*/XGM/*")
    # sel includes only XGM sources
    # run still includes all the data

.. class:: DataCollection

   .. automethod:: select

   .. automethod:: deselect

   .. automethod:: select_trains

   .. automethod:: union

Writing selected data
---------------------

.. class:: DataCollection

   .. automethod:: write

   .. automethod:: write_virtual

Missing data
------------

What happens if some data was not recorded for a given train?

Control data is duplicated for each train until it changes.
If the device cannot send changes, the last values will be recorded for each
subsequent train until it sends changes again.
There is no general way to distinguish this scenario from values which
genuinely aren't changing.

Parts of instrument data may be missing from the file. These will also be
missing from the data returned by ``karabo_data``:

- The train-oriented methods :meth:`~.DataCollection.trains`,
  :meth:`~.DataCollection.train_from_id`, and
  :meth:`~.DataCollection.train_from_index` give you dictionaries keyed by
  source and key name. Sources and keys are only included if they have
  data for that train.
- :meth:`~.DataCollection.get_array`, and
  :meth:`~.DataCollection.get_series` skip over trains which are missing data.
  The indexes on the returned DataArray or Series objects link the returned
  data to train IDs. Further operations with xarray or pandas may drop
  misaligned data or introduce fill values.
- :meth:`~.DataCollection.get_dataframe` includes rows for which any column has
  data. Where some but not all columns have data, the missing values are filled
  with ``NaN`` by pandas' `missing data handling
  <http://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html>`__.

Missing data does not necessarily mean that something has gone wrong:
some devices send data at less than 10 Hz (the train rate), so they always
have gaps between updates.

Data problems
-------------

If you encounter problems accessing data with ``karabo_data``, there may be
problems with the data files themselves. Use the ``karabo-data-validate``
command to check for this (see :doc:`validation`).

Here are some problems we've seen, and possible solutions or workarounds:

- Indexes point to data beyond the end of datasets:
  this has previously been caused by bugs in the detector calibration pipeline.
  If you see this in calibrated data (in the ``proc/`` folder),
  ask for the relevant runs to be re-calibrated.
- Train IDs are not strictly increasing:
  issues with the timing system when the data is recorded can create an
  occasional train ID which is completely out of sequence.
  Usually it seems to be possible to ignore this and use the remaining data,
  but if you have any issues, please let us know.

  - In one case, a train ID had the maximum possible value (2\ :sup:`64` - 1),
    causing :meth:`~.info` to fail. You can select everything except this train
    using :meth:`~.select_trains`::

        from karabo_data import by_id
        sel = run.select_trains(by_id[:2**64-1])

If you're having problems with karabo_data, you can also try searching
`previously reported issues <https://github.com/European-XFEL/karabo_data/issues?q=is%3Aissue>`_
to see if anyone has encountered similar symptoms.

Cached run data maps
--------------------

When you open a run in karabo_data, it needs to know what data is in each file.
Each file has metadata describing its contents, but reading this from every file
is slow, especially on GPFS. karabo_data therefore tries to cache this
information the first time a run is opened, and reuse it when opening that run
again.

This should happen automatically, without the user needing to know about it.
You only need these details if you think caching may be causing problems.

- Caching is triggered when you use :func:`RunDirectory` or :func:`open_run`.
- There are two possible locations for the cached data map:

  - In the run directory: ``(run dir)/karabo_data_map.json``.
  - In the proposal scratch directory:
    ``(proposal dir)/scratch/.karabo_data_maps/raw_r0032.json``.
    This will normally be the one used on Maxwell, as users can't write to the
    run directory.

- The format is a JSON array, with an object for each file in the run.

  - This holds the list of train IDs in the file, and the lists of control and
    instrument sources.
  - It also stores the file size and last modified time of each data file, to
    check if the file has changed since the cache was created. If either of
    these attributes doesn't match, karabo_data ignores the cached information
    and reads the metadata from the HDF5 file.

- If any file in the run wasn't listed in the data map, or its entry was
  outdated, a new data map is written automatically. It tries the same two
  locations described above, but it will continue without error if it can't
  write to either.

JSON was chosen as it can be easily inspected manually, and it's reasonably
efficient to load the entire file.
