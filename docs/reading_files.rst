Reading data files
==================

Data is available in *trains*, which arrive ten times per second.
Each train has a unique train ID, which is used to match up corresponding data
stored separately. Many data sources will make one reading per train, but some,
including the main detectors, record data more frequently.

.. module:: karabo_data

.. autofunction:: RunDirectory

.. autofunction:: open_run

   .. versionadded:: 0.5

.. autofunction:: H5File

.. autoclass:: DataCollection

   .. attribute:: train_ids

      A list of the train IDs for which there is any data in this run.
      The data recorded may not be the same for each train.

   .. attribute:: control_sources

      A set of the control source names in this data, in the format
      ``"SA3_XTD10_VAC/TSENS/S30100K"``.

   .. attribute:: instrument_sources

      A set of the instrument source names in this data,
      in the format ``"FXE_DET_LPD1M-1/DET/15CH0:xtdf"``.

   .. attribute:: all_sources

      A set of names for both instrument and control sources.
      This is the union of the two sets above.

   .. automethod:: keys_for_source

   .. automethod:: info

   .. automethod:: trains

   .. automethod:: train_from_id

   .. automethod:: train_from_index

   .. automethod:: get_dataframe

   .. automethod:: get_series

   .. automethod:: get_array

   .. automethod:: select

   .. automethod:: deselect

   .. automethod:: select_trains

   .. automethod:: union

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
