Reading data files
==================

Data is available in *trains*, which arrive ten times per second.
Each train has a unique train ID, which is used to match up corresponding data
stored separately. Many data sources will make one reading per train, but some,
including the main detectors, record data more frequently.

.. module:: karabo_data

.. autofunction:: RunDirectory

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
