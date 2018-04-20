Reading data files
==================

Data is available in *trains*, which arrive ten times per second.
Each train has a unique train ID, which is used to match up corresponding data
stored separately. Many data sources will make one reading per train, but some,
including the main detectors, record data more frequently.

.. module:: euxfel

.. autoclass:: H5File

   .. attribute:: control_devices

      A set of the control device names in this file, in the format
      ``"SA3_XTD10_VAC/TSENS/S30100K"``.

   .. attribute:: instrument_device_channels

      A set of the instrument devices and output channel names in this file,
      in the format ``"FXE_DET_LPD1M-1/DET/15CH0:xtdf"``.

   .. automethod:: trains

   .. automethod:: train_from_id

   .. automethod:: train_from_index

.. autoclass:: RunDirectory

   .. automethod:: trains

   .. automethod:: train_from_id

   .. automethod:: train_from_index
