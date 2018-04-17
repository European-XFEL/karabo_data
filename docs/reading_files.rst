Reading data files
==================

Data is available in *trains*, which arrive ten times per second.
Each train has a unique train ID, which is used to match up corresponding data
stored separately. Many data sources will make one reading per train, but some,
including the main detectors, record data more frequently.

.. module:: euxfel

.. autoclass:: H5File

   .. attribute:: devices

      A list of the available device names in this file.

   .. automethod:: trains

   .. automethod:: train_from_id

   .. automethod:: train_from_index

.. autoclass:: RunDirectory

   .. automethod:: trains

   .. automethod:: train_from_id

   .. automethod:: train_from_index
