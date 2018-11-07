Checking data files
===================

*karabo_data* includes a tool to check the integrity of data files.
You can pass it a run::

    karabo-data-validate /gpfs/exfel/exp/XMPL/201750/p700000/raw/r0803

Or a single data file::

    karabo-data-validate /gpfs/exfel/exp/XMPL/201750/p700000/raw/r0803/RAW-R0803-AGIPD00-S00000.h5

The checks are informed by problems we have encountered with data files in the
past. Currently, it checks that:

- All ``.h5`` files in a run can be opened, and the run contains at least one
  usable file.
- The list of train IDs in a file has no zeros except for padding at the end.
- Each train ID in a file is greater than the one before it.
- The indexes do not point to data beyond the end of a dataset.
- The indexes point to the start of the dataset, and then to successive chunks
  for successive trains, without gaps or overlaps between them.
