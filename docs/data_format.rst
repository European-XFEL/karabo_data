Data files format
=================

The main unit of data this tool works with is a *run*. A run is data collected
in a specific period, and each research proposal given beantime at European XFEL
may collect hundreds of runs.

A run is stored as a directory containing HDF5 data files from different
sources. These fall into two important categories:

1. Detector data, from the main X-ray detectors in the various experiments.

   - Each detector module writes separate files, e.g. ``RAW-R0348-AGIPD00-S00000.h5``.
     The number in the third part of the filename identifies the module (0 in
     this example).
   - The detectors in use as of April 2018 are *LPD* and *AGIPD* in the file
     names. Each has 16 modules numbered 0–15.

2. All the other data, such as motor positions, beam measurements, etc., are
   recorded through a *data aggregator*, and stored in a file with the letters
   *DA* in the name, e.g. ``RAW-R0450-DA01-S00000.h5``.

The last part of the file name (e.g. ``S00000``) is a sequence number. The
data within a run may be broken into a number of sequences. So
``RAW-R0450-DA01-S00000.h5`` and ``RAW-R0450-DA01-S00001.h5`` will contain data
from the same set of devices, with sequence 1 continuing just after the end of
sequence 0. Though all data within a run may be broken into sequences, different
data sets do not necessarily break at the same point, so the various 'sequence 0'
data files in a run do not have corresponding data.


HDF5 file structure
-------------------

METADATA
~~~~~~~~

The ``METADATA`` group in an HDF5 file contains three datasets, each of which
is a 1D array of strings:

* ``METADATA/dataSourceId`` lists data groups in the file. The values are either:

  * ``CONTROL/`` followed by a Karabo device name, e.g.
    ``CONTROL/SA1_XTD2_XGM/DOOCS/MAIN``.
  * ``INSTRUMENT/`` followed by a Karabo device name, a colon, the name of the
    output channel, a slash, and the name of a data group (?), e.g.
    ``INSTRUMENT/SA1_XTD2_XGM/DOOCS/MAIN:output/data``

* ``METADATA/deviceId`` lists the part of each *dataSourceId* after the first
  slash.
* ``METADATA/root`` lists the parts before the first slash, so
  ``concat(root, "/", deviceId) == dataSourceId``.

These three data sets always have the same number of values. They may be padded
with empty strings, so empty entries are ignored.

INDEX
~~~~~

``INDEX/trainId`` is a 1D array of uint64, listing the pulse trains which the
file holds data for. This is crucial, since all other data has to be matched up
according to train IDs.

For each entry in ``METADATA/deviceId``, the ``INDEX`` group contains two
datasets, both uint64 data with the same length as the train IDs:

* ``INDEX/{ deviceId }/count``: for each train ID, how many data samples did
  this device record. This may be 0 if no data was recorded for this train.
* ``INDEX/{ deviceId }/first``: for each train ID, the index at which the
  corresponding data starts in the arrays for this device.

Thus, to find the data for a given train ID, we could do::

    train_index = trainIds.index(train_id)
    first = device_firsts[train_index]
    count = device_counts[train_index]
    train_data = data[first : first+count]

Control data is always (?) recorded once per train, so *count* is 1 and *first*
counts up from 0 to the number of trains. Instrument data is more variable.

Some older files use a different index format with first/last/status instead of
first/count. In this case, a status of 0 means that no data was recorded
for that train.

CONTROL and RUN
~~~~~~~~~~~~~~~

For each *CONTROL* entry in ``METADATA/dataSourceId``, there is a group with
that name in the file. This may have further arbitrarily nested subgroups
representing different properties of that device, e.g.
``/CONTROL/SA1_XTD2_XGM/DOOCS/MAIN/current/bottom/output``.

The leaves of this tree are pairs of datasets called ``timestamp`` and ``value``.
Each dataset has one entry per train, and the ``timestamp`` record when the
value was updated, which is typically less than once per train. The ``value``
dataset may have extra dimensions, but in most cases it is 1D.

(Does timestamp update if value is re-read but doesn't change?)

``RUN`` holds a complete duplicate of the ``CONTROL`` hierarchy, but each pair
of ``timestamp`` and ``value`` contain only one entry, taken at the start of
the run. There is still a dimension for this, so 2D value datasets in CONTROL
have corresponding 2D datasets in RUN, but the first dimension has length 1.

(Is RUN exactly duplicated in subsequent sequence files?)

INSTRUMENT
~~~~~~~~~~

For each *INSTRUMENT* entry in ``METADATA/dataSourceId``, there is a group with
that name in the file. Each such group holds a 1D ``trainId`` dataset, and a
number of other datasets (possibly nested in subgroups). All these datasets have
the same length in the first dimension: this represents the successive readings
taken. The slices defined by the corresponding datasets in *INDEX* work on
this dimension.

The ``trainId`` dataset for each instrument group thus appears to be redundant
with the information in INDEX.
