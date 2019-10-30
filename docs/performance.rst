Performance notes
=================

These are some notes on how to load and process data efficiently.

Load data into memory
---------------------

Where the data you need can fit into memory, it's more efficient to load it
in one go using :meth:`~.DataCollection.get_array`,
:meth:`~.DataCollection.get_series` or :meth:`~.DataCollection.get_dataframe`,
and then work with it using xarray, numpy or pandas.
:doc:`xpd_examples` has some examples of this.
The methods to get data by trains—:meth:`~.DataCollection.trains`,
:meth:`~.DataCollection.train_from_id` and
:meth:`~.DataCollection.train_from_index`—only load the data for one train
at once, which saves memory for big data but is slower to process.

Machines in the Maxwell cluster have hundreds of gigabytes of RAM, so it's
practical to load many kinds of data completely into memory.
However, data for a full run from megahertz detectors such as AGIPD, LPD or DSSC
can easily be too much.

The command ``free -h`` will show the amount of memory on any Linux machine.

Select sources before getting trains
------------------------------------

If you do need to use :meth:`~.DataCollection.trains`,
:meth:`~.DataCollection.train_from_id` or
:meth:`~.DataCollection.train_from_index` to get data for one train at a time,
first pick the sources and keys you need with :meth:`~.DataCollection.select`.
Otherwise, you will load the data for every source in the run, which could
be very slow.

::

    run = RunDirectory("/gpfs/exfel/exp/XMPL/201750/p700000/raw/r0004")

    # SLOW: Don't do this!
    for tid, train_data in run.trains():
        ...

    # Better option: select image data from all detector modules first.
    for tid, train_data in run.select('*/DET/*', 'image.data').trains():
        ...

The ``devices=`` parameter for all three train methods does the same thing
as using :meth:`~.DataCollection.select` like this.

Reduce before assembling
------------------------

Assembling detector images (see :doc:`geometry`) is relatively slow.
If your analysis involves a reduction step like summing or averaging over
a number of images, try to do this on the data from separate modules before
assembling them into images.

This also applies more generally: if a step in your processing makes the data
smaller, you want to do that step as near the start as possible.
