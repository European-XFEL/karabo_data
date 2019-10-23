European XFEL Python data tools
===============================

**karabo_data** is a Python library for accessing and working with data
produced at `European XFEL <https://www.xfel.eu/>`_.

Installation
------------

karabo_data is available on our Anaconda installation on the Maxwell cluster::

    module load exfel exfel_anaconda3

You can also install it `from PyPI <https://pypi.org/project/karabo-data/>`__
to use in other environments with Python 3.5 or later::

    pip install karabo_data

If you get a permissions error, add the ``--user`` flag to that command.

Quickstart
----------

Open a run or a file - see :ref:`opening-files` for more::

    from karabo_data import open_run, RunDirectory, H5File

    # Find a run on the Maxwell cluster
    run = open_run(proposal=700000, run=1)

    # Open a run with a directory path
    run = RunDirectory("/gpfs/exfel/exp/XMPL/201750/p700000/raw/r0001")

    # Open an individual file
    file = H5File("RAW-R0017-DA01-S00000.h5")

After this step, you'll use the same methods to get data whether you opened a
run or a file.

Load data into memory - see :ref:`data-by-source-and-key` for more::

    # Get a labelled array
    arr = run.get_array("SA3_XTD10_PES/ADC/1:network", "digitizers.channel_4_A.raw.samples")

    # Get a pandas dataframe of 1D fields
    df = run.get_dataframe(fields=[
        ("*_XGM/*", "*.i[xy]Pos"),
        ("*_XGM/*", "*.photonFlux")
    ])

Iterate through data for each pulse train - see :ref:`data-by-train` for more::

    for train_id, data in run.select("*/DET/*", "image.data").trains():
        mod0 = data["FXE_DET_LPD1M-1/DET/0CH0:xtdf"]["image.data"]

These are not the only ways to get data: :doc:`reading_files` describes
various other options.
karabo_data also has classes to work with detector geometry,
described in :doc:`geometry`.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2

   reading_files
   agipd_lpd_data
   streaming
   validation
   geometry
   cli
   data_format
   performance

.. toctree::
   :caption: Examples

   Demo
   lpd_data
   apply_geometry
   examine_geometry
   agipd_geometry
   dssc_geometry
   xpd_examples
   xpd_examples2
   parallel_example
   dask_averaging

.. toctree::
   :caption: Development
   :maxdepth: 1

   changelog

.. seealso::

   `Data Analysis at European XFEL
   <https://in.xfel.eu/readthedocs/docs/data-analysis-user-documentation/en/latest/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

