Karabo Python data tools
========================

**karabo_data** is a Python library for accessing and working with data
produced at `European XFEL <https://www.xfel.eu/>`_.
It is available on our Anaconda installation on the Maxwell cluster::

    module load exfel exfel_anaconda3

You can also install it `from PyPI <https://pypi.org/project/karabo-data/>`__
to use in other environments with Python 3.5 or later::

    pip install karabo_data

If you get a permissions error, add the ``--user`` flag to that command.

Contents:

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

