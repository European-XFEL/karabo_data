Karabo Python data tools
========================

**karabo_data** is a Python library for accessing and working with data
produced at `European XFEL <https://www.xfel.eu/>`_. To install it on
the Maxwell cluster::

    module load anaconda/3
    pip install --user karabo_data

If this causes problems for Jupyter, you may need to upgrade
``ipykernel`` to fix them::

    pip install --user --upgrade ipykernel


Contents:

.. toctree::
   :maxdepth: 2

   reading_files
   streaming
   validation
   data_format

.. toctree::
   :caption: Examples

   Demo
   apply_geometry
   examine_geometry
   agipd_geometry
   xpd_examples

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

