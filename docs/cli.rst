Command line tools
==================

.. _cmd-lsxfel:

``lsxfel``
----------

Examine the contents of an EuXFEL proposal directory, run directory, or HDF5
file:

.. code-block:: shell

   # Proposal directory
   lsxfel /gpfs/exfel/exp/XMPL/201750/p700000

   # Run directory
   lsxfel /gpfs/exfel/exp/XMPL/201750/p700000/raw/r0002

   # Single file
   lsxfel /gpfs/exfel/exp/XMPL/201750/p700000/proc/r0002/CORR-R0034-AGIPD00-S00000.h5


``karabo-data-validate``
------------------------

.. note::

   This tool has been renamed to ``extra-data-validate``,
   provided `as part of the EXtra-data package
   <https://extra-data.readthedocs.io/en/latest/validation.html>`_.

Check the structure of an EuXFEL run or HDF5 file:

.. code-block:: shell

   karabo-data-validate /gpfs/exfel/exp/XMPL/201750/p700000/raw/r0002

If it finds problems with the data, the program will produce a list of them and
exit with status 1.

.. _cmd-serve-files:

``karabo-bridge-serve-files``
-----------------------------

Stream data from files in the `Karabo bridge
<https://in.xfel.eu/readthedocs/docs/data-analysis-user-documentation/en/latest/online.html#data-stream-to-user-tools>`_
format. See :doc:`streaming` for more information.

``karabo-data-make-virtual-cxi``
--------------------------------

.. note::

   This tool has been renamed to ``extra-data-make-virtual-cxi``,
   provided `as part of the EXtra-data package
   <https://extra-data.readthedocs.io/en/latest/cli.html#extra-data-make-virtual-cxi>`_.

Make a virtual CXI file to access AGIPD/LPD detector data from a specified run:

.. code-block:: shell

   karabo-data-make-virtual-cxi /gpfs/exfel/exp/XMPL/201750/p700000/proc/r0003 -o xmpl-3.cxi

.. program:: karabo-data-make-virtual-cxi

.. option:: -o <path>, --output <path>

   The filename to write. Defaults to creating a file in the proposal's
   scratch directory.

.. option:: --min-modules <number>

   Include trains where at least N modules have data (default 9).
