Streaming data over ZeroMQ
==========================

*Karabo Bridge* provides access to live data during the experiment over a ZeroMQ
socket. The ``euxfel`` Python package can stream data from files using the same
protocol. You can use this to test code which expects to receive data from
Karabo Bridge, or use the same code for analysing live data and stored data.

.. note::

   The class described here is for sending data with the Karabo Bridge protocol.
   To receive Karabo Bridge data in Python, use the
   `euxfel_karabo_bridge package <https://github.com/European-XFEL/karabo-bridge-py>`__.

.. currentmodule:: euxfel

.. autoclass:: ZMQStreamer

   .. automethod:: start

   .. automethod:: feed
