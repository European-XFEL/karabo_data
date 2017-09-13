"""
These tests are part of the euxfel-h5tools package
To run all tests, execute:
    nosetests -v euxfel_h5tools
"""

import unittest
import numpy as np
from euxfel_h5tools.quickview import QuickView


class TestQuickView(unittest.TestCase):

    def setUp(self):
        self.qv = QuickView()

    def test_init(self):
        """Tests that the QuickView object initalizes properly"""
        self.assertIs(self.qv.data, None)
        self.qv.data = np.empty((1, 1, 1), dtype=np.int8)
        self.assertEqual(len(self.qv), 1)
        self.assertEqual(self.qv.pos, 0)

        with self.assertRaises(TypeError):
            self.qv.data = 4

        with self.assertRaises(TypeError):
            self.qv.data = np.empty((1, 1, 1, 1), dtype=np.int8)


if __name__ == "__main__":
    unittest.main()
