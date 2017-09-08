"""
These tests are part of the euxfel-h5tools package
To run all tests, execute:
    nosetests -v euxfel_h5tools
"""
import io
import sys
import unittest
from euxfel_h5tools import main

class TestMain(unittest.TestCase):

    def setUp(self):
        self.helloer = main.PlaceHolder()

    def test_run(self):
        """Test that the placehoder does correct printing"""
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.helloer.run()
        self.assertEqual(capturedOutput.getvalue().strip(), "hello")

if __name__ == "__main__":
    unittest.main()
