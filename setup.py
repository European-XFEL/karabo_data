#!/usr/bin/env python
import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name="euxfel_h5tools",
      author="The European XFEL",
      author_email="usp-support@xfel.eu",
      description="Python tools for reading European XFEL's h5 files",
      long_description=read("README.md"),
      url="https://github.com/European-XFEL/h5tools-py",
      license = "To Be Confirmed",
      scripts = ["bin/euxfel_h5tool.py", "bin/euxfel_h5tool_tmp.py"],
      install_requires=["h5py", "numpy", "matplotlib", "docopt"]
      )
