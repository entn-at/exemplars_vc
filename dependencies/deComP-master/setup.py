#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re

# load version form _version.py
VERSIONFILE = "decomp/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# module

setup(name='decomp',
      version=verstr,
      author="Keisuke Fujii",
      author_email="fujii@me.kyoto-u.ac.jp",
      description=("Python library for Large Scale Matrix Decomposition"),
      license="BSD 3-clause",
      keywords="plasma-fusion machine-learning",
      url="http://github.com/fujii-team/deComP",
      include_package_data=True,
      ext_modules=[],
      packages=["decomp", ],
      package_dir={'decomp': 'decomp', 'decomp.utils': 'decomp/utils'},
      py_modules=['decomp.__init__', 'decomp.utils.__init__'],
      test_suite='tests',
      install_requires="""
        numpy>=1.11
        chainer>=3.0
        """,
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Physics']
      )
