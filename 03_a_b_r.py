# -*- coding: utf-8 -*-

""" Created on 3:29 PM 12/11/18
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function

import logging

try:
    import coloredlogs

    coloredlogs.install()
except ImportError:
    pass

import os
import sys
