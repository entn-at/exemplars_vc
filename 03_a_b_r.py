# -*- coding: utf-8 -*-

""" Created on 3:29 PM 12/11/18
    @author: ngunhuconchocon
    @brief: Pickling everything needed for run-time conversion
    2 item will be pickled:
        - W (the mapping function - the trained neural network)
        - R (residual difference between warped spectrum and the target spectrum: W(A) - B)

        For the source exemplars A, it is the input to the conversion system. It will be calculate in conversion phase runtime
"""

from __future__ import print_function

from models import Net
from utils import config_get_config, io_read_data

import os
import subprocess

import sys
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

import logging
import datetime
logging.basicConfig(
    filename="logs/" + ":".join(str(datetime.datetime.now()).split(":")[:-1]),
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)

try:
    import coloredlogs
    coloredlogs.install(level=logging.DEBUG, fmt='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')
except ModuleNotFoundError:
    pass

SEED = 10
args = config_get_config("config/config")


def get_A(*args, **kwargs):
    """
    Get A (FW-ed source dictionary) for conversion
    :param args:
    :param kwargs:
    :return: source dictionary (FW-ed)
    """
    raise NotImplementedError


def get_W(*args, **kwargs):
    """
    Get W (warping function dictionary) for conversion. Note that W is belong to R^(MxN)
    :param args:
    :param kwargs:
    :return: W (warping function dictionary)
    """

    raise NotImplementedError


def get_R(*args, **kwargs):
    """
    Get R (residual compensation dictionary) for conversion
    :param args:
    :param kwargs:
    :return: R (residual compensation dictionary)
    """
    raise NotImplementedError


if __name__ == "__main__":





