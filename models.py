# -*- coding: utf-8 -*-

""" Created on 5:12 PM 12/11/18
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function

import os
import sys
import pickle
import pprint

import cProfile
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary.torchsummary import summary
from tensorboardX import SummaryWriter

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