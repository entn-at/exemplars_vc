# -*- coding: utf-8 -*-

""" Created on 1:44 PM 12/10/18
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