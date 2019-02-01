# -*- coding: utf-8 -*-

""" Created on 11:25 AM 12/15/18
    @author: ngunhuconchocon
    @brief: Ok i give up. Have to use DFW :'(
"""

from __future__ import print_function

import os
import sys

from utils import config_get_config

import pysptk
import librosa as lbr
import numpy as np

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

# parse the configuration
args = config_get_config("config/config")

frame_length = int(args['feat_frame_length'])
overlap = float(args['feat_overlap'])
hop_length = int(frame_length * overlap)
order = int(args['feat_order'])
alpha = float(args['feat_alpha'])
gamma = float(args['feat_gamma'])

data_path = args['DataPath']
speakerA = args['SpeakerA']
speakerB = args['SpeakerB']
feature_path = args['feature_path']
sr = int(args['sampling_rate'])

def dfw()
