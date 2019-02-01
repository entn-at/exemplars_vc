# -*- coding: utf-8 -*-

""" Created on 2:13 PM 12/14/18
    @author: ngunhuconchocon
    @brief:
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


def __dummy_windowed_frames(source, frame_len=512, hopsize=80):
    np.random.seed(98765)
    n_frames = int(len(source) / hopsize) + 1
    windowed = np.random.randn(n_frames, frame_len) * pysptk.blackman(frame_len)
    return 0.5 * 32768.0 * windowed


# audio, _ = lbr.load("data/SF1/100008.wav", sr=None, dtype=np.float64)
audio, _ = lbr.load(lbr.util.example_audio_file(), sr=None)

hopsize = 80
windowed = __dummy_windowed_frames(
    audio, frame_len=frame_length, hopsize=hop_length)

frames = lbr.util.frame(audio, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
frames *= pysptk.hamming(frame_length, normalize=0)
print(type(frames), frames.shape)

# mcep = np.apply_along_axis(pysptk.mcep, 1, frames, order=order, alpha=alpha).T
lpcs = np.apply_along_axis(pysptk.lpc, 1, windowed, order=order)
lpcs1 = np.apply_along_axis(pysptk.lpc, 1, frames, order=order)

# for frame in frames:
#     print(frame.shape)
#     lpc = pysptk.lpc(frame)
print(type(lpcs), lpcs.shape)

lsp = pysptk.lpc2lsp(lpcs)
print(type(lsp), lsp.shape)
