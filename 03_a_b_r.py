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
from utils import config_get_config, io_read_data,io_read_speaker_data, io_save_to_disk
from tqdm import tqdm

import pyworld as pw

import os
import pdb
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

data_path = args['DataPath']
speakerA = args['SpeakerA']
speakerB = args['SpeakerB']
feature_path = args['feature_path']
sr = int(args['sampling_rate'])

refine_f0 = args['is_refined']
frame_period = args['frame_period']


def get_conversion_data(audiodata, fs, refine_f0):
    """
    Get A (without warping source dictionary) feature for conversion (sp, ap, f0)
    :param args:
    :param kwargs:
    :return: source dictionary (without warping)
    """
    features = []

    logging.info("Start building speaker A dictionary: Extracting feature for conversion (sp, ap, f0)")
    for audio in tqdm(audiodata):
        # Extract feature
        _f0, t = pw.dio(audio, fs)  # raw pitch extractor

        if refine_f0:
            f0 = pw.stonemask(audio, _f0, t, fs)  # pitch refinement
        else:
            f0 = _f0

        sp = pw.cheaptrick(audio, f0, t, fs)  # extract smoothed spectrogram
        ap = pw.d4c(audio, f0, t, fs)  # extract aperiodicity
        # y = pw.synthesize(f0, sp, ap, fs)

        features.append({
            'sp': sp,
            'ap': ap,
            'f0': f0,
            'fs': fs,
            'sr': fs
        })

    return features


def warp(A, W_A):
    """
    warp raw spectral feature (FW-ed source dictionary) for conversion
    :param args:
    :param kwargs:
    :return: source dictionary (FW-ed)
    """
    logging.info("Align spectral feature with Dynamic-MCEP-warping warping matrix")

    for idx, file in enumerate(A):  # Iterate through all wav files
        for i in range(len(W[idx])):
            pass
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
    filename = "exemplar_W"

    # Read audio time-series from npy
    speakerAdata = io_read_speaker_data(data_path, speakerA, savetype='npy')
    speakerBdata = io_read_speaker_data(data_path, speakerB, savetype='npy')

    A = get_conversion_data(speakerAdata, fs=sr, refine_f0=refine_f0)
    # B = get_conversion_data(speakerBdata, fs=sr, refine_f0=refine_f0)

    with open(os.path.join("data/vc/exem_dict", filename + "_A"), "rb") as f:
        W_A = pickle.load(f)

    with open(os.path.join("data/vc/exem_dict", filename + "_B"), "rb") as f:
        W_B = pickle.load(f)

    A_warped = warp(A, W_A)
    pdb.set_trace()
