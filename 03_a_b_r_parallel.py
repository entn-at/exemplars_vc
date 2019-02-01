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

# from models import Net
from utils import config_get_config, io_read_data,io_read_speaker_data, io_save_to_disk
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

import itertools
import pyworld as pw
import librosa as lbr

import os
import pdb
import subprocess

import sys
import pickle
import numpy as np
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

frame_length = 400
hop_length = int(frame_length/5)
window = 'hann'

data_path = args['DataPath']
speakerA = args['SpeakerA']
speakerB = args['SpeakerB']
feature_path = args['feature_path']
sr = int(args['sampling_rate'])

refine_f0 = args['is_refined']
frame_period = args['frame_period']
f0_floor = args['f0_floor']

cpu_rate = float(args['cpu_rate'])
nb_file = int(args['nb_file'])
use_stft = args['use_stft']

logging.info("{}% cpu resources ({} cores) will be used to run this script".format(cpu_rate * 100, int(cpu_rate * cpu_count())))


def _get_conversion_data(audiodatum, fs, refine_f0, f0_floor, use_stft):
    """
        Note: this is file-based implementation for multiprocessing. Different from non-parallel version
    Get A (without warping source dictionary) feature for conversion (sp, ap, f0)
    :param audiodatum: 1 file audio data. not all 162
    :param args:
    :param kwargs:
    :return: source dictionary (without warping)
    """

    # Extract feature
    # _f0, t = pw.dio(audiodatum, fs)  # raw pitch extractor
    if not use_stft:
        _f0, t = pw.harvest(audiodatum, fs)  # raw pitch extractor

        if refine_f0:
            f0 = pw.stonemask(audiodatum, _f0, t, fs)  # pitch refinement
        else:
            f0 = _f0

        sp = pw.cheaptrick(audiodatum, f0, t, fs)  # extract smoothed spectrogram
        ap = pw.d4c(audiodatum, f0, t, fs)  # extract aperiodicity
        # y = pw.synthesize(f0, sp, ap, fs)

        features = {'sp': sp, 'ap': ap, 'f0': f0, 'fs': fs, 'sr': fs}

        return features
    else:
        return {
            'stft': lbr.core.stft(y=audiodatum, n_fft=frame_length, hop_length=hop_length, window=window).T,
            'fs': fs
        }


def get_conversion_data(audiodata, fs, refine_f0, f0_floor, speaker, use_stft):
    """
        This will use multiprocess.Pool to parallel call _extract_features
    Get A (without warping source dictionary) feature for conversion (sp, ap, f0)

    :return: source dictionary (without warping)
    """
    n_workers = int(cpu_rate * cpu_count())
    p = Pool(n_workers)

    if not use_stft:
        logging.info("Using sp, ap, f0 as features ...")
        logging.info("Speaker {} ...".format(speaker))

        filepath = os.path.join(feature_path, "exem_dict/{}_feat_sp_ap_f0.pkl".format(speaker))
        # check if features is exist
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            # Extract feature from time-series
            results = p.starmap(_get_conversion_data, zip(audiodata, itertools.repeat(fs), itertools.repeat(refine_f0), itertools.repeat(f0_floor), itertools.repeat(use_stft)))

            # save to disk for later used
            with open(filepath, "wb") as f:
                pickle.dump(results, f, protocol=3)

            return results
    else:
        logging.info("Using raw stft as features ...")
        logging.info("Speaker {} ...".format(speaker))

        filepath = os.path.join(feature_path, "exem_dict/{}_feat_stft.pkl".format(speaker))
        # check if features is exist
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            # Extract feature from time-series
            results = p.starmap(_get_conversion_data, zip(audiodata, itertools.repeat(fs), itertools.repeat(refine_f0), itertools.repeat(f0_floor), itertools.repeat(use_stft)))

            # save to disk for later used
            with open(filepath, "wb") as f:
                pickle.dump(results, f, protocol=3)

            return results


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
    logging.info("===================================================")
    logging.info("Start reading audio time-series")
    speakerAdata = io_read_speaker_data(data_path, speakerA, savetype='npy')[:nb_file]
    speakerBdata = io_read_speaker_data(data_path, speakerB, savetype='npy')[:nb_file]

    # Get data for conversion
    logging.info("===================================================")
    # logging.info("Extracting features for exemplar dictionaries: sp, ap, f0 ...")
    # A = get_conversion_data(speakerAdata, fs=sr, refine_f0=refine_f0, f0_floor=f0_floor, speaker=speakerA)
    # B = get_conversion_data(speakerBdata, fs=sr, refine_f0=refine_f0, f0_floor=f0_floor, speaker=speakerB)
    A = get_conversion_data(speakerAdata, fs=sr, refine_f0=refine_f0, f0_floor=f0_floor, speaker=speakerA, use_stft=use_stft)
    B = get_conversion_data(speakerBdata, fs=sr, refine_f0=refine_f0, f0_floor=f0_floor, speaker=speakerB, use_stft=use_stft)

    print(len(A), len(B))
    # with open(os.path.join("data/vc/exem_dict", filename + "_A"), "rb") as f:
    #     W_A = pickle.load(f)
    #
    # with open(os.path.join("data/vc/exem_dict", filename + "_B"), "rb") as f:
    #     W_B = pickle.load(f)
    #
    # A_warped = warp(A, W_A)
    # pdb.set_trace()
