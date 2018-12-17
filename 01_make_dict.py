# -*- coding: utf-8 -*-

""" Created on 9:21 AM 11/30/18
    @author: ngunhuconchocon
    @brief: This script is made to construct frame-wise dictionary between src and tar spectra
    For short, it create dicts of exemplars ( a_i, b_i pairs)
"""

from __future__ import print_function
from tqdm import tqdm
from dtw import dtw
from librosa import display
from utils import config_get_config

import pickle
import configparser

import os
import librosa as lbr
import numpy as np
import matplotlib.pyplot as plt

# for debugging
import logging
import cProfile

import pysptk

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

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


def io_read_audio(filepath):
    y, sr = lbr.load(filepath, sr=None)
    return y, sr


def io_save_to_disk(datapath, speaker='SF1', savetype='npy'):
    """
    this function will save all available audio of a speaker to npy/bin/pkl file
    see this link for selecting best type for saving ndarray to disk
    https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
    :param datapath: path to data, the wav path should be in "speakerpath/speaker/*.wav"
    :return: True if no error occurs. False otherwise
    """
    try:
        # Read
        speakerdir = os.path.join(datapath, speaker)

        yA = []
        print("=======================")
        logging.debug("Read {} data".format(speaker))

        # print("Read", speaker, "data")
        for filename in tqdm(os.listdir(speakerdir)):
            y, _ = io_read_audio(os.path.join(speakerdir, filename))
            yA.append(y)

        # Save all to dick
        # uh oh ...
        # print(np.asarray(yA).shape)
        if savetype == 'npy':
            os.system("mkdir -p npy")
            np.save(os.path.join("npy", speaker), np.asarray(yA))
        else:
            print(savetype, "is not supported")
            exit()

    except Exception as e:
        return False


def io_read_speaker_data(datapath, speaker, savetype='npy'):
    """
    this function is used to read saved data (npy, npz, pkl, ...) of a speaker to a ndarray
    if the path is not exist, which mean there is no saved data, read and create one.
    :param alldatapath:
    :param savetype:
    :return:
    """
    path = os.path.join(savetype, speaker) + ".npy"
    if savetype == 'npy':
        if os.path.isfile(path):
            return np.load(path)
        else:
            io_save_to_disk(datapath, speaker)
            return np.load(path)
    else:
        print(savetype, "filetype is not supported yet")
        exit()


# This is wrong (as we need MCEP, not MFCC). Nevertheless, preserving this is necessary.
# Updated 2018 Dec 14: Edit feat_mfccs to extract_features, with multiple choice of choosing feature to extract. Default is mcep
# TODO feat argument need to be implement as a list, support multiple featuretype return
def extract_features(audiodata, sr=16000, feat='mcep'):
    """
    Feature extraction. For each type of feature, see corresponding case below
    Currently support: MCEP, MFCC. Will be updated when needed
    :param audiodata: 162 files time-series data
    :param sr: sampling rate.
    :param feat: type of currently supported feature
    :return:
    """
    if feat.lower() == 'mfcc':
        """
            extract mfcc from audio time series data (from librosa.load)
        """
        mfccs = []
        for audio in tqdm(audiodata):
            mfccs.append(lbr.util.normalize(lbr.feature.mfcc(audio, sr=sr, n_fft=frame_length, hop_length=hop_length), norm=1, axis=0))

        # return np.stack(mfccs)
        return mfccs, feat

    elif feat.lower() == 'mcep' or feat.lower() == 'mcc':
        """ 
            MCEP is extracted via pysptk. See the link below for more details
            https://github.com/eYSIP-2017/eYSIP-2017_Speech_Spoofing_and_Verification/wiki/Feature-Extraction-for-Speech-Spoofing
            
            Example of using pysptk to extract mcep (copied from the above link):             
                frameLength = 1024
                overlap = 0.25
                hop_length = frameLength * overlap
                order = 25
                alpha = 0.42
                gamma = -0.35
            
                sourceframes = librosa.util.frame(speech, frame_length=frameLength, hop_length=hop_length).astype(np.float64).T
                sourceframes *= pysptk.blackman(frameLength)
                sourcemcepvectors = np.apply_along_axis(pysptk.mcep, 1, sourceframes, order, alpha)
        """
        mceps = []
        for audio in tqdm(audiodata):
            frame = lbr.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T
            frame *= pysptk.blackman(frame_length)

            mceps.append(np.apply_along_axis(pysptk.mcep, 1, frame, order=order, alpha=alpha).T)

        return mceps, feat
    else:
        logging.critical('{} feature is not supported yet, exiting ...')
        exit()


# EDIT 2018 Dec 17: This function will be removed. It will later be split to 3 separated function: `make_A`, `make_R`, `make_W`
def make_dict_from_feat(feat_A, feat_B):
    """
    Final function: return the "dictionary" of exemplars, which is construct by alignment of DTW
    Tentative: return a list, each item is a tuple size of 2, which is A and B, for src and tar speaker
    :param dtw_path: path[0], path[1]
    :return:
    """

    dtw_paths = []

    for i in range(len(feat_A)):
        dist, cost, cum_cost, path = dtw(feat_A[i].T, feat_B[i].T, lambda x, y: np.linalg.norm(x - y, ord=1))
        dtw_paths.append(path)

    exemplars = []
    for idx_file, path in tqdm(enumerate(dtw_paths)):
        a = []
        b = []

        for it in range(len(path[0])):
            try:
                a.append(feat_A[idx_file].T[path[0][it]])
                b.append(feat_B[idx_file].T[path[1][it]])
            except Exception as e:
                input("Error occur. Press any key to exit ...")
                exit()

        exemplars.append(np.stack([np.asarray(a), np.asarray(b)], axis=0))
    return exemplars


def final_make_dict():
    # TODO should add argument to python call
    # TODO to specify which speaker to cover

    # Read audio time-series from npy
    speakerAdata = io_read_speaker_data(data_path, speakerA, savetype='npy')
    speakerBdata = io_read_speaker_data(data_path, speakerB, savetype='npy')

    # Extract features from time-series
    feat_A, feat_type_A = extract_features(speakerAdata, sr=sr, feat='mcep')
    feat_B, feat_type_B = extract_features(speakerBdata, sr=sr, feat='mcep')

    assert feat_type_A == feat_type_B, "Inconsistent feature type. 2 speaker must have the same type of extracted features."

    exemplars = make_dict_from_feat(feat_A, feat_B)
    print(exemplars[0].shape, exemplars[1].shape)

    # Dump to npy
    os.system("mkdir -p " + feature_path)
    # with open(os.path.join(feature_path, speakerA + "2" + speakerB + "_mfcc_25ms_10ms_norm" + ".pkl"), "wb") as f:
    with open(os.path.join(feature_path, "{}2{}_{}_{}ms_{}ms.pkl".format(
            speakerA, speakerB, 'mcep', int(frame_length * 1000 / sr), int(hop_length * 1000 / sr))), "wb") as f:
        pickle.dump(exemplars, f, protocol=3)

    # np.save(os.path.join(feature_path, speakerA + "2" + speakerB + "_mfcc" + ".npy"), exemplars)


def debug_profiling_main():
    cProfile.run('final_make_dict()')


if __name__ == "__main__":
    final_make_dict()
    # cProfile.run('final_make_dict()', )
