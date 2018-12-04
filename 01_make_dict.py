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


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


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


def feat_mfcc(audiodata, sr=16000):
    """
    extract mfcc from audio time series data (from librosa.load)
    :param audiodata: 162 files time-series data
    :return:
    """
    mfccs = []
    for audio in audiodata:
        mfccs.append(lbr.feature.mfcc(audio, sr=sr))

    # return np.stack(mfccs)
    return mfccs


def make_dict_from_mfcc(feat_mfcc_A, feat_mfcc_B):
    """
    Final function: return the "dictionary" of exemplars
    Tentative: return a list, each item is a tuple size of 2, which is A and B, for src and tar speaker
    :param dtw_path: path[0], path[1]
    :return:
    """
    dtw_paths = []
    for i in range(len(feat_mfcc_A)):
        dist, cost, cum_cost, path = dtw(feat_mfcc_A[i].T, feat_mfcc_B[i].T, lambda x, y: np.linalg.norm(x - y, ord=1))
        dtw_paths.append(path)

    exemplars = []
    for idx_file, path in tqdm(enumerate(dtw_paths)):
        a = []
        b = []
        for it in range(len(path[0])):
            try:
                a.append(feat_mfcc_A[idx_file].T[path[0][it]])
                b.append(feat_mfcc_B[idx_file].T[path[1][it]])
            except Exception as e:
                input("Error occur. Press any key to exit ...")
                exit()

        exemplars.append(np.stack([np.asarray(a), np.asarray(b)], axis=0))

    return exemplars


def final_make_dict():
    args = config_get_config("config/config")

    # parse the configuration
    data_path = args['DataPath']
    speakerA = args['SpeakerA']
    speakerB = args['SpeakerB']
    feature_path = args['feature_path']
    sr = args['sampling_rate']

    # TODO should add argument to python call
    # TODO to specify which speaker to cover

    # Read audio time-series from npy
    speakerAdata = io_read_speaker_data(data_path, speakerA, savetype='npy')
    speakerBdata = io_read_speaker_data(data_path, speakerB, savetype='npy')

    # Extract the MFCCs from time-series
    feat_mfcc_A = feat_mfcc(speakerAdata, sr=sr)
    feat_mfcc_B = feat_mfcc(speakerBdata, sr=sr)

    # for i in range(len(feat_mfcc_B)):
    #     print(feat_mfcc_A[i].shape, feat_mfcc_B[i].shape)

    exemplars = make_dict_from_mfcc(feat_mfcc_A, feat_mfcc_B)
    print(exemplars[0].shape, exemplars[1].shape)

    # Dump to npy
    os.system("mkdir -p " + feature_path)
    with open(os.path.join(feature_path, speakerA + "2" + speakerB + "_mfcc" + ".pkl"), "wb") as f:
        pickle.dump(exemplars, f, protocol=3)

    # np.save(os.path.join(feature_path, speakerA + "2" + speakerB + "_mfcc" + ".npy"), exemplars)


def debug_profiling_main():
    cProfile.run('final_make_dict()')


if __name__ == "__main__":
    # final_make_dict()
    cProfile.run('final_make_dict()', )

    # args = config_get_config("config/config")
    #
    # # parse the configuration
    # data_path = args['DataPath']
    # speakerA = args['SpeakerA']
    # speakerB = args['SpeakerB']
    # sr = args['sampling_rate']
    #
    # # TODO should add argument to python call
    # # TODO to specify which speaker to cover
    #
    # # Read audio time-series from npy
    # speakerAdata = io_read_speaker_data("data", speakerA, savetype='npy')
    # speakerBdata = io_read_speaker_data("data", speakerB, savetype='npy')
    #
    # # Extract the MFCCs from time-series
    # feat_mfcc_A = feat_mfcc(speakerAdata, sr=sr)
    # feat_mfcc_B = feat_mfcc(speakerBdata, sr=sr)
    #
    # # plt.subplot(2, 1, 1)
    # # display.specshow(feat_mfcc_A[0])
    # #
    # # plt.subplot(2, 1, 2)
    # # display.specshow(feat_mfcc_B[0])
    #
    #
    #
    # # print(path[0].shape, path[1].shape)
    # # plt.imshow(cost.T, origin='lower', interpolation='nearest')
    # # plt.plot(path[0], path[1])
    # # plt.show()
    #
    #
