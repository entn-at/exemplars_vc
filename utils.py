# -*- coding: utf-8 -*-

""" Created on 3:19 PM 12/3/18
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function
import configparser

import os
import pickle
import logging

import numpy as np
import librosa as lbr

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 10


def logdir():
    """
    return an available name for tensorboard dir. The logdir will look like this
    --- runs
         |--1
         |--2
         |--3
         |--4
        To this case, "5" will be returned. Even with the case
    :return:
    """
    listdir = [xx for xx in os.listdir("runs/") if os.path.isdir(os.path.join("runs", xx))]

    max_ = 0
    for i in range(len(listdir)):
        try:
            filename = int(listdir[i])
            if max_ < filename:
                max_ = filename
        except ValueError:
            continue

    return str(max_ + 1)


def config_get_config(configpath):
    args = {}

    config = configparser.ConfigParser()
    config.read(configpath)

    args['DataPath'] = config['PATH']['DataPath']

    args['SpeakerA'] = config['VAR']['src']
    args['SpeakerB'] = config['VAR']['tar']
    args['sampling_rate'] = config['VAR']['sr']
    args['feature_path'] = config['VAR']['feature_path']
    args['use_stft'] = bool(int(config['VAR']['use_stft']))

    args['checkpoint_name'] = config['NET']['checkpoint_name']
    args['nb_frame_in_batch'] = config['NET']['nb_frame_in_batch']
    args['patience'] = config['NET']['patience']
    args['nb_epoch'] = config['NET']['nb_epoch']
    args['batch_size'] = config['NET']['batch_size']
    args['dropout_rate'] = config['NET']['dropout_rate']
    args['nb_lstm_layers'] = config['NET']['nb_lstm_layers']
    args['bidirectional'] = config['NET']['bidirectional']

    args['in_size'] = 20
    args['hidden_size'] = 20
    args['out_size'] = 20

    args['feat_frame_length'] = config['MCEP']['feat_frameLength']
    args['feat_overlap'] = config['MCEP']['feat_overlap']
    args['feat_hop_length'] = config['MCEP']['feat_hop_length']
    args['feat_order'] = config['MCEP']['feat_order']
    args['feat_alpha'] = config['MCEP']['feat_alpha']
    args['feat_gamma'] = config['MCEP']['feat_gamma']

    args['is_refined'] = int(bool(config['PYWORLD']['f0_is_refined']))
    args['f0_floor'] = int(config['PYWORLD']['f0_floor'])
    args['frame_period'] = float(config['PYWORLD']['frame_period'])

    args['cpu_rate'] = float(config['MISC']['cpu_rate'])
    args['nb_file'] = int(config['MISC']['nb_file'])
    return args


def io_read_data(filename, filetype):
    """
    read data from npy. MFCCs only by far
    :param datapath:
    :return:
    """
    filepath = os.path.join(os.path.join(ROOT_DIR, 'data/vc'), filename)
    logging.info("Reading from {}".format(filename))

    if filetype == 'npy':
        return np.load(filepath)
    elif filetype == 'pkl':
        with open(os.path.join(filepath), "rb") as f:
            temp = pickle.load(f)
            data_train, data_test = train_test_split(temp, test_size=0.2, random_state=SEED)
            return data_train, data_test
    else:
        logging.critical(filetype, "is not supported by far. Exiting ...")
        exit()


def io_read_speaker_data(datapath, speaker, savetype='npy', parallel=False):
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
            io_save_to_disk(datapath, speaker, savetype, parallel=parallel)
            return np.load(path)
    else:
        print(savetype, "filetype is not supported yet")
        exit()


def io_save_to_disk(datapath, speaker, savetype, parallel):
    """
    this function will save all available audio of a speaker to npy/bin/pkl file
    see this link for selecting best type for saving ndarray to disk
    https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
    :param datapath: path to data, the wav path should be in "speakerpath/speaker/*.wav"
    :return: True if no error occurs. False otherwise
    """
    if parallel:
        io_save_to_disk_parallel(datapath, speaker=speaker, savetype=savetype)
    else:
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


def io_save_to_disk_parallel(datapath, speaker='SF1', savetype='npy'):
    """
        Note: this is file-based implementation for multiprocessing. Different from non-parallel version
    this function will save all available audio of a speaker to npy/bin/pkl file
    see this link for selecting best type for saving ndarray to disk
    https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
    :param datapath: path to data, the wav path should be in "speakerpath/speaker/*.wav"
    :return: True if no error occurs. False otherwise
    """
    n_workers = int(0.8 * cpu_count())
    # try:
    # Read
    speakerdir = os.path.join(datapath, speaker)

    yA = []
    print("=======================")
    logging.debug("Parallel: Read {} data".format(speaker))

    # print("Read", speaker, "data")
    # for filename in tqdm(os.listdir(speakerdir)):
    #     y, _ = io_read_audio(os.path.join(speakerdir, filename))
    #     yA.append(y)

    p = Pool(n_workers)
    results = p.map(io_read_audio, [os.path.join(speakerdir, filename) for filename in os.listdir(speakerdir)])

    yA, sr = [xxx[0] for xxx in results], results[0][1]

    # import pdb
    # pdb.set_trace()
    # Save all to dick
    # uh oh ...
    # print(np.asarray(yA).shape)
    if savetype == 'npy':
        os.system("mkdir -p npy")
        np.save(os.path.join("npy", speaker), np.asarray(yA))
    else:
        print(savetype, "is not supported")
        exit()

    # except Exception as e:
    #     return False


def io_read_audio(filepath):
    y, sr = lbr.load(filepath, sr=None, dtype=np.float64)
    return y, sr