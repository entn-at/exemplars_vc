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

from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 10


def config_get_config(configpath):
    args = {}

    config = configparser.ConfigParser()
    config.read(configpath)

    args['DataPath'] = config['PATH']['DataPath']
    args['SpeakerA'] = config['VAR']['src']
    args['SpeakerB'] = config['VAR']['tar']
    args['sampling_rate'] = config['VAR']['sr']
    args['feature_path'] = config['VAR']['feature_path']
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
