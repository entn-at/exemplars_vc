# -*- coding: utf-8 -*-

""" Created on 3:19 PM 12/3/18
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function
import configparser


def config_get_config(configpath):
    args = {}

    config = configparser.ConfigParser()
    config.read(configpath)

    args['DataPath'] = config['PATH']['DataPath']
    args['SpeakerA'] = config['VAR']['src']
    args['SpeakerB'] = config['VAR']['tar']
    args['sampling_rate'] = config['VAR']['sr']
    args['feature_path'] = config['VAR']['feature_path']

    # ars['in_size' =

    return args