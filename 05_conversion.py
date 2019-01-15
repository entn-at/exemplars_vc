# -*- coding: utf-8 -*-

""" Created on 10:10 AM 12/24/18
    @author: ngunhuconchocon
    @brief: This script calculate the `residual compensation` of source and target voice:
                    log(r_n) = log(b_n) - log(bhat_n), with b_n is target-exemplar, bhat_n = w(b_n), with w is the warping function
"""

from __future__ import print_function
from utils import config_get_config
from sklearn.decomposition import NMF

import os
import sys
import pdb

import logging
import datetime
import pickle

import pyworld as pw
import librosa as lbr
import numpy as np
import pprint

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

args = config_get_config("config/config")
data_path = args['DataPath']
feature_path = args['feature_path']
speakerA = args['SpeakerA']
speakerB = args['SpeakerB']


def io_load_from_pickle(speaker):
    pickle_path = os.path.join(feature_path, "exem_dict")
    with open(os.path.join(pickle_path, "{}_feat_sp_ap_f0.pkl".format(speaker)), "rb") as f:
        exemplar_speaker = pickle.load(f)

    with open(os.path.join(pickle_path, "{}_feat_sp_ap_f0.pkl".format(speaker)), "rb") as f:
        exemplar_W_speaker = pickle.load(f)

    return exemplar_speaker, exemplar_W_speaker


if __name__ == "__main__":
    filedata, sr = lbr.load("data/Full_data/SF1/100162.wav", sr=None, dtype=np.float64)
    src_for_conversion = _get_conversion_data(filedata, fs=sr, refine_f0=True)

    exemplar_A, exemplar_W_A = io_load_from_pickle(speakerA)

    # we will use NMF to factorize each frame of src_for_conversion (with respect to each type of features)
    # in exemplar_A to a product of exemplar_A and an activation matrix (named H).
    # H is consider to be a presentation of linear combination of frames in exemplar_A

    _W = []
    for i in range(len(exemplar_W_A[:-150])):
        _W.extend(exemplar_W_A[i]['sp'])

    _W = np.array(_W)
    print(len(_W))
    model = NMF(n_components=_W.T.shape[1])

    # for frame in src_for_conversion['sp']:
        # Each frame have size of (1, 513). Need to transpose:
        #       frame.T = exemplar_A[:]['sp'] x h (h is activation matrix)

    output = model.fit_transform(src_for_conversion['sp'][0].T[:, np.newaxis], W=_W.T)
    print(output)
    pdb.set_trace()

    # exemplar_B, exemplar_W_B = io_load_from_pickle(speakerB)
    #
    # pdb.set_trace()


