# -*- coding: utf-8 -*-

""" Created on 2:19 PM 1/14/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!

"""

from __future__ import print_function
from dtw import dtw
from tqdm import tqdm
from sklearn.decomposition import NMF, non_negative_factorization
from utils import config_get_config

import numpy as np
import pyworld as pw
import librosa as lbr

import os
import sys
import pdb
import pickle

import time
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

args = config_get_config("config/config")

data_path = args['DataPath']
speakerA = args['SpeakerA']
speakerB = args['SpeakerB']
feature_path = args['feature_path']
sr = int(args['sampling_rate'])

refine_f0 = args['is_refined']
frame_period = args['frame_period']

cpu_rate = float(args['cpu_rate'])


def io_load_from_pickle():
    with open("./data/vc/exem_dict/SF1_feat_sp_ap_f0.pkl", "rb") as f:
        src_feat = pickle.load(f)

    with open("./data/vc/exem_dict/TF1_feat_sp_ap_f0.pkl", "rb") as f:
        tar_feat = pickle.load(f)

    with open("./data/vc/exem_dict/exemplar_W_A", "rb") as f:
        src_W = pickle.load(f)

    with open("./data/vc/exem_dict/exemplar_W_B", "rb") as f:
        tar_W = pickle.load(f)

    return src_feat, tar_feat, src_W, tar_W


def _spectral_dtw(x, y, dist):
    """
        Note: this is file-based implementation for multiprocessing. Different from non-parallel version
    Calculate spectral distance by using dtw
    :param x:
    :param y:
    :param dist:
    :return:
    """
    return dtw(x, y, dist=dist)


def align_sp_ap_f0(feat=None, dtw_path=None):
    """
        Align spectral feature, based on dtw_path. Check if an available pickled version is existed
    :param feat: spectral features
    :param dtw_path: dtw_path of the audio
    :return: `feat`-like object, but aligned
    """
    src_feat, tar_feat, src_W, tar_W = io_load_from_pickle()

    aligned_src_feat = []
    aligned_tar_feat = []

    for i in tqdm(range(len(src_W))):  # iterate through each audio file
        sp1 = []; ap1 = []; f01 = []
        sp2 = []; ap2 = []; f02 = []

        for j in range(len(src_W[i])):
            sp1.append(src_feat[i]['sp'][src_W[i][j]])
            ap1.append(src_feat[i]['ap'][src_W[i][j]])
            f01.append(src_feat[i]['f0'][src_W[i][j]])

            sp2.append(tar_feat[i]['sp'][tar_W[i][j]])
            ap2.append(tar_feat[i]['ap'][tar_W[i][j]])
            f02.append(tar_feat[i]['f0'][tar_W[i][j]])

        aligned_src_feat.append({
            'sp': np.asarray(sp1),
            'ap': np.asarray(ap1),
            'f0': np.asarray(f01),
            'fs': src_feat[i]['fs']
        })

        aligned_tar_feat.append({
            'sp': np.asarray(sp2),
            'ap': np.asarray(ap2),
            'f0': np.asarray(f02),
            'fs': tar_feat[i]['fs']
        })

    return aligned_src_feat, aligned_tar_feat


def synthesize(f0, sp, ap, fs, filename=str(datetime.datetime.now())):
    # y = pw.synthesize(aligned_src_feat[0]['f0'], aligned_src_feat[0]['sp'], aligned_src_feat[0]['ap'], aligned_src_feat[0]['fs'])
    os.system("mkdir -p wav")
    y = pw.synthesize(f0, sp, ap, fs)

    lbr.output.write_wav("wav/{}.wav".format(filename), y, sr=fs)


def _factorize(X, W, beta_loss="kullback-leibler"):
    """
        Calculate matrix `H`, which `W x H ~ X`
        Note: In this function, `sklearn.decomposition's` non_negative_factorization is used. This function's implementation only
                allow we to feed in a fixed H, not W. So we need to used a trick here to feed in a fixed W (we need to fix W, it's our exemplar dictionary).
                Instead of calling `non_negative_factorization(X=X, H=H)` directly, we feed in `X.T as X, W.T as H. Notice that:
                        X.T = H.T x W.T
                    So we transpose H and W to swap their roles. Remember to swap the result back

    :param X: original matrix, which we need to decompose
    :param W: We have precomputed W.
    :return: H: W ~ W x H
    """
    # Note: This statement should be true, but the input of this function is (n_samples, n_features), so we don't need to transpose (X.T). See their shapes for better understand
    # _W, _H, n_iter = non_negative_factorization(X=X.T, H=W.T, init="custom", update_H=False, n_components=W.T.shape[1], beta_loss=beta_loss, solver='mu')

    _W, _H, n_iter = non_negative_factorization(X=X, H=W, init="custom", update_H=False, n_components=W.shape[0], beta_loss=beta_loss, solver='mu')

    return _W.T


def factorize(tobe_converted, src_feat):
    """
        Perform Non-negative Matrix Factorization. Audio-file level.
        1st attempt: perform nmf on each feature type (sp, ap, f0)

        Note: NMF is best used with the fit_transform method, which returns the matrix W.
        The matrix H is stored into the fitted model in the components_ attribute;

            Note of note: X^T=H^T W^T. sklearn NMF decompose X = WH, but can fix H only (use precomputed H)
    :return:
    """
    if os.path.isfile("data/vc/exem_dict/{}.pkl".format("H_test_8")):
        with open("data/vc/exem_dict/{}.pkl".format("H_test_8"), "rb") as f:
            return pickle.load(f)
    else:
        conv_sp, conv_ap, conv_f0 = tobe_converted['sp'], tobe_converted['ap'], tobe_converted['f0'][:, np.newaxis]

        A_sp = []  # np.asarray([src_feat[i]['sp'] for i in range(len(src_feat))])
        A_ap = []  # np.asarray([src_feat[i]['ap'] for i in range(len(src_feat))])
        A_f0 = []  # np.asarray([src_feat[i]['f0'] for i in range(len(src_feat))])

        for i in range(len(src_feat)):
            A_sp.extend(src_feat[i]['sp'])
            A_ap.extend(src_feat[i]['ap'])
            A_f0.extend(src_feat[i]['f0'])

        A_sp = np.asarray(A_sp)
        A_ap = np.asarray(A_ap)
        A_f0 = np.asarray(A_f0)[:, np.newaxis]

        H_sp = _factorize(X=conv_sp, W=A_sp)
        H_ap = _factorize(X=conv_ap, W=A_ap)
        H_f0 = _factorize(X=conv_f0, W=A_f0)

        H = {'H_sp': H_sp, 'H_ap': H_ap, 'H_f0': H_f0}

        with open("data/vc/exem_dict/{}.pkl".format("H_test_8"), "wb") as f:
            pickle.dump(H, f)

        return H


def convert(H, tar_feat):
    # TODO MAI VIET DOC
    """

    :param H:
    :param tar_feat:
    :return:
    """
    H_sp, H_ap, H_f0 = H['H_sp'], H['H_ap'], H['H_f0']

    B_sp = []  # np.asarray([src_feat[i]['sp'] for i in range(len(src_feat))])
    B_ap = []  # np.asarray([src_feat[i]['ap'] for i in range(len(src_feat))])
    B_f0 = []  # np.asarray([src_feat[i]['f0'] for i in range(len(src_feat))])

    for i in range(len(tar_feat)):
        B_sp.extend(tar_feat[i]['sp'])
        B_ap.extend(tar_feat[i]['ap'])
        B_f0.extend(tar_feat[i]['f0'])

    B_sp = np.asarray(B_sp)
    B_ap = np.asarray(B_ap)
    B_f0 = np.asarray(B_f0)[:, np.newaxis]

    converted_sp = np.matmul(H_sp.T, B_sp)
    converted_ap = np.matmul(H_ap.T, B_ap)
    converted_f0 = np.matmul(H_f0.T, B_f0)

    converted_feature = {
        'sp': converted_sp,
        'ap': converted_ap,
        'f0': np.squeeze(converted_f0)
    }

    return converted_feature


if __name__ == "__main__":
    start = time.time()

    tobe_converted, sr = lbr.load("data/Full_data/SF1/100162.wav", sr=None, dtype=np.double)

    # Extract feature
    _f0, t = pw.dio(tobe_converted, fs=sr)  # raw pitch extractor

    if refine_f0:
        f0 = pw.stonemask(tobe_converted, _f0, t, fs=sr)  # pitch refinement
    else:
        f0 = _f0

    sp = pw.cheaptrick(tobe_converted, f0, t, fs=sr)  # extract smoothed spectrogram
    ap = pw.d4c(tobe_converted, f0, t, fs=sr)  # extract aperiodicity
    # y = pw.synthesize(f0, sp, ap, fs)

    tobe_converted_features = {'sp': sp, 'ap': ap, 'f0': f0, 'fs': sr, 'sr': sr}

    aligned_src_feat, aligned_tar_feat = align_sp_ap_f0()
    assert len(aligned_tar_feat) == len(aligned_src_feat)

    # TODO parallel
    logging.info("====================")
    H = factorize(tobe_converted_features, aligned_src_feat)

    print("Elapsed time: {}".format(time.time() - start))

    #
    converted = convert(H, aligned_tar_feat)

    synthesize(converted['f0'], converted['sp'], converted['ap'], fs=sr)
# Сою́з Сове́тских Социалисти́ческих Респу́блик
# союз советский
# советский союз
# советский союз


