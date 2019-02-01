# -*- coding: utf-8 -*-

""" Created on 2:52 PM 1/31/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь!
"""

from __future__ import print_function

import os
import sys
import pdb

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

import librosa as lbr
import numpy as np
import librosa as lbr

from zz_audio_utilities import reconstruct_signal_griffin_lim

hop_length_in_ms = 5
n_fft = 2048


if __name__ == "__main__":
    # Extract stft
    print("---------------------------------------")
    # data, sr = lbr.load(lbr.util.example_audio_file())  # , sr=None)
    data, sr = lbr.load("/home/enamoria/Desktop/samples/tgt_kinhph/xu-ong-nguyen-khac-thuy-dam-o-tre-em-o-vung-tau_4.wav", sr=None)
    # hop_length = int(sr / 1000.0 * hop_length_in_ms)
    hop_length = 256
    stft = lbr.core.stft(data, n_fft=n_fft, hop_length=hop_length).T

    # Get magnitude
    # The spectrogram shows the the intensity of frequencies over time. A spectrogram is simply the squared magnitude of the STFT
    print("---------------------------------------")
    S = lbr.amplitude_to_db(abs(stft))
    # pdb.set_trace()

    # Reconstructing waveform
    # Using griffin_lim from audio_utilities.reconstruct_signal_griffin_lim
    #           x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(stft_modified_scaled,
    #                                                       args.fft_size, hopsamp,
    #                                                       args.iterations)
    print("---------------------------------------")
    stft_2_wav = reconstruct_signal_griffin_lim(np.abs(stft), n_fft, hop_length, 300)

    lbr.output.write_wav("reconstructed_wav.wav", stft_2_wav, sr=sr, norm=True)
    pdb.set_trace()