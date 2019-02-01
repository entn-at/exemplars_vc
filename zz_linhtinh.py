# -*- coding: utf-8 -*-

""" Created on 10:48 AM 1/14/19
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function

import logging
import datetime

logging.basicConfig(
    filename="logs/" + ":".join(str(datetime.datetime.now()).split(":")[:-1]),
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
import pysptk
try:
    import coloredlogs

    coloredlogs.install(level=logging.DEBUG, fmt='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')
except ModuleNotFoundError:
    pass

pysptk.
# import numpy as np
# from nmf_tool.nmf import NMF
#
# # data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
# data = np.random.rand(5, 10)
#
# # nmf = NMF(data, num_bases=data.shape[0])
# # nmf = pymf.cnmf.RNMF(data, num_bases=data.shape[0])
# W = np.random.rand(5, 15)
#
# model = NMF(max_iter=2000, learning_rate=0.1, display_step=10, optimizer='mu', initW=True)
# W, H = model.fit_transform(data, r_components=2, initW=False, givenW=W)
# print(data)
# print("=======")
# print(np.matmul(W, H))
# print("=======")
# x = np.square(model.inverse_transform(W, H) - data)
# print(np.sum(x))
# import pdb
# import torch
# import librosa
# from torchnmf import NMF
# from torchnmf.metrics import KL_divergence

# y, sr = librosa.load(librosa.util.example_audio_file())
# y = torch.from_numpy(y)
# windowsize = 2048
# S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
#
# R = 8   # number of components
#
# net = NMF(S.shape, n_components=R)
# run extremely fast on gpu

data = torch.Tensor(np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]]))
W = torch.Tensor([[2,3], [4,2]])
net = NMF(data.shape, n_components=W.shape[1])

n_iter, H = net.fit_transform(data, update_W=False, W=W)
print(KL_divergence(H, data))
print(n_iter
      )
pdb.set_trace()