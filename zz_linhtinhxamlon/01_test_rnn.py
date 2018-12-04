# -*- coding: utf-8 -*-

""" Created on 11:42 AM 12/3/18
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary.torchsummary import summary

rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)

print(output)