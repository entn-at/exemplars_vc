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

# #In this example i have 2 different nets. The input is fed to the first net.
# #The output of the first net is fed to the second net.
# #I only want to train the second net in this example
#
# net1_input = get_data()
# net1_output = net1(net1_input)
# #This creates a computation graph from input to output (i.e. all operations performed on the input to acquire the output). This is used when we want to get gradients for the weights we want to improve.
# #As we only want to train the second net we dont need the computation graph from the first net. Therefore we detach the output from the previous computation graph
# net2_input = net1_output.detach()
# net2_output = net2(net2_input)
# target = get_target()
# loss = loss_function(net2_output,target)
# loss.backward()
# #loss.backward() only computes gradients for the second net as we detached the output of the first net before we fed it to the second net
