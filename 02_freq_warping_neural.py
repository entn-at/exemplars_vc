# -*- coding: utf-8 -*-

""" Created on 3:06 PM 11/30/18
    @author: ngunhuconchocon
    @brief: The frequency_warping family determine a warping function w'_i in R_M space, from each exemplar pair [a_i, b_i] in dictionary (constructed in 01_make_dict.py)
    This script implement a neural-based approach.
"""

from __future__ import print_function
from utils import config_get_config
from torchsummary.torchsummary import summary
from tqdm import tqdm

import os
import pickle

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TODO config here, like in 01_make_dict.py
args = config_get_config("config/config")

# parse the configuration
## Path and some basic variable
data_path = args['DataPath']
speakerA = args['SpeakerA']
speakerB = args['SpeakerB']
feature_path = args['feature_path']
sr = args['sampling_rate']

logging.getLogger().setLevel(logging.DEBUG)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

use_cuda = torch.cuda.is_available()


def io_read_data(filename, filetype='npy'):
    """
    read data from npy. MFCCs only by far
    :param datapath:
    :return:
    """
    filepath = os.path.join(os.path.join(ROOT_DIR, 'data/vc'), filename)

    if filetype == 'npy':
        return np.load(filepath)
    elif filetype == 'pkl':
        with open(os.path.join(filename), "rb") as f:
            return pickle.load(f)
    else:
        print(filetype, "is not supported by far. Exiting ...")
        exit()


# First try: 2018 Dec 03
def fw_neural_network(*args, **kwargs):
    """
    This define neural-based approach for frequency mapping
    :param args: placeholder
    :param kwargs: placeholder
    :return: kind of f(x)
    """
    raise NotImplementedError


class Net(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        # self.fc1 = nn.Linear()
        self.lstm = nn.LSTM(input_size=self.in_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        # self.fc = nn.Linear(self.hidden_size, self.out_size)
        self.fc1 = nn.Linear(self.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, self.out_size)

    def forward(self, x, h_state):
        # h_state = torch.zeros(len(x))
        print(x.shape)
        out, h_state = self.lstm(x, h_state)
        print(out.shape, h_state[0].shape, h_state[1].shape)
        # print(len(h_state), h_state[0].shape, h_state[1].shape)
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        # out = self.fc(out)
        return out


def get_batches(data0, data1, ii, batch_size, nb_frame_in_batch):
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        batch_x.append(data0[ii + i: ii + i + nb_frame_in_batch])
        batch_y.append(data1[ii + i: ii + i + nb_frame_in_batch])
    # print(len(batch_x), batch_x[0].shape, type(batch_x), type(np.array(batch_x)))

    # return torch.from_numpy(np.array(batch_x)), torch.from_numpy(np.array(batch_y))
    return torch.stack(batch_x, dim=0), torch.stack(batch_y, dim=0)


def train():
    """
    See this script for more information
    https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/403_RNN_regressor.py
    :return:
    """
    if use_cuda:
        net = Net(20, 20, 20).cuda()
    else:
        net = Net(20, 20, 20)

    # summary(net, (4, 20))
    # exit()

    optimizer = optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    net.train()

    data = io_read_data(speakerA + '2' + speakerB + '_mfcc.pkl')

    batch_size = 4
    nb_frame_in_batch = 8

    h_state = [torch.zeros(2, 4, 20) for _ in range(2)]

    for epoch in range(10):
        count = 0

        loss_sum = 0
        for i in tqdm(range(len(data))):
            # each batch contains `batch_size` of frames
            # for batch_x, batch_y in zip(torch.split(torch.tensor(data[i][0]), batch_size), torch.split(torch.tensor(data[i][1]), batch_size)):
            # for ii in range(0, data[i][0].shape[0] - batch_size):
            #     batch_x, batch_y = torch.tensor(data[i][0][ii: ii + batch_size]), torch.tensor(data[i][1][ii: ii + batch_size])
            if use_cuda:
                temp_x = torch.tensor(data[i][0]).cuda()
                temp_y = torch.tensor(data[i][1]).cuda()
            else:
                temp_x = torch.tensor(data[i][0])
                temp_y = torch.tensor(data[i][1])

            for ii in range(0, data[i][0].shape[0] - nb_frame_in_batch):
                # batch_x, batch_y = temp_x[ii: ii + batch_size], temp_y[ii: ii + batch_size]
                batch_x, batch_y = get_batches(temp_x, temp_y, ii, batch_size, nb_frame_in_batch)
                optimizer.zero_grad()

                prediction = net(batch_x.float(), h_state)
                # prediction = net(batch_x.unsqueeze(0).float(), None)

                # Do we need h_state?
                # h_state = h_state.data

                # print(batch_x.shape, batch_x.unsqueeze(0).shape)
                # print(prediction.shape)
                # print(batch_y.shape)
                # input()
                loss = criterion(prediction.float()[0], batch_y.float())

                loss.backward()
                optimizer.step()

                loss_sum += loss

                count += 1

        print("Epoch {}: average loss = {}".format(epoch, loss_sum / count))


def main():
    ## For the net

    # data = io_read_data(speakerA + '2' + speakerB + '_mfcc.pkl')
    #
    # print(data[0].shape, data[1].shape)
    # net = Net()

    train()


if __name__ == "__main__":
    main()
