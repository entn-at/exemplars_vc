# -*- coding: utf-8 -*-

""" Created on 1:44 PM 12/10/18
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function
from utils import config_get_config, io_read_data
from tqdm import tqdm

import os
import pickle
import pprint

import cProfile
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary.torchsummary import summary
from tensorboardX import SummaryWriter

from sklearn.model_selection import train_test_split

import logging
import datetime

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU id={}. Device name: {}".format(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
else:
    print("GPU is not available. Using CPU ...")

print("====================================================================")
args = config_get_config("config/config")

pprint.pprint(args)
print("====================================================================")

# parse the configuration
## Path and some basic variable
data_path = args['DataPath']
speakerA = args['SpeakerA']
speakerB = args['SpeakerB']
feature_path = args['feature_path']
sr = args['sampling_rate']
MODEL_PTH_NAME = args['checkpoint_name']

nb_lstm_layers = int(args['nb_lstm_layers'])
batch_size = int(args['batch_size'])
nb_frame_in_batch = int(args['nb_frame_in_batch'])
patience = int(args['patience'])
nb_epoch = int(args['nb_epoch'])
dropout_rate = float(args['dropout_rate'])
bidirectional = int(bool(args['bidirectional']))

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


class Net(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, nb_lstm_layers):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.nb_lstm_layers = nb_lstm_layers
        self.bi = bidirectional + 1

        # self.fc1 = nn.Linear()
        self.lstm = nn.LSTM(input_size=self.in_size, hidden_size=self.hidden_size, num_layers=self.nb_lstm_layers, batch_first=False, bidirectional=False)
        # self.fc = nn.Linear(self.hidden_size, self.out_size)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.out_size)

    def forward(self, x, h_state):
        out, h_state = self.lstm(x.view(len(x), 1, -1))
        # print(embeds.view(len(sentence), 1, -1))

        output_fc = []

        for frame in out:
            # output_fc.append(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(frame))))))
            output_fc.append(self.fc3(torch.tanh(self.fc1(frame))))

        return torch.stack(output_fc), h_state

    def hidden_init(self, temp_x):
        if use_cuda:
            h_state = torch.stack((torch.zeros(nb_lstm_layers * self.bi, int(temp_x.shape[0]/batch_size), self.hidden_size * self.bi), torch.zeros(nb_lstm_layers * self.bi, int(temp_x.shape[0]/batch_size), self.hidden_size * self.bi))).cuda()
        #     h_state = torch.stack([torch.zeros(nb_lstm_layers, batch_size, 20) for _ in range(2)]).cuda()
        else:
            h_state = torch.stack((torch.zeros(nb_lstm_layers * self.bi, int(temp_x.shape[0]/batch_size), self.hidden_size * self.bi), torch.zeros(nb_lstm_layers * self.bi, int(temp_x.shape[0]/batch_size), self.hidden_size * self.bi)))
        #     h_state = torch.stack([torch.zeros(nb_lstm_layers, batch_size, 20) for _ in range(2)])

        return h_state


def evaluate(model, dataset):
    pass


def main():
    data_train, data_test = io_read_data(speakerA + '2' + speakerB + '_mfcc_25ms_10ms.pkl', 'pkl')

    model = Net(20, 20, 20, nb_lstm_layers)

    model.load_state_dict(torch.load("checkpoints/events.out.tfevents.1544417266.voice-lab-04"))

    for i in range(len(data_test)):
        prediction = evaluate(model, data_test[i])
        print(nn.L1Loss(prediction, data_test[i][]))

    # cProfile("train()")
    # print(logdir())


if __name__ == "__main__":
    main()