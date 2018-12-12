# -*- coding: utf-8 -*-

""" Created on 5:12 PM 12/11/18
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function

import datetime
import logging

import torch
import torch.nn as nn

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

# print("====================================================================")
# args = config_get_config("config/config")
# pprint.pprint(args)
# print("====================================================================")
#
# # parse the configuration
# ## Path and some basic variable
# data_path = args['DataPath']
# speakerA = args['SpeakerA']
# speakerB = args['SpeakerB']
# feature_path = args['feature_path']
# sr = args['sampling_rate']
# MODEL_PTH_NAME = args['checkpoint_name']
#
# nb_lstm_layers = int(args['nb_lstm_layers'])
# batch_size = int(args['batch_size'])
# nb_frame_in_batch = int(args['nb_frame_in_batch'])
# patience = int(args['patience'])
# nb_epoch = int(args['nb_epoch'])
# dropout_rate = float(args['dropout_rate'])
# bidirectional = int(bool(args['bidirectional']))
#
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# print("====================================================================")
# logging.debug("Start training frequency warping")
# use_cuda = torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, nb_lstm_layers, batch_size, bidirectional=False):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.nb_lstm_layers = nb_lstm_layers
        self.bi = int(bool(bidirectional)) + 1
        self.batch_size = batch_size

        # self.fc1 = nn.Linear()
        self.lstm = nn.LSTM(input_size=self.in_size, hidden_size=self.hidden_size, num_layers=self.nb_lstm_layers, batch_first=False, bidirectional=False)
        # self.fc = nn.Linear(self.hidden_size, self.out_size)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        # self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.out_size)
        self.fc4 = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x, h_state):
        out, h_state = self.lstm(x.view(len(x), 1, -1))
        # print(embeds.view(len(sentence), 1, -1))

        output_fc = []

        for frame in out:
            # output_fc.append(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(frame))))))
            # output_fc.append(self.fc3(torch.tanh(self.fc1(frame))))
            output_fc.append(self.fc4(frame))

        return torch.stack(output_fc), h_state

    def hidden_init(self, temp_x):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            h_state = torch.stack((torch.zeros(self.nb_lstm_layers * self.bi, int(temp_x.shape[0]/self.batch_size), self.hidden_size * self.bi), torch.zeros(self.nb_lstm_layers * self.bi, int(temp_x.shape[0]/self.batch_size), self.hidden_size * self.bi))).cuda()
        #     h_state = torch.stack([torch.zeros(nb_lstm_layers, batch_size, 20) for _ in range(2)]).cuda()
        else:
            h_state = torch.stack((torch.zeros(self.nb_lstm_layers * self.bi, int(temp_x.shape[0]/self.batch_size), self.hidden_size * self.bi), torch.zeros(self.nb_lstm_layers * self.bi, int(temp_x.shape[0]/self.batch_size), self.hidden_size * self.bi)))
        #     h_state = torch.stack([torch.zeros(nb_lstm_layers, batch_size, 20) for _ in range(2)])

        return h_state
