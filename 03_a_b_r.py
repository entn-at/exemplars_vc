# -*- coding: utf-8 -*-

""" Created on 3:29 PM 12/11/18
    @author: ngunhuconchocon
    @brief: after training a neural network for FW, pickling everything needed for run-time conversion
    2 item will be pickled:
        - W (the mapping function - the trained neural network)
        - R (residual difference between warped spectrum and the target spectrum: W(A) - B)
        For the source exemplars A, it is the input to the conversion system. It will be calculate in conversion phase runtime
"""

from __future__ import print_function

from models import Net
from utils import config_get_config, io_read_data

import os
import subprocess

import sys
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

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

SEED = 10
args = config_get_config("config/config")


def get_epoch_from_filename(filename):
    """
    Input: checkpoint_epoch51_15.482
    Output: 51, 15.482
    :param filename:
    :return:
    """
    parts = filename.strip().split("_")

    return int(parts[1].strip("epoch")), float(parts[2])


def load_model():
    """
    Get the newest checkpoint (best one) in the checkpoint directory
    For more detail about torch's load/save progress, see:
        https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
    :param filename:
    :return: model (with training progress, can be used for either inference or further training)
    """

    model = Net(int(args['in_size']), int(args['hidden_size']), int(args['out_size']),
                int(args['nb_lstm_layers']), int(args['batch_size']), int(bool(args['bidirectional'])))

    newest_dir = [xxx for xxx in subprocess.check_output("ls -t checkpoints/".split(" ")).decode("utf8").strip().split("\n") if os.path.isdir("checkpoints/" + xxx)][0]  # | head -n 1"])
    print(newest_dir)
    newest_pth = ["checkpoints/{}/{}".format(newest_dir, xxx) for xxx in subprocess.check_output("ls -t checkpoints/{}/".format(newest_dir).split(" ")).decode("utf8").strip().split("\n")
                  if os.path.isfile("checkpoints/{}/{}".format(newest_dir, xxx))]

    list_file = {get_epoch_from_filename(filename)[0]: get_epoch_from_filename(filename)[1] for filename in newest_pth}
    best_epoch = max(list_file.keys())
    best_loss = list_file[best_epoch]
    print(best_epoch, best_loss)

    state = torch.load("checkpoints/{}/checkpoint_epoch{}_{}".format(newest_dir, best_epoch, best_loss), map_location='cpu')
    model.load_state_dict(state['state_dict'])

    optimizer = optim.RMSprop(model.parameters())
    optimizer.load_state_dict(state['optimizer'])

    return model, optimizer


if __name__ == "__main__":
    model, optimizer = load_model()
    loss = nn.L1Loss(reduce=False)

    print(model)
    print(optimizer)

    data_train, data_test = io_read_data(args['SpeakerA'] + '2' + args['SpeakerB'] + '_mfcc_25ms_10ms.pkl', 'pkl')
    model.eval()

    with torch.no_grad():
        for i in range(len(data_test)):
            temp_x = torch.tensor(data_test[i][0]).float()
            temp_y = torch.tensor(data_test[i][1]).float()

            h_state = model.hidden_init(temp_x)

            prediction, h_state = model(torch.tensor(data_test[i][0]).float(), h_state)

            # print(prediction[:5])
            print("===")

            l = loss(prediction, temp_y.float().view(len(temp_y), int(args['batch_size']), -1))
            print(l[:5])
            # print(temp_y.float().view(len(temp_y), int(args['batch_size']), -1)[:5])





