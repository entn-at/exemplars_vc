# -*- coding: utf-8 -*-

""" Created on 3:06 PM 11/30/18
    @author: ngunhuconchocon
    @brief: The frequency_warping family determines a warping function w'_i in R_M space, from each exemplar pair [a_i, b_i] in dictionary (constructed in 01_make_dict.py)
    This script implements a neural-based approach.
"""

from __future__ import print_function

from utils import config_get_config, io_read_data
from models import Net
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
from scipy.stats import describe

import logging
import datetime

os.system("mkdir -p logs")
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

# TODO config here, like in 01_make_dict.py
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print("====================================================================")
logging.debug("Start training frequency warping")
use_cuda = torch.cuda.is_available()

if use_cuda:
    print("Using GPU id={}. Device name: {}".format(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
else:
    print("GPU is not available. Using CPU ...")


def get_batches(data0, data1, ii, batch_size, nb_frame_in_batch):
    assert len(data0) == len(data1), "Inconsistent input size"
    assert batch_size + nb_frame_in_batch - 1 <= len(data0), "...asdoiasjdfioqwhfoqwhjifjqwidf"

    batch_x = []
    batch_y = []
    for i in range(batch_size):
        batch_x.append(data0[ii + i: ii + i + nb_frame_in_batch])
        batch_y.append(data1[ii + i: ii + i + nb_frame_in_batch])

    # return torch.from_numpy(np.array(batch_x)), torch.from_numpy(np.arraybatch(batch_y))
    return torch.stack(batch_x, dim=0), torch.stack(batch_y, dim=0)


def save_checkpoint(filename, model, state=None):
    """
    Save the torch model. If state is not None, save all the model, state of the training progress (lr, epoch, ...). Else, save the model only
    For more detail about torch's load/save progress, see:
            https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
    :param filename: name of the checkpoint
    :param model: the model
    :param state: the state of the training progress, contain the model itself. Optional
    :return:
    """
    if not state:
        torch.save(model.state_dict(), os.path.join('checkpoints/', filename))
    else:
        _state = {
            'epoch': state['epoch'],
            'state_dict': state['state_dict'].state_dict(),
            'optimizer': state['optimizer'].state_dict()
        }

        torch.save(_state, os.path.join('checkpoints/', filename))


def logdir():
    """
    return an available name for tensorboard dir. The logdir will look like this
    --- runs
         |--1
         |--2
         |--3
         |--4
        To this case, "5" will be returned. Even with the case
    :return:
    """
    listdir = [xx for xx in os.listdir("runs/") if os.path.isdir(os.path.join("runs", xx))]

    max_ = 0
    for i in range(len(listdir)):
        try:
            filename = int(listdir[i])
            if max_ < filename:
                max_ = filename
        except ValueError:
            continue

    return str(max_ + 1)


def train(data_train, data_test):
    """
    See this script for more information
    https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/403_RNN_regressor.py
    :return:
    """
    data = data_train
    # # xxx = [item for xx in data for item in xx]
    # xxx = []
    # for xx in data:
    #     xxx.extend(xx.flatten())

    checkpoint_and_write_save_dir = logdir()

    os.system("mkdir -p checkpoints")
    os.system("mkdir -p checkpoints/{}".format(checkpoint_and_write_save_dir))

    writer = SummaryWriter(os.path.join("runs", checkpoint_and_write_save_dir), comment="FreqWarp")

    logging.info("Building architecture...")

    if use_cuda:
        net = Net(20, 20, 20, nb_lstm_layers, batch_size).cuda()
    else:
        net = Net(20, 20, 20, nb_lstm_layers, batch_size)
    net.train()

    # optimizer = optim.SGD(net.parameters(), lr=0.001)
    # optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=0.0001)
    optimizer = optim.RMSprop(net.parameters(), lr=0.005, weight_decay=0.0001)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss(size_average=False)

    logging.info("Reading data ...")

    best_avg_loss = 1000000
    best_avg_loss_at_epoch = 0

    logging.info("START TRAINING ... MAX EPOCH: " + str(nb_epoch))
    for epoch in range(nb_epoch):
        print("====================================================================")
        count = 0
        loss_sum = 0

        for i in range(len(data)):
            if use_cuda:
                temp_x = torch.tensor(data[i][0]).cuda()
                temp_y = torch.tensor(data[i][1]).cuda()
            else:
                temp_x = torch.tensor(data[i][0])
                temp_y = torch.tensor(data[i][1])
                
            # exit()
            # for ii in range(0, data[i][0].shape[0] - nb_frame_in_batch*2 + 1):
            optimizer.zero_grad()

            h_state = net.hidden_init(temp_x)  # New added Dec 07: They say hidden state need to be clear before each step

            # prediction, h_state = net(batch_x.float(), h_state)
            prediction, h_state = net(temp_x.float(), h_state)
            # prediction = net(batch_x.unsqueeze(0).float(), None)

            loss = criterion(prediction.float(), temp_y.float().view(len(temp_y), batch_size, -1))

            # h_state = (h_state[0].detach(), h_state[1].detach())

            loss.backward()
            optimizer.step()

            loss_sum += loss
            count += 1

        else:
            with torch.no_grad():
                losses = []
                for i in range(len(data_test)):
                    if use_cuda:
                        temp_x = torch.tensor(data_test[i][0]).cuda()
                        temp_y = torch.tensor(data_test[i][1]).cuda()
                    else:
                        temp_x = torch.tensor(data_test[i][0])
                        temp_y = torch.tensor(data_test[i][1])

                    h_state = net.hidden_init(temp_x)
                    prediction, h_state = net(temp_x.float(), h_state)
                    loss = criterion(prediction.float(), temp_y.float().view(len(temp_y), batch_size, -1))

                    losses.append(loss.data.item())
            logging.info(describe(losses))

            writer.add_scalar("loss/minibatch", loss_sum / count, global_step=epoch)
            # writer.add_graph(net, (temp_x.float(), h_state), verbose=True)

        # for m_index, m in enumerate(net.parameters()):
        #     print(m_index)
        #     print(net_modules[m_index])
        #     writer.add_histogram('histogram/', m.data, global_step=epoch)
        for name, param in net.named_parameters():
            writer.add_histogram('histogram/' + name, param.data, global_step=epoch)

        avg_loss = loss_sum / count
        if avg_loss < best_avg_loss:
            state = {
                'epoch': epoch,
                'state_dict': net,
                'optimizer': optimizer
            }

            save_checkpoint(checkpoint_and_write_save_dir + "/" + MODEL_PTH_NAME + "_epoch" + str(epoch) + "_" + str(round(float(avg_loss), 3)), model=net, state=state)

            logging.info("Epoch {}: average loss = {:.3f}, improve {:.3f} from {:.3f}. Model saved at checkpoints/{}/{}.pth"
                         .format(epoch, avg_loss, best_avg_loss - avg_loss, best_avg_loss, checkpoint_and_write_save_dir, MODEL_PTH_NAME + "_epoch" + str(epoch) + "_" + str(round(float(avg_loss), 3))))

            best_avg_loss = avg_loss
            best_avg_loss_at_epoch = epoch

        elif epoch - best_avg_loss_at_epoch > patience:
            logging.info("Model hasn't improved since epoch {}. Stop training ...".format(best_avg_loss_at_epoch))
            break
        else:
            logging.info("Epoch {}: average loss = {:.3f}. No improvement since epoch {}".format(epoch, avg_loss, best_avg_loss_at_epoch))

    writer.close()

    return net


def evaluate(model, dataset):
    pass


def norm(data):
    pass


def main():
    data_train, data_test = io_read_data(speakerA + '2' + speakerB + '_mfcc_25ms_10ms.pkl', 'pkl')
    train(data_train, data_test)


if __name__ == "__main__":
    main()
