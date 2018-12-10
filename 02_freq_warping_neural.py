# -*- coding: utf-8 -*-

""" Created on 3:06 PM 11/30/18
    @author: ngunhuconchocon
    @brief: The frequency_warping family determines a warping function w'_i in R_M space, from each exemplar pair [a_i, b_i] in dictionary (constructed in 01_make_dict.py)
    This script implements a neural-based approach.
"""

from __future__ import print_function
from utils import config_get_config
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print("====================================================================")
logging.debug("Start training frequency warping")
use_cuda = torch.cuda.is_available()

if use_cuda:
    print("Using GPU id={}. Device name: {}".format(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
else:
    print("GPU is not available. Using CPU ...")


def io_read_data(filename, filetype='npy'):
    """
    read data from npy. MFCCs only by far
    :param datapath:
    :return:
    """
    filepath = os.path.join(os.path.join(ROOT_DIR, 'data/vc'), filename)
    logging.info("Reading from {}".format(filename))

    if filetype == 'npy':
        return np.load(filepath)
    elif filetype == 'pkl':
        with open(os.path.join(filename), "rb") as f:
            return pickle.load(f)
    else:
        logging.critical(filetype, "is not supported by far. Exiting ...")
        exit()


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


def save_checkpoint(filename, model):
    torch.save(model.state_dict(), os.path.join('checkpoints/', filename))


def logdir():
    """
    return an available name for tensorboard dir. The logdir will look like this
    --- runs
         |--1
         |--2
         |--3
         |--4
        To this case, "5" will be returned
    :return:
    """
    listdir = [xx for xx in os.listdir("runs/") if os.path.isdir(os.path.join("runs", xx))]

    max = 0
    for i in range(len(listdir)):
        try:
            filename = int(listdir[i])
            if max < filename:
                max = filename
        except ValueError:
            continue

    return str(max + 1)
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
    def __init__(self, in_size, hidden_size, out_size, nb_lstm_layers):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.nb_lstm_layers = nb_lstm_layers

        # self.fc1 = nn.Linear()
        self.lstm = nn.LSTM(input_size=self.in_size, hidden_size=self.hidden_size, num_layers=self.nb_lstm_layers, batch_first=True, bias=True)
        # self.fc = nn.Linear(self.hidden_size, self.out_size)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.out_size)

    def forward(self, x, h_state):
        out, h_state = self.lstm(x, h_state)
        output_fc = []

        for frame in out:
            # output_fc.append(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(frame))))))
            output_fc.append(self.fc3(torch.tanh(self.fc1(frame))))

        return torch.stack(output_fc), h_state

    def hidden_init(self):
        if use_cuda:
            h_state = torch.stack([torch.zeros(nb_lstm_layers, batch_size, 20) for _ in range(2)]).cuda()
        else:
            h_state = torch.stack([torch.zeros(nb_lstm_layers, batch_size, 20) for _ in range(2)])

        return h_state

        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)
        # out = self.fc(out)
        # return out


def train():
    """
    See this script for more information
    https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/403_RNN_regressor.py
    :return:
    """
    checkpoint_and_write_save_dir = logdir()

    os.system("mkdir -p checkpoints")
    os.system("mkdir -p checkpoints/{}".format(checkpoint_and_write_save_dir))

    writer = SummaryWriter(os.path.join("runs", checkpoint_and_write_save_dir), comment="FreqWarp")

    logging.info("Building architecture...")

    if use_cuda:
        net = Net(20, 20, 20, nb_lstm_layers).cuda()
    else:
        net = Net(20, 20, 20, nb_lstm_layers)
    net.train()

    # optimizer = optim.SGD(net.parameters(), lr=0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.0001)
    criterion = nn.MSELoss()

    logging.info("Reading data ...")
    data = io_read_data(speakerA + '2' + speakerB + '_mfcc_25ms_10ms.pkl')

    best_avg_loss = 10000
    best_avg_loss_at_epoch = 0

    logging.info("START TRAINING ... MAX EPOCH: " + str(nb_epoch))
    for epoch in range(nb_epoch):
        count = 0
        loss_sum = 0

        batch_x = None
        for i in (range(len(data))[:30]):
            if use_cuda:
                temp_x = torch.tensor(data[i][0]).cuda()
                temp_y = torch.tensor(data[i][1]).cuda()
            else:
                temp_x = torch.tensor(data[i][0])
                temp_y = torch.tensor(data[i][1])

            # print(temp_x.shape, temp_y.shape)

            for ii in range(0, data[i][0].shape[0] - nb_frame_in_batch*2 + 1):
                # batch_x, batch_y = temp_x[ii: ii + batch_size], temp_y[ii: ii + batch_size]
                batch_x, batch_y = get_batches(temp_x, temp_y, ii, batch_size, nb_frame_in_batch)
                optimizer.zero_grad()

                h_state = net.hidden_init()  # New added Dec 07: They say hidden state need to be clear before each step

                prediction, h_state = net(batch_x.float(), h_state)
                # prediction = net(batch_x.unsqueeze(0).float(), None)

                # Do we need h_state?
                # h_state = h_state.data

                loss = criterion(prediction.float(), batch_y.float())

                h_state = (h_state[0].detach(), h_state[1].detach())

                loss.backward()
                optimizer.step()

                loss_sum += loss
                count += 1

                if ii % 50 == 0:
                    logging.debug("Step {}: loss: {}".format(ii, float(loss.data)))
                    writer.add_scalar("loss/minibatch", loss_sum / count, global_step=epoch * i + ii)
            else:
                print("====================================================================")
                # input()
                pass

        else:
            writer.add_scalar("loss/minibatch", loss_sum / count, global_step=epoch)
            writer.add_graph(net, (batch_x.float(), h_state), verbose=False)

        # for m_index, m in enumerate(net.parameters()):
        #     print(m_index)
        #     print(net_modules[m_index])
        #     writer.add_histogram('histogram/', m.data, global_step=epoch)
        for name, param in net.named_parameters():
            writer.add_histogram('histogram/' + name, param.data, global_step=epoch)

        avg_loss = loss_sum / count
        if avg_loss < best_avg_loss:
            save_checkpoint(MODEL_PTH_NAME + "_epoch" + str(epoch) + "_" + str(round(float(avg_loss), 3)), model=net)

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


def main():
    train()
    # cProfile("train()")
    # print(logdir())


if __name__ == "__main__":
    main()
