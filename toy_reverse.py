import os, sys
import numpy as np
import copy
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt

plt.rc("font", size=8)

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.utils
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

from dataloader import ReverseDataset
from models import RNNAutoEncoder
from data_utils import pad_collate

# comment out warnings if you are testing it out
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="RNNAutoEncoder")

parser.add_argument(
    "--save-freq",
    type=int,
    default=5,
    help="every x epochs save weights",
)
parser.add_argument(
    "--batch",
    type=int,
    default=5,
    help="train batch size",
)
parser.add_argument(
    "--vs",
    type=float,
    default=0.05,
    help="val split",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    help="initial lr",
)

args = parser.parse_args()

reverse_dataset = ReverseDataset()

noisy_dataloader = DataLoader(
    noisy_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    collate_fn=pad_collate,
)


mse_loss = nn.MSELoss()


def train_model(model, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for ((_, noisy, noisy_lens), (_, clean, clean_lens)) in tqdm(
            zip(noisy_dataloader, clean_dataloader)
        ):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # forward
            optimizer.zero_grad()

            h_n, c_n = model(noisy, noisy_lens)
            decoded_clean = model.decode(
                r_h_n=h_n, r_c_n=c_n, r_max=int(max(clean_lens))
            )
            # print(num.shape, rnn_output.shape)
            loss = mse_loss(clean, decoded_clean)

            loss.backward()
            optimizer.step()
        print(loss)
        print(clean[0], decoded_clean[0])

        # deep copy the model
        # if epoch % args.save_freq == 0:
        #     print("saving model")
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     weight_name = "model_weights_" + str(epoch) + ".pt"
        #     torch.save(best_model_wts, weight_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNAutoEncoder(input_dim=1).to(device)
# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = None
# if args.schedule:
#     scheduler = StepLR(optimizer, step_size=5000, gamma=0.1)
train_model(model, optimizer, scheduler, num_epochs=300)
