import os, sys
import numpy as np
import random
import copy
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
plt.rc("font", size=8)

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloader import ReverseDataset
from models import RNNDecoderNoAttn, RNNEncoder, RNNDecoder, Seq2SeqAttn
from data_utils import pad_collate

# comment out warnings if you are testing it out
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="RNNAutoEncoder")

parser.add_argument(
    "--save-freq",
    type=int,
    default=50,
    help="every x epochs save weights",
)
parser.add_argument(
    "--batch",
    type=int,
    default=20,
    help="train batch size",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="initial lr",
)
parser.add_argument(
    "--exp-name",
    default="testing_nn_hq",
    help="Experiment name",
)
args = parser.parse_args()


import wandb

os.environ["WANDB_NAME"] = args.exp_name
wandb.init(project="sub_seq")
wandb.config.update(args)
wandb.config.update({"dataset": "reverse_dataset"})


#fix random seeds 
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


reverse_dataset = ReverseDataset()

rev_dataloader = DataLoader(
        reverse_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=pad_collate,
    )

round_3 = lambda x: (x * 10**3).round() / (10**3)


def calc_loss(
    prediction,
    target,
    metrics,
):
    mse_loss = F.mse_loss(prediction, target)
    mae_loss = F.l1_loss(prediction, target)
    metrics["MSE"] += mse_loss.data.cpu().numpy() * target.size(0)
    metrics["MAE"] += mae_loss.data.cpu().numpy() * target.size(0)

    return mse_loss


def print_metrics(metrics, epoch_samples,  epoch):
    outputs = []
    outputs.append("{}:".format(str(epoch)))
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        wandb.log({k: metrics[k] / epoch_samples})
    print("{}".format( ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        metrics = defaultdict(float)
        epoch_samples = 0
        for normal, rev, lens in tqdm(rev_dataloader):
            normal = normal.to(device)
            rev = rev.to(device)

            optimizer.zero_grad()

            decoded_rev, all_attn = model(normal, lens)
            
            loss = calc_loss(rev, decoded_rev, metrics)
            loss.backward()
            optimizer.step()
        
            epoch_samples += normal.size(0)
        
        print_metrics(metrics, epoch_samples, epoch)
        print(round_3(rev[0]), round_3(decoded_rev[0]))
        

        # deep copy the model
        if epoch % args.save_freq == 0:
            print("saving model")
            best_model_wts = copy.deepcopy(model.state_dict())
            weight_name = "seq2seq_all_attn_weights_" + str(epoch) + ".pt"
            torch.save(best_model_wts, weight_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
e = RNNEncoder(input_dim=1)
d= RNNDecoder(input_dim=(e.input_size + e.hidden_size), hidden_size= e.hidden_size)
# d= RNNDecoderNoAttn(input_dim=1, hidden_size= e.hidden_size)

model = Seq2SeqAttn(encoder=e, decoder=d).to(device)
# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = None
# if args.schedule:
#     scheduler = StepLR(optimizer, step_size=5000, gamma=0.1)
train_model(model, optimizer, scheduler, num_epochs=151)
