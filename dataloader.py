import random
import glob
import os
import math
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset


class NoisySineDataset(Dataset):
    def __init__(
        self,
        size: int = 1000, 
        max_chunks: int = 5
    ):

        self.size = size 
        self.max_chunks = max_chunks

    def __len__(self):
        return self.size

    def get_noise(self, shape = 20, mean = 0.5):
        return np.random.normal(0, mean, shape)

    def get_sine(self, interval):
        x = np.linspace(interval[0]*np.pi, interval[1]*np.pi, 50)
        return np.sin(x)

    def __getitem__(self, i):

        #each chunk is either sin or noise
        num_chunks = random.randint(2,self.max_chunks)
        chunk_pos = [0 for i in range(num_chunks)]
        chunk_pos[random.sample(list(range(num_chunks)),1)[0]] = 1

        intervals = [[i*8, (i+1)* 8] for i in range(num_chunks)]
        
        x = []
        y = []

        for i in range(num_chunks):
            #if noise 
            if chunk_pos[i] == 0:
                x.append(np.linspace(intervals[i][0]*np.pi, intervals[i][1]*np.pi, 20))
                y.append(self.get_noise(shape=20))
            else:
                x.append(np.linspace(intervals[i][0]*np.pi, intervals[i][1]*np.pi, 50))
                y.append(self.get_sine(intervals[i]))

        np_x = np.concatenate(x)
        np_y = np.concatenate(y)

        return np_x, np_y

class CleanSineDataset(Dataset):
    def __init__(
        self,
        size: int = 1000
    ):
        self.size = size

    def __len__(self):
        return self.size

    def get_sine(self, interval):
        x = np.linspace(interval[0]*np.pi, interval[1]*np.pi, 50)
        return np.sin(x)

    def __getitem__(self, i):
        num_chunks = random.randint(2,10)
        chunk_pos = [0 for i in range(num_chunks)]
        chunk_pos[random.sample(list(range(num_chunks)),1)[0]] = 1

        intervals = [[i*8, (i+1)* 8] for i in range(num_chunks)]
        
        x = []
        y = []

        for i in range(num_chunks):
            if chunk_pos[i] == 1:
                x.append(np.linspace(intervals[i][0]*np.pi, intervals[i][1]*np.pi, 50))
                y.append(self.get_sine(intervals[i]))

        np_x = np.concatenate(x)
        np_y = np.concatenate(y)

        return np_x, np_y


if __name__ == "__main__":
    k = NoisySineDataset()
    for i in range(10):
        next(iter(k))
