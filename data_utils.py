from typing import List, Dict
from collections import defaultdict, Counter
from statistics import mean, stdev
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

"""
Includes code for preprocessing for torch
"""


def pad_collate_dummy_1(batch):
    """
    collate function that pads with zeros for variable lenght data-points.
    pass into the dataloader object.
    """
    # nums = zip(*batch)
    nums = batch
    lens = [x.shape[0] for x in nums]
    nums = pad_sequence(nums, batch_first=True, padding_value=0)

    return nums, torch.tensor(lens, dtype=torch.float)


def pad_collate_dummy_2(batch):
    """
    collate function that pads with zeros for variable lenght data-points.
    pass into the dataloader object.
    """
    nums, rev = zip(*batch)

    lens = [x.shape[0] for x in nums]
    nums = pad_sequence(nums, batch_first=True, padding_value=0)
    rev = pad_sequence(rev, batch_first=True, padding_value=0)

    return nums, rev, torch.tensor(lens, dtype=torch.float)


def pad_collate_1hot(batch):
    """
    collate function that pads with zeros for variable lenght data-points.
    pass into the dataloader object.
    """
    s_ns, letters, users = zip(*batch)

    s_n_lens = [x.shape[0] for x in s_ns]
    l_lens = [x.shape[0] for x in letters]

    s_ns = pad_sequence(s_ns, batch_first=True, padding_value=0)
    letters = pad_sequence(letters, batch_first=True, padding_value=0)
    users = pad_sequence(users, batch_first=True, padding_value=0)

    to_torch = lambda x: torch.tensor(x, dtype=torch.float)

    return s_ns, letters, users, to_torch(s_n_lens), to_torch(l_lens)
