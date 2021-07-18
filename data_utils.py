import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence


# def pad_collate_clean_sin(batch):
    # """
    # collate function that pads with zeros for variable lenght data-points.
    # pass into the dataloader object.
    # """
    # # nums = zip(*batch)
    # nums = batch
    # lens = [x.shape[0] for x in nums]
    # nums = pad_sequence(nums, batch_first=True, padding_value=0)

    # return nums, torch.tensor(lens, dtype=torch.float)


def pad_collate(batch):
    """
    collate function that pads with zeros for variable lenght data-points.
    pass into the dataloader object.
    """
    np_x, noisy_sin = zip(*batch)
    # print(np_x)
    lens = [x.shape[0] for x in np_x]
    np_x = pad_sequence(np_x, batch_first=True, padding_value=0)
    noisy_sin = pad_sequence(noisy_sin, batch_first=True, padding_value=0)

    return np_x, noisy_sin, torch.tensor(lens, dtype=torch.float)




# def pad_collate_1hot(batch):
#     """
#     collate function that pads with zeros for variable lenght data-points.
#     pass into the dataloader object.
#     """
#     s_ns, letters, users = zip(*batch)

#     s_n_lens = [x.shape[0] for x in s_ns]
#     l_lens = [x.shape[0] for x in letters]

#     s_ns = pad_sequence(s_ns, batch_first=True, padding_value=0)
#     letters = pad_sequence(letters, batch_first=True, padding_value=0)
#     users = pad_sequence(users, batch_first=True, padding_value=0)

#     to_torch = lambda x: torch.tensor(x, dtype=torch.float)

#     return s_ns, letters, users, to_torch(s_n_lens), to_torch(l_lens)
