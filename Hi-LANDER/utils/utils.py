from torch.utils.data import random_split
import torch
import math

def split_batch(x, lbs, batch_size, seed=100):
    fea_list = []
    lbs_list = []
    n_obs = x.shape[0]

    split_list = [batch_size] * math.floor(n_obs / batch_size)
    if n_obs % batch_size != 0:
        split_list.append(n_obs % batch_size)

    batch_idx = random_split(range(n_obs),
                             split_list,
                             generator=torch.Generator().manual_seed(seed))
    for i in range(len(batch_idx)):
        batch_idx_i = batch_idx[i].indices
        fea_batch = x[batch_idx_i, :]
        lbs_batch = lbs[batch_idx_i]
        fea_list.append(fea_batch)
        lbs_list.append(lbs_batch)

    return fea_list, lbs_list