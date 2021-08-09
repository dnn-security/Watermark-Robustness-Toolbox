import os

import GPUtil as GPUtil
import numpy as np
from torch.utils.data.dataset import Dataset


class LimitDataset(Dataset):
    """ Class used for debugging attacks and defenses. Creates a mockup dataset from
    a subset of the given dataset.
    """

    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = min(len(dataset), n)

        # Randomize indices to load from all classes.
        self.idx = np.arange(len(dataset))
        np.random.shuffle(self.idx)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[self.idx[i]]


def get_max_index(data_dir, suffix):
    """ Lists all files from a folder and checks the largest integer prefix for a filename (in snake_case) that
    contains the suffix
    """
    index = 0
    for filename in os.listdir(data_dir):
        if suffix in filename:
            index = int(filename.split("_")[0]) + 1 if int(filename.split("_")[0]) >= index else index
    return str(index)


def pick_gpu():
    """
    Picks a GPU with the least memory load.
    :return:
    """
    try:
        gpu = GPUtil.getFirstAvailable(order='memory', maxLoad=2, maxMemory=0.8, includeNan=False,
                                       excludeID=[], excludeUUID=[])[0]
        return gpu
    except Exception as e:
        print(e)
        return "0"


def reserve_gpu(mode_or_id):
    """ Chooses a GPU.
    If None, uses the GPU with the least memory load.
    """
    if mode_or_id:
        gpu_id = mode_or_id
        os.environ["CUDA_VISIBLE_DEVICES"] = mode_or_id
    else:
        gpu_id = str(pick_gpu())
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Selecting GPU id {gpu_id}")
