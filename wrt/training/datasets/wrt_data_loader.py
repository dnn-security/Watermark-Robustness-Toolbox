import abc
from copy import deepcopy
from typing import List

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms


class WRTMixedNumpyDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, transform, x: np.ndarray, y: np.ndarray, boost_factor: float = 1.0):
        """ Creates a new dataset concatenated with numpy arrays
        :param dataset The dataset to concatenate all values with.
        :param x The input images.
        :param y The labels
        :param boost_factor Number of times to repeat the numpy arrays
        :param transform Transformation applied to the numpy data.
        """
        self.dataset = dataset
        self.x_data = x
        self.y_data = y.astype(np.float32)
        self.breakpoint = len(dataset)
        self.idx = np.arange(len(dataset) + int(boost_factor * x.shape[0]))
        self.transform = deepcopy(transform)

        if self.transform is not None:
            self.transform.transforms.insert(0, transforms.ToPILImage())

        # Make the numpy data the same shape as the remaining training data.
        if type(self.dataset[0][1]) is torch.Tensor:
            if np.isscalar(self.y_data[0]):
                self.y_data = torch.eye(self.dataset[0][1].shape[0])[self.y_data]
            else:
                self.y_data = torch.from_numpy(self.y_data)
        else:
            if not np.isscalar(self.y_data[0]):
                self.y_data = self.y_data.argmax(1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if i < self.breakpoint:
            x, y = self.dataset[i]
            if type(y) == torch.Tensor:
                y = y.float()
        else:
            x = self.x_data[(i % self.breakpoint) % self.x_data.shape[0]].astype(np.float32)
            x = self.transform(np.uint8(x*255).transpose(1, 2, 0))
            y = self.y_data[(i % self.breakpoint) % self.y_data.shape[0]]
        return x, y


class WRTSplitDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, split_id, total_splits, shuffle=True):
        """ Creates a new dataset split into non-overlapping subsets.
        """
        self.dataset = dataset
        split_size = (len(dataset) // total_splits)
        start, end = split_id * split_size, (split_id+1)*split_size
        self.idx = np.arange(start, end)
        if shuffle:
            np.random.shuffle(self.idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.dataset[self.idx[i]]


class WRTDataLoader(data.DataLoader):
    def __init__(self, dataset, mean, std, batch_size, transform=None, **kwargs):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.transform = transform

        # Always normalize on default.
        if self.transform is None:
            print("[INFO] No transformation given .. adding normalization. ")
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean.squeeze(), std=self.std.squeeze())
            ])

        super().__init__(dataset, batch_size=batch_size, **kwargs)

    def normalize(self, x: np.ndarray):
        """ Takes a numpy array x and normalizes it
        """
        x_norm = (x-self.mean)/self.std
        return x_norm.astype(np.float32)

    def unnormalize(self, x: np.ndarray):
        """ Takes a numpy array x and unnormalizes it
        """
        return (x * self.std) + self.mean

    def split(self, splits: int) -> List:
        """ Splits a dataset into non-overlapping subsets
        """
        datasets = [WRTSplitDataset(self.dataset, split_id=i, total_splits=splits) for i in range(splits)]
        loaders = [WRTDataLoader(dataset, self.mean, self.std, batch_size=self.batch_size, **self.kwargs) for dataset in datasets]
        return loaders

    def add_numpy_data(self, x: np.ndarray, y: np.ndarray, boost_factor: float = 1.0):
        """ Creates a new data loader that is same as the old one but also indexes x and y data.
        The numpy data is expected to be unnormalized in the range [0, 1].

        :param x: Image data to append.
        :param y Label to append (preferably in the same output dim as the training loader)
        :param boost_factor Number of times to repeat x and y. If this is not an integer, some elements
        will be repeated more often than others.
        """
        if np.min(x) < 0 or np.max(x) > 1:
            print("[WARNING] Added numpy data should be normalized into the range [0, 1], but it seems like it is not.")

        mixed_dataset = WRTMixedNumpyDataset(self.dataset, self.transform, x, y, boost_factor)
        return WRTDataLoader(mixed_dataset, self.mean, self.std, batch_size=self.batch_size, transform=self.transform,
                             **self.kwargs)

    def merge_with_wrt_loader(self, data_loader: data.DataLoader):
        """ Merges this data loader with another one.
        """
        pass


