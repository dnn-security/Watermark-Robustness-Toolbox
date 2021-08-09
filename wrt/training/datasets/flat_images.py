import os
import warnings

import mlconfig
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Normalize
from tqdm import tqdm

from wrt.classifiers import PyTorchClassifier
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.datasets.imagenet import ImageNetShuffledSubset, ImageNetStolenDataset


class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.items = os.listdir(self.root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.items[idx]
        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def _normalize(apply_normalization):
    if apply_normalization:
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        return mean, std
    return np.array([0, 0, 0]).reshape((1, 3, 1, 1)), np.array([1, 1, 1]).reshape((1, 3, 1, 1))


def _augment(apply_augmentation: bool, image_size: int, normalize: Normalize) -> Compose:
    if apply_augmentation:
        return transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


@mlconfig.register
class FlatImagesDataLoader(WRTDataLoader):
    def __init__(self, root: str, image_size: int, batch_size: int, shuffle: bool = True,
                 apply_augmentation=True, subset="all", n_train=np.inf, num_workers=16,
                 apply_normalization=True, source_model=None, class_labels=None, **kwargs):
        root = os.path.expanduser(root)
        print(f"Loading Flat Images at: {root}")
        self.mean, self.std = _normalize(apply_normalization)
        normalize = transforms.Normalize(mean=self.mean.squeeze(), std=self.std.squeeze())
        warnings.filterwarnings('ignore', category=UserWarning)

        print("Flat images Data loader!")
        if source_model is not None:
            # Predict stolen labels for the training data without augmentation, but always with normalization.
            mean, std = _normalize(True)
            predict_normalize = transforms.Normalize(mean=mean.squeeze(), std=std.squeeze())
            transform: Compose = _augment(False, image_size, predict_normalize)
            predict_dataset = FlatImageDataset(root=root, transform=transform)
            predict_dataset = ImageNetShuffledSubset(dataset=predict_dataset, mode=subset, n_max=n_train)

            transform: Compose = _augment(apply_augmentation, image_size, normalize)
            augmented_dataset = FlatImageDataset(root=root, transform=transform)
            augmented_dataset = ImageNetShuffledSubset(dataset=augmented_dataset, mode=subset, n_max=n_train)

            dataset = ImageNetStolenDataset(source_model=source_model,
                                            predict_dataset=predict_dataset,
                                            augmented_dataset=augmented_dataset,
                                            num_workers=num_workers,
                                            batch_size=batch_size)
        else:
            # No stolen labels.
            transform: Compose = _augment(apply_augmentation, image_size, normalize)
            dataset = FlatImageDataset(root=root, transform=transform)
            dataset = ImageNetShuffledSubset(dataset=dataset, mode=subset, n_max=n_train)

        super(FlatImagesDataLoader, self).__init__(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers,
                                                   mean=self.mean,
                                                   std=self.std,
                                                   **kwargs)
