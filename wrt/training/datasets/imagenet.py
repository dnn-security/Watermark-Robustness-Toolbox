import contextlib
import os
import warnings
from typing import List, Union, Tuple

import mlconfig
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Normalize
from tqdm import tqdm

from wrt.classifiers import PyTorchClassifier
from wrt.training.datasets.wrt_data_loader import WRTDataLoader


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class ImageNetSingleClassShuffledSubset(datasets.ImageFolder):

    def __init__(self, root: str, class_labels: Union[int, List[int]], transform=None,
                 mode: str = "all", n_max: int = np.inf, seed=1337):
        """ Loads only images from one class (the given class_label)
        Note: This breaks up the order of the dataset (!)
        """
        super(ImageNetSingleClassShuffledSubset, self).__init__(root, transform=transform)

        if isinstance(class_labels, int):
            class_labels = [class_labels]

        dataset: List[Tuple[str, int]] = self.imgs

        # First apply the subset rule.
        idx = np.arange(len(dataset))

        with temp_seed(seed):
            np.random.shuffle(idx)

        if mode == "attacker":
            idx = idx[:min(n_max, len(dataset) // 3)]
        elif mode == "defender":
            idx = idx[len(dataset) // 3:min(len(idx), len(dataset) // 3 + n_max)]
        elif mode == "debug":
            idx = idx[:min(n_max, 10000)]
        else:
            if n_max == np.inf:
                n_max = -1
            idx = idx[:n_max]

        # Now filter only the allowed class labels.
        filtered_imgs = []
        for i in idx:
            img_path, label = self.imgs[i]
            if label in class_labels:
                filtered_imgs.append((img_path, label))

        # Set this loader's classes to the given value.
        self.imgs = filtered_imgs
        self.samples = filtered_imgs
        self.targets = [s[1] for s in self.samples]
        print(f"filtered {len(self.samples)} images")


class ImageNetSingleClassDataset(data.Dataset):
    def __init__(self, root: str, class_labels: List[int], transform=None):
        """ Loads only images from one class (the given class_label)
        Note: This breaks up the order of the dataset (!)
        """
        if isinstance(class_labels, int):
            class_labels = [class_labels]

        self.class_labels = class_labels
        self.transform = transform
        self.loader = default_loader

        root = os.path.expanduser(root)
        folders = [d.name for d in os.scandir(root) if d.is_dir()]
        folders.sort()
        target_folders = [folders[c] for c in class_labels]

        self.filepaths, self.class_labels = [], []
        for class_label, target_folder in zip(class_labels, target_folders):
            class_filepaths = [os.path.join(root, target_folder, x) for x in
                               os.listdir(os.path.join(root, target_folder))]
            self.filepaths.extend(class_filepaths)
            self.class_labels.extend([class_label] * len(class_filepaths))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        x = self.loader(self.filepaths[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x, self.class_labels[idx]


class ImageNetShuffledSubset(data.Dataset):
    """ Wrapper for the defender's or attacker's subsets for the whole ImageNet dataset. """

    def __init__(self,
                 dataset: data.Dataset,
                 mode: str = "all",
                 n_max: int = np.inf,
                 seed=1337):
        """ Loads a shuffled imagenet dataset. """
        self.dataset = dataset
        self.idx = np.arange(len(dataset))

        with temp_seed(seed):
            np.random.shuffle(self.idx)

        if mode == "attacker":
            self.idx = self.idx[:min(n_max, len(dataset) // 3)]
        elif mode == "defender":
            self.idx = self.idx[len(dataset) // 3:min(len(self.idx), len(dataset) // 3 + n_max)]
        elif mode == "debug":
            self.idx = self.idx[:min(n_max, 10000)]
        else:
            if n_max == np.inf:
                n_max = -1
            self.idx = self.idx[:n_max]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.dataset[self.idx[i]]


class ImageNetAveragingStolenDataset(data.Dataset):
    def __init__(self,
                 source_model: PyTorchClassifier,
                 predict_dataset: data.Dataset,
                 augmented_dataset: data.Dataset,
                 batch_size: int,
                 num_workers: int,
                 query_labels_n_times: int,
                 top_k: int = None,
                 **kwargs):
        """ Replaces the labels from the given dataset with those predicted from the source model.
        """
        super().__init__(**kwargs)
        self.predict_dataset = predict_dataset
        self.augmented_dataset = augmented_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.__replace_labels_with_source(source_model, query_labels_n_times)
        self.top_k = top_k
        if self.top_k is not None:
            self.targets = torch.eye(source_model.nb_classes())[self.targets.argmax(1)]

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, i):
        return self.augmented_dataset[i][0], self.targets[i]

    def __replace_labels_with_source(self, source_model, query_n_times):
        """ Predicts all labels using the source model of this dataset.
        (Showcased to eliminate dawn)
        :param source_model
        """
        data_loader = data.DataLoader(dataset=self.predict_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False)

        self.targets = np.zeros((len(self), source_model.nb_classes()))
        batch_size = data_loader.batch_size
        for i in range(query_n_times):
            with torch.no_grad(), tqdm(data_loader, desc=f"Predict Stolen Labels Round [{i+1}/{query_n_times}]") as pbar:
                accs = []
                for batch_id, (batch_x, y) in enumerate(pbar):
                    batch_y = source_model.predict(batch_x.cuda())
                    self.targets[batch_id * batch_size:batch_id * batch_size + batch_y.shape[0]] += (1/query_n_times)*batch_y

                    if (batch_id < 50) or batch_id % 100 == 99:  # Compute accuracy every 100th batch.
                        preds = np.argmax(batch_y, axis=1)
                        accs.append(len(np.where(preds == y.numpy())[0]) / preds.shape[0])
                        pbar.set_description(f"Predict Stolen Labels [Round {i+1}/{query_n_times}] ({100 * np.mean(accs):.4f}% Accuracy)")
        self.targets = torch.from_numpy(self.targets)
        self.targets = self.targets.softmax(dim=1)


class ImageNetStolenDataset(data.Dataset):
    def __init__(self,
                 source_model: PyTorchClassifier,
                 predict_dataset: data.Dataset,
                 augmented_dataset: data.Dataset,
                 batch_size: int,
                 num_workers: int,
                 top_k: int = None,
                 apply_softmax=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.predict_dataset = predict_dataset
        self.augmented_dataset = augmented_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.__replace_labels_with_source(source_model, apply_softmax=apply_softmax)
        self.top_k = top_k
        if self.top_k is not None:
            self.targets = torch.eye(source_model.nb_classes())[self.targets.argmax(1)]

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, i):
        return self.augmented_dataset[i][0], self.targets[i]

    def __replace_labels_with_source(self, source_model: PyTorchClassifier, apply_softmax=True):
        """ Predicts all labels using the source model of this dataset. """
        batch_size = self.batch_size  # Prediction batch size can be higher.
        data_loader = data.DataLoader(dataset=self.predict_dataset,
                                      batch_size=batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False)
        self.targets = np.empty((len(self), source_model.nb_classes()))
        with torch.no_grad(), tqdm(data_loader, desc="Predict Stolen Labels") as pbar:
            accs = []
            for batch_id, (batch_x, y) in enumerate(pbar):
                batch_y = source_model.predict(batch_x.cuda())
                self.targets[batch_id * batch_size:batch_id * batch_size + batch_y.shape[0]] = batch_y

                if (batch_id < 50) or batch_id % 100 == 99:  # Compute accuracy every 100th batch.
                    preds = np.argmax(batch_y, axis=1)
                    accs.append(len(np.where(preds == y.numpy())[0]) / preds.shape[0])
                    pbar.set_description(f"Stolen Labels ({100 * np.mean(accs):.4f}% Accuracy)")
        self.targets = torch.from_numpy(self.targets)
        if apply_softmax:
            self.targets = self.targets.softmax(dim=1)


def _normalize(apply_normalization):
    if apply_normalization:
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        return mean, std
    return np.array([0, 0, 0]).reshape((1, 3, 1, 1)), np.array([1, 1, 1]).reshape((1, 3, 1, 1))


def _augment(apply_augmentation: bool, train: bool, image_size: int, normalize: Normalize) -> Compose:
    if apply_augmentation:
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size + 32, interpolation=Image.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


@mlconfig.register
class ImageNetDataLoader(WRTDataLoader):
    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool = True,
                 apply_augmentation=True, subset="all", n_test=np.inf, n_train=np.inf, num_workers=16,
                 apply_normalization=True, apply_softmax=True, class_labels: List[int] = None, source_model=None,
                 query_labels_n_times: int = 1, top_k=None, **kwargs):

        self.mean, self.std = _normalize(apply_normalization)
        normalize = transforms.Normalize(mean=self.mean.squeeze(), std=self.std.squeeze())
        warnings.filterwarnings('ignore', category=UserWarning)

        phase = 'train' if train else 'val'
        if (source_model is not None) and train:
            # Predict stolen labels for the training data without augmentation.
            mean, std = _normalize(True)
            predict_normalize = transforms.Normalize(mean=mean.squeeze(), std=std.squeeze())

            transform: Compose = _augment(apply_augmentation, train, image_size, normalize)
            if class_labels is not None:
                augmented_dataset = ImageNetSingleClassShuffledSubset(os.path.join(root, phase), transform=transform,
                                                            mode=subset, class_labels=class_labels, n_max=n_train)
            else:
                augmented_dataset = datasets.ImageFolder(os.path.join(root, phase), transform=transform)
                augmented_dataset = ImageNetShuffledSubset(dataset=augmented_dataset, mode=subset, n_max=n_train)

            if query_labels_n_times > 1:
                transform = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.RandomResizedCrop(image_size+32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    predict_normalize
                ])
                if class_labels is not None:
                    predict_dataset = ImageNetSingleClassShuffledSubset(os.path.join(root, phase), transform=transform,
                                                            mode=subset, class_labels=class_labels, n_max=n_train)
                else:
                    predict_dataset = datasets.ImageFolder(os.path.join(root, phase), transform=transform)
                    predict_dataset = ImageNetShuffledSubset(dataset=predict_dataset, mode=subset, n_max=n_train)

                dataset = ImageNetAveragingStolenDataset(source_model=source_model,
                                                         predict_dataset=predict_dataset,
                                                         augmented_dataset=augmented_dataset,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         top_k=top_k,
                                                         query_labels_n_times=query_labels_n_times)
            else:
                transform: Compose = _augment(False, train, image_size, predict_normalize)
                if class_labels is not None:
                    predict_dataset = ImageNetSingleClassShuffledSubset(os.path.join(root, phase), transform=transform,
                                                      mode=subset, class_labels=class_labels, n_max=n_train)
                else:
                    predict_dataset = datasets.ImageFolder(os.path.join(root, phase), transform=transform)
                    predict_dataset = ImageNetShuffledSubset(dataset=predict_dataset, mode=subset, n_max=n_train)

                dataset = ImageNetStolenDataset(source_model=source_model,
                                                predict_dataset=predict_dataset,
                                                augmented_dataset=augmented_dataset,
                                                num_workers=num_workers,
                                                top_k=top_k,
                                                apply_softmax=apply_softmax,
                                                batch_size=batch_size)
        else:
            # No stolen labels.
            transform: Compose = _augment(apply_augmentation, train, image_size, normalize)
            dataset = datasets.ImageFolder(os.path.join(root, phase), transform=transform)
            if train:
                dataset = ImageNetShuffledSubset(dataset=dataset, mode=subset, n_max=n_train)
            else:
                dataset = ImageNetShuffledSubset(dataset=dataset, mode="all", n_max=n_test)

        transform = _augment(apply_augmentation, train, image_size, normalize)

        super(ImageNetDataLoader, self).__init__(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 mean=self.mean,
                                                 std=self.std,
                                                 transform=transform,
                                                 **kwargs)
