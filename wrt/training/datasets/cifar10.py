import contextlib
import os

import mlconfig
import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import Normalize, Compose
from tqdm import tqdm
from PIL import Image

from wrt.classifiers import PyTorchClassifier
from wrt.training.datasets.wrt_data_loader import WRTDataLoader

cifar_classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class CIFAR10ShuffledSubset(data.Dataset):
    """ Wrapper for the defender's or attacker's subsets for the whole CIFAR-10 dataset. """

    def __init__(self,
                 dataset: data.Dataset,
                 mode: str = "all",
                 n_max: int = np.inf,
                 seed=1337):
        """ Shuffles a dataset with a random permutation (given a seed).
        :param mode 'attacker', 'defender', 'debug' have special meaning.
        :param n_max Maximum number of elements to load.
        """
        self.dataset = dataset
        self.idx = np.arange(len(dataset))

        with temp_seed(seed):
            np.random.shuffle(self.idx)

        if mode == "attacker":
            self.idx = self.idx[:min(n_max, len(dataset) // 3)]
        elif mode == "defender":
            self.idx = self.idx[min(n_max, len(dataset) // 3):]
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


class CIFAR10AveragingStolenDataset(data.Dataset):
    def __init__(self,
                 source_model: PyTorchClassifier,
                 predict_dataset: data.Dataset,
                 augmented_dataset: data.Dataset,
                 batch_size: int,
                 num_workers: int,
                 query_labels_n_times: int,
                 transform: Compose = None,
                 top_k=None,
                 **kwargs):
        """ Replaces the labels from the given dataset with those predicted from the source model.
        """
        super().__init__(**kwargs)
        self.predict_dataset = predict_dataset
        self.augmented_dataset = augmented_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        mean, std = _normalize(True)
        normalize = transforms.Normalize(mean=mean.squeeze(), std=std.squeeze())
        if transform is None:
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

        self.__replace_labels_with_source(source_model, query_labels_n_times, transform)
        if top_k is not None:
            self.targets = torch.eye(source_model.nb_classes())[self.targets.argmax(1)]

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, i):
        return self.augmented_dataset[i][0], self.targets[i]

    def __replace_labels_with_source(self, source_model, query_n_times, transform):
        """ Predicts all labels using the source model of this dataset.
        (Showcased to eliminate dawn)
        :param source_model
        """
        data_loader = data.DataLoader(dataset=self.predict_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False)

        self.targets = torch.empty((len(self), source_model.nb_classes()))
        batch_size = data_loader.batch_size
        with torch.no_grad(), tqdm(data_loader, desc="Predict Stolen Labels") as pbar:
            accs = []
            counter = 0
            for batch_id, (x_batch, y) in enumerate(pbar):
                x_batch = x_batch.cpu().numpy()
                x_batch = np.uint8(x_batch * 255).transpose((0, 2, 3, 1))
                x_batch = [Image.fromarray(x) for x in x_batch]

                for j, x in enumerate(x_batch):
                    batch_y = 0
                    for i in range(query_n_times):
                        xt = torch.unsqueeze(transform(x), 0).to("cuda")
                        batch_y = (1 / query_n_times) * torch.from_numpy(
                            source_model.predict(xt, batch_size=batch_size)).softmax(dim=1) + batch_y
                    counter += 1
                    self.targets[batch_id * batch_size + j] = batch_y

                if (batch_id < 50) or batch_id % 100 == 99:  # Compute accuracy every 100 batches.
                    preds = np.argmax(self.targets[counter-len(x_batch):counter].numpy(), axis=1)
                    accs.append(len(np.where(preds == y.numpy())[0]) / preds.shape[0])
                    pbar.set_description(f"Stolen Labels ({100 * np.mean(accs):.4f}% Accuracy)")



class CIFAR10StolenDataset(data.Dataset):
    def __init__(self,
                 source_model: PyTorchClassifier,
                 predict_dataset: data.Dataset,
                 augmented_dataset: data.Dataset,
                 batch_size: int,
                 num_workers: int,
                 top_k=None,
                 apply_softmax=True,
                 **kwargs):
        """ Replaces the labels from the given dataset with those predicted from the source model.
        """
        super().__init__(**kwargs)
        self.predict_dataset = predict_dataset
        self.augmented_dataset = augmented_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.__replace_labels_with_source(source_model, apply_softmax=apply_softmax)
        if top_k is not None:
            self.targets = torch.eye(source_model.nb_classes())[self.targets.argmax(1)]

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, i):
        return self.augmented_dataset[i][0], self.targets[i]

    def __replace_labels_with_source(self, source_model, apply_softmax=True):
        """ Predicts all labels using the source model of this dataset.
        :param source_model
        """
        data_loader = data.DataLoader(dataset=self.predict_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False)
        self.targets = torch.empty((len(self), source_model.nb_classes()))
        batch_size = data_loader.batch_size
        with torch.no_grad(), tqdm(data_loader, desc="Predict Stolen Labels") as pbar:
            accs = []
            for batch_id, (batch_x, y) in enumerate(pbar):
                x = batch_x.cuda()
                batch_y = torch.from_numpy(source_model.predict(x, batch_size=batch_size))
                if apply_softmax:
                    batch_y = batch_y.softmax(dim=1)
                self.targets[batch_id * batch_size:batch_id * batch_size + batch_y.shape[0]] = batch_y
                if (batch_id < 50) or batch_id % 100 == 99:  # Compute accuracy every 100 batches.
                    preds = np.argmax(batch_y.numpy(), axis=1)
                    accs.append(len(np.where(preds == y.numpy())[0]) / preds.shape[0])
                    pbar.set_description(f"Stolen Labels ({100 * np.mean(accs):.4f}% Accuracy)")


def _normalize(apply_normalization):
    if apply_normalization:
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))
        std = np.array([0.247, 0.243, 0.261]).reshape((1, 3, 1, 1))
        return mean, std
    return np.array([0, 0, 0]).reshape((1, 3, 1, 1)), np.array([1, 1, 1]).reshape((1, 3, 1, 1))


def _augment(apply_augmentation: bool, train: bool, image_size: int, normalize: Normalize) -> Compose:
    if apply_augmentation:
        if train:
            return transforms.Compose([
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                # transforms.Resize(int(image_size * 256 / 224)),
                # transforms.CenterCrop(image_size),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
            ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])


@mlconfig.register
class CIFAR10DataLoader(WRTDataLoader):
    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool = True,
                 apply_augmentation=True, subset="all", n_test=np.inf, n_train=np.inf, num_workers=2,
                 apply_normalization=True, apply_softmax=True, source_model=None, class_labels=None, download: bool = True,
                 query_labels_n_times: int = 1, top_k=None, **kwargs):

        self.mean, self.std = _normalize(apply_normalization and query_labels_n_times <= 1)
        normalize = transforms.Normalize(mean=self.mean.squeeze(), std=self.std.squeeze())

        if (source_model is not None) and train:
            # Predict stolen labels for the training data without augmentation.
            transform: Compose = _augment(apply_augmentation=False,
                                          train=train,
                                          image_size=image_size,
                                          normalize=normalize)
            predict_dataset = datasets.CIFAR10(root, train=train, transform=transform, download=download)
            predict_dataset = CIFAR10ShuffledSubset(dataset=predict_dataset, mode=subset, n_max=n_train)

            augmented_dataset = predict_dataset
            if apply_augmentation:
                # Load images with augmentation during training (if desired)
                transform: Compose = _augment(apply_augmentation=apply_augmentation,
                                              train=train,
                                              image_size=image_size,
                                              normalize=normalize)
                augmented_dataset = datasets.CIFAR10(root, train=train, transform=transform, download=download)
                augmented_dataset = CIFAR10ShuffledSubset(dataset=augmented_dataset, mode=subset, n_max=n_train)

            if query_labels_n_times > 1:
                dataset = CIFAR10AveragingStolenDataset(source_model=source_model,
                                                        predict_dataset=predict_dataset,
                                                        augmented_dataset=augmented_dataset,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        top_k=top_k,
                                                        query_labels_n_times=query_labels_n_times)
            else:
                dataset = CIFAR10StolenDataset(source_model=source_model,
                                               predict_dataset=predict_dataset,
                                               augmented_dataset=augmented_dataset,
                                               batch_size=batch_size,
                                               top_k=top_k,
                                               apply_softmax=apply_softmax,
                                               num_workers=num_workers)
        else:
            # No stolen labels.
            transform: Compose = _augment(apply_augmentation, train, image_size, normalize)
            dataset = datasets.CIFAR10(root, train=train, transform=transform, download=download)
            if train:
                dataset = CIFAR10ShuffledSubset(dataset=dataset, mode=subset, n_max=n_train)
            else:
                dataset = CIFAR10ShuffledSubset(dataset=dataset, mode="all", n_max=n_test)

        super(CIFAR10DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=num_workers, mean=self.mean, std=self.std, **kwargs)


class CIFAR10Extended(datasets.ImageFolder):
    url = "https://www.dropbox.com/s/aom04ukmra5bcrk/cifar_extended.zip?dl=1"
    filename = "cifar_extended.zip"
    folder_name = "cifar_extended"

    def __init__(self,
                 root: str,
                 transform=None,
                 download=True):
        self.root = os.path.join(root, self.folder_name)

        if download:
            self.download()
        super().__init__(self.root, transform)

    def _check_integrity(self):
        return os.path.isdir(self.root)

    def download(self):
        if self._check_integrity():
            print("CIFAR10 Extended already downloaded.")
            return
        os.makedirs(self.root, exist_ok=True)
        download_and_extract_archive(self.url, self.root, filename=self.filename, remove_finished=True)


@mlconfig.register
class CIFAR10ExtendedDataLoader(WRTDataLoader):
    """ Loads the extended CIFAR-10 image dataset that contains 3k additional images per class.
    This data is sampled and downsized from Flickr and there is no overlap with CIFAR-10 data.
    """

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool = True,
                 apply_augmentation=True, subset="all", n_test=np.inf, n_train=np.inf, num_workers=2,
                 apply_normalization=True, source_model=None, download: bool = True, **kwargs):

        self.mean, self.std = _normalize(apply_normalization)
        normalize = transforms.Normalize(mean=self.mean.squeeze(), std=self.std.squeeze())

        if source_model is not None and train:
            # Predict stolen labels for the training data without augmentation.
            transform: Compose = _augment(apply_augmentation=False,
                                          train=train,
                                          image_size=image_size,
                                          normalize=normalize)
            predict_dataset = CIFAR10Extended(root, transform=transform)
            predict_dataset = CIFAR10ShuffledSubset(dataset=predict_dataset, mode=subset, n_max=n_train)

            augmented_dataset = predict_dataset
            if apply_augmentation:
                # Load images with augmentation during training (if desired)
                transform: Compose = _augment(apply_augmentation=apply_augmentation,
                                              train=train,
                                              image_size=image_size,
                                              normalize=normalize)
                augmented_dataset = CIFAR10Extended(root, transform=transform)
                augmented_dataset = CIFAR10ShuffledSubset(dataset=augmented_dataset, mode=subset, n_max=n_train)

            dataset = CIFAR10StolenDataset(source_model=source_model,
                                           predict_dataset=predict_dataset,
                                           augmented_dataset=augmented_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
        else:
            # No stolen labels.
            transform: Compose = _augment(apply_augmentation, train, image_size, normalize)
            if train:
                dataset = CIFAR10Extended(root, transform=transform)
                dataset = CIFAR10ShuffledSubset(dataset=dataset, mode=subset, n_max=n_train)
            else:
                dataset = datasets.CIFAR10(root, train=train, transform=transform, download=download)
                dataset = CIFAR10ShuffledSubset(dataset=dataset, mode="all", n_max=n_test)

        super(CIFAR10ExtendedDataLoader, self).__init__(dataset=dataset,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        mean=self.mean,
                                                        std=self.std,
                                                        num_workers=num_workers,
                                                        **kwargs)
