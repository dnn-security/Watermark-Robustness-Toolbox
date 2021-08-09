import contextlib

import mlconfig
import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, Compose
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


class OmniglotShuffledSubset(data.Dataset):
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


class OmniglotStolenDataset(data.Dataset):
    def __init__(self,
                 source_model: PyTorchClassifier,
                 predict_dataset: data.Dataset,
                 augmented_dataset: data.Dataset,
                 batch_size: int,
                 num_workers: int,
                 **kwargs):
        """ Replaces the labels from the given dataset with those predicted from the source model.
        """
        super().__init__(**kwargs)
        self.predict_dataset = predict_dataset
        self.augmented_dataset = augmented_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.__replace_labels_with_source(source_model)

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, i):
        return self.augmented_dataset[i][0], self.targets[i]

    def __replace_labels_with_source(self, source_model):
        """ Predicts all labels using the source model of this dataset. """
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
                batch_y = torch.from_numpy(source_model.predict(x, batch_size=batch_size)).softmax(dim=1)
                self.targets[batch_id * batch_size:batch_id * batch_size + batch_y.shape[0]] = batch_y
                if (batch_id < 50) or batch_id % 100 == 99:  # Compute accuracy every 100 batches.
                    preds = np.argmax(batch_y.numpy(), axis=1)
                    accs.append(len(np.where(preds == y.numpy())[0]) / preds.shape[0])
                    pbar.set_description(f"Stolen Labels ({100 * np.mean(accs):.4f}% Accuracy)")


def _normalize(apply_normalization):
    if apply_normalization:
        mean = np.array([0.13066, 0.13066, 0.13066]).reshape((1, 3, 1, 1))
        std = np.array([0.30131, 0.30131, 0.30131]).reshape((1, 3, 1, 1))
        return mean, std
    return np.array([0, 0, 0]).reshape((1, 3, 1, 1)), np.array([1, 1, 1]).reshape((1, 3, 1, 1))


def _augment(apply_augmentation: bool, image_size: int, normalize: Normalize) -> Compose:
    if apply_augmentation:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])


@mlconfig.register
class OmniglotDataLoader(WRTDataLoader):
    def __init__(self, root: str, image_size: int, batch_size: int, shuffle: bool = True,
                 apply_augmentation=True, subset="all", n_max=np.inf, num_workers=2,
                 apply_normalization=True, source_model=None, class_label=None, **kwargs):

        self.mean, self.std = _normalize(apply_normalization)
        normalize = transforms.Normalize(mean=self.mean.squeeze(), std=self.std.squeeze())

        if source_model is not None:
            # Predict stolen labels for the training data without augmentation.
            transform: Compose = _augment(apply_augmentation=False,
                                          image_size=image_size,
                                          normalize=normalize)
            predict_dataset = datasets.Omniglot(root, transform=transform, download=True)
            predict_dataset = OmniglotShuffledSubset(dataset=predict_dataset, mode=subset, n_max=n_max)

            augmented_dataset = predict_dataset
            if apply_augmentation:
                # Load images with augmentation during training (if desired)
                transform: Compose = _augment(apply_augmentation=apply_augmentation,
                                              image_size=image_size,
                                              normalize=normalize)
                augmented_dataset = datasets.Omniglot(root, transform=transform, download=True)
                augmented_dataset = OmniglotShuffledSubset(dataset=augmented_dataset, mode=subset, n_max=n_max)

            dataset = OmniglotStolenDataset(source_model=source_model,
                                            predict_dataset=predict_dataset,
                                            augmented_dataset=augmented_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers)
        else:
            # No stolen labels.
            transform: Compose = _augment(apply_augmentation, image_size, normalize)
            dataset = datasets.Omniglot(root, transform=transform, download=True)
            dataset = OmniglotShuffledSubset(dataset=dataset, mode=subset, n_max=n_max)

        super(OmniglotDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, mean=self.mean, std=self.std, **kwargs)
