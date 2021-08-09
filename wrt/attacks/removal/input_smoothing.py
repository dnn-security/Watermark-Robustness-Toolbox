"""
This module implements the White box attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc

import logging
from typing import Callable

import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter, median_filter

from wrt.classifiers import Loss
from wrt.classifiers.classifier import Classifier
from wrt.attacks.attack import RemovalAttack
from wrt.exceptions import ClassifierError
from wrt.preprocessors import Preprocessor
from wrt.training.callbacks import EvaluateWmAccCallback
from wrt.training.datasets import WRTDataLoader
import mlconfig

from wrt.training.trainer import Trainer

logger = logging.getLogger(__name__)


class SmoothingPreprocessor(Preprocessor):

    def __init__(self, normalize_fn: Callable, unnormalize_fn: Callable):
        super().__init__()
        self.normalize_fn = normalize_fn
        self.unnormalize_fn = unnormalize_fn

    @property
    def apply_fit(self):
        return False

    @property
    def apply_predict(self):
        return True

    def __call__(self, x, y=None):
        """
        Perform data preprocessing and return preprocessed data as tuple.
        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        x_unnormalized = self.unnormalize_fn(x)
        x_smoothed = self.do_smoothing(x_unnormalized)
        x_normalized = self.normalize_fn(x_smoothed)
        return x_normalized, y

    @abc.abstractmethod
    def do_smoothing(self, x):
        """
        Perform the smoothing
        :param x: np.ndarray; Input data
        :return: np.ndarray; smoothed input data
        """
        raise NotImplementedError

    def fit(self, x, y=None, **kwargs):
        pass


class MeanSmoothingPreprocessor(SmoothingPreprocessor):

    def __init__(self, kernel_size, normalize_fn: Callable, unnormalize_fn: Callable):
        super().__init__(normalize_fn, unnormalize_fn)

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        self.kernel = np.ones(shape=(1, 1, kernel_size, kernel_size)) / (kernel_size * kernel_size)
        self.kernel = self.kernel.astype(np.float32)
        self.padding = (kernel_size - 1) // 2

    def do_smoothing(self, x):
        """
        Perform the smoothing
        :param x: np.ndarray; Input data
        :return: np.ndarray; smoothed input data
        """
        x = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)])
        x = convolve(x, self.kernel, mode='valid')
        return x


class GaussianSmoothingPreprocessor(SmoothingPreprocessor):

    def __init__(self, smooth_std, normalize_fn: Callable, unnormalize_fn: Callable):
        super().__init__(normalize_fn, unnormalize_fn)

        kernel_size = int(np.ceil(6 * smooth_std + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1

        dist = np.zeros(shape=(kernel_size, kernel_size))
        dist[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1

        self.kernel = gaussian_filter(dist, smooth_std, truncate=3).reshape((1, 1, kernel_size, kernel_size)).astype(np.float32)
        self.padding = (kernel_size - 1) // 2

    def do_smoothing(self, x):
        """
        Perform the smoothing
        :param x: np.ndarray; Input data
        :return: np.ndarray; smoothed input data
        """
        x = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)])
        return convolve(x, self.kernel, mode='valid')


class MedianSmoothingPreprocessor(SmoothingPreprocessor):

    def __init__(self, kernel_size, normalize_fn: Callable, unnormalize_fn: Callable):
        super().__init__(normalize_fn, unnormalize_fn)

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        self.kernel_size = kernel_size

    def do_smoothing(self, x):
        """
        Perform the smoothing
        :param x: np.ndarray; Input data
        :return: np.ndarray; smoothed input data
        """
        x = median_filter(x, size=(1, 1, self.kernel_size, self.kernel_size), mode="reflect")
        return x


class InputSmoothing(RemovalAttack):
    """
    The attack consists of a whitebox attack
    """

    class CELoss(Loss):
        """
        Cross-entropy loss with support for soft labels, since PyTorch cross-entropy loss
        requires labels to be argmax-encoded
        """

        def reduce_labels(self):
            return False

        def compute_loss(self, pred, true, x=None):
            import torch

            if len(true.shape) == 1:
                return torch.nn.functional.cross_entropy(pred, true)

            logprobs = torch.nn.functional.log_softmax(pred, dim=1)
            return -(true * logprobs).sum() / pred.shape[0]

    def __init__(self, classifier):
        """
        :param classifier: Classifier
        """
        super(InputSmoothing, self).__init__(classifier)


    @abc.abstractmethod
    def get_preprocessor(self, normalize_fn: Callable, unnormalize_fn: Callable):
        raise NotImplementedError

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int = 0,
               epsilon: float = 0.0,
               wm_data = None,
               check_every_n_batches: int = None,
               callbacks = None,
               device="cuda",
               **kwargs):
        """Apply the attack

        :param train_loader: Training data loader.
        """
        if callbacks is None:
            callbacks = []

        preprocessor = self.get_preprocessor(train_loader.normalize, train_loader.unnormalize)
        self.classifier.add_preprocessor(preprocessor, 'smoothing')

        # Change to loss over soft labels.
        history = {}
        if epochs > 0:
            old_loss = self.classifier.loss
            self.classifier.loss = InputSmoothing.CELoss(self.classifier)

            if wm_data:
                print("Found wm data! Adding callback")
                callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data, log_after_n_batches=check_every_n_batches))

            trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                              device=device, num_epochs=epochs, epsilon=epsilon, callbacks=callbacks)
            trainer.evaluate()
            history = trainer.fit()

            self.classifier.loss = old_loss
        return history



class InputMeanSmoothing(InputSmoothing):

    def __init__(self, classifier, kernel_size=3, **kwargs):
        super().__init__(classifier,)
        self.kernel_size = kernel_size

    def get_preprocessor(self, normalize_fn: Callable, unnormalize_fn: Callable):
        return MeanSmoothingPreprocessor(self.kernel_size, normalize_fn, unnormalize_fn)


class InputGaussianSmoothing(InputSmoothing):

    def __init__(self, classifier, std=1, **kwargs):
        super().__init__(classifier)
        self.std = std

    def get_preprocessor(self, normalize_fn: Callable, unnormalize_fn: Callable):
        return GaussianSmoothingPreprocessor(self.std, normalize_fn, unnormalize_fn)


class InputMedianSmoothing(InputSmoothing):

    def __init__(self, classifier, kernel_size=3, **kwargs):
        super().__init__(classifier)
        self.kernel_size = kernel_size

    def get_preprocessor(self, normalize_fn: Callable, unnormalize_fn: Callable):
        return MedianSmoothingPreprocessor(self.kernel_size, normalize_fn, unnormalize_fn)


@mlconfig.register
def input_gaussian_smoothing_attack(classifier, **kwargs):
    return InputGaussianSmoothing(classifier, **kwargs)


@mlconfig.register
def input_gaussian_smoothing_removal(attack: InputGaussianSmoothing,
                       train_loader,
                       **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)


@mlconfig.register
def input_median_smoothing_attack(classifier, **kwargs):
    return InputMedianSmoothing(classifier, **kwargs)


@mlconfig.register
def input_median_smoothing_removal(attack: InputMedianSmoothing,
                       train_loader,
                       **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)


@mlconfig.register
def input_mean_smoothing_attack(classifier, **kwargs):
    return InputMeanSmoothing(classifier, **kwargs)


@mlconfig.register
def input_mean_smoothing_removal(attack: InputMeanSmoothing,
                       train_loader,
                       **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)