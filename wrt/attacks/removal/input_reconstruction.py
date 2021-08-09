"""
This module implements the White box attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable

import numpy as np
import mlconfig

from wrt.attacks.attack import RemovalAttack
from wrt.classifiers import PyTorchClassifier
from wrt.config import WRT_DATA_PATH
from wrt.preprocessors import Preprocessor
from wrt.training.datasets import WRTDataLoader
from wrt.training.models.torch.autoencoder.imagenet_autoencoder import cifar_autoencoder

logger = logging.getLogger(__name__)


class InputReconstruction(RemovalAttack):
    """
    The attack reconstructs the input using an Autoencoder.

    Note: Input Reconstruction is only defined for CIFAR-10, because Autoencoders for ImageNet are not good enough.
    """

    class ReconstructionPreprocessor(Preprocessor):

        def __init__(self,
                     autoencoder,
                     normalize_fn: Callable,
                     unnormalize_fn: Callable, batch_size):
            super().__init__()
            self.autoencoder = autoencoder
            self.normalize_fn = normalize_fn
            self.unnormalize_fn = unnormalize_fn
            self.batch_size = batch_size

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
            x = self.unnormalize_fn(x)
            x_reconstructed = self.autoencoder.predict(x.astype(np.float32), batch_size=self.batch_size)
            x_normalized = self.normalize_fn(x_reconstructed)
            return x_normalized, y

        def fit(self, x, y=None, **kwargs):
            pass

    def __init__(self,
                 classifier: PyTorchClassifier,
                 autoencoder: Callable[[int], PyTorchClassifier],
                 complexity: int,
                 lr: float = 0.001,
                 **kwargs):
        """
        :param classifier: Classifier
        :param autoencoder: function; must have exactly one parameter (the complexity), and must
                            return a Classifier object representing the autoencoder
        :param complexity: Any; single argument passed to the autoencoder function
        :param lr: float; learning rate, only needs to be provided if train is True
        :param epochs: int; num epochs to train the autoencoder, only needs to be provided
                       if train is True
        :param batch_size: int; batch size to train, only needs to be provided if train is True
        """
        super(InputReconstruction, self).__init__(classifier)

        self.autoencoder = autoencoder(complexity)
        self.lr = lr

    def remove(self,
               train_loader: WRTDataLoader,
               train_autoencoder: bool = False,
               batch_size: int = 32,
               **kwargs):
        """ Attach a preprocessor to this classifier.
        :param train_loader: Training data loader.
        :param train_autoencoder Whether to train the autoencoder.
        :param batch_size The batch size for the autoencoder
        """
        if train_autoencoder:
            raise NotImplementedError

        preprocessor = InputReconstruction.ReconstructionPreprocessor(autoencoder=self.autoencoder,
                                                                      normalize_fn=train_loader.normalize,
                                                                      unnormalize_fn=train_loader.unnormalize,
                                                                      batch_size=batch_size)
        self.classifier.add_preprocessor(preprocessor, 'reconstruction')


@mlconfig.register
def input_reconstruction_attack(classifier, dataset, **kwargs):
    assert dataset in ["cifar10"], f"No pretrained model found for {dataset}"

    def get_autoencoder(complexity):
        if dataset == "cifar10":
            autoencoder = cifar_autoencoder(complexity)
            autoencoder.load(f'cifar_autoencoder_{complexity}_0', path=WRT_DATA_PATH)
        else:
            raise FileNotFoundError
        return autoencoder

    return InputReconstruction(classifier, autoencoder=get_autoencoder, **kwargs)


@mlconfig.register
def input_reconstruction_removal(attack: InputReconstruction, train_loader, **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)