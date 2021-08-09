"""
This module implements the White box attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from collections import Callable

import numpy as np
import mlconfig

from wrt.classifiers.classifier import Classifier
from wrt.attacks.attack import RemovalAttack
from wrt.exceptions import ClassifierError
from wrt.preprocessors import Preprocessor
from wrt.training.datasets import WRTDataLoader

logger = logging.getLogger(__name__)


class InputNoising(RemovalAttack):
    """
    Input noising adds random noise to an image in range [0,1]
    """

    attack_params = RemovalAttack.attack_params

    class NoisingPreprocessor(Preprocessor):

        def __init__(self,
                     noise_mean: np.ndarray,
                     noise_std: np.ndarray,
                     normalize_fn: Callable,
                     unnormalize_fn: Callable):
            super().__init__()
            self.noise_mean = noise_mean
            self.noise_std = noise_std
            self.normalize_fn = normalize_fn
            self.unnormalize_fn = unnormalize_fn

        @property
        def apply_fit(self):
            return False

        @property
        def apply_predict(self):
            return True

        def __call__(self,
                     x: np.ndarray,
                     y: np.ndarray = None):
            """
            Perform data preprocessing and return preprocessed data as tuple.
            :param x: Dataset to be preprocessed.
            :param y: Labels to be preprocessed.
            :return: Preprocessed data.
            """
            # Sample the noise.
            noise = np.random.normal(self.noise_mean, self.noise_std, size=x.shape)

            # Unnormalize the data, add noise, clip to [0,1] and normalize.
            x = self.unnormalize_fn(x)
            x += noise
            x = np.clip(x, 0, 1)
            x = self.normalize_fn(x)

            return x, y

        def fit(self, x, y=None, **kwargs):
            pass

    def __init__(self, classifier, mean=0, std=1, **kwargs):
        """
        :param classifier: Classifier
        :param mean: float
        :param std: float
        """
        super(InputNoising, self).__init__(classifier)

        if not isinstance(classifier, Classifier):
            raise ClassifierError(self.__class__, [Classifier], classifier)

        kwargs = {}
        InputNoising.set_params(self, **kwargs)

        self.mean = mean
        self.std = std

    def remove(self,
               train_loader: WRTDataLoader,
               **kwargs):
        """ Attach a preprocessor to this classifier.
        :param train_loader: Training data loader.
        """
        preprocessor = InputNoising.NoisingPreprocessor(self.mean,
                                                        self.std,
                                                        train_loader.normalize,
                                                        train_loader.unnormalize)
        self.classifier.add_preprocessor(preprocessor, 'noise')

        return None

    def predict(self, x, **kwargs):
        """
        Perform prediction using the watermarked classifier.

        :param x: Test set.
        :type x: `np.ndarray`
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :type kwargs: `dict`
        :return: Predictions for test set.
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)


@mlconfig.register
def input_noising_attack(classifier, **kwargs):
    return InputNoising(classifier, **kwargs)


@mlconfig.register
def input_noising_removal(attack: InputNoising,
                       train_loader,
                       **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)
