"""
This module implements the White box attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable

import numpy as np

from wrt.classifiers.classifier import Classifier
from wrt.attacks.attack import RemovalAttack
from wrt.exceptions import ClassifierError
from wrt.preprocessors import Preprocessor
from wrt.training.datasets import WRTDataLoader
import mlconfig

logger = logging.getLogger(__name__)


class InputQuantization(RemovalAttack):
    """
    Input Quantization splits the input space (between [0-1]) into bins and maps values inside of the mean value of
    each bin.
    """

    attack_params = RemovalAttack.attack_params

    class QuantizationPreprocessor(Preprocessor):

        def __init__(self,
                     num_divisions: int,
                     normalize_fn: Callable,
                     unnormalize_fn: Callable):
            super().__init__()
            self.num_divisions = num_divisions
            self.quantas = np.linspace(0, 1, num_divisions, endpoint=False)
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
            x = self.unnormalize_fn(x)
            x[x < 0] = 0
            for i in range(self.num_divisions-1):
                x[np.logical_and(x >= self.quantas[i], x < self.quantas[i + 1])] = (self.quantas[i] + self.quantas[i+1])/2
            x_quantized = x.astype(np.float32)
            x_normalized = self.normalize_fn(x_quantized)
            return x_normalized, y

        def fit(self, x, y=None, **kwargs):
            pass

    def __init__(self, classifier, num_divisions=16, **kwargs):
        """
        :param classifier: Classifier
        :param num_divisions: int
        """
        super(InputQuantization, self).__init__(classifier)

        if not isinstance(classifier, Classifier):
            raise ClassifierError(self.__class__, [Classifier], classifier)

        kwargs = {}
        InputQuantization.set_params(self, **kwargs)

        self.num_divisions = num_divisions

    @classmethod
    def is_valid_classifier_type(cls, classifier):
        """
        Checks whether the classifier provided is a classifier which this class can perform an attack on
        :param classifier:
        :return:
        """
        return True if isinstance(classifier, Classifier) else False

    def remove(self,
               train_loader: WRTDataLoader,
               **kwargs):
        """ Attach a preprocessor to this classifier.
        :param train_loader: Training data loader.
        """
        preprocessor = InputQuantization.QuantizationPreprocessor(self.num_divisions,
                                                                  train_loader.normalize,
                                                                  train_loader.unnormalize)
        self.classifier.add_preprocessor(preprocessor, 'quantization')
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
def input_quantization_attack(classifier, **kwargs):
    return InputQuantization(classifier, **kwargs)


@mlconfig.register
def input_quantization_removal(attack: InputQuantization,
                       train_loader,
                       **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)