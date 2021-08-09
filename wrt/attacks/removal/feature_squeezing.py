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


class FeatureSqueezing(RemovalAttack):
    """
    The attack consists of a whitebox attack
    """

    attack_params = RemovalAttack.attack_params

    class SqueezingPreprocessor(Preprocessor):

        def __init__(self,
                     bit_depth: int,
                     normalize_fn: Callable[[np.ndarray], np.ndarray],
                     unnormalize_fn: Callable[[np.ndarray], np.ndarray]):
            super().__init__()
            self.bit_depth = bit_depth
            self.normalize_fn = normalize_fn
            self.unnormalize_fn = unnormalize_fn

        @property
        def apply_fit(self):
            return False

        @property
        def apply_predict(self):
            return True

        def __squeeze(self, x):
            max_value = np.rint(2 ** self.bit_depth - 1)
            x = np.rint(x * max_value) / max_value
            return x

        def __call__(self, x, y=None):
            """
            Perform data preprocessing and return preprocessed data as tuple.
            :param x: Dataset to be preprocessed.
            :param y: Labels to be preprocessed.
            :return: Preprocessed data.
            """
            x = self.unnormalize_fn(x)
            x_squeezed = self.__squeeze(x)
            x_normalized = self.normalize_fn(x_squeezed).astype(np.float32)
            return x_normalized, y

        def fit(self, x, y=None, **kwargs):
            pass

    def __init__(self,
                 classifier,
                 bit_depth: int = 8,
                 **kwargs):
        """
        :param classifier: Classifier
        :param bit_depth: int; New bit depth of data
        """
        super(FeatureSqueezing, self).__init__(classifier)

        if not isinstance(classifier, Classifier):
            raise ClassifierError(self.__class__, [Classifier], classifier)

        kwargs = {}
        FeatureSqueezing.set_params(self, **kwargs)

        self.bit_depth = bit_depth

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
        """Apply the attack

        :param train_loader: Training data loader.
        """
        preprocessor = FeatureSqueezing.SqueezingPreprocessor(self.bit_depth,
                                                              train_loader.normalize,
                                                              train_loader.unnormalize)
        self.classifier.add_preprocessor(preprocessor, 'squeeze')

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
def feature_squeezing_attack(classifier, **kwargs):
    return FeatureSqueezing(classifier, **kwargs)


@mlconfig.register
def feature_squeezing_removal(attack: FeatureSqueezing,
                       train_loader,
                       **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)