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


class InputHorizontalFlipping(RemovalAttack):
    """  Input HorizontalFlipping flips the image horizontally.
    """

    class HorizontalFlippingPreprocessor(Preprocessor):

        def __init__(self,
                     normalize_fn: Callable,
                     unnormalize_fn: Callable):
            super().__init__()
            self.normalize_fn = normalize_fn
            self.unnormalize_fn = unnormalize_fn

        @property
        def apply_fit(self):
            return True

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
            x = np.ascontiguousarray(np.fliplr(x.transpose((0,3,2,1)))).transpose((0,3,2,1))
            return x, y

        def fit(self, x, y=None, **kwargs):
            pass

    def __init__(self, classifier, **kwargs):
        """
        :param classifier: Classifier
        :param num_divisions: int
        """
        super(InputHorizontalFlipping, self).__init__(classifier)

    def remove(self,
               train_loader: WRTDataLoader,
               **kwargs):
        """ Attach a preprocessor to this classifier.
        :param train_loader: Training data loader.
        """
        preprocessor = InputHorizontalFlipping.HorizontalFlippingPreprocessor(train_loader.normalize,
                                                                    train_loader.unnormalize)
        self.classifier.add_preprocessor(preprocessor, 'flipping')
        return None


@mlconfig.register
def input_flipping_attack(classifier, **kwargs):
    return InputHorizontalFlipping(classifier, **kwargs)


@mlconfig.register
def input_flipping_removal(attack: InputHorizontalFlipping, train_loader, **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)