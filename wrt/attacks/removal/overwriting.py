# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Watermark overwriting attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import logging
from abc import ABC

import numpy as np

from wrt.classifiers.classifier import Classifier
from wrt.attacks.attack import RemovalAttack
from wrt.exceptions import ClassifierError

logger = logging.getLogger(__name__)


class Overwriting(RemovalAttack):
    """
    The attack consists of overwriting a watermark by embedding a different one using
    the same scheme
    """

    def __init__(
            self,
            classifier,
            defense,
            init_kwargs=None,
            embed_args=None,
            embed_kwargs=None,
            other_kwargs=None
    ):
        """
        Create a :class:`.Regularization` instance.

        :param classifier: Classifier; A trained classifier.
        :param defense: Watermark; the watermark scheme to overwrite
        """
        super(Overwriting, self).__init__(classifier)

        if not isinstance(classifier, Classifier):
            raise ClassifierError(self.__class__, [Classifier], classifier)

        self.defense = defense
        self.init_kwargs = init_kwargs if init_kwargs is not None else {}
        self.embed_args = embed_args if embed_args is not None else ()
        self.embed_kwargs = embed_kwargs if embed_kwargs is not None else {}
        self.other_kwargs = other_kwargs if other_kwargs is not None else {}

    @classmethod
    def is_valid_classifier_type(cls, classifier):
        """
        Checks whether the classifier provided is a classifier which this class can perform an attack on
        :param classifier:
        :return:
        """
        return True if isinstance(classifier, Classifier) else False

    def preprocess_data(self, x, y):
        """
        Preprocess the training data and labels for embedding
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :return: (np.ndarray, np.ndarray); the processed data
        """
        if 'normalize' in self.other_kwargs:
            mean, std = self.other_kwargs['normalize']
        else:
            mean, std = np.tile(0, x.shape[1]), np.tile(1, x.shape[1])
        mean, std = mean.reshape((1, x.shape[1], 1, 1)), std.reshape((1, x.shape[1], 1, 1))
        return ((x - mean) / std).astype(np.float32), y

    def remove(self, x, y=None, **kwargs):
        """
        Apply the overwriting attack

        :param x: An array with the target inputs.
        :type x: `np.ndarray`
        :param y: np.ndarray; Corresponding labels
        :return: None
        """
        if y is None:
            raise ValueError("Labels must be provided")

        x, y = self.preprocess_data(x, y)

        # Drop unused labels.
        print("Predicting labels for overwriting!")
        print(y.shape, type(y), y[0])
        y = self.classifier.predict(x, batch_size=32,
                                    learning_phase=kwargs.setdefault("learning_phase", False))
        y = np.eye(self.classifier.nb_classes())[np.argmax(y, axis=1)]
        print(y.shape, type(y), y[0])

        defense_overwrite_instance = self.defense(self.classifier, **self.init_kwargs)
        embed_kwargs = {**self.embed_kwargs, 'x_train': x, 'y_train': y}
        defense_overwrite_instance.embed(**embed_kwargs)


