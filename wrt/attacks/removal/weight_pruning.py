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
This module implements weight pruning attack.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List

import numpy as np

import torch
import mlconfig

from wrt.classifiers.classifier import Classifier
from wrt.attacks.attack import RemovalAttack
from wrt.exceptions import ClassifierError
from collections import OrderedDict

from wrt.training import WRTCallback
from wrt.training.datasets import WRTDataLoader

logger = logging.getLogger(__name__)


class WeightPruning(RemovalAttack):
    """
    The attack consists of a weight pruning attack on a given model.
    """

    attack_params = RemovalAttack.attack_params + [
        "epochs",
        "batch_size",
    ]

    def __init__(
        self,
        classifier,
        threshold=None,
        sparsity=None,
        prune_last_layer=True,
        **kwargs
    ):
        """
        Create a :class:`.AdversarialTraining` instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param threshold: Threshold for weight pruning. If the absolute value of a weight is less than threshold,
                          it will be pruned.
        :type threshold: : `float`
        :param sparsity: Percentage of weights to be pruned for a layer.
        :type sparsity: `float`
        :param prune_last_layer: Whether to prune the last layer of the model.
        :type prune_last_layer: `bool`
        """
        super(WeightPruning, self).__init__(classifier)

        if not isinstance(classifier, Classifier):
            raise ClassifierError(self.__class__, [Classifier], classifier)

        if threshold is None and sparsity is None:
            raise ValueError("Either 'threshold' or 'sparsity' has to be set.")
        if threshold is not None and sparsity is not None:
            raise ValueError("'threshold' and 'sparsity' are mutually exclusive.")

        self.threshold = threshold
        self.sparsity = sparsity
        self.prune_last_layer = prune_last_layer

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               output_dir: str = None,
               device: str = "cuda",
               **kwargs):
        """ Perform weight pruning on a model.

        :param train_loader: Data loader for the training dataset.
        :param valid_loader: Loads normalized testing data.
        :param output_dir: (optional) The output directory to store intermediate results
        :param device Which device to train on.
        """
        weights = self.classifier._model.state_dict()
        # Init an empty dict
        pruned_weights = OrderedDict()
        # Looping through all layers
        for i, (l, w) in enumerate(weights.items()):
            # skip batch norm layers
            if 'bn' in l:
                pruned_weights[l] = w
                continue

            if self.prune_last_layer is False and i == len(weights) - 1:
                pruned_weights[l] = w
                break

            w = w.cpu().numpy()
            if self.threshold is not None:
                w[np.abs(w) <= self.threshold] = 0
            else:
                if len(w.shape) > 0:
                    num_pruned = np.ceil(w.size * self.sparsity).astype('int')
                    idx = np.unravel_index(np.argsort(np.abs(w).ravel()), w.shape)
                    idx = tuple(i[:num_pruned] for i in idx)
                    w[idx] = 0
            pruned_weights[l] = torch.from_numpy(w)

        self.classifier._model.load_state_dict(pruned_weights)
        return None


@mlconfig.register
def weight_pruning_attack(classifier, **kwargs):
    return WeightPruning(classifier, **kwargs)


@mlconfig.register
def weight_pruning_removal(attack: WeightPruning, train_loader, **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)
