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
from copy import deepcopy

import mlconfig

from wrt.attacks.attack import RemovalAttack
from wrt.training.datasets import WRTDataLoader

logger = logging.getLogger(__name__)


class FeatureShuffling(RemovalAttack):
    """
    The attack consists of a feature shuffling attack on a given model.
    """

    attack_params = RemovalAttack.attack_params + [
        "epochs",
        "batch_size",
    ]

    def __init__(
        self,
        classifier,
        **kwargs
    ):
        """
        Create a :class:`.WeightShuffling` instance.
        Randomly permutes all weights.

        :param classifier: A trained classifier.
        """
        super(FeatureShuffling, self).__init__(classifier)

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
        self.get_classifier().shuffle_intermediate_features = True


@mlconfig.register
def feature_shuffling_attack(classifier, **kwargs):
    return FeatureShuffling(classifier, **kwargs)


@mlconfig.register
def feature_shuffling_removal(attack: FeatureShuffling, train_loader, **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)
