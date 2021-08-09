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
This module implements a weight shifting attack.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import mlconfig
import numpy as np
import torch
from tqdm import tqdm

from wrt.attacks.attack import RemovalAttack
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer

logger = logging.getLogger(__name__)


class WeightShifting(RemovalAttack):
    """
    The attack consists of a weight pruning attack on a given model.
    """

    def __init__(self, classifier, device="cuda", **kwargs):
        """
        Create a :class:`.WeightShifting` instance.
        Shifts the mean of a single filter in each conv layer.

        :param classifier: A trained classifier.
        """
        super(WeightShifting, self).__init__(classifier)
        self.device = device

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int = 1,
               lmbda: float = 1.5,
               lmbda2: float = 1.0,
               epsilon: float = 0.0,
               wm_data=None,
               check_every_n_batches: int = None,
               callbacks=None,
               **kwargs):
        """ Perform weight shifting on a model.

        :param train_loader Loader for the training data.
        :param valid_loader Loader for the testing data.
        :param epochs Number of epochs for fine-tuning.
        :param lmbda Strength of the perturbation to each filter.
        :param wm_data Tuple of the defense, x_wm and y_wm.
        :param check_every_n_batches Run debug callbacks every n batches.
        :param callbacks List of callbacks to call during fine-tuning.
        :param device Device to run on.
        """
        if callbacks is None:
            callbacks = []

        all_params = list(self.get_classifier().model.named_parameters())
        print("Params: ", len(all_params))
        for name, param in tqdm(all_params, desc="Noising"):
            if "conv" in name and "weight" in name:
                param.data -= param.data.mean(0)*lmbda + lmbda2 * torch.normal(torch.zeros_like(param.data), param.data.std())

        if wm_data:
            defense, x_wm, y_wm = wm_data
            print(f"Wm Acc: {defense.verify(x_wm, y_wm, classifier=self.get_classifier())[0]}")
            callbacks.append(
                DebugWRTCallback(lambda: defense.verify(x_wm, y_wm, classifier=self.get_classifier())[0],
                                 message="wm_acc",
                                 check_every_n_batches=check_every_n_batches))

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          num_epochs=epochs, callbacks=callbacks, device=self.device, epsilon=epsilon)
        trainer.evaluate()
        return trainer.fit()


@mlconfig.register
def weight_shifting_attack(classifier, **kwargs):
    return WeightShifting(classifier, **kwargs)


@mlconfig.register
def weight_shifting_removal(attack: WeightShifting, train_loader, **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)
