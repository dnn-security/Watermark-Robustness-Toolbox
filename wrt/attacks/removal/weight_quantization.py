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
This module implements a weight quantization attack.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import mlconfig
from tqdm import tqdm
import numpy as np
import torch

from wrt.attacks.attack import RemovalAttack
from wrt.training.callbacks import EvaluateWmAccCallback, DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer

logger = logging.getLogger(__name__)


class WeightQuantization(RemovalAttack):
    """
    The attack consists of a weight pruning attack on a given model.
    """

    def __init__(self, classifier, **kwargs):
        """
        Create a :class:`.WeightQuantization` instance.
        Randomly permutes all weights.

        :param classifier: A trained classifier.
        """
        super(WeightQuantization, self).__init__(classifier)

    def quantization(self, param, bits):
        quantata = int(np.math.pow(2, bits))
        min_weight, max_weight = param.data.min(), param.data.max()
        qranges = torch.linspace(min_weight, max_weight, quantata)

        ones = torch.ones_like(param.data)
        zeros = torch.zeros_like(param.data)
        for i in range(len(qranges) - 1):
            t1 = torch.where(param.data > qranges[i], zeros, ones)
            t2 = torch.where(param.data < qranges[i + 1], zeros, ones)
            t3 = torch.where((t1 + t2) == 0, ones * (qranges[i] + qranges[i + 1]) / 2, zeros)
            t4 = torch.where((t1 + t2) == 0, zeros, ones)

            param.data = t4 * param.data + t3
        return param

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int = 1,
               bits: int = 5,
               epsilon: float = 0.0,
               wm_data=None,
               callbacks=None,
               device="cuda",
               **kwargs):
        """ Perform weight pruning on a model.

        :param bits Number of bits for the quantization.
        """
        if callbacks is None:
            callbacks = []

        print(f"Removing with bits: {bits}")
        for name, param in tqdm(self.get_classifier().model.named_parameters(), desc="Quantization"):
            #if "conv" in name and "weight" in name:
            self.quantization(param, bits=bits)

        if wm_data:
            print("Found wm data! Adding callback")
            defense, x_wm, y_wm = wm_data
            print(f"Wm Acc: {defense.verify(x_wm, y_wm, classifier=self.get_classifier())[0]}")
            callbacks.append(
                DebugWRTCallback(lambda: defense.verify(x_wm, y_wm, classifier=self.get_classifier())[0],
                                 message="wm_acc",
                                 check_every_n_batches=200))

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          num_epochs=epochs, callbacks=callbacks, device=device, epsilon=epsilon)
        trainer.evaluate()
        return trainer.fit()


@mlconfig.register
def weight_quantization_attack(classifier, **kwargs):
    return WeightQuantization(classifier, **kwargs)


@mlconfig.register
def weight_quantization_removal(attack: WeightQuantization, train_loader, **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)
