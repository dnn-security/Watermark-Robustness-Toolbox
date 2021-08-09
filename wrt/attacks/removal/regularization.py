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
This module implements the White box attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os
from typing import List, Tuple

import mlconfig
import numpy as np
import torch

from wrt.classifiers import Loss, PyTorchClassifier
from wrt.classifiers.classifier import Classifier
from wrt.attacks.attack import RemovalAttack
from wrt.defenses import Watermark
from wrt.exceptions import ClassifierError
from wrt.training import WRTCallback, EarlyStoppingWRTCallback
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer
from wrt.training.utils import convert_to_augmented_data_generator, sample_dataset, replace_labels, \
    convert_to_data_generator

logger = logging.getLogger(__name__)


class Regularization(RemovalAttack):
    """
    The attack consists of a whitebox attack
    """

    class CELoss(Loss):
        """
        Cross-entropy loss with soft labels, since PyTorch cross-entropy loss
        requires labels to be argmax-encoded
        """

        def reduce_labels(self):
            return False

        def compute_loss(self, pred, true, x=None):
            if len(true.shape) == 1:
                return torch.nn.functional.cross_entropy(pred, true)

            logprobs = torch.nn.functional.log_softmax(pred, dim=1)
            return -(true * logprobs).sum() / pred.shape[0]

    def __init__(self, classifier: PyTorchClassifier, **kwargs):
        """
        Create a :class:`.Regularization` instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param l2_decay Strength of the l2 decay.
        """
        super(Regularization, self).__init__(classifier)

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               l2_decay: float,
               reg_epochs: int,
               max_ft_epochs: int,
               ft_patience: int,
               check_every_n_batches: int = None,
               callbacks: List[WRTCallback] = None,
               wm_data: Tuple[Watermark, np.ndarray, np.ndarray] = None,
               output_dir=None,
               device="cuda",
               **kwargs):
        """ The regularization attack first applies strong l2 regularization to the model and then fine-tunes the
        model.

        :param train_loader Data loader for the training data.
        :param valid_loader Data loader for the validation data
        :param l2_decay Strength of the l2 decay.
        :param reg_epochs Number of epochs to run the regularization
        :param max_ft_epochs: Maximum number of fine-tuning epochs
        :param ft_patience Patience for early stopping during the fine-tuning phase
        :param check_every_n_batches Evaluates the early stopping condition every n batches. If None, called at the
        end of an epoch.
        :param callbacks Callbacks during training.
        :param wm_data Data used to evaluate the watermark accuracy as a callback.
        :param output_dir Save results in this directory.
        :param device Device to run model training on.
        :rtype: `np.ndarray`
        """
        if callbacks is None:
            callbacks = []

        if wm_data is not None:
            print("Tracking wm data!")
            defense, x_wm, y_wm = wm_data
            callbacks.append(
                DebugWRTCallback(debug_fn=lambda: defense.verify(x_wm, y_wm, classifier=self.get_classifier())[0],
                                 message="wm acc",
                                 check_every_n_batches=check_every_n_batches))

        # Allow soft labels.
        old_loss = self.classifier.loss
        self.classifier.loss = self.classifier.loss = Regularization.CELoss(self.classifier)

        print("Regularization Phase!")
        pre_reg = self.classifier.optimizer.param_groups[0]['weight_decay']
        print('pre_reg: ' + str(pre_reg))
        pre_regs = []
        for i, g in enumerate(self.classifier.optimizer.param_groups):
            print(f"g['weight_decay'] -> {l2_decay}")
            pre_regs.append(g['weight_decay'])
            g['weight_decay'] = l2_decay

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=device, num_epochs=reg_epochs, callbacks=callbacks)
        reg_hist: dict = trainer.fit()

        print("Fine-Tuning Phase!")
        for i, g in enumerate(self.classifier.optimizer.param_groups):
            print(f"g['weight_decay'] -> {pre_regs[i]}")
            g['weight_decay'] = pre_regs[i]

        callbacks.append(EarlyStoppingWRTCallback(lambda: trainer.evaluate()[0].value,
                                                  check_every_n_batches=check_every_n_batches,
                                                  patience=ft_patience,
                                                  mode='min'))
        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=device, num_epochs=max_ft_epochs, callbacks=callbacks)
        ft_hist: dict = trainer.fit()

        # Reset the model's loss function.
        self.classifier.loss = old_loss
        history = {"reg_hist": reg_hist, "ft_hist": ft_hist}

        # Write the history to the training file.
        if output_dir is not None:
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(history, f)

        return history


@mlconfig.register
def regularization_attack(classifier, **kwargs):
    return Regularization(classifier, **kwargs)


@mlconfig.register
def regularization_removal(attack: Regularization, **kwargs):
    return attack, attack.remove(**kwargs)
