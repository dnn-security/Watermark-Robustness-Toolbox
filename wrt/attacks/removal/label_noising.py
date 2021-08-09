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
This module implements the Fine-Tuning attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List, Tuple

import mlconfig
import numpy as np
import torch

from wrt.attacks.attack import RemovalAttack
from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import Loss
from wrt.training.callbacks import EvaluateWmAccCallback, WRTCallback, DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer


class LabelNoisingAttack(RemovalAttack):
    """
    The attack consists of fine-tuning a watermarked classifier on more target data.
    (Examplary Implementation)
    """

    class CELoss(Loss):
        """
        Cross-entropy loss with support for soft labels, since PyTorch cross-entropy loss
        requires labels to be argmax-encoded
        """

        def __init__(self, classifier, epsilon, random_perm):
            super().__init__(classifier)
            assert 0 <= epsilon <= 1, f"Epsilon must be between 0 and 1. '{epsilon}' is not allowed."
            self.epsilon = epsilon

            self.random_perm = random_perm

        def reduce_labels(self):
            return False

        def compute_loss(self, pred, true, x=None):

            if true.dim() == 1:
                true = torch.eye(pred.shape[-1])[true].to(true.device)

            rnd_probs = true[:, torch.randperm(true.shape[1])] #torch.rand_like(true)

            true = self.epsilon * (rnd_probs / rnd_probs.sum(1, keepdim=True)) + (1 - self.epsilon) * true

            logprobs = torch.nn.functional.log_softmax(pred, dim=1)
            return -(true * logprobs).sum() / pred.shape[0]

    def __init__(
            self,
            classifier,
            num_classes,
            **kwargs
    ):
        """
        Create a :class:`.RemovalAttack` instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param lr: Learning rate for the fine-tuning process
        :type lr: `float`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(LabelNoisingAttack, self).__init__(classifier)
        self.num_classes = num_classes

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int,
               epsilon: float = 0.0,
               callbacks: List[WRTCallback] = None,
               wm_data: Tuple = None,
               check_every_n_batches: int = None,
               device="cuda",
               **kwargs):
        """
        The random occlusion defense is similar to adversarial training, but
        generates random occlusion maps instead of adversarial examples.

        :param train_loader Data loader for the training data.
        :param valid_loader Data loader for the validation data
        :param callbacks Callbacks during training.
        :param wm_data Data used to evaluate the watermark accuracy as a callback.
        :param device Device to run model training on.
        :rtype: `np.ndarray`
        """
        if callbacks is None:
            callbacks = []

        # Change to loss over soft labels.
        self.classifier.loss = LabelNoisingAttack.CELoss(self.classifier, epsilon=epsilon,
                                                         random_perm=torch.randperm(self.get_classifier().nb_classes()))

        if wm_data:
            print("Found wm data! Adding callback")
            callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data, log_after_n_batches=check_every_n_batches))


        callbacks.append(DebugWRTCallback(lambda: evaluate_test_accuracy(self.get_classifier(), valid_loader,
                                                                         limit_batches=50, verbose=False),
                                          message="test acc",
                                          check_every_n_batches=check_every_n_batches))

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=device, num_epochs=epochs, callbacks=callbacks)
        trainer.evaluate()
        history = trainer.fit()

        return history


@mlconfig.register
def label_noising_attack(config,
                         **kwargs):
    return LabelNoisingAttack(**kwargs)


@mlconfig.register
def label_noising_removal(attack: LabelNoisingAttack,
                          config,
                          **kwargs):
    scheduler = None
    if "scheduler" in config.keys():
        scheduler = config.scheduler(attack.get_classifier().optimizer)
    return attack, attack.remove(scheduler=scheduler, **kwargs)
