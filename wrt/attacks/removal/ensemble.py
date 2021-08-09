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
This module implements the Label Smoothing attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List

import mlconfig

from embed import __load_model
from wrt.attacks.attack import RemovalAttack
from wrt.classifiers import Loss, PyTorchClassifier
from wrt.training.callbacks import EvaluateWmAccCallback, WRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer


class EnsembleAttack(RemovalAttack):
    """
    The attack consists of training an ensemble on n non-overlapping subsets of the data.
    (Examplary Implementation)
    """

    class CELoss(Loss):
        """
        Cross-entropy loss with support for soft labels, since PyTorch cross-entropy loss
        requires labels to be argmax-encoded
        """

        def reduce_labels(self):
            return False

        def compute_loss(self, pred, true, x=None):
            import torch

            if len(true.shape) == 1:
                return torch.nn.functional.cross_entropy(pred, true)

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
        """
        super(EnsembleAttack, self).__init__(classifier)
        self.num_classes = num_classes

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               surrogate_models: List[PyTorchClassifier],
               schedulers: List = None,
               splits: int = 3,
               epochs: int = 1,
               epsilon: float = 0.0,
               wm_data=None,
               callbacks: List[WRTCallback] = None,
               device="cuda",
               **kwargs):
        """Attempt to remove the watermark
        :param train_loader The loader for the training data.
        :param valid_loader Test data loader
        :param splits Number of splits for the ensemble.
        :param epochs Number of epochs to fine-tune.
        :param epsilon Label smoothing parameter
        :param wm_data Watermark data.
        :param callbacks Callbacks during training.
        :param device Device to train on.
        """
        if callbacks is None:
            callbacks = []

        # Change to loss over soft labels.
        self.classifier.loss = EnsembleAttack.CELoss(self.classifier)

        # Split the training data into non-overlapping subsets.
        loaders: List[WRTDataLoader] = train_loader.split(splits)

        for i, (surrogate_model, loader) in enumerate(zip(surrogate_models, loaders)):
            if wm_data:
                print("Found wm data! Adding callback")
                callbacks.append(EvaluateWmAccCallback(surrogate_model, wm_data))

            scheduler = None
            if schedulers is not None:
                scheduler = schedulers[i]

            print(f"Loader {i+1}/{splits}")
            trainer = Trainer(model=surrogate_model, train_loader=loader, epsilon=epsilon, scheduler=scheduler,
                              valid_loader=valid_loader, device=device, num_epochs=epochs, callbacks=callbacks)
            trainer.fit()
        self.classifier = surrogate_models
        return None


#################### Configuration callable through mlconfig

@mlconfig.register
def ensemble_attack(**kwargs):
    return EnsembleAttack(**kwargs)


@mlconfig.register
def ensemble_removal(attack: EnsembleAttack, config, **kwargs):
    splits = config.remove.splits

    surrogate_models, schedulers = [], []
    for i in range(splits):
        model = config.surrogate_model()
        optimizer = config.optimizer(model.parameters())
        surrogate_models.append(__load_model(model, optimizer=optimizer, image_size=config.dataset.image_size,
                                             num_classes=config.create.num_classes))
        schedulers.append(config.scheduler(optimizer))
    return attack, attack.remove(surrogate_models=surrogate_models, schedulers=schedulers, **kwargs)
