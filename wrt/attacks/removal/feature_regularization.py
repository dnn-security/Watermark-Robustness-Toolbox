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

import abc

import mlconfig
from torch.nn import Conv2d

from wrt.attacks.attack import RemovalAttack
from wrt.attacks.removal import InputHorizontalFlipping
from wrt.classifiers import Loss
from wrt.training.callbacks import EvaluateWmAccCallback, DebugWRTCallback
from wrt.attacks.util import evaluate_test_accuracy
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer
import torch


class FeatureRegularization(RemovalAttack):
    """
    The attack consists of fine-tuning a watermarked classifier on more target data.
    (Examplary Implementation)
    """

    class CELoss(Loss):
        """
        Cross-entropy loss with support for soft labels, since PyTorch cross-entropy loss
        requires labels to be argmax-encoded
        """

        def __init__(self, classifier, threshold=0, norm=1, f_reg=0.001):
            super().__init__(classifier)
            self.f_reg = f_reg
            self.norm = norm
            self.threshold = threshold
            self.dent = torch.rand(classifier.nb_classes()).to(classifier.device) * self.f_reg

            print(f"3: {self.dent[3]}, 4: {self.dent[4]}")

            print(f"Using f_reg: {f_reg}")

        def reduce_labels(self):
            return False

        def compute_loss(self, pred: list, true, x=None):

            if true.dim() == 1:
                true = torch.eye(pred[-1].shape[1])[true].to(true.device)

            # Cross-entropy loss
            logprobs = torch.nn.functional.log_softmax(pred[-1], dim=1)
            loss = -(true * logprobs).sum() / pred[-1].shape[0]

            if self.norm == 1:
                for features in pred[:-1]:
                    loss_f = ((self.dent[true.argmax(1)]) * torch.flatten(features).mean(-1)).sum()  # l_1
                    loss += loss_f
            elif self.norm == 2:
                for features in pred[:-1]:
                    f = (self.dent[true.argmax(1)])*torch.flatten(features).abs().pow(2).sum(-1)
                    loss_f = (1/torch.flatten(features).shape[0]) * torch.sqrt(f)   # l_2
                    loss += loss_f.sum()
            elif self.norm == 3:
                for features in pred[:-1]:
                    loss -= self.f_reg*torch.distributions.normal.Normal(0, 1, validate_args=None).log_prob(torch.flatten(features)).mean(-1).sum()
            else:
                raise NotImplementedError

            return loss

    def __init__(
            self,
            classifier,
            num_classes,
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
        super(FeatureRegularization, self).__init__(classifier)
        self.num_classes = num_classes

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int,
               f_reg: float = 0.001,
               norm=1,
               lr = None,
               scheduler=None,
               epsilon: float = 0.0,
               threshold=0,
               check_every_n_batches: int = None,
               wm_data=None,
               callbacks=None,
               **kwargs):
        """Attempt to remove the watermark
        :param train_loader The loader for the training data.
        :param epochs Number of epochs to train for.
        :param scheduler
        :param valid_loader
        :param wm_data
        :param callbacks
        """
        if callbacks is None:
            callbacks = []

        # Change to loss over soft labels.
        old_loss = self.classifier.loss
        self.classifier.loss = FeatureRegularization.CELoss(self.classifier, threshold=threshold, norm=norm, f_reg=f_reg)

        if wm_data:
            print("Found wm data! Adding callback")
            callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data, log_after_n_batches=check_every_n_batches))

        callbacks.append(DebugWRTCallback(lambda: evaluate_test_accuracy(self.get_classifier(), valid_loader,
                                                                         limit_batches=50, verbose=False),
                                          message="test acc",
                                          check_every_n_batches=check_every_n_batches))
        old_lr = None
        if lr is not None:
            old_lr = self.classifier.lr
            self.classifier.lr = lr

        preprocessor = InputHorizontalFlipping.HorizontalFlippingPreprocessor(train_loader.normalize,
                                                                              train_loader.unnormalize)
        self.classifier.add_preprocessor(preprocessor, 'flipping')

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          scheduler=scheduler, device=self.get_classifier().device, num_epochs=1,
                          epsilon=epsilon, callbacks=callbacks, train_all_features=True)
        history = trainer.fit()

        self.classifier.loss = old_loss
        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          scheduler=scheduler, device=self.get_classifier().device, num_epochs=3,
                          epsilon=epsilon, callbacks=callbacks, train_all_features=False)
        history = trainer.fit()


        if old_lr is not None:
            self.classifier.lr = old_lr

        return history


#################### Configuration functions callable through mlconfig

@mlconfig.register
def feature_regularization_attack(config,
                **kwargs):
    return FeatureRegularization(**kwargs)


@mlconfig.register
def feature_regularization_removal(attack: FeatureRegularization,
                 config,
                 **kwargs):
    scheduler = None
    if "optimizer" in config.keys():
        print("Initializing custom optimizer ...")
        attack.get_classifier()._optimizer = config.optimizer(attack.get_classifier().model.parameters())
    if "scheduler" in config.keys():
        scheduler = config.scheduler(attack.get_classifier().optimizer)
    return attack, attack.remove(scheduler=scheduler, **kwargs)

