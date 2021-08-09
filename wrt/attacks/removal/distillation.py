"""
This module implements Model Distillation attacks
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple

import mlconfig
import torch

from wrt.attacks.attack import RemovalAttack
from wrt.classifiers import Loss
from wrt.training import WRTCallback
from wrt.training.callbacks import EvaluateWmAccCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer

logger = logging.getLogger(__name__)


class ModelDistillation(RemovalAttack):
    """
    Whitebox distillation attack.
    """
    class KDLoss(Loss):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha.
        """

        def __init__(self, classifier, alpha, T):
            super().__init__(classifier)
            self.alpha = alpha
            self.T = T

        def reduce_labels(self):
            return False

        def compute_loss(self, pred, true, x=None):
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            # Expand hard labels.
            if true.ndim == 1:
                true = torch.eye(pred.shape[0])[true].to(true.device)

            labels = torch.argmax(true, dim=1).to(true.device)
            kd_loss = nn.KLDivLoss()(F.log_softmax(pred / self.T, dim=1),
                                     F.softmax(true / self.T, dim=1)) * (self.alpha * self.T * self.T) + \
                      F.cross_entropy(pred, labels) * (1. - self.alpha)

            return kd_loss

    def __init__(self, classifier, alpha=1.0, T=1, device="cuda", **kwargs):
        """
        Create a :class:`.ModelDistillation` instance.

        :param classifier: The teacher classifier.
        :param alpha: float; the parameter for distillation controlling the amount of knowledge used from the teacher
        :param T: float; the temperature for distillation
        """
        super(ModelDistillation, self).__init__(classifier)

        self.classifier = classifier
        self.alpha = alpha
        self.T = T
        self.device = device

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int = 60,
               scheduler: Optional = None,
               callbacks: Optional[List[WRTCallback]] = None,
               wm_data: Optional[Tuple] = None,
               output_dir: Optional[str] = None,
               **kwargs):
        """
        Train a new classifier with the given data with labels predicted by
        the pre-trained classifier

        :param train_loader Loads normalized training data images.
        :param valid_loader Loads normalized testing data images.
        :param epochs Epochs to train with KD loss.
        :param scheduler Scheduler called during training
        :param callbacks List of callbacks to call during training
        :param wm_data Tuple consisting of [Watermark Defense, x_wm, y_wm]
        :param output_dir Output dir to save checkpoints during training.
        :return: An array holding the loss and accuracy
        """
        if callbacks is None:
            callbacks = []

        old_loss = self.classifier.loss
        self.classifier.loss = ModelDistillation.KDLoss(self.classifier, alpha=self.alpha, T=self.T)

        if wm_data:
            print("Found wm data! Adding callback")
            callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data))

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader,
                          valid_loader=valid_loader, scheduler=scheduler, device=self.device,
                          num_epochs=epochs, output_dir=output_dir, callbacks=callbacks)
        trainer.evaluate()
        history = trainer.fit()

        self.classifier.loss = old_loss
        return history

#####################################
### Configs for cmd line scripts. ###
#####################################


@mlconfig.register
def distillation_attack(classifier, **kwargs):
    return ModelDistillation(classifier, **kwargs)


@mlconfig.register
def distillation_removal(attack: ModelDistillation,
                       config,
                       **kwargs):
    optimizer = attack.get_classifier().optimizer
    return attack, attack.remove(scheduler=config.scheduler(optimizer), **kwargs)
