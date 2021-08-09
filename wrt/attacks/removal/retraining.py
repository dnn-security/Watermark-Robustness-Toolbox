"""
This module implements model extraction attacks
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Optional, List, Tuple

import mlconfig
import torch

from wrt.attacks.attack import RemovalAttack
from wrt.classifiers import Loss
from wrt.training.callbacks import EvaluateWmAccCallback, WRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer


class ModelExtraction(RemovalAttack):
    """
    Superclass for the black-box model extraction attacks
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

    def __init__(self, classifier, device="cuda", **kwargs):
        """
        Create a :class:`.ModelExtraction` instance.

        :param classifier: A trained classifier.
        """
        super().__init__(classifier)
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

        # Change to loss over soft labels.
        self.classifier.loss = ModelExtraction.CELoss(self.classifier)

        if wm_data:
            print("Found wm data! Adding callback")
            callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data))

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader,
                          valid_loader=valid_loader, scheduler=scheduler, device=self.device,
                          num_epochs=epochs, output_dir=output_dir, callbacks=callbacks)
        trainer.evaluate()
        return trainer.fit()


class ModelExtractionBB(ModelExtraction):
    """
    Black-box variant of the model extraction attack. Train a new classifier
    from scratch using labels predicted by the pre-trained classifier
    """

    def __init__(self, classifier, **kwargs):
        """
        Create a :class:`ModelExtractionBB` instance.

        :param classifier: A trained classifier.
        :param surrogate_classifier: An untrained classifier
        :param use_logits: bool; whether to use logit labels or argmax labels
        :param epochs: int; number of epochs to train for
        :param batch_size: int; batch size
        """
        super(ModelExtractionBB, self).__init__(classifier)

        self.classifier = classifier


@mlconfig.register
def retraining_attack(classifier, **kwargs):
    return ModelExtractionBB(classifier, **kwargs)


@mlconfig.register
def retraining_removal(attack: ModelExtractionBB,
                       train_loader,
                       valid_loader,
                       output_dir,
                       config,
                       **kwargs):
    optimizer = attack.get_classifier().optimizer
    return attack, attack.remove(train_loader=train_loader, valid_loader=valid_loader,
                                 scheduler=config.scheduler(optimizer),
                                 output_dir=output_dir, **kwargs)
