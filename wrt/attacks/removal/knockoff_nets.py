"""
This module implements the Knockoff attack.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List, Tuple

import mlconfig
import numpy as np
import torch

from wrt.attacks.attack import RemovalAttack
from wrt.classifiers import Loss
from wrt.defenses import Watermark
from wrt.training import WRTCallback
from wrt.training.callbacks import EvaluateWmAccCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.datasets.cifar10 import cifar_classes
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer


class KnockoffNets(RemovalAttack):
    """
    Superclass for the black-box and white-box model extraction attacks
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

    def __init__(self, classifier, num_classes: int, approach="random_selection"):
        """
        Create a :class:`.ModelExtraction` instance.

        :param classifier: A trained classifier.
        :param use_logits: bool; whether to use logit labels or argmax labels
        :param augment: bool; if True, then perform data augmentation
        :param epochs: int; number of epochs to train for
        :param batch_size: int; batch size
        """
        super(KnockoffNets, self).__init__(classifier)
        self.num_classes = num_classes
        self.approach = approach

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs,
               scheduler: torch.optim.lr_scheduler = None,
               output_dir: str = None,
               callbacks: List[WRTCallback] = None,
               wm_data: Tuple[Watermark, np.ndarray, np.ndarray] = None,
               device: str ="cuda",
               **kwargs):
        """
        Train a new classifier with the given data with labels predicted by
        the pre-trained source model on a transfer set.

        :param train_loader The loader data from the transfer set.
        :param valid_loader Loads the validation data.
        :param epochs Number of training epochs.
        :param scheduler Scheduler for adjusting lr
        :param output_dir Output directory to store intermediate data.
        :param callbacks List of callbacks during training.
        :param wm_data Watermark data to verify the watermark accuracy during training.
        :param device The device to train on
        """
        if callbacks is None:
            callbacks = []

        assert self.approach == "random_selection", "Only the random selection approach is implemented for Knockoff."
        assert scheduler is not None, "Knockoff requires a scheduler"
        assert output_dir is not None, "Knockoff requires an output dir"

        # Change to loss over soft labels.
        self.classifier.loss = KnockoffNets.CELoss(self.classifier)

        if wm_data:
            print("Found wm data! Adding callback")
            callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data))

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader,
                          valid_loader=valid_loader, scheduler=scheduler, device=device,
                          num_epochs=epochs, output_dir=output_dir, callbacks=callbacks)
        trainer.evaluate()
        return trainer.fit()

    def predict(self, x, **kwargs):
        """
        Perform prediction using the classifier.

        :param x: Test set.
        :type x: `np.ndarray`
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :type kwargs: `dict`
        :return: Predictions for test set.
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)


@mlconfig.register
def knockoff_attack(config,
                **kwargs):
    return KnockoffNets(**kwargs)


@mlconfig.register
def knockoff_removal(attack: KnockoffNets,
                     source_model,
                     train_loader,  # Replace the train_loader with the transfer set.
                     config,
                 **kwargs):
    transfer_dataset = config.transfer_dataset(source_model=source_model)
    optimizer = attack.get_classifier().optimizer
    return attack, attack.remove(train_loader=transfer_dataset, scheduler=config.scheduler(optimizer), **kwargs)

