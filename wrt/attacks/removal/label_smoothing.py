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

from wrt.attacks.attack import RemovalAttack
from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import Loss
from wrt.training.callbacks import EvaluateWmAccCallback, WRTCallback, DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer
import torch
import numpy as np
from tqdm import tqdm


class LabelSmoothingAttack(RemovalAttack):
    """
    The attack consists of fine-tuning a watermarked classifier on more target data with label smoothing.
    (Examplary Implementation)
    """

    class CELoss(Loss):
        """
        Cross-entropy loss with support for soft labels, since PyTorch cross-entropy loss
        requires labels to be argmax-encoded
        """
        def __init__(self, classifier, epsilon=0.0, cluster_to_smooth_label=None, top_k=None):
            super().__init__(classifier)
            self.top_k = top_k
            self.epsilon = epsilon

            if cluster_to_smooth_label is not None:
                self.cluster_to_smooth_label = torch.stack(cluster_to_smooth_label)

        def reduce_labels(self):
            return False

        def compute_loss(self, pred, true, x=None):

            if true.dim() == 1:
                true = torch.eye(pred.shape[1])[true].to(true.device)

            # We keep the probability for the top_1 example, but remove the next k-1 examples.
            # Get the top-k values and indices.
            top_vals, top_idx = torch.topk(true, k=self.top_k, dim=1)
            # top_idx[:, 1:] += 20

            # Top-1 scaled label.
            t1 = torch.eye(pred.shape[1])[top_idx[:, 0]].to(true.device)
            t1 *= top_vals[:, 0].view(top_vals.shape[0], 1)

            # Remove top-2 to top-k labels.
            t2 = torch.eye(pred.shape[1])[top_idx[:, 1]].to(true.device) * top_vals[:, 1].view(top_vals.shape[0], 1)
            normalizer = top_vals[:, 1].view(top_vals.shape[0], 1)
            for i in range(2, self.top_k):
                t2 += torch.eye(pred.shape[1])[top_idx[:, i]].to(true.device) * top_vals[:, i].view(top_vals.shape[0], 1)
                normalizer += top_vals[:, i].view(top_vals.shape[0], 1)
            true = (true - t2) / (1-normalizer)

            ''''# Top-3 to top-k scaled labels
            scales = (1-top_vals[:, 0])/(self.top_k-2)
            t2 = torch.eye(pred.shape[1])[top_idx[:, 2]].to(true.device)
            for i in range(3, self.top_k):
                t2 += torch.eye(pred.shape[1])[top_idx[:, i]].to(true.device)
            t2 *= scales.view(top_vals.shape[0], 1)
            true = t1 + t2'''

            #true = (1-self.epsilon) * true + self.epsilon * self.cluster_to_smooth_label[true.argmax(1)]

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
        super(LabelSmoothingAttack, self).__init__(classifier)
        self.num_classes = num_classes

    def get_class_similarity(self, train_loader):
        pass

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epsilon: float = 0.1,
               epochs: int = 1,
               top_k: int = None,
               check_every_n_batches: int = None,
               wm_data=None,
               callbacks: List[WRTCallback] = None,
               device="cuda",
               **kwargs):
        """Attempt to remove the watermark
        :param train_loader The loader for the training data.
        :param valid_loader Test data loader
        :param epsilon Epsilon for the label smoothing
        :param epochs Number of epochs to fine-tune.
        :param check_every_n_batches Callback for evaluating the watermark accuracy every n batches.
        :param wm_data Watermark data.
        :param callbacks Callbacks during training.
        :param device Device to train on.
        """
        if callbacks is None:
            callbacks = []

        # Create nb_classes many tensors with random choices.
        '''def create_random_maps():
            print("Creating random maps!")
            tensors = []
            for _ in tqdm(range(self.classifier.nb_classes())):
                idx = np.random.choice(self.classifier.nb_classes(), size=top_k-1, replace=False) #np.arange(top_k-1)#
                t = torch.eye(self.classifier.nb_classes())[idx[0]].to(self.classifier.device)
                for i in idx[1:]:
                    t += torch.eye(self.classifier.nb_classes())[i].to(self.classifier.device)
                t /= len(idx)
                tensors.append(t)
            return tensors'''

        # Change to loss over soft labels.
        #tensors = create_random_maps()
        # self.classifier.loss = LabelSmoothingAttack.CELoss(self.classifier, epsilon=epsilon, top_k=top_k, cluster_to_smooth_label=tensors)
        self.classifier.loss = LabelSmoothingAttack.CELoss(self.classifier, epsilon=epsilon, top_k=top_k)
        self.classifier.lr = self.classifier.lr * 10

        if wm_data:
            print("Found wm data! Adding callback")
            callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data, log_after_n_batches=check_every_n_batches))

        callbacks.append(DebugWRTCallback(lambda: evaluate_test_accuracy(self.get_classifier(), valid_loader,
                                                                         limit_batches=50, verbose=False),
                                          message="test acc",
                                          check_every_n_batches=check_every_n_batches))

        '''def reset_loss():
            self.classifier.loss = LabelSmoothingAttack.CELoss(self.classifier, epsilon=epsilon, top_k=top_k, cluster_to_smooth_label=create_random_maps())
            return 1

        callbacks.append(DebugWRTCallback(lambda: reset_loss(),
                                          message="reset loss"))'''

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=device, num_epochs=epochs, callbacks=callbacks, epsilon=epsilon)
        trainer.evaluate()
        history = trainer.fit()

        self.classifier.lr = self.classifier.lr / 10
        return history


#################### Configuration callable through mlconfig

@mlconfig.register
def label_smoothing_attack(**kwargs):
    return LabelSmoothingAttack(**kwargs)


@mlconfig.register
def label_smoothing_removal(attack: LabelSmoothingAttack, **kwargs):
    return attack, attack.remove(**kwargs)
