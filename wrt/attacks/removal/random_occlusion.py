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
from tqdm import tqdm

from wrt.attacks.attack import RemovalAttack
from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import Loss, PyTorchClassifier
from wrt.defenses import Watermark
from wrt.training import WRTCallback
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.trainer import Trainer

logger = logging.getLogger(__name__)


def generate_occlusion(train_loader: WRTDataLoader, max_occlusion_ratio=0.25, boost_factor=1) -> Tuple[np.ndarray, np.ndarray]:
    max_occlusion_ratio *= 2  # We apply the max_occlusion ratio separately on the width and height of each image.

    # Get the correct shapes.
    example_input = next(iter(train_loader))
    x_shape, y_shape = example_input[0].shape[1:], example_input[1].shape[1:]

    # Allocate space for the occluded inputs and labels.
    n = len(train_loader) * train_loader.batch_size * boost_factor
    x_all, y_all = np.empty((n, *x_shape)), np.empty((n, *y_shape))

    counter = 0
    for b in range(boost_factor):
        for x, y in tqdm(train_loader, f"Loading Occlusions [Round {b+1}/{boost_factor}]"):
            max_width, max_height = max_occlusion_ratio * x.shape[2], max_occlusion_ratio * x.shape[3]
            width, height = np.random.randint(1, max_width), np.random.randint(1, max_height)
            pos_x, pos_y = int(np.random.rand() * x.shape[2]), int(np.random.rand() * x.shape[3])

            occlusion = torch.rand_like(x[0])
            mask = torch.zeros_like(occlusion)
            mask[:, pos_x:pos_x+width, pos_y:pos_y+height] = 1
            occlusion *= mask

            if type(y) == torch.Tensor:
                y = y.cpu().numpy()

            x_all[counter:counter+x.shape[0]] = (x * (1-mask) + occlusion).numpy()
            y_all[counter:counter+x.shape[0]] = y
            counter += x.shape[0]

    return x_all[:counter], y_all[:counter]


class RandomOcclusion(RemovalAttack):
    """
    The attack consists of a whitebox attack
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

            return loss

    def __init__(self, classifier: PyTorchClassifier, **kwargs):
        """
        Create a :class:`.Regularization` instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param l2_decay Strength of the l2 decay.
        """
        super(RandomOcclusion, self).__init__(classifier)

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               train_loader_subset: WRTDataLoader,
               epochs: int,
               repeats: int = 1,
               epsilon: float = 0.0,
               noise_level: float = 0.0,
               max_occlusion_ratio: float = 0.25,
               callbacks: List[WRTCallback] = None,
               wm_data: Tuple[Watermark, np.ndarray, np.ndarray] = None,
               check_every_n_batches: int = None,
               boost_factor: int = 1,
               output_dir=None,
               device="cuda",
               **kwargs):
        """
        The random occlusion defense is similar to adversarial training, but
        generates random occlusion maps instead of adversarial examples.

        :param train_loader Data loader for the training data.
        :param valid_loader Data loader for the validation data
        :param train_loader_subset Data loader for data that should be occluded.
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
        callbacks.append(DebugWRTCallback(lambda: evaluate_test_accuracy(self.get_classifier(), valid_loader,
                                                                         limit_batches=50, verbose=False),
                                          message="test acc",
                                          check_every_n_batches=check_every_n_batches))

        # Allow soft labels.
        old_loss = self.classifier.loss
        self.classifier.loss = self.classifier.loss = RandomOcclusion.CELoss(self.classifier)

        history = {}
        for _ in range(repeats):
            # Generate the occluded images and add the non-normalized images to the training loader.
            x_oc, y_oc = generate_occlusion(train_loader_subset, boost_factor=boost_factor)
            train_loader_pegged = train_loader.add_numpy_data(x_oc, y_oc, boost_factor=1)

            # Fine-Tune the model.
            trainer = Trainer(model=self.get_classifier(), train_loader=train_loader_pegged, valid_loader=valid_loader,
                              device=device, num_epochs=epochs, epsilon=epsilon, callbacks=callbacks, train_all_features=True)
            history: dict = {**history, **trainer.fit()}

        # Reset the model's loss function.
        self.classifier.loss = old_loss

        # Write the history to the training file.
        if output_dir is not None:
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(history, f)

        return history


@mlconfig.register
def random_occlusion_attack(classifier, **kwargs):
    return RandomOcclusion(classifier, **kwargs)


@mlconfig.register
def random_occlusion_removal(attack: RandomOcclusion, config, **kwargs):
    train_loader_subset = config.subset_dataset(source_model=attack.get_classifier(), train=True)
    return attack, attack.remove(train_loader_subset=train_loader_subset, **kwargs)
