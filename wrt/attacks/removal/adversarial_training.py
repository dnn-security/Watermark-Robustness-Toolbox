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
This module implements adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Union

import mlconfig
from tqdm import tqdm
import numpy as np

from wrt.art_classes import ProjectedGradientDescent, FastGradientMethod
from wrt.attacks.attack import RemovalAttack
from wrt.classifiers import PyTorchClassifier
from wrt.defenses.utils import NormalizingPreprocessor
from wrt.training import WRTCallback
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer

logger = logging.getLogger(__name__)


class AdversarialTraining(RemovalAttack):
    """
    The attack consists of a Adversarial training on a given model.
    """

    def __init__(
            self,
            classifier: PyTorchClassifier,
            eps: Union[List[float], float],
            method: str = "pgd",
            eps_step: float = 0.01,
            norm: float = np.inf,
            max_iter: int = 40,
            **kwargs
    ):
        """
        Create a :class:`.AdversarialTraining` instance.

        :param classifier: A trained classifier.
        :param norm: Type of norm to use (supporting np.inf, 1 or 2).
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations for searching.
        :type batch_size: `int`
        """
        super(AdversarialTraining, self).__init__(classifier)

        assert (method in ["pgd", "fgm"]), print("Method has to be 'pgd' or 'fgm'.")

        if not type(eps) is list:
            eps = [eps]

        self.norm = norm
        self.method = method
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.device = self.classifier.device

    def generate_adversarial_examples(self, attack, train_loader: WRTDataLoader, n_max: int, target_label=None,
                                      batch_size: int = 32):
        """ Generate the adversarial examples.
        This method expects a train loader that returns normalized images between [0,1].

        :param attack The adversarial attack to generate the adversarial examples
        :param train_loader Training data loader that loads normalized images.
        :param n_max Maximum number of adversarial examples to generate
        :param batch_size Batch size for the generation of adversarial examples.
        """
        classifier = self.get_classifier()

        # Collect input data.
        x_train, y_train = collect_n_samples(n=n_max, data_loader=train_loader, verbose=False)
        x_train = train_loader.unnormalize(x_train)

        if len(y_train.shape) == 1:  # Convert from hard to soft labels.
            y_train = np.eye(classifier.nb_classes())[y_train]

        if target_label is not None:
            target_label = x_train.shape[0]*[target_label]
            target_label = np.eye(classifier.nb_classes())[target_label]

        print(f"Generating adversarial examples from input shape '{x_train.shape}'. ")
        preprocessor = NormalizingPreprocessor(mean=train_loader.mean, std=train_loader.std)
        classifier.add_preprocessor(preprocessor, "adv_training_preprocessor")
        x_adv = attack.generate(x_train.astype(np.float32), y=target_label, batch_size=batch_size)
        classifier.remove_preprocessor("adv_training_preprocessor")
        return x_adv, y_train

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int,
               n_max: int = 1000,
               boost_factor: int = 1,
               check_every_n_batches: int = None,
               pgd_batch_size: int = 32,
               target_label = None,
               output_dir: str = None,
               scheduler=None,
               wm_data=None,
               callbacks: List[WRTCallback] = None,
               **kwargs):
        """ Perform adversarial training on the model.

        :param train_loader: Data loader for the training dataset.
        :param valid_loader: Loads normalized testing data.
        :param epochs: Number of total epochs for training.
        :param method Either "pgd" or "fgm"
        :param boost_factor Number of times to repeat the adversarial examples.
        :param n_max Number of adversarial examples to generate.
        :param scheduler Scheduler during adversarial training.
        :param check_every_n_batches: Validate watermark accuracy every n batches.
        :param wm_data: Watermark data. Consists of a tuple [Defense, x_wm, y_wm]
        :param callbacks: Callbacks during training
        :param output_dir: (optional) The output directory to store intermediate results
        :param pgd_batch_size Batch size to generate pgd adversarial examples.
        """
        print(f"Targeted? {target_label is not None}")
        if callbacks is None:
            callbacks = []

        classifier = self.get_classifier()

        # The adversarial attack.
        if self.method == "pgd":
            attacks = [ProjectedGradientDescent(classifier=self.classifier,
                                                norm=self.norm,
                                                eps=eps,
                                                eps_step=self.eps_step,
                                                max_iter=self.max_iter,
                                                batch_size=pgd_batch_size) for eps in self.eps]
        else:
            attacks = [FastGradientMethod(classifier=self.classifier,
                                          norm=self.norm,
                                          eps=eps,
                                          targeted=target_label is not None,
                                          eps_step=self.eps_step,
                                          batch_size=pgd_batch_size) for eps in self.eps]

        # Log the watermarking accuracy during training.
        if wm_data is not None:
            defense, x_wm, y_wm = wm_data
            callbacks.append(DebugWRTCallback(lambda: defense.verify(x_wm, y_wm, classifier=classifier)[0],
                                              message="wm_acc",
                                              check_every_n_batches=check_every_n_batches))

        for e in range(epochs):
            print(f"Epoch {e}/{epochs}")
            x_adv, y_adv = None, None
            for attack in attacks:
                x, y = self.generate_adversarial_examples(attack, train_loader, target_label=target_label, n_max=n_max, batch_size=32)
                if x_adv is None:
                    x_adv, y_adv = x, y
                else:
                    x_adv, y_adv = np.vstack((x_adv, x)), np.vstack((y_adv, y))

            # Add the normalized data to the training loader.
            train_loader_pegged = train_loader.add_numpy_data(x_adv, y_adv.astype(np.float32),
                                                              boost_factor=boost_factor)

            trainer = Trainer(model=self.get_classifier(), train_loader=train_loader_pegged, valid_loader=valid_loader,
                              device=self.get_classifier().device, num_epochs=1, scheduler=scheduler, callbacks=callbacks,
                              disable_progress=False)
            trainer.fit()
        return None

    def predict(self, x, **kwargs):
        """
        Perform prediction using the watermarked classifier.

        :param x: Test set.
        :type x: `np.ndarray`
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :type kwargs: `dict`
        :return: Predictions for test set.
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)


@mlconfig.register
def adversarial_training_attack(classifier, **kwargs):
    return AdversarialTraining(classifier, **kwargs)


@mlconfig.register
def adversarial_training_removal(attack: AdversarialTraining, train_loader, config, **kwargs):
    # Add a scheduler if found (for adversarial model extraction)
    scheduler = None
    if "scheduler" in config.keys():
        scheduler = config.scheduler(attack.get_classifier().optimizer)
    return attack, attack.remove(train_loader=train_loader, scheduler=scheduler, **kwargs)
