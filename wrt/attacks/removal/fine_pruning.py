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
This module implements the Fine-Pruning attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List

from tqdm import tqdm

import numpy as np

from wrt.classifiers.classifier import Classifier
from wrt.attacks.attack import RemovalAttack
from wrt.exceptions import ClassifierError
from wrt.config import WRT_NUMPY_DTYPE
from wrt.training.callbacks import DebugWRTCallback, WRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer
from wrt.training.utils import convert_to_augmented_data_generator, convert_to_data_generator, sample_dataset
import mlconfig

logger = logging.getLogger(__name__)


# original FineTuningAttack moved to FTALAttack
class FinePruningAttack(RemovalAttack):
    """
    The attack consists of a fine-pruning attack on a watermarked classifier
    """

    attack_params = RemovalAttack.attack_params + [
        "epochs",
        "batch_size",
    ]

    def __init__(self, classifier, ratio, layer_index, mask_function, batch_size=32, lr=0.001, epochs=1, device="cuda",
                 **kwargs):
        """
        Create a :class:`.RemovalAttack` instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param ratio: float; pruning ratio
        :param layer_index: int; index of the layer to prune
        :param mask_function: function; must have exactly two parameters: the first is the
                              model, and the second is an np.ndarray of 0's and 1's, indicating
                              if the neuron activations in the specified layer should be kept or zeroed.
                              This function is called exactly once in the remove() call,
                              and must modify the classifier in-place to mask the given layer.
        :param lr: Learning rate for the fine-tuning process
        :type lr: `float`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(FinePruningAttack, self).__init__(classifier)

        if not isinstance(classifier, Classifier):
            raise ClassifierError(self.__class__, [Classifier], classifier)

        kwargs = {
            "epochs": epochs,
            "batch_size": batch_size,
        }
        FinePruningAttack.set_params(self, **kwargs)

        self.ratio = ratio
        self.layer_index = layer_index
        self.mask_function = mask_function
        self.lr = lr
        self.device = device

    @classmethod
    def is_valid_classifier_type(cls, classifier):
        """
        Checks whether the classifier provided is a classifier which this class can perform an attack on
        :param classifier:
        :return:
        """
        return True if isinstance(classifier, Classifier) else False

    def __get_avg_activations(self, x):
        num_batches = int(np.ceil(x.shape[0] // self.batch_size))
        activations = []
        loop = tqdm(range(num_batches))
        for batch in loop:
            loop.set_description(f"Determining average activations: {batch + 1}/{num_batches}")
            activation = \
                self.classifier.get_all_activations(x[batch * self.batch_size: (batch + 1) * self.batch_size])[
                    self.layer_index]
            activation = self.classifier.functional.numpy(activation)
            activations.append(activation)
        return np.average(np.vstack(activations), axis=0)

    def __do_pruning(self, x_normalized):
        avg_activations = self.__get_avg_activations(x_normalized)
        activation_shape = avg_activations.shape

        prune_amount = int(np.floor(np.prod(activation_shape) * self.ratio))

        avg_activations = avg_activations.reshape(-1)
        mask_indices = np.argpartition(avg_activations, prune_amount)[:prune_amount]

        mask = np.ones(np.prod(activation_shape), dtype=WRT_NUMPY_DTYPE)
        mask[mask_indices] = 0
        mask = mask.reshape(activation_shape)

        self.mask_function(self.classifier.model, mask)

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int,
               patience: int = 5,
               check_every_n_batches: int = None,
               wm_data=None,
               output_dir: str = None,
               callbacks: List[WRTCallback] = None,
               **kwargs):
        """
        Performs the fine-pruning attack.

        """
        if callbacks is None:
            callbacks = []

        # Sample n data elements (for compatiblity with the old interface)
        x_normalized, y = collect_n_samples(n=50000,
                                            data_loader=train_loader,
                                            has_labels=True,
                                            verbose=False)
        self.__do_pruning(x_normalized)

        original_lr = self.classifier.lr
        self.classifier.lr = self.lr

        if wm_data:
            defense, x_wm, y_wm = wm_data
            callbacks.append(DebugWRTCallback(lambda: defense.verify(x_wm, y_wm, classifier=self.get_classifier())[0],
                                              message="wm_acc",
                                              check_every_n_batches=check_every_n_batches))

        # Fine-Tuning phase.
        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=self.device, num_epochs=epochs, output_dir=output_dir, callbacks=callbacks)

        history = trainer.fit()
        self.classifier.lr = original_lr

        return history

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param epochs: The epochs for fine-tuning.
        :type epochs: `int`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(FinePruningAttack, self).set_params(**kwargs)
        return True

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
def fine_pruning_attack(classifier, layer_name: str, layer_index: int, **kwargs):
    import torch

    def create_mask_function(layer_name):
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            device = torch.device("cuda:{}".format(cuda_idx))

        class MaskedLayer(torch.nn.Module):

            def __init__(self, layer, mask):
                super(MaskedLayer, self).__init__()

                self.layer = layer
                self.mask = mask

            def forward(self, x):
                x = self.layer(x)
                if isinstance(x, list):
                    x = x[:-1] + [x[-1] * self.mask]
                else:
                    x = x * self.mask
                return x

        def mask_function(model, mask):
            mask = torch.from_numpy(mask).to(device)
            layer = getattr(model, layer_name)
            setattr(model, layer_name, MaskedLayer(layer, mask))

        return mask_function

    # prune the last convolutional layer
    mask_function = create_mask_function(layer_name)

    return FinePruningAttack(classifier, mask_function=mask_function, layer_index=layer_index, **kwargs)


@mlconfig.register
def fine_pruning_removal(attack: FinePruningAttack,
                         train_loader,
                         **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)
