"""
This module implements the Neural Cleanse patching attacks
http://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Tuple, Optional

import mlconfig
import numpy as np

from wrt.attacks.attack import RemovalAttack
from wrt.attacks.extraction import NeuralCleanse, NeuralCleansePartial, NeuralCleanseMulti
from wrt.attacks.util import soft_argmax, to_onehot
from wrt.config import WRT_NUMPY_DTYPE
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer

logger = logging.getLogger(__name__)


class NeuralCleanseUnlearning(RemovalAttack):
    """  Implement the Neural Cleanse unlearning attack
    """
    # cache the results of extracts so we don't have to keep computing them for the same
    # kwargs passed to Neural Cleanse
    last_extract_kwargs = {}
    last_triggers = {}

    def __init__(self,
                 classifier,
                 triggers=None,
                 extract=True,
                 extract_kwargs: dict = None,
                 **kwargs):
        """
        Create a :class:`.NeuralCleanseUnlearning` instance.

        :param classifier: A trained classifier.
        :param extract: bool; whether or not to run Neural Cleanse extraction
                              from scratch. If False, then triggers must be
                              provided. If True, then the triggers are ignored
        :param triggers: list of tuples returned by NeuralCleanse extract
        :param extract_kwargs: dict; keyword arguments passed to the initialization
                               of NeuralCleanse if extract is True
        :param epochs: int; number of epochs to train for
        :param batch_size: int; batch size
        """
        super(NeuralCleanseUnlearning, self).__init__(classifier)

        if not extract and triggers is None:
            raise ValueError("Error: triggers must be provided if extraction is not performed")

        self.extract = extract
        self.triggers = triggers
        self.extract_kwargs = extract_kwargs if extract_kwargs is not None else {}

    def get_triggers(self, x, y):
        """
        Reverse-engineer the triggers using Neural Cleanse, given training data.

        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :return: list of (int, np.ndarray) tuples, where each tuple represents an
                 infected target index and the corresponding trigger
        """
        neural_cleanse = NeuralCleanse(self.classifier, **self.extract_kwargs)
        triggers = neural_cleanse.extract(x, y)

        # sort by L1 norm
        return sorted(triggers, key=lambda t: np.sum(np.abs(t[1])))

    def get_unlearning_dataset(self, x: np.ndarray, y: np.ndarray, triggers: List[Tuple[int, np.ndarray]]):
        """
        Get a dataset for unlearning the backdoor given the reverse-engineered triggers.

        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :param triggers: list of (int, np.ndarray) tuples; the reverse-engineered triggers
        :return: (np.ndarray, np.ndarray); dataset and labels
        """
        trigger = triggers[0][1]

        trigger_indices = np.random.choice(x.shape[0], x.shape[0] // 5, replace=False)
        x_trigger = x[trigger_indices]
        x_trigger = x_trigger + trigger
        x_trigger[x_trigger > 1] = 1
        y_trigger = y[trigger_indices]
        return np.vstack([x, x_trigger]).astype(np.float32), np.vstack([y, y_trigger])

    def remove(self,
               train_loader: WRTDataLoader,
               train_loader_subset: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int,
               source_classes: Optional[List[int]] = None,
               target_classes: Optional[List[int]] = None,
               n_samples: int = 10000,
               boost_factor: int = 1,
               wm_data=None,
               batch_size: int = 64,
               check_wm_acc_every_n_batches: int = None,
               **kwargs):
        """ Perform neural cleanse on a model.

        :param train_loader: Data loader for the training dataset.
        :param valid_loader: Loads normalized testing data.
        :param epochs Number of epochs for training.
        :param source_classes Source classes to inspect for infection.
        :param target_classes Target classes to inspect for infection.
        :param n_samples Get average activations from this number of samples.
        :param batch_size The batch size for training.
        :param check_wm_acc_every_n_batches Check callbacks every n batches.
        :param wm_data Watermarking data consists of a tuple of [defense, x_wm, y_wm]
        :param device Which device to train on.
        """
        callbacks = []

        # Randomly sample the training data.
        x, y = collect_n_samples(n=n_samples, data_loader=train_loader_subset, has_labels=True)

        if self.extract:
            print("Extracting trigger patterns between pairs of classes ...")
            triggers = self.get_triggers(x, y)
        else:
            triggers = self.triggers

        x_unlearn, y_unlearn = self.get_unlearning_dataset(x, y, triggers)

        if wm_data is not None:
            defense, x_wm, y_wm = wm_data
            callbacks.append(DebugWRTCallback(lambda: defense.verify(x_wm, y_wm, classifier=self.get_classifier())[0],
                                              message="wm_acc",
                                              check_every_n_batches=check_wm_acc_every_n_batches))

        x_unlearn = train_loader.unnormalize(x_unlearn)
        train_loader_pegged = train_loader.add_numpy_data(x_unlearn, y_unlearn, boost_factor=boost_factor)
        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader_pegged, valid_loader=valid_loader,
                          device=self.classifier.device, num_epochs=epochs, callbacks=callbacks)
        return trainer.fit()


class NeuralCleansePartialUnlearning(NeuralCleanseUnlearning):

    def get_triggers(self, x, y):
        """
        Reverse-engineer the triggers using Neural Cleanse, given training data
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :return: list of (int, int, np.ndarray) tuples, where each tuple represents
                 a source index, a target index, and the corresponding trigger
        """
        neural_cleanse = NeuralCleansePartial(self.classifier, **{**self.extract_kwargs})
        return sorted(neural_cleanse.extract(x, y), key=lambda t: np.sum(np.abs(t[2])))

    def get_unlearning_dataset(self, x: np.ndarray, y: np.ndarray, triggers):
        """
        Get a dataset for unlearning the backdoor given the reverse-engineered triggers
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :param triggers: list of (int, int, np.ndarray) tuples; the reverse-engineered triggers
        :return: (np.ndarray, np.ndarray); dataset and labels
        """
        source_index, _, trigger = triggers[0]

        # Convert to one hot if necessary.
        y = to_onehot(y, self.classifier.nb_classes())

        x_source_indices = np.argmax(y, axis=1) == source_index
        x_source = x[x_source_indices]
        x_trigger = x_source + trigger
        x_trigger[x_trigger > 1] = 1
        y_trigger = y[x_source_indices]
        return np.vstack([x, x_trigger]).astype(np.float32), np.vstack([y, y_trigger])


@mlconfig.register
def neural_cleanse_partial_unlearning_attack(classifier, **kwargs):
    return NeuralCleansePartialUnlearning(classifier, **kwargs)


@mlconfig.register
def neural_cleanse_partial_unlearning_removal(attack: NeuralCleansePartialUnlearning, train_loader, config, **kwargs):
    train_loader_subset = config.subset_dataset(train=True)
    return attack, attack.remove(train_loader=train_loader, train_loader_subset=train_loader_subset, **kwargs)


class NeuralCleanseMultiUnlearning(NeuralCleanseUnlearning):
    def get_triggers(self, x, y):
        """
        Reverse-engineer the triggers using Neural Cleanse, given training data
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :return: list of (int, np.ndarray) tuples, where each tuple represents an
                 infected target index and the corresponding trigger
        """
        neural_cleanse = NeuralCleanseMulti(self.classifier, **{**self.extract_kwargs})
        return sorted(neural_cleanse.extract(x, y), key=lambda t: np.sum(np.abs(t[1])))

    def get_unlearning_dataset(self, x, y, triggers):
        """
        Get a dataset for unlearning the backdoor given the reverse-engineered triggers
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :param triggers: list of (int, np.ndarray) tuples; the reverse-engineered triggers
        :return: (np.ndarray, np.ndarray); dataset and labels
        """
        x_train, y_train = x, y
        num_samples_per_trigger = x.shape[0] // self.classifier.nb_classes()
        for _, trigger in triggers:
            trigger_indices = np.random.choice(x.shape[0], num_samples_per_trigger, replace=False)
            x_trigger = x[trigger_indices]
            x_trigger = x_trigger + trigger
            x_trigger[x_trigger > 1] = 1
            y_trigger = y[trigger_indices]
            x_train, y_train = np.vstack([x_train, x_trigger]), np.vstack([y_train, y_trigger])

        return x_train.astype(np.float32), y_train


class NeuralCleansePruning(RemovalAttack):
    """
    Implement the Neural Cleanse pruning attack
    """

    attack_params = RemovalAttack.attack_params + ["batch_size"]

    # cache the results of extracts so we don't have to keep computing them for the same
    # kwargs passed to Neural Cleanse
    last_extract_kwargs = {}
    last_triggers = {}

    def __init__(self, classifier, ratio, layer_index, mask_function, extract=False, triggers=None, extract_kwargs=None,
                 batch_size=32, **kwargs):
        """
        Create a :class:`.NeuralCleansePruning` instance.

        :param classifier: A trained classifier.
        :param ratio: float; pruning ratio
        :param layer_index: int; index of the layer to prune
        :param mask_function: function; must have exactly two parameters: the first is the
                              model, and the second is an np.ndarray of 0's and 1's, indicating
                              if the neuron activations in the specified layer should be kept or zeroed.
                              This function is called exactly once in the remove() call,
                              and must modify the classifier in-place to mask the given layer.
        :param extract: bool; whether or not to run Neural Cleanse extraction
                              from scratch. If False, then triggers must be
                              provided. If True, then the triggers are ignored
        :param triggers: list of tuples returned by NeuralCleanse extract
        :param extract_kwargs: dict; keyword arguments passed to the initialization
                               of NeuralCleanse if extract is True
        :param batch_size: int; batch size
        """
        super(NeuralCleansePruning, self).__init__(classifier)

        if not extract and triggers is None:
            raise ValueError("Error: trigger must be provided if extraction is not performed")

        self.ratio = ratio
        self.layer_index = layer_index
        self.mask_function = mask_function

        self.extract = extract
        self.triggers = triggers
        self.extract_kwargs = extract_kwargs if extract_kwargs is not None else {}
        self.batch_size = batch_size

    def get_triggers(self, x, y):
        """
        Reverse-engineer the triggers using Neural Cleanse, given training data
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :return: list of (int, np.ndarray) tuples, where each tuple represents an
                 infected target index and the corresponding trigger
        """
        neural_cleanse = NeuralCleanse(self.classifier, **{**self.extract_kwargs})
        triggers = neural_cleanse.extract(x, y)

        # sort by L1 norm
        return sorted(triggers, key=lambda t: np.sum(np.abs(t[1])))

    def get_trigger_set(self, x, y, triggers):
        """
        Return reconstructed backdoor inputs into the model using the given
        reverse-engineered triggers
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding data
        :param triggers: list of (int, np.ndarray) tuples; reverse-engineered triggers
        :return: np.ndarray
        """
        trigger = triggers[0][1]

        x_trigger = x + trigger
        x_trigger[x_trigger > 1] = 1
        return x_trigger

    def __activation_difference(self, x, x_trigger):
        """
        Return the average difference between activations of clean inputs
        and activations of adversarial inputs
        :return: np.ndarray
        """
        clean_activations = []
        num_clean_batches = int(np.ceil(x.shape[0] // self.batch_size))
        for batch in range(num_clean_batches):
            activation = self.classifier.get_all_activations(x[batch * self.batch_size: (batch + 1) * self.batch_size])[self.layer_index]
            activation = self.classifier.functional.numpy(activation)
            clean_activations.append(np.mean(activation, axis=0))

        num_adv_batches = int(np.ceil(x_trigger.shape[0] // self.batch_size))
        adv_activations = []
        for batch in range(num_adv_batches):
            activation = self.classifier.get_all_activations(x_trigger[batch * self.batch_size: (batch + 1) * self.batch_size])[self.layer_index]
            activation = self.classifier.functional.numpy(activation)
            adv_activations.append(np.mean(activation, axis=0))

        avg_clean_activation = np.average(np.array(clean_activations), axis=0)
        avg_adv_activation = np.average(np.array(adv_activations), axis=0)

        return avg_adv_activation - avg_clean_activation

    def remove(self,
               train_loader: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int,
               n_samples: int = 10000,
               output_dir: str = None,
               batch_size: int = 64,
               check_every_n_batches: int = None,
               wm_data=None,
               device: str = "cuda",
               **kwargs):
        """ Perform neural cleanse on a model.

        :param train_loader: Data loader for the training dataset.
        :param valid_loader: Loads normalized testing data.
        :param epochs Number of epochs for training.
        :param output_dir: (optional) The output directory to store intermediate results
        :param batch_size The batch size for training.
        :param check_every_n_batches Check callbacks every n batches.
        :param wm_data Watermarking data consists of a tuple of [defense, x_wm, y_wm]
        :param device Which device to train on.
        """
        x, y = collect_n_samples(n_samples, data_loader=train_loader, has_labels=True)

        if self.extract:
            triggers = self.get_triggers(x, y)
        else:
            triggers = self.triggers

        x_trigger = self.get_trigger_set(x, y, triggers)

        diffs = self.__activation_difference(x, x_trigger)
        diff_shape = diffs.shape

        prune_amount = int(np.floor(np.prod(diff_shape) * self.ratio))

        diffs = diffs.reshape(-1)
        mask_indices = np.argpartition(diffs, -prune_amount)[-prune_amount:]

        mask = np.ones(np.prod(diff_shape), dtype=WRT_NUMPY_DTYPE)
        mask[mask_indices] = 0
        mask = mask.reshape(diff_shape)

        self.mask_function(self.classifier.model, mask)


class NeuralCleansePartialPruning(NeuralCleansePruning):

    def get_triggers(self, x, y):
        """
        Reverse-engineer the triggers using Neural Cleanse, given training data
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :return: list of (int, int, np.ndarray) tuples, where each tuple represents
                 a source index, a target index, and the corresponding trigger
        """
        neural_cleanse = NeuralCleansePartial(self.classifier, **{**self.extract_kwargs})
        return sorted(neural_cleanse.extract(x, y), key=lambda t: np.sum(np.abs(t[2])))

    def get_trigger_set(self, x, y, triggers):
        """
        Return reconstructed backdoor inputs into the model using the given
        reverse-engineered triggers
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding data
        :param triggers: list of (int, int, np.ndarray) tuples; reverse-engineered triggers
        :return: np.ndarray
        """
        source_index, _, trigger = triggers[0]

        x_source_indices = soft_argmax(y, 1) == source_index
        x_source = x[x_source_indices]

        x_trigger = x_source + trigger
        x_trigger[x_trigger > 1] = 1
        return x_trigger


@mlconfig.register
def neural_cleanse_partial_pruning_attack(classifier, layer_name, layer_index, config, **kwargs):
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
                    return x[:-1] + [x[-1] * self.mask]
                else:
                    return x * self.mask

        def mask_function(model, mask):
            mask = torch.from_numpy(mask).to(device)
            layer = getattr(model, layer_name)
            setattr(model, layer_name, MaskedLayer(layer, mask))

        return mask_function

    # prune the last convolutional layer
    mask_function = create_mask_function(layer_name)

    return NeuralCleansePartialPruning(classifier, mask_function=mask_function, layer_index=layer_index,
                                       **kwargs)


@mlconfig.register
def neural_cleanse_partial_pruning_removal(attack: NeuralCleansePartialPruning, train_loader, **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)


class NeuralCleanseMultiPruning(NeuralCleansePruning):

    def get_triggers(self, x, y):
        """
        Reverse-engineer the triggers using Neural Cleanse, given training data
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :return: list of (int, np.ndarray) tuples, where each tuple represents an
                 infected target index and the corresponding trigger
        """
        neural_cleanse = NeuralCleanseMulti(self.classifier, **{**self.extract_kwargs})
        return sorted(neural_cleanse.extract(x, y), key=lambda t: np.sum(np.abs(t[1])))

    def get_trigger_set(self, x, y, triggers):
        """
        Return reconstructed backdoor inputs into the model using the given
        reverse-engineered triggers
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding data
        :param triggers: list of (int, np.ndarray) tuples; reverse-engineered triggers
        :return: np.ndarray
        """
        trigger_set = []
        for _, trigger in triggers:
            num_samples_per_trigger = x.shape[0] // self.classifier.nb_classes()
            trigger_indices = np.random.choice(x.shape[0], num_samples_per_trigger, replace=False)
            x_trigger = x[trigger_indices]
            x_trigger = x_trigger + trigger
            x_trigger[x_trigger > 1] = 1
            trigger_set.append(x_trigger)

        return np.vstack(trigger_set).astype(np.float32)
