"""
This module implements the Neural Network Laundering attack
https://arxiv.org/pdf/2004.11368.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import pickle
from typing import List

import mlconfig
import numpy as np
import torch

from wrt.attacks.attack import RemovalAttack
from wrt.attacks.extraction import NeuralCleanse, NeuralCleansePartial, NeuralCleanseMulti
from wrt.attacks.util import soft_argmax, to_onehot
from wrt.config import WRT_DATA_PATH
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets import WRTDataLoader
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer
from wrt.training.utils import sample_dataset

logger = logging.getLogger(__name__)


class Laundering(RemovalAttack):
    """
    Implement the Neural Laundering Attack
    """

    # cache the results of extracts so we don't have to keep computing them for the same
    # kwargs passed to Neural Cleanse
    last_extract_kwargs = {}
    last_triggers = {}

    def __init__(self, classifier,
                 reset_function,
                 extract=False,
                 triggers=None,
                 extract_kwargs=None,
                 n_samples: int = 10000,
                 dt=2,
                 ct=1,
                 **kwargs):
        """
        Create a :class:`.Laundering` instance.

        :param classifier: A trained classifier.
        :param reset_function: function; must take in three parameters: the first is the
                               model, the second is the layer index of the neuron/channel to
                               remove, and the third is the index of the neuron/channel in the
                               layer. This function is called for every neuron/channel that will
                               be reset, and must modify the classifier in-place.
        :param extract: bool; whether or not to run Neural Cleanse extraction
                              from scratch. If False, then triggers must be
                              provided. If True, then the triggers are ignored
        :param triggers: list of tuples returned by NeuralCleanse extract
        :param extract_kwargs: dict; keyword arguments passed to the initialization
                               of NeuralCleanse if extract is True
        :param dt: float; threshold to prune fully-connected neurons
        :param ct: float; threshold to prune convolutional layer neurons
        :param batch_size: int; batch size
        :param epochs: int; number of fine-tuning epochs
        """
        super(Laundering, self).__init__(classifier)

        self.extract = extract
        self.triggers = triggers
        self.extract_kwargs = extract_kwargs if extract_kwargs is not None else {}
        self.max_nb_samples = n_samples

        self.reset_function = reset_function
        self.dt = dt
        self.ct = ct

    def __get_id(self, y):
        import hashlib
        import json

        x = np.zeros((1, *self.classifier.input_shape)).astype(np.float32)
        pred = self.classifier.predict(x, batch_size=1)

        y = y if y is not None else np.array(-1)
        d = json.dumps(self.extract_kwargs, sort_keys=True).encode('utf-8')

        hash = hashlib.md5()
        hash.update(pred)
        hash.update(y)
        hash.update(d)

        return hash.hexdigest()

    def __save_triggers(self, y, triggers):
        filename = f'neural_cleanse_{self.__get_id(y)}.trigger'
        filepath = os.path.join(WRT_DATA_PATH, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(triggers, f)

    def __load_triggers(self, y):
        filename = f'neural_cleanse_{self.__get_id(y)}.trigger'
        filepath = os.path.join(WRT_DATA_PATH, filename)
        with open(filepath, 'rb') as f:
            triggers = pickle.load(f)
            return triggers

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

    def get_unlearning_dataset(self, x, y, triggers):
        """
        Get a dataset for unlearning the backdoor given the reverse-engineered triggers
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :param triggers: list of (int, np.ndarray) tuples; the reverse-engineered triggers
        :return: (np.ndarray, np.ndarray); dataset and labels
        """
        trigger = triggers[0][1]

        trigger_indices = np.random.choice(x.shape[0], x.shape[0], replace=False)
        x_trigger = x[trigger_indices]
        x_trigger = x_trigger + trigger
        x_trigger[x_trigger > 1] = 1
        y_trigger = y[trigger_indices]
        return np.vstack([x, x_trigger]).astype(np.float32), np.vstack([y, y_trigger])

    def __activation_difference(self, x, x_wm):
        """
        Return the average difference between activations of clean inputs
        and activations of adversarial inputs, for every layer
        :return: list of np.ndarray
        """
        functional = self.classifier.functional
        num_layers = len(self.classifier.get_all_activations(np.expand_dims(x_wm[0], axis=0)))
        batch_size = 64

        # slower but more memory-efficient way to get the mean activation:
        # store only a running mean
        avg_clean_activations = [None for _ in range(num_layers)]
        num_datapoints = 0

        num_clean_batches = int(np.ceil(x.shape[0] // batch_size))
        for batch in range(num_clean_batches):
            activations = self.classifier.get_all_activations(x[batch * batch_size: (batch + 1) * batch_size])
            for layer_index, activation in enumerate(activations):
                activation = np.mean(functional.numpy(activation), axis=0)
                if avg_clean_activations[layer_index] is None:
                    avg_clean_activations[layer_index] = activation
                else:
                    avg_clean_activations[layer_index] = \
                        (activation + num_datapoints * avg_clean_activations[layer_index]) / (num_datapoints + 1)
            num_datapoints += 1

        avg_adv_activations = [None for _ in range(num_layers)]
        num_datapoints = 0
        num_adv_batches = int(np.ceil(x_wm.shape[0] / batch_size))
        for batch in range(num_adv_batches):
            activations = self.classifier.get_all_activations(x_wm[batch * batch_size: (batch + 1) * batch_size])
            for layer_index, activation in enumerate(activations):
                activation = np.mean(functional.numpy(activation), axis=0)
                if avg_adv_activations[layer_index] is None:
                    avg_adv_activations[layer_index] = activation
                else:
                    avg_adv_activations[layer_index] = \
                        (activation + num_datapoints * avg_adv_activations[layer_index]) / (num_datapoints + 1)
            num_datapoints += 1

        return [adv_activations - clean_activations for clean_activations, adv_activations in zip(avg_clean_activations, avg_adv_activations)]

    def remove(self,
               train_loader: WRTDataLoader,
               train_loader_subset: WRTDataLoader,
               valid_loader: WRTDataLoader,
               epochs: int,
               output_dir: str = None,
               batch_size: int = 64,
               check_every_n_batches: int = None,
               boost_factor: int = 1,
               wm_data=None,
               callbacks: List = None,
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
        if callbacks is None:
            callbacks = []

        # Collect data as numpy arrays from the training data loader.
        x, y = collect_n_samples(n=self.max_nb_samples, data_loader=train_loader_subset, has_labels=True)

        triggers = self.get_triggers(x, y)
        x_trigger = self.get_trigger_set(x, y, triggers)

        '''for trigger in train_loader.unnormalize(x_trigger):
            import matplotlib.pyplot as plt
            plt.imshow(trigger.transpose((1,2,0)))
            plt.show()'''

        diffs = self.__activation_difference(x, x_trigger)
        prune_fc_count, total_fc = 0, 0
        prune_conv_count, total_conv = 0, 0
        for layer_index, diff in enumerate(diffs):
            diff_shape = diff.shape

            if len(diff_shape) == 3:
                # for convolutional layers, zero out entire channels
                diff = np.max(diff, axis=(1, 2))
                threshold = self.ct
            else:
                # for fc layers, zero out individual neurons
                diff = diff.reshape(-1)
                threshold = self.dt

            reset_indices = diff > threshold
            reset_indices = np.arange(reset_indices.shape[0])[reset_indices]
            for index in reset_indices:
                self.reset_function(self.classifier.model, layer_index, index)

            if len(diff_shape) == 3:
                prune_conv_count += len(reset_indices)
                total_conv += len(diff)
            else:
                prune_fc_count += len(reset_indices)
                total_fc += len(diff)

        print(f"Pruned {prune_conv_count}/{total_conv} channels and {prune_fc_count}/{total_fc} neurons")

        x_retrain, y_retrain = self.get_unlearning_dataset(x, y, triggers)

        if wm_data is not None:
            defense, x_wm, y_wm = wm_data
            callbacks.append(DebugWRTCallback(lambda: defense.verify(x_wm, y_wm, classifier=self.get_classifier())[0],
                                              message="wm_acc",
                                              check_every_n_batches=check_every_n_batches))

        train_loader_pegged = train_loader.add_numpy_data(x_retrain, y_retrain.astype(np.float64), boost_factor=boost_factor)
        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader_pegged, valid_loader=valid_loader,
                          device=device, num_epochs=epochs, callbacks=callbacks, disable_progress=False)
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


class PartialLaundering(Laundering):

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

        x_source_indices = soft_argmax(y, axis=1) == source_index
        x_source = x[x_source_indices]

        x_trigger = x_source + trigger
        x_trigger[x_trigger > 1] = 1
        return x_trigger

    def get_unlearning_dataset(self, x, y, triggers):
        """
        Get a dataset for unlearning the backdoor given the reverse-engineered triggers
        :param x: np.ndarray; training data
        :param y: np.ndarray; corresponding labels
        :param triggers: list of (int, int, np.ndarray) tuples; the reverse-engineered triggers
        :return: (np.ndarray, np.ndarray); dataset and labels
        """
        source_index, _, trigger = triggers[0]
        uninfected_target = triggers[-1][1]

        y = to_onehot(y, self.classifier.nb_classes())  # Ensure y is encoded as one-hot

        x_source_indices = np.argmax(y, axis=1) == source_index
        x_source = x[x_source_indices]
        x_trigger = x_source + trigger
        x_trigger[x_trigger > 1] = 1
        y_trigger = y[x_source_indices]

        # label reconstructed images not belonging to the source class to the least infected
        # target class
        x_other = x[np.logical_not(x_source_indices)]
        x_other = x_other + trigger
        x_other[x_other > 1] = 1
        y_other = np.eye(self.classifier.nb_classes())[uninfected_target]
        y_other = np.tile(y_other, (x_other.shape[0], 1))

        return np.vstack([x, x_trigger, x_other]).astype(np.float32), np.vstack([y, y_trigger, y_other])


class MultiLaundering(Laundering):

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


@mlconfig.register
def neural_laundering_attack(classifier, dataset, **kwargs):
    assert dataset in ["cifar10", "imagenet"], f"Dataset '{dataset}' is not defined"

    if dataset == "cifar10":
        def reset_function(model, layer_index, neuron_index):
            """
            Reset (and retrain) a neuron or channel by setting the weights
            of the input into the neuron/channel to zero
            """
            layer_index_to_module = {
                # original paper only resets weights in the second half
                # 1: model.layer1[0].conv1,
                # 2: model.layer1[0].conv2,
                # 4: model.layer1[1].conv1,
                # 5: model.layer1[1].conv2,
                7: model.layer2[0].conv1,
                8: model.layer2[0].conv2,
                10: model.layer2[1].conv1,
                11: model.layer2[1].conv2,
                13: model.layer3[0].conv1,
                14: model.layer3[0].conv2,
                16: model.layer3[1].conv1,
                17: model.layer3[1].conv2,
            }

            if layer_index not in layer_index_to_module:
                return

            with torch.no_grad():
                module = layer_index_to_module[layer_index]
                module.weight[neuron_index] = 0
    elif dataset == "imagenet":
        def reset_function(model, layer_index, neuron_index):
            """
            Reset (and retrain) a neuron or channel by setting the weights
            of the input into the neuron/channel to zero
            """
            layer_index_to_module = {
                # original paper only resets weights in the second half
                29: model.layer3[0].conv1,
                30: model.layer3[0].conv2,
                31: model.layer3[0].conv3,
                33: model.layer3[1].conv1,
                34: model.layer3[1].conv2,
                35: model.layer3[1].conv3,
                37: model.layer3[2].conv1,
                38: model.layer3[2].conv2,
                39: model.layer3[2].conv3,
                41: model.layer3[3].conv1,
                42: model.layer3[3].conv2,
                43: model.layer3[3].conv3,
                45: model.layer3[4].conv1,
                46: model.layer3[4].conv2,
                47: model.layer3[4].conv3,
                49: model.layer3[5].conv1,
                50: model.layer3[5].conv2,
                51: model.layer3[5].conv3,
                53: model.layer4[0].conv1,
                54: model.layer4[0].conv2,
                55: model.layer4[0].conv3,
                57: model.layer4[1].conv1,
                58: model.layer4[1].conv2,
                59: model.layer4[1].conv3,
                61: model.layer4[2].conv1,
                62: model.layer4[2].conv2,
                63: model.layer4[2].conv3
            }

            if layer_index not in layer_index_to_module:
                return

            with torch.no_grad():
                module = layer_index_to_module[layer_index]
                module.weight[neuron_index] = 0
    else:
        raise ValueError

    return PartialLaundering(classifier, reset_function=reset_function, **kwargs)


@mlconfig.register
def neural_laundering_partial_removal(attack: Laundering, train_loader, config, **kwargs):
    train_loader_subset = config.subset_dataset(train=True)
    return attack, attack.remove(train_loader=train_loader, train_loader_subset=train_loader_subset, **kwargs)