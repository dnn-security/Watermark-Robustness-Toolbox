"""
This module implements the Neural Cleanse Attack
http://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple, List, Union, Optional

import numpy as np
from tqdm import tqdm

from wrt.attacks.attack import ExtractionAttack
from wrt.classifiers import PyTorchClassifier
from wrt.config import WRT_NUMPY_DTYPE
import torch.optim as optim
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class NeuralCleanse(ExtractionAttack):
    """
    Implement the Neural Cleanse Attack.
    """

    class CustomAdam:
        """
        A custom Adam Optimizer for optimizing the mask and trigger
        """

        def __init__(self, mask, trigger, lr, beta_1=0.9, beta_2=0.999, eps=1e-8):
            """
            Create an instance initialized with the given mask and trigger
            :param mask: np.ndarray with shape (rows, cols)
            :param trigger: np.ndarray with shape (channels, rows, cols)
            :param lr: float
            :param beta_1: float
            :param beta_2: float
            :param eps: float
            """
            self.mask = mask
            self.trigger = trigger
            self.lr = lr
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.eps = eps

            self.t = 1
            self.m_mask, self.v_mask = np.zeros(mask.shape), np.zeros(mask.shape)
            self.m_trigger, self.v_trigger = np.zeros(trigger.shape), np.zeros(trigger.shape)

        def update(self, mask_grad, trigger_grad):
            """
            Update the mask and trigger using the gradients given
            :param mask_grad: np.ndarray; mask gradient
            :param trigger_grad: np.ndarray; trigger gradient
            :return: (np.ndarray, np.ndarray); the new mask and trigger
            """
            self.m_mask = self.beta_1 * self.m_mask + (1 - self.beta_1) * mask_grad
            self.v_mask = self.beta_2 * self.v_mask + (1 - self.beta_2) * np.power(mask_grad, 2)
            m_hat = self.m_mask / (1 - np.power(self.beta_1, self.t))
            v_hat = self.v_mask / (1 - np.power(self.beta_2, self.t))
            self.mask -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.m_trigger = self.beta_1 * self.m_trigger + (1 - self.beta_1) * trigger_grad
            self.v_trigger = self.beta_2 * self.v_trigger + (1 - self.beta_2) * np.power(trigger_grad, 2)
            m_hat = self.m_trigger / (1 - np.power(self.beta_1, self.t))
            v_hat = self.v_trigger / (1 - np.power(self.beta_2, self.t))
            self.trigger -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.t += 1
            return self.mask, self.trigger

    def __init__(self,
                 classifier: PyTorchClassifier,
                 extract_epochs: int = 1,
                 finetune_epochs: int = 1,
                 lmbda: float = 1e-3,
                 source_classes: Optional[List[int]] = None,
                 target_classes: Optional[List[int]] = None,
                 batch_size: int = 32,
                 **kwargs):
        """
        Create a :class:`.NeuralCleanse` instance.

        :param classifier: A trained classifier.
        :param extract_epochs: int; number of epochs to run the Neural Cleanse extraction and detect outliers
        :param finetune_epochs: int; number of epochs to fine-tune the trigger when an
                                infected label is found; if set to None, no fine-tuning is
                                performed and the current detected trigger is returned
        :param lmbda: float; weight of the mask regularization term
        :param source_classes List of source classes to check if they are infected.
        :param target_classes List of target classes to check if they are infected.
        :param batch_size: int; batch size
        """
        super(NeuralCleanse, self).__init__(classifier)

        # Initialize the source and target classes.
        if source_classes is None:
            source_classes = np.arange(classifier.nb_classes())

        if target_classes is None:
            target_classes = np.arange(classifier.nb_classes())

        self.extract_epochs = extract_epochs
        self.finetune_epochs = finetune_epochs
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.source_classes = source_classes
        self.target_classes = target_classes

        self.input_shape = self.classifier.input_shape
        self.num_classes = self.classifier.nb_classes()

    def _reverse_engineer_trigger(self,
                                  x: np.ndarray,
                                  y: np.ndarray,
                                  target_index: int,
                                  mask_init: np.ndarray,
                                  trigger_init: np.ndarray,
                                  epochs: int,
                                  source_index: int = None) -> Tuple[np.ndarray, np.ndarray]:
        def sigmoid(x):
            return 1 / (1 + self.classifier.functional.exp(-x))

        def apply_trigger(x, mask, trigger):
            mask = sigmoid(mask)
            trigger = sigmoid(trigger)
            return mask * trigger + x * (1 - mask)

        functional = self.classifier.functional

        mask = Variable(functional.tensor(np.copy(mask_init)))
        trigger = Variable(functional.tensor(np.copy(trigger_init)))
        mask.requires_grad = True
        trigger.requires_grad = True
        optimizer = optim.Adam([mask, trigger], lr=0.1, betas=(0.5, 0.9))

        # If source_index is specified, only find the minimum trigger to mis-classify
        # data of the source class into the target class. Otherwise, find the minimum
        # trigger to mis-classify data from all classes, other than the target class
        # itself, into the target class
        if source_index is not None:
            if len(y.shape) > 1:
                y = y.argmax(1)

            indices, = np.where(y == source_index)
            x = x[indices]
        else:
            indices, = np.where(y != target_index)
            x = x[indices]

        print(f"Working on {x.shape[0]} examples for source index {source_index}")

        # Freeze the classifier.
        ce_loss_function = self.classifier.loss
        for param in self.classifier.model.parameters():
            param.requires_grad = False

        # Optimize the trigger.
        for e in range(epochs):
            num_batches = int(np.ceil(x.shape[0] / self.batch_size))
            ind = np.arange(x.shape[0])
            np.random.shuffle(ind)

            train_loop = tqdm(range(num_batches))
            for batch in train_loop:
                x_batch = functional.tensor(x[ind[batch * self.batch_size: (batch + 1) * self.batch_size]])
                y_batch = functional.tensor(np.tile(target_index, x_batch.shape[0]).astype(np.int64))

                x_batch = apply_trigger(x_batch, mask, trigger)
                optimizer.zero_grad()
                outputs = self.classifier.get_all_activations(x_batch)[-1]
                ce_loss = ce_loss_function.compute_loss(outputs, y_batch)
                reg_loss = functional.sum(sigmoid(mask))
                loss = ce_loss + self.lmbda * reg_loss

                loss.backward()
                optimizer.step()

                mask_np = mask.clone().detach().cpu().numpy()

                mask_norm = np.sum(1 / (1 + np.exp(-mask_np)))
                if source_index is None:
                    description = f"Label {target_index}: ({e + 1}/{epochs}): Loss ({loss:.4f}) Mask Norm ({mask_norm:.4f})"
                else:
                    description = f"Label {source_index} -> {target_index}: ({e + 1}/{epochs}): Loss ({loss:.4f}) Mask Norm ({mask_norm:.4f})"
                train_loop.set_description(description)

        # Unfreeze the classifier.
        for param in self.classifier.model.parameters():
            param.requires_grad = True

        mask = sigmoid(mask)
        trigger = sigmoid(trigger)
        return functional.numpy(mask), functional.numpy(trigger)

    @staticmethod
    def _get_infected_labels(triggers: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """ Given all triggers, extracts those labels which are most likely infected.
        """
        k = 1.4826  # scaling factor for fitting the data to a normal distribution

        norms = np.sum(np.abs(triggers), axis=(1, 2, 3))
        median = np.median(norms)
        mad = k * np.median(np.abs(norms - median))
        anomaly_indices = np.abs(norms - median) / mad

        infected_labels = []
        for index in range(triggers.shape[0]):
            if anomaly_indices[index] > 2 and norms[index] <= median:
                infected_labels.append((index, triggers[index]))

        return infected_labels

    def extract(self, x: np.ndarray, y: np.ndarray = None, **kwargs) -> List[Tuple[int, np.ndarray]]:
        """
        Detect whether the model is watermarked or not, and return a list
        containing the triggers
        :param x: np.ndarray; original training data
        :param y: np.ndarray; original training labels
        :param kwargs: ignored
        :return: list of (int, np.ndarray) tuples, with each element indicating
                 an infected label index and the corresponding reversed trigger.
        """
        triggers = []
        mask_init = np.random.random(self.input_shape[1:]).astype(WRT_NUMPY_DTYPE)
        trigger_init = np.random.random(self.input_shape).astype(WRT_NUMPY_DTYPE)

        for target_index in range(self.num_classes):
            mask, trigger = self._reverse_engineer_trigger(x, y, target_index, mask_init, trigger_init, self.extract_epochs)
            triggers.append(mask * trigger)

        infected_labels = self._get_infected_labels(np.array(triggers))

        if self.finetune_epochs is None:
            return infected_labels
        else:
            triggers = []
            for target_index, trigger in infected_labels:
                mask, trigger = self._reverse_engineer_trigger(x, y, target_index, mask_init, trigger_init, self.extract_epochs + self.finetune_epochs)
                triggers.append((target_index, mask * trigger))
            return triggers


class NeuralCleansePartial(NeuralCleanse):
    """
    Implement the Neural Cleanse "partial" variant, in which we assume there is a single infected class.
    """

    def extract(self, x, y=None,
                **kwargs) -> List[Tuple[int, int, np.ndarray]]:
        """
        Implement a Neural Cleanse variant where we assume there is a single infected (target) classes.
        This attack has a square complexity in the number of source and target classes.

        :param x: np.ndarray; original training data
        :param y: np.ndarray; original training labels
        :param kwargs: ignored
        :return: list of (int, int, np.ndarray) tuples, with each element
                 containing a source label, a target label, and the corresponding
                 reversed trigger. The list is sorted by likelihood to be infected, with
                 the most likely items in front.
        """
        # Randomly initialize the trigger and masks.
        triggers = []
        mask_init = np.random.random(self.input_shape[1:]).astype(WRT_NUMPY_DTYPE)
        trigger_init = np.random.random(self.input_shape).astype(WRT_NUMPY_DTYPE)

        for source_index in self.source_classes:
            for target_index in self.target_classes:
                if source_index == target_index:
                    continue

                mask, trigger = self._reverse_engineer_trigger(x, y, target_index, mask_init, trigger_init, self.extract_epochs,
                                                               source_index=source_index)
                triggers.append((source_index, target_index, mask * trigger))

        triggers = sorted(triggers, key=lambda t: np.sum(np.abs(t[2])))

        if self.finetune_epochs is None:
            return triggers
        else:
            candidate_triggers = triggers[:self.num_classes]
            new_candidate_triggers = []

            for source_index, target_index, trigger in candidate_triggers:
                mask, trigger = self._reverse_engineer_trigger(x, y, target_index, mask_init, trigger_init, self.extract_epochs + self.finetune_epochs,
                                                               source_index=source_index)
                new_candidate_triggers.append((source_index, target_index, mask * trigger))

            new_candidate_triggers = sorted(new_candidate_triggers, key=lambda t: np.sum(np.abs(t[2])))
            triggers[:self.num_classes] = new_candidate_triggers
            return triggers


class NeuralCleanseMulti(NeuralCleanse):
    """
    Implement a Neural Cleanse variant where we assume there are multiple infected (target) classes.
    In this variant, we find for every target label the minimum trigger needed
    to misclassify data into the label.
    """

    def extract(self, x, y=None, **kwargs):
        """
        Return a list containing the reversed-engineered triggers for every target label
        :param x: np.ndarray; original training data
        :param y: np.ndarray; original training labels
        :param kwargs: ignored
        :return: list of (int, np.ndarray) tuples, with each element indicating
                 an infected label index and the corresponding reversed trigger.
        """
        triggers = []
        mask_init = np.random.random(self.input_shape[1:]).astype(WRT_NUMPY_DTYPE)
        trigger_init = np.random.random(self.input_shape).astype(WRT_NUMPY_DTYPE)

        num_epochs = self.extract_epochs if self.finetune_epochs is None else self.extract_epochs + self.finetune_epochs

        for target_index in self.target_classes:
            mask, trigger = self._reverse_engineer_trigger(x, y, target_index, mask_init, trigger_init, num_epochs)
            triggers.append((target_index, mask * trigger))

        return triggers
