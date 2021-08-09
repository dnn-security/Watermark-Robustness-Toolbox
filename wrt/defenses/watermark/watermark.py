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
This module implements the abstract base class for defenses that watermark a neural network.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import Tuple
import numpy as np
import os

from wrt.classifiers import PyTorchClassifier
from wrt.training.utils import compute_accuracy

import torch


class Watermark(abc.ABC):
    """
    Abstract base class for watermarking defenses.
    """

    def __init__(self, classifier, **kwargs):
        """
        Create a watermarking defense object
        """
        self.classifier = classifier

    @staticmethod
    def get_name():
        return "No Name Provided"

    def keygen(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a secret key and message.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, **kwargs):
        """
        Embed a message into a model using a secret key.
        """
        raise NotImplementedError

    def get_classifier(self) -> PyTorchClassifier:
        """
        Return the classifier instance
        """
        return self.classifier

    def save(self, filename: str, path: str = None, checkpoint: dict = None):
        """ Persist this instance without watermarking keys.
        This is a default loader that should be overridden by the subclass.

        :param filename The filename ('.pth') to save this defense.
        :param path The path to the folder to which this defense is saved. Defaults to the
        :param checkpoint Data to save.
        default WRT data path.
        """

        if not filename.endswith('.pth'):
            print(f"[WARNING] Defense instance saved as '{filename}', but should end in '.pth'.")

        if path is None:
            from wrt.config import WRT_DATA_PATH
            full_path = os.path.join(WRT_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)

        print(f"Saving defense instance at {full_path}")

        checkpoint = {
            **checkpoint
        }

        torch.save(checkpoint, full_path)

    def load(self, filename, path=None, **load_kwargs: dict):
        """ Load this instance.

        :param filename The filename ('.pth') to load this defense.
        :param path The path to the folder to which this defense is saved. Defaults to the
        default WRT data path.
        """
        if not filename.endswith('.pth'):
            print(f"[WARNING] Defense instance loaded from a '{filename}', which should end in '.pth'.")

        if path is None:
            from wrt.config import WRT_DATA_PATH
            full_path = os.path.join(WRT_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)

        return torch.load(full_path)

    @abc.abstractmethod
    def extract(self,
                x: np.ndarray,
                classifier: PyTorchClassifier = None,
                **kwargs) -> np.ndarray:
        """
        Extract a message from the model given the watermarking key.

        :param x: The secret watermarking key.
        :param classifier: Classifier; if provided, extract the watermark from
                           the given classifier instead of this object's classifier
        :param kwargs: Other optional parameters.
        :return: The secret message.
        """
        raise NotImplementedError

    def verify(self,
               x: np.ndarray,
               y: np.ndarray = None,
               classifier: PyTorchClassifier = None,
               **kwargs) -> Tuple[float, bool]:
        """ Verifies whether the given classifier retains the watermark. Returns the watermark
        accuracy and whether it is higher than the decision threshold.

        :param x: The secret watermarking key.
        :param y The expected message.
        :param classifier The classifier to verify.
        :param kwargs: Other parameters for the extraction.
        :return A tuple of the watermark accuracy and whether it is larger than the decision threshold.
        """
        if classifier is None:
            classifier = self.get_classifier()

        classifier.model.eval()
        msg = self.extract(x, classifier=classifier, **kwargs)
        wm_acc = compute_accuracy(msg, y)

        return wm_acc, wm_acc > 0  # ToDo: Implement decision boundary as a second parameter

    def predict(self,
                x: np.ndarray,
                **kwargs):
        """
        Perform prediction using the watermarked classifier.

        :param x: The prediction set.
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :return: Predictions for test set.
        """
        return self.classifier.predict(x, **kwargs)
