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
This module implements the abstract base classes for all attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import abc
import numpy as np

from wrt.classifiers import PyTorchClassifier
from wrt.classifiers.classifier import Classifier
from wrt.exceptions import ClassifierError

logger = logging.getLogger(__name__)


class input_filter(abc.ABCMeta):
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls
    """

    def __init__(cls, name, bases, clsdict):
        """
        This function overrides any existing generate or extract methods with a new method that
        ensures the input is an ndarray. There is an assumption that the input object has implemented
        __array__ with np.array calls.
        """

        def make_replacement(fdict, func_name):
            """
            This function overrides creates replacement functions dynamically
            """

            def replacement_function(self, *args, **kwargs):
                if len(args) > 0:
                    lst = list(args)

                if "x" in kwargs:
                    if not isinstance(kwargs["x"], np.ndarray):
                        kwargs["x"] = np.array(kwargs["x"])
                else:
                    if not isinstance(args[0], np.ndarray):
                        lst[0] = np.array(args[0])

                if "y" in kwargs:
                    if kwargs["y"] is not None and not isinstance(kwargs["y"], np.ndarray):
                        kwargs["y"] = np.array(kwargs["y"])
                elif len(args) == 2:
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list = ["generate"]
        for item in replacement_list:
            if item in clsdict:
                new_function = make_replacement(clsdict, item)
                setattr(cls, item, new_function)


class Attack(abc.ABC, metaclass=input_filter):
    """
    Abstract base class for all attack abstract base classes.
    """

    attack_params = list()

    def __init__(self, classifier):
        """
        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        """
        if not isinstance(classifier, Classifier) and classifier is not None:
            raise ClassifierError(self.__class__, [Classifier], classifier)

        self.classifier = classifier

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True

    def get_classifier(self) -> PyTorchClassifier:
        """
        Return the classifier instance
        :return: Classifier
        """
        return self.classifier

    @staticmethod
    def get_hyperparameters():
        """
        Return the tunable hyperparameters of the attack that can be passed
        to the constructor call, along with sensible default values
        :return: dict containing hyperparameter names mapped to list of
                 possible values
        """
        return {}


class RemovalAttack(Attack):
    """
    Abstract base class for watermark removal attack classes.
    """

    def __init__(self, classifier):
        """
        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def remove(self, **kwargs):
        """
        Removes watermarks from models and returns them as WRT classifiers. This method should be overridden by all
        concrete watermark removal attack implementations.

        :return: ART classifier of the extracted model.
        :rtype: :class:`.Classifier`
        """
        raise NotImplementedError


class ExtractionAttack(Attack):
    """
    Abstract base class for watermark extraction attack classes.
    """

    def __init__(self, classifier):
        """
        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def extract(self, x, y=None, **kwargs):
        """
        Extract watermarks from models. See subclass documentation for details. This
        method should be overridden by all concrete watermark extraction attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :type y: `np.ndarray`
        """
        raise NotImplementedError
