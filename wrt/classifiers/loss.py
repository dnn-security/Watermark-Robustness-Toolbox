"""
Module adding an extra abstraction layer to framework-specific loss classes
"""

import abc


class Loss(abc.ABC):

    def __init__(self, classifier):
        self.classifier = classifier
        self._functional = None

    def reduce_labels(self):
        """
        Return True if the loss requires reduced labels instead of
            one-hot encoded labels
        :return: bool
        """
        return False

    def get_functional(self):
        """
        Return the Functional instance associated with this object
        :return: Functional
        """
        return self._functional

    def set_functional(self, functional):
        """
        Set the Functional instance associated with this object
        :param functional: Functional
        :return: None
        """
        self._functional = functional
        self.on_functional_change()

    def on_functional_change(self):
        """
        Called when the Functional instance is changed
        :return: None
        """
        pass

    @abc.abstractmethod
    def compute_loss(self, pred, true, x=None):
        """
        Return the loss between predicted labels pred and actual
            labels true. x, if provided, are the inputs to the classifier
            and may be used for additional loss terms.
        Note that self._functional may be used for framework-specific
            operations
        :param pred: Framework-specific tensor
        :param true: Framework-specific tensor
        :param x: Framework-specific tensor
        :return: Framework-specific tensor
        """
        raise NotImplementedError
