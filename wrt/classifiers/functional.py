"""
Module adding an extra abstraction layer to framework-specific functions
"""

import abc


class Functional(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def tensor(self, x, diff=False):
        """
        Return a framework-specific tensor object with the same
            data as x
        :param x: number type or np.ndarray; data to convert to tensor
        :param diff: bool; set to True to allow the returned tensor to
            have its own gradient
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def numpy(self, x):
        """
        Return a numpy array with the same data as x
        :param x: Framework-specific tensor object
        :return: np.ndarray or python number type
        """
        raise NotImplementedError

    @abc.abstractmethod
    def shape(self, x):
        """
        Return the shape of x
        :param x: Framework-specific tensor object
        :return: tuple-like type
        """
        raise NotImplementedError

    @abc.abstractmethod
    def matmul(self, x, y):
        """
        Return the product of x and y
        :param x: Framework-specific tensor object
        :param y: Framework-specific tensor object
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reshape(self, x, *dims):
        """
        Return x reshaped to dims
        :param x: Framework-specific tensor object
        :param dims: ints
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transpose(self, x):
        """
        Return the transpose of a 2-D array x
        :param x: 2-D framework-specific tensor object
        :return: 2-D framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tile(self, x, *dims):
        """
        Return x repeated dims[i] times along axis i
        :param x: Framework-specific tensor object
        :param dims: ints
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def equal(self, x, y, return_bool=False):
        """
        Return element-wise equality of x with y after
            broadcasting shapes
        :param x: Framework-specific tensor object
        :param y: Framework-specific tensor object
        :param return_bool: bool; set to True to return a boolean
            tensor instead of an int tensor
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, x, axis=None):
        """
        Return the mean of x
        :param x: Framework-specific tensor object
        :param axis: int or list of ints; axis to take the mean
            (if None, the mean is taken over the entire tensor)
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sum(self, x, axis=None, keep_dims=False):
        """
        Return the sum of x
        :param x: Framework-specific tensor object
        :param axis: int or list of ints; axis to sum over
            (if None, the sum is taken over the entire tensor)
        :param keep_dims: bool; if set to True is axis is not None,
            keep all dimensions of x in the resulting tensor
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pow(self, x, exp):
        """
        Return element-wise power of x to the exponent exp
        :param x: Framework-specific tensor object
        :param exp: int or framework-specific int type
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def exp(self, x):
        """
        Return element-wise exponential of x base e
        :param x: Framework-specific tensor object
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log(self, x):
        """
        Return element-wise log of x base e
        :param x: Framework-specific tensor object
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def abs(self, x):
        """
        Return element-wise absolute value of x
        :param x: Framework-specific tensor object
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sigmoid(self, x):
        """
        Return element-wise sigmoid of x
        :param x: Framework-specific tensor object
        :return: Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cross_entropy_loss(self, pred, true):
        """
        Return the cross-entropy loss between predicted labels
            pred and actual labels true
        :param pred: 2-D Framework-specific tensor object
        :param true: 2-D Framework-specific tensor object
        :return: 1-D Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def binary_cross_entropy_loss(self, pred, true):
        """
        Return the binary cross-entropy loss between predicted labels
            pred and actual labels true
        :param pred: 1-D Framework-specific tensor object
        :param true: 1-D Framework-specific tensor object
        :return: Framework-specific scalar
        """

    @abc.abstractmethod
    def mse_loss(self, pred, true):
        """
        Return the mean-squared loss between predicted labels
            pred and actual labels true
        :param pred: 2-D Framework-specific tensor object
        :param true: 2-D Framework-specific tensor object
        :return: 1-D Framework-specific tensor object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, function, base):
        """
        Return the gradient of the given function with respect
            to base. The base must be a framework-specific tensor
            object created with diff=True
        :param function: Framework-specific tensor object
        :param base: Framework-specific tensor object
        :return: python int or numpy int type
        """
        raise NotImplementedError
