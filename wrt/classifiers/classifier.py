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
This module implements abstract base classes defining to properties for all classifiers.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc

import numpy as np

from wrt.classifiers.utils import check_and_transform_label_format


class input_filter(abc.ABCMeta):
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls.
    """

    def __init__(cls, name, bases, clsdict):
        """
        This function overrides any existing generate or extract methods with a new method that
        ensures the input is an ndarray. There is an assumption that the input object has implemented
        __array__ with np.array calls.
        """

        def make_replacement(fdict, func_name, has_y):
            """
            This function overrides creates replacement functions dynamically.
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
                elif has_y:
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list_no_y = ["get_activations", "class_gradient"]
        replacement_list_has_y = ["fit", "loss_gradient"]

        for item in replacement_list_no_y:
            if item in clsdict:
                new_function = make_replacement(clsdict, item, False)
                setattr(cls, item, new_function)
        for item in replacement_list_has_y:
            if item in clsdict:
                new_function = make_replacement(clsdict, item, True)
                setattr(cls, item, new_function)


class Classifier(abc.ABC, metaclass=input_filter):
    """
    Base class defining the minimum classifier functionality and is required for all classifiers. A classifier of this
    type can be combined with black-box attacks.
    """

    def __init__(
        self, clip_values=None, channel_index=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=None, **kwargs
    ):
        """
        Initialize a `Classifier` object.

        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param channel_index: Index of the axis in input (feature) array `x` representing the color channels.
        :type channel_index: `int`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        self._clip_values = clip_values
        if clip_values is not None:
            if len(clip_values) != 2:
                raise ValueError(
                    "`clip_values` should be a tuple of 2 floats or arrays containing the allowed data range."
                )
            if np.array(clip_values[0] >= clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")

        self._channel_index = channel_index
        self.preprocessing_defences = preprocessing_defences
        self.postprocessing_defences = postprocessing_defences
        self.preprocessing = preprocessing

        super().__init__(**kwargs)

    @abc.abstractmethod
    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction of the classifier for input `x`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :type x: `np.ndarray`
        :param batch_size: The batch size used for evaluating the classifer's `model`.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, x, y, batch_size=128, **kwargs):
        """
        Evaluate accuracy of the classifier on `x, y`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2)
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: The batch size used for evaluating the classifer's `model`.
        :type batch_size: `int`
        :return: Training metrics. See class implementation for details
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x, y, batch_size=128, nb_epochs=20, shuffle=True, verbose=True, callbacks=None, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: The batch size used for evaluating the classifer's `model`.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param shuffle: Whether or not to shuffle data before training
        :param verbose: Whether or not to log verbosely
        :param callbacks: list of objects with the methods on_batch_end() and on_epoch_end(); called at
                          the corresponding times during training
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, verbose=True, callbacks=None, **kwargs):
        """
        Fit the classifier using `generator` yielding training batches as specified. Framework implementations can
        provide framework-specific versions of this function to speed-up computation.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param verbose: Whether or not to log verbosely
        :param callbacks: list of objects with the methods on_batch_end() and on_epoch_end(); called at
                          the corresponding times during training
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        from wrt.data_generators import DataGenerator

        if not isinstance(generator, DataGenerator):
            raise ValueError(
                "Expected instance of `DataGenerator` for `fit_generator`, got %s instead." % str(type(generator))
            )

        for _ in range(nb_epochs):
            for _ in range(int(generator.size / generator.batch_size)):
                x, y = generator.get_batch()

                # Fit for current batch
                self.fit_batch(x, y)

    def fit_batch(self, x, y):
        """
        Fit the classifier on a batch x and labels y
        :param x: np.ndarray; inputs to classifier
        :param y: np.ndarray; corresponding labels
        :return: int; loss value for the batch
        """
        self.fit(x, y, batch_size=x.shape[0], nb_epochs=1, shuffle=False, verbose=False)
        return 0

    @abc.abstractmethod
    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Input with shape as expected by the classifier's model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Input with shape as expected by the classifier's model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_activations(self, x):
        """
        Return the output of every layer for the input x
        :param x: np.ndarray or framework-specific tensor
        :return: list of framework-specific tensors
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_weights(self):
        """
        Return the layer weights of the neural network in sequence, ordered in the same
        way as a forward-pass through the network. The return type depends on the backend
        framework.
        :return: Framework-specific list of Tensors
        """
        raise NotImplementedError

    @property
    def clip_values(self):
        """
        :return: Tuple of form `(min, max)` containing the minimum and maximum values allowed for the input features.
        :rtype: `tuple`
        """
        return self._clip_values

    @property
    def channel_index(self):
        """
        :return: Index of the axis in input data containing the color channels.
        :rtype: `int`
        """
        return self._channel_index

    @property
    @abc.abstractmethod
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_shape(self):
        """
        Return the shape of one input.

        :return: Shape of one input for the classifier.
        :rtype: `tuple`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def learning_phase(self):
        """
        Return the learning phase set by the user for the current classifier. Possible values are `True` for training,
        `False` for prediction and `None` if it has not been set through the library. In the latter case, the library
        does not do any explicit learning phase manipulation and the current value of the backend framework is used.
        If a value has been set by the user for this property, it will impact all following computations for
        model fitting, prediction and gradients.

        :return: Value of the learning phase.
        :rtype: `bool` or `None`
        """
        raise NotImplementedError

    @learning_phase.setter
    @abc.abstractmethod
    def learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: `True` if the learning phase is training, `False` if learning phase is not training.
        :type train: `bool`
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loss(self):
        """
        Get the current loss function
        :return: Loss instance
        """
        raise NotImplementedError

    @loss.setter
    @abc.abstractmethod
    def loss(self, loss):
        """
        Set the loss function to the one given. If the loss function is not an
        instance of `Loss`, then it is wrapped as an instance
        :param loss: Framework-specific loss function or `Loss` instance
        :return: None
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def functional(self):
        """
        Return the Functional instance associated with this classifier in order
        to call framework-specific tensor functions
        :return: Functional
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model(self):
        """
        Return the framework-specific model of this neural network
        :return: Framework-specific model
        """
        raise NotImplementedError

    def add_preprocessor(self, preprocessor, name):
        """
        Add a preprocessing defense with the given name
        :param preprocessor: Preprocessor
        :param name: str
        :return: None
        """
        if self.preprocessing_defences is None:
            self.preprocessing_defences = {}

        if name in self.preprocessing_defences:
            raise NameError(f"Error: preprocessor with name {name} already exists")

        self.preprocessing_defences[name] = preprocessor

    def remove_preprocessor(self, name):
        """
        Remove the preprocessing defense with the given name
        :param name: str
        :return: None
        """
        if self.preprocessing_defences is None:
            self.preprocessing_defences = {}

        if name not in self.preprocessing_defences:
            return

        del self.preprocessing_defences[name]

    def add_postprocessor(self, postprocessor, name):
        """
        Add a postprocessing defense with the given name
        :param postprocessor: Postprocessor
        :param name: str
        :return: None
        """
        if self.postprocessing_defences is None:
            self.postprocessing_defences = {}

        if name in self.postprocessing_defences:
            raise NameError(f"Error: postprocessor with name {name} already exists")

        self.postprocessing_defences[name] = postprocessor

    def remove_postprocessor(self, name):
        """
        Remove the postprocessing defense with the given name
        :param name: str
        :return: None
        """
        if self.postprocessing_defences is None:
            self.postprocessing_defences = {}

        if name not in self.postprocessing_defences:
            return

        del self.postprocessing_defences[name]

    @abc.abstractmethod
    def save(self, filename, path=None):
        """
        Save a model to file specific to the backend framework.

        :param filename: Name of the file where to save the model.
        :type filename: `str`
        :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in
                     the default data location of ART at `ART_DATA_PATH`.
        :type path: `str`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, filename, path=None):
        """
        Load a model from a file specific to the backend framework.

        :param filename: Name of the file where to save the model.
        :type filename: `str`
        :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in
                     the default data location of ART at `ART_DATA_PATH`.
        :type path: `str`
        :return: None
        """
        raise NotImplementedError

    def _apply_preprocessing(self, x, y, fit):
        """
        Apply all defenses and preprocessing operations on the inputs `(x, y)`. This function has to be applied to all
        raw inputs (x, y) provided to the classifier.

        :param x: Features, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :param y: Target values (class labels), where first dimension is the number of samples.
        :type y: `np.ndarray` or `None`
        :param fit: `True` if the defenses are applied during training.
        :type fit: `bool`
        :return: Value of the data after applying the defenses.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.nb_classes())

        if self.preprocessing_defences is not None:
            for name, defence in self.preprocessing_defences.items():
                if fit:
                    if defence.apply_fit:
                        x, y = defence(x, y)
                else:
                    if defence.apply_predict:
                        x, y = defence(x, y)

        x = self._apply_preprocessing_standardisation(x)
        return x, y

    def _apply_preprocessing_standardisation(self, x):
        """
        Apply standardisation to input data `x`.

        :param x: Input data, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :return: Array for `x` with the standardized data.
        :rtype: `np.ndarray`
        :raises: `TypeError`
        """
        if x.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            raise TypeError(
                "The data type of input data `x` is {} and cannot represent negative values. Consider "
                "changing the data type of the input data `x` to a type that supports negative values e.g. "
                "np.float32.".format(x.dtype)
            )

        if self.preprocessing is not None:
            sub, div = self.preprocessing
            sub = np.asarray(sub, dtype=x.dtype)
            div = np.asarray(div, dtype=x.dtype)

            res = x - sub
            res = res / div

        else:
            res = x

        return res

    def _apply_postprocessing(self, preds, fit, x=None):
        """
        Apply all defenses operations on model output.

        :param preds: model output to be postprocessed.
        :type preds: `np.ndarray`
        :param fit: `True` if the defenses are applied during training.
        :type fit: `bool`
        :return: Postprocessed model output.
        :rtype: `np.ndarray`
        """
        post_preds = preds.copy()

        if self.postprocessing_defences is not None:
            for name, defence in self.postprocessing_defences.items():
                if fit:
                    if defence.apply_fit:
                        post_preds = defence(post_preds, x=x)
                else:
                    if defence.apply_predict:
                        post_preds = defence(post_preds, x=x)

        return post_preds

    def _apply_preprocessing_gradient(self, x, gradients):
        """
        Apply the backward pass through all preprocessing operations to the gradients.

        Apply the backward pass through all preprocessing operations and defenses on the inputs `(x, y)`. This function
        has to be applied to all gradients returned by the classifier.

        :param x: Features, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :param gradients: Input gradients.
        :type gradients: `np.ndarray`
        :return: Gradients after backward step through preprocessing operations and defenses.
        :rtype: `np.ndarray`
        """
        gradients = self._apply_preprocessing_normalization_gradient(gradients)
        gradients = self._apply_preprocessing_defences_gradient(x, gradients)
        return gradients

    def _apply_preprocessing_defences_gradient(self, x, gradients, fit=False):
        """
        Apply the backward pass through the preprocessing defenses.

        Apply the backward pass through all preprocessing defenses of the classifier on the gradients. This function is
        intended to only be called from function `_apply_preprocessing_gradient`.

        :param x: Features, where first dimension is the number of samples.
        :type x: `np.ndarray`
        :param gradients: Input gradient.
        :type gradients: `np.ndarray`
        :param fit: `True` if the gradient is computed during training.
        :return: Gradients after backward step through defenses.
        :rtype: `np.ndarray`
        """
        if self.preprocessing_defences is not None:
            for defence in list(self.preprocessing_defences.values())[::-1]:
                if fit:
                    if defence.apply_fit:
                        gradients = defence.estimate_gradient(x, gradients)
                else:
                    if defence.apply_predict:
                        gradients = defence.estimate_gradient(x, gradients)

        return gradients

    def _apply_preprocessing_normalization_gradient(self, gradients):
        """
        Apply the backward pass through standardisation of `x` to `gradients`.

        :param gradients: Input gradients.
        :type gradients: `np.ndarray`
        :return: Gradients after backward step through standardisation.
        :rtype: `np.ndarray
        """
        if self.preprocessing is not None:
            _, div = self.preprocessing
            div = np.asarray(div, dtype=gradients.dtype)
            res = gradients / div
        else:
            res = gradients

        return res
