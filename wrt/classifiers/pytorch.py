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
This module implements the classifier `PyTorchClassifier` for PyTorch models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
from typing import Union, Tuple

import numpy as np
import torch
from tqdm import tqdm

from wrt.classifiers.classifier import Classifier
from wrt.classifiers.functional import Functional
from wrt.classifiers.loss import Loss

logger = logging.getLogger(__name__)

# weight constant for the exponential moving average
ALPHA = 0.05


class StopTrainingException(Exception):
    """
    Exception that should be raised to indicate that training should be stopped
    """
    pass


class PyTorchClassifier(Classifier):  # lgtm [py/missing-call-to-init]
    """
    This class implements a classifier with the PyTorch framework.
    """

    def __init__(
            self,
            model,
            loss,
            optimizer,
            input_shape,
            nb_classes,
            channel_index=1,
            clip_values=None,
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=None,
            device_type="gpu",
    ):
        """
        Initialization specifically for the PyTorch-based implementation.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :type model: `torch.nn.Module`
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :type loss: `torch.nn.modules.loss._Loss`
        :param optimizer: The optimizer used to train the classifier.
        :type optimizer: `torch.optim.Optimizer`
        :param input_shape: The shape of one input instance.
        :type input_shape: `tuple`
        :param nb_classes: The number of classes of the model.
        :type nb_classes: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :type device_type: `string`
        """
        super(PyTorchClassifier, self).__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._nb_classes = nb_classes
        self._input_shape = input_shape
        self._model = self._make_model_wrapper(model)
        self._loss = self._make_loss_wrapper(loss)
        self._optimizer = optimizer
        self._learning_phase = None
        self.override_learning_phase = False

        # Get the internal layers
        self._layer_names = self._model.get_layers

        # Set device
        import torch

        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        self._model.to(self._device)

        # Index of layer at which the class gradients should be calculated
        self._layer_idx_gradients = -1

        # Shuffle intermediate feature activations
        self.shuffle_intermediate_features = False
        self.shuffle_indices = []

        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        self._reduce_labels = self._loss.reduce_labels()

        # Framework-specific functions for this classifier
        self._functional = PyTorchFunctional()

    @staticmethod
    def accuracy(true: torch.Tensor,
                 pred: torch.Tensor) -> float:
        """ Computes the accuracy.
        """
        if pred.dim() > 1:
            pred = pred.argmax(dim=1)

        if true.dim() > 1:
            true = true.argmax(dim=1)

        acc = (true == pred)
        return float(acc.sum() / acc.shape[0])

    def predict(self,
                x: Union[np.ndarray, torch.Tensor],
                batch_size: int = 64,
                learning_phase: bool = False,
                verbose: bool = False,
                **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :param batch_size: Size of batches.
        :param learning_phase Set the model into training mode
        :param verbose Whether to print a progress bar
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        self.learning_phase = learning_phase or self.override_learning_phase

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if self.preprocessing_defences is not None:
            x, _ = self._apply_preprocessing(x=x.cpu().numpy(), y=None, fit=False)
            x = torch.from_numpy(x).to(self.device)

        # Speed up if only a single batch is used.
        with torch.no_grad():
            if x.shape[0] <= batch_size:
                results: np.ndarray = self._model(x)[-1].detach().cpu().numpy()
            else:
                results: list = []
                num_batches = int(np.ceil(len(x) / float(batch_size)))
                with tqdm(range(num_batches), disable=not verbose) as pbar:
                    for m in pbar:
                        begin, end = m * batch_size, min((m + 1) * batch_size, x.shape[0])
                        data = x[begin:end].to(self._device)
                        output = self._model(data)[-1]
                        results.append(output.detach().cpu().numpy())
                results = np.vstack(results)

        if self.postprocessing_defences is not None:
            results = self._apply_postprocessing(preds=results, fit=False, x=x)
        return results

    def evaluate(self,
                 x: Union[torch.Tensor, np.ndarray],
                 y: Union[torch.Tensor, np.ndarray],
                 batch_size: int = 64,
                 verbose: bool = False,
                 **kwargs) -> Tuple[float, float]:
        """
        Evaluate loss and accuracy of the classifier on `x, y`.

        :param x: Input images NCHW
        :param y: Target values as integer values. Can also be one-hot encoded, but will be argmax reduced.
        :param batch_size: The batch size used for evaluating the classifer's `model`.
        :return: loss, accuracy
        """
        self.learning_phase = False or self.override_learning_phase

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(self.device)

        # Apply preprocessing
        if self.preprocessing:
            x, _ = self._apply_preprocessing(x.cpu().numpy(),
                                             y=None,
                                             fit=False)
            x = torch.from_numpy(x).to(self.device)

        if self._reduce_labels and len(y.shape) > 1:
            y = torch.argmax(y, dim=1)

        with torch.no_grad():
            results: list = []
            num_batches = int(np.ceil(len(x) / float(batch_size)))
            with tqdm(range(num_batches), disable=not verbose) as pbar:
                for m in pbar:
                    begin, end = m * batch_size, min((m + 1) * batch_size, x.shape[0])
                    data = x[begin:end].to(self._device)
                    output = self._model(data)[-1]
                    results.append(output.detach().cpu().numpy())
            results = np.vstack(results)

        # Apply postprocessing.
        if self.postprocessing_defences is not None:
            results = self._apply_postprocessing(preds=results, fit=False, x=x)

        preds = np.argmax(results, axis=1)
        acc = len(np.where(y.cpu().numpy() == preds)[0]) / len(y)

        results = torch.from_numpy(results).to(self.device)
        loss = self.loss.compute_loss(results, y, x=x).cpu().numpy()

        return loss, acc

    def fit_batch(self,
                  x: Union[torch.Tensor, np.ndarray],
                  y: torch.Tensor,
                  eval_mode: bool = False,
                  all_features: bool = False,
                  **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Fits the model on a batch of data.
        """
        self.learning_phase = True and (not eval_mode)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(self.device)

        # Apply preprocessing
        if self.preprocessing:
            print(self.preprocessing)
            x, _ = self._apply_preprocessing(x.cpu().numpy(),
                                             y=None,
                                             fit=False)
            x = torch.from_numpy(x).to(self.device)

        if self._reduce_labels and len(y.shape) > 1:
            y = torch.argmax(y, dim=1)

        self.optimizer.zero_grad()
        if all_features:
            out = self._model(x)
            loss = self.loss.compute_loss(out, y, x=x)
            loss.backward()
            self.optimizer.step()
            out = out[-1]
        else:
            out = self._model(x)[-1]
            loss = self.loss.compute_loss(out, y, x=x)
            loss.backward()
            self.optimizer.step()

        return loss, out

    def fit(self,
            x, y, batch_size=128, nb_epochs=10, shuffle=True, verbose=True, callbacks=None, catch=True, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param shuffle: Whether or not to shuffle data before training
        :param verbose: Whether or not to log verbosely
        :param callbacks: list of objects with the methods on_batch_end() and on_epoch_end(); called at
                          the corresponding times during training
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        import torch

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for e in range(nb_epochs):
            self.learning_phase = True

            moving_acc = 0
            moving_loss = 0

            # Shuffle the examples
            if shuffle:
                random.shuffle(ind)

            # Train for one epoch
            train_loop = tqdm(range(num_batch), desc="Fit")
            for m in train_loop:
                i_batch = torch.from_numpy(x_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self._model(i_batch)

                # Form the loss function
                loss = self._loss.compute_loss(model_outputs[-1], o_batch, x=i_batch)

                # Actual training
                loss.backward()
                self._optimizer.step()

                if verbose:
                    if m == 0:
                        moving_acc = self.accuracy(o_batch, model_outputs[-1])
                        moving_loss = loss
                    else:
                        moving_acc = ALPHA * self.accuracy(o_batch, model_outputs[-1]) + (1 - ALPHA) * moving_acc
                        moving_loss = ALPHA * loss + (1 - ALPHA) * moving_loss

                    train_loop.set_description(f"({e + 1}/{nb_epochs}): Acc ({moving_acc:.4f}%) Loss ({moving_loss:.4f})")

                if callbacks:
                    for callback in callbacks:
                        callback.on_batch_end(m)

            try:
                if callbacks:
                    for callback in callbacks:
                        callback.on_epoch_end(e, loss=moving_loss, lr=self.lr)
            except StopTrainingException:
                if catch:
                    print("Stopping training as StopTrainingException was raised")
                    break
                else:
                    raise

    def fit_generator(self, generator, nb_epochs=20, verbose=True, callbacks=None, catch=True, **kwargs):
        """
        Fit the classifier using the generator that yields batches as specified.

        :param generator: Batch generator providing `(x, y)` for each epoch.
        :type generator: :class:`.DataGenerator`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        :param verbose: Whether or not to log verbosely
        :param callbacks: list of objects with the methods on_batch_end() and on_epoch_end(); called at
                          the corresponding times during training
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        :type kwargs: `dict`
        :return: `None`
        """
        import torch
        from wrt.data_generators import PyTorchDataGenerator

        # Train directly in PyTorch
        if isinstance(generator, PyTorchDataGenerator) and \
                (self.preprocessing_defences is None or self.preprocessing_defences == []) and \
                self.preprocessing == (0, 1):
            for e in range(nb_epochs):
                self.learning_phase = True

                moving_acc = 0
                moving_loss = 0

                train_loop = tqdm(enumerate(generator.iterator), desc="Fit", total=(generator.size//generator.batch_size))
                for batch_id, (i_batch, o_batch) in train_loop:
                    if isinstance(i_batch, np.ndarray):
                        i_batch = torch.from_numpy(i_batch).to(self._device)
                    else:
                        i_batch = i_batch.to(self._device)

                    if isinstance(o_batch, np.ndarray):
                        o_batch = torch.from_numpy(o_batch)
                    o_batch = o_batch.to(self._device)

                    if self._reduce_labels and len(o_batch.shape) > 1:
                        o_batch = torch.argmax(o_batch, dim=1)

                    # Zero the parameter gradients
                    self._optimizer.zero_grad()

                    # Perform prediction
                    model_outputs = self._model(i_batch)

                    # Form the loss function
                    loss = self._loss.compute_loss(model_outputs[-1], o_batch, x=i_batch)

                    # Actual training
                    loss.backward()
                    self._optimizer.step()

                    if verbose:
                        if batch_id == 0:
                            moving_acc = self.accuracy(o_batch, model_outputs[-1])
                            moving_loss = loss
                        else:
                            moving_acc = ALPHA * self.accuracy(o_batch, model_outputs[-1]) + (1 - ALPHA) * moving_acc
                            moving_loss = ALPHA * loss + (1 - ALPHA) * moving_loss

                        train_loop.set_description(f"({e + 1}/{nb_epochs}): Acc ({moving_acc:.4f}%) Loss ({moving_loss:.4f})")

                    if callbacks:
                        for callback in callbacks:
                            callback.on_batch_end(batch_id)

                try:
                    if callbacks:
                        for callback in callbacks:
                            callback.on_epoch_end(e, loss=moving_loss, lr=self.lr)
                except StopTrainingException:
                    if catch:
                        print("Stopping training as StopTrainingException was raised")
                        break
                    else:
                        raise
                if verbose and ("x_test" in kwargs) and ("y_test" in kwargs):
                    print("Test Accuracy: {}".format(self.accuracy(np.argmax(kwargs.get("y_test"), axis=1), self.predict(kwargs.get("x_test")))))
        else:
            # Fit a generic data generator through the API
            super(PyTorchClassifier, self).fit_generator(generator, nb_epochs=nb_epochs)

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
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
        import torch

        if not (
                (label is None)
                or (isinstance(label, (int, np.integer)) and label in range(self._nb_classes))
                or (
                        isinstance(label, np.ndarray)
                        and len(label.shape) == 1
                        and (label < self._nb_classes).all()
                        and label.shape[0] == x.shape[0]
                )
        ):
            raise ValueError("Label %s is out of range." % label)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        x_preprocessed = torch.from_numpy(x_preprocessed).to(self._device)

        # Compute gradients
        if self._layer_idx_gradients < 0:
            x_preprocessed.requires_grad = True

        # Run prediction
        model_outputs = self._model(x_preprocessed)

        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_preprocessed

        # Set where to get gradient from
        preds = model_outputs[-1]

        # Compute the gradient
        grads = []

        def save_grad():
            def hook(grad):
                grads.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self._model.zero_grad()
        if label is None:
            for i in range(self.nb_classes()):
                torch.autograd.backward(
                    preds[:, i], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
                )

        elif isinstance(label, (int, np.integer)):
            torch.autograd.backward(
                preds[:, label], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
            )
        else:
            unique_label = list(np.unique(label))
            for i in unique_label:
                torch.autograd.backward(
                    preds[:, i], torch.Tensor([1.0] * len(preds[:, 0])).to(self._device), retain_graph=True
                )

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = grads[np.arange(len(grads)), lst]

            grads = grads[None, ...]

        grads = np.swapaxes(np.array(grads), 0, 1)
        grads = self._apply_preprocessing_gradient(x, grads)

        return grads

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        import torch

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Convert the inputs to Tensors
        inputs_t = torch.from_numpy(x_preprocessed).to(self._device)
        inputs_t.requires_grad = True

        # Convert the labels to Tensors
        labels_t = torch.from_numpy(y_preprocessed).to(self._device)

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        loss = self._loss.compute_loss(model_outputs[-1], labels_t, x=inputs_t)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = inputs_t.grad.cpu().numpy().copy()
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x.shape

        return grads

    def loss_gradient_framework(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: Gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]
        from torch.autograd import Variable

        # Check label shape
        if self._reduce_labels:
            y = torch.argmax(y, dim=1)

        # Convert the inputs to Variable
        x = Variable(x, requires_grad=True)

        # Compute the gradient and return
        model_outputs = self._model(x)
        loss = self._loss(model_outputs[-1], y)

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = x.grad
        assert grads.shape == x.shape  # type: ignore

        return grads  # type: ignore

    def get_all_activations(self, x):
        """
        Return the output of every layer for the input x
        :param x: np.ndarray or torch Tensor
        :return: list of torch Tensors
        """
        import torch

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if hasattr(self._model._model, 'return_hidden_activations'):
            self._model._model.return_hidden_activations = True

        outputs = self._model(x.to(self.device))

        if self.shuffle_intermediate_features:
            for i, output in enumerate(outputs[:-1]):
                if len(self.shuffle_idx) < i:
                    idx = np.random.arange(output.shape[0])
                    np.random.shuffle(idx)
                    self.shuffle_indices[i] = idx
                outputs[i] = output[self.shuffle_indices[i]]

        if hasattr(self._model._model, 'return_hidden_activations'):
            self._model._model.return_hidden_activations = False

        return outputs

    def get_weights(self):
        """
        Return the layer weights of the neural network in sequence, ordered in the same
        way as a forward-pass through the network.
        :return: list of torch Tensors
        """
        return list(self._model._model.parameters())

    @property
    def device(self):
        return self._device

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either. In addition, the function can only infer the internal layers if the input
                     model is of type `nn.Sequential`, otherwise, it will only return the logit layer.
        """
        return self._layer_names

    @property
    def input_shape(self):
        """
        Return the shape of one input.

        :return: Shape of one input for the classifier.
        :rtype: `tuple`
        """
        return self._input_shape

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    @property
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
        return self._learning_phase

    @learning_phase.setter
    def learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: `True` if the learning phase is training, `False` if learning phase is not training.
        :type train: `bool`
        """
        if isinstance(train, bool):
            self._learning_phase = train
            self._model.train(train)

    @property
    def loss(self):
        """
        Get the current loss function
        :return: Loss
        """
        return self._loss

    @loss.setter
    def loss(self, loss):
        """
        Set the loss function to the one given
        :param loss: Loss
        :return: None
        """
        self._loss = self._make_loss_wrapper(loss)
        self._reduce_labels = self._loss.reduce_labels()

    @property
    def lr(self):
        """
        Get the current learning rate
        :return: float
        """
        return self._optimizer.param_groups[0]['lr']

    @lr.setter
    def lr(self, value):
        for g in self._optimizer.param_groups:
            g['lr'] = value

    @property
    def optimizer(self):
        """
        Get the optimizer
        :return: PyTorch optimizer
        """
        return self._optimizer

    @property
    def functional(self):
        """
        Return the Functional instance associated with this classifier
        :return: PyTorchFunctional
        """
        return self._functional

    @property
    def model(self):
        """
        Return the classifier model
        :return: PyTorch Module
        """
        return self._model._model

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `WRT_DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os
        import torch

        if path is None:
            from wrt.config import WRT_DATA_PATH

            full_path = os.path.join(WRT_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        # pylint: disable=W0212
        # disable pylint because access to _modules required
        try:
            torch.save(self._model._model.state_dict(), full_path + ".model")
            torch.save(self._optimizer.state_dict(), full_path + ".optimizer")
        except:
            pass  # Deprecated

        logger.info("Model state dict saved in path: %s.", full_path + ".model")
        logger.info("Optimizer state dict saved in path: %s.", full_path + ".optimizer")

    def load(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `WRT_DATA_PATH`.
        :type path: `str`
        :return: None
        """
        import os
        import torch

        if path is None:
            from wrt.config import WRT_DATA_PATH

            full_path = os.path.join(WRT_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        # pylint: disable=W0212
        # disable pylint because access to _modules required
        self._model._model.load_state_dict(torch.load(full_path + ".model"), strict=False)
        self._optimizer.load_state_dict(torch.load(full_path + ".optimizer"))
        logger.info("Model state dict loaded from path: %s.", full_path + ".model")
        logger.info("Optimizer state dict loaded from path: %s.", full_path + ".optimizer")

    def __getstate__(self):
        """
        Use to ensure `PytorchClassifier` can be pickled.

        :return: State dictionary with instance parameters.
        :rtype: `dict`
        """
        import time
        import copy

        # pylint: disable=W0212
        # disable pylint because access to _model required
        state = self.__dict__.copy()
        state["inner_model"] = copy.copy(state["_model"]._model)

        if not isinstance(self._loss, PyTorchClassifier.LossWrapper):
            state["inner_loss"] = copy.copy(state["_loss"])
        else:
            state["inner_loss"] = copy.copy(state["_loss"].loss)

        # Remove the unpicklable entries
        del state["_model_wrapper"]
        del state["_device"]
        del state["_model"]
        del state["_loss"]

        model_name = str(time.time())
        state["model_name"] = model_name
        self.save(model_name)

        return state

    def __setstate__(self, state):
        """
        Use to ensure `PytorchClassifier` can be unpickled.

        :param state: State dictionary with instance parameters to restore.
        :type state: `dict`
        """
        self.__dict__.update(state)

        # Load and update all functionality related to Pytorch
        import os
        import torch
        from wrt.config import WRT_DATA_PATH

        # Recover model
        full_path = os.path.join(WRT_DATA_PATH, state["model_name"])
        model = state["inner_model"]
        model.load_state_dict(torch.load(str(full_path) + ".model"))
        self._model = self._make_model_wrapper(model)

        # Recover loss function
        self._loss = self._make_loss_wrapper(state["inner_loss"])
        self._reduce_labels = self._loss.reduce_labels()

        # Recover device
        if not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        self._model.to(self._device)

        # Recover optimizer
        try:
            self._optimizer.load_state_dict(torch.load(str(full_path) + ".optimizer"))
        except:
            pass  # Deprecated

        # experimentally. We delete models once they are loaded.
        try:
            if os.path.exists(full_path + ".model"):
                os.remove(full_path + ".model")
            if os.path.exists(full_path + ".optimizer"):
                os.remove(full_path + ".optimizer")
        except:
            pass

        self.__dict__.pop("model_name", None)
        self.__dict__.pop("inner_model", None)

    def __repr__(self):
        repr_ = (
                "%s(model=%r, loss=%r, optimizer=%r, input_shape=%r, nb_classes=%r, channel_index=%r, "
                "clip_values=%r, preprocessing_defences=%r, postprocessing_defences=%r, preprocessing=%r)"
                % (
                    self.__module__ + "." + self.__class__.__name__,
                    self._model,
                    self._loss,
                    self._optimizer,
                    self._input_shape,
                    self.nb_classes(),
                    self.channel_index,
                    self.clip_values,
                    self.preprocessing_defences,
                    self.postprocessing_defences,
                    self.preprocessing,
                )
        )

        return repr_

    class LossWrapper(Loss):
        """
        Wrapper for the loss function
        """
        def __init__(self, loss):
            self.loss = loss

        def reduce_labels(self):
            import torch
            return isinstance(self.loss, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss))

        def compute_loss(self, pred, true, x=None):
            return self.loss(pred, true)

        def __call__(self, *args, **kwargs):
            return self.compute_loss(*args, **kwargs)

    def _make_loss_wrapper(self, loss):
        import torch

        if not isinstance(loss, Loss):
            assert isinstance(loss, torch.nn.modules.loss._Loss)
            return PyTorchClassifier.LossWrapper(loss)
        else:
            loss.set_functional(PyTorchFunctional())
            return loss

    def _make_model_wrapper(self, model):
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        try:
            import torch.nn as nn

            # Define model wrapping class only if not defined before
            if not hasattr(self, "_model_wrapper"):

                class ModelWrapper(nn.Module):
                    """
                    This is a wrapper for the input model.
                    """

                    def __init__(self, model):
                        """
                        Initialization by storing the input model.

                        :param model: PyTorch model. The forward function of the model must return the logit output.
                        :type model: is instance of `torch.nn.Module`
                        """
                        super(ModelWrapper, self).__init__()
                        self._model = model

                    # pylint: disable=W0221
                    # disable pylint because of API requirements for function
                    def forward(self, x):
                        """
                        This is where we get outputs from the input model.

                        :param x: Input data.
                        :type x: `torch.Tensor`
                        :return: a list of output layers, where the last 2 layers are logit and final outputs.
                        :rtype: `list`
                        """
                        # pylint: disable=W0212
                        # disable pylint because access to _model required
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            for _, module_ in self._model._modules.items():
                                x = module_(x)
                                result.append(x)

                        elif isinstance(self._model, nn.Module):
                            x = self._model(x)
                            if isinstance(x, list):
                                result.extend(x)
                            else:
                                result.append(x)

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")

                        return result

                    @property
                    def get_layers(self):
                        """
                        Return the hidden layers in the model, if applicable.

                        :return: The hidden layers in the model, input and output layers excluded.
                        :rtype: `list`

                        .. warning:: `get_layers` tries to infer the internal structure of the model.
                                     This feature comes with no guarantees on the correctness of the result.
                                     The intended order of the layers tries to match their order in the model, but this
                                     is not guaranteed either. In addition, the function can only infer the internal
                                     layers if the input model is of type `nn.Sequential`, otherwise, it will only
                                     return the logit layer.
                        """
                        import torch.nn as nn

                        result = []
                        if isinstance(self._model, nn.Sequential):
                            # pylint: disable=W0212
                            # disable pylint because access to _modules required
                            for name, module_ in self._model._modules.items():
                                result.append(name + "_" + str(module_))

                        elif isinstance(self._model, nn.Module):
                            result.append("final_layer")

                        else:
                            raise TypeError("The input model must inherit from `nn.Module`.")
                        logger.info("Inferred %i hidden layers on PyTorch classifier.", len(result))

                        return result

                # Set newly created class as private attribute
                self._model_wrapper = ModelWrapper

            # Use model wrapping class to wrap the PyTorch model received as argument
            return self._model_wrapper(model)

        except ImportError:
            raise ImportError("Could not find PyTorch (`torch`) installation.")


class PyTorchFunctional(Functional):
    def __init__(self):
        import torch

        super(PyTorchFunctional, self).__init__()

        if not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

    def tensor(self, x, diff=False):
        import torch
        x = torch.tensor(x).to(self._device)
        if diff:
            x = x.float()
            x.requires_grad = True
        return x

    def numpy(self, x):
        return x.clone().detach().cpu().numpy()

    def shape(self, x):
        return x.size()

    def matmul(self, x, y):
        import torch
        return torch.matmul(x, y)

    def reshape(self, x, *dims):
        import torch
        return torch.reshape(x, dims)

    def transpose(self, x):
        import torch
        return torch.transpose(x, dim0=0, dim1=1)

    def tile(self, x, *dims):
        import torch
        return torch.Tensor.repeat(x, *dims)

    def equal(self, x, y, return_bool=False):
        import torch
        if return_bool:
            return torch.eq(x, y)
        else:
            return torch.eq(x, y).int()

    def mean(self, x, axis=None):
        import torch
        if axis is None:
            return torch.mean(x)
        else:
            return torch.mean(x, axis)

    def sum(self, x, axis=None, keep_dims=False):
        import torch
        if axis is None:
            return torch.sum(x)
        else:
            return torch.sum(x, dim=axis, keepdim=keep_dims)

    def pow(self, x, exp):
        import torch
        return torch.pow(x, exp)

    def exp(self, x):
        import torch
        return torch.exp(x)

    def log(self, x):
        import torch
        return torch.log(x)

    def abs(self, x):
        import torch
        return torch.abs(x)

    def softmax(self, x, axis=None):
        import torch.nn.functional as F
        return F.softmax(x, dim=axis)

    def sigmoid(self, x):
        import torch
        return torch.sigmoid(x)

    def cross_entropy_loss(self, pred, true):
        import torch.nn.functional as F
        return F.cross_entropy(pred, true, reduction='mean')

    def binary_cross_entropy_loss(self, pred, true, reduction='mean'):
        import torch.nn.functional as F
        return F.binary_cross_entropy(pred, true, reduction=reduction)

    def mse_loss(self, pred, true):
        import torch.nn.functional as F
        return F.mse_loss(pred, true, reduction='mean')

    def gradient(self, function, base):
        if base.grad is not None:
            base.grad.zero_()
        function.backward(retain_graph=True)
        return base.grad.detach().cpu().numpy().copy()
