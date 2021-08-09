from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from typing import List

import mlconfig
import numpy as np

from wrt.classifiers import PyTorchClassifier
from wrt.classifiers.loss import Loss
from wrt.defenses.watermark.watermark import Watermark
from wrt.training import WRTCallback, EarlyStoppingWRTCallback
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.trainer import Trainer


class Uchida(Watermark):
    """
    Implements the Uchida watermarking scheme.
    https://arxiv.org/pdf/1701.04082.pdf
    """

    class RegularizedLoss(Loss):
        """
        Regularization loss used for embedding a watermark
        """

        def __init__(self, classifier, layer_index, b, X, lmbda=0.01):
            """
            Initiate an instance
            :param classifier: Classifier instance
            :param layer_index: index to embed
            :param b: vector to embed
            :param X: secret random projection matrix
            :param lmbda: strength of regularization term
            """
            super(Uchida.RegularizedLoss, self).__init__(classifier)

            self._weights = classifier.get_weights()[layer_index]
            if len(self._weights.shape) == 4:
                # average the output channels
                self._weights = self._weights.mean(0).reshape(-1)
            else:
                self._weights = self._weights.reshape(-1)

            self._b_np = b
            self._X_np = X
            self._lambda = lmbda

            self._b = None
            self._X = None

        def on_functional_change(self):
            self._b = self._functional.tensor(self._b_np)
            self._X = self._functional.tensor(self._X_np)

        def reduce_labels(self):
            return True

        def compute_loss(self, pred, true, x=None):
            w = self._functional.reshape(self._weights, -1)

            ce_loss = self._functional.cross_entropy_loss(pred, true)

            y = self._functional.matmul(self._X, w)
            y = self._functional.sigmoid(y)

            reg_loss = self._functional.binary_cross_entropy_loss(y, self._b.float())

            # the paper doesn't seem to do mean reduction on the regularization loss
            reg_loss = reg_loss * self._functional.shape(self._b)[0]

            return ce_loss + self._lambda * reg_loss

    def __init__(self, classifier, layer_idx=0, lmbda=0.01, **kwargs):
        """
        Create a :class:`Uchida` instance.

        :param classifier: Model to train.
        :type classifier: :class:`.Classifier`
        :param b_length: Length of the vector to embed
        :param layer_idx: Index of the layer to embed the fingerprint.
        :param lmbda: float; strength of the regularization term
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier

        # embed the watermark into the first layer
        self.layer_idx = layer_idx
        self.lmbda = lmbda

    @staticmethod
    def get_name():
        return "Uchida"

    def keygen(self,
               keylength: int,
               layer_idx: int = None,
               y_wm: np.ndarray = None,
               **kwargs):
        if layer_idx is None:
            layer_idx = self.layer_idx

        w = self.classifier.get_weights()[layer_idx]
        w = self.classifier.functional.numpy(w)
        if len(w.shape) == 4:
            # average the output channels
            w = np.mean(w, axis=0).reshape(-1)
        else:
            w = w.reshape(-1)

        # Sample message if none is provided.
        if y_wm is None:
            y_wm = np.random.randint(0, 2, size=keylength)
        # Generate a random projection matrix
        x_wm = np.random.normal(0, 1, [keylength, w.shape[0]]).astype(np.float32)

        return x_wm, y_wm

    def embed(self,
              train_loader: WRTDataLoader,
              valid_loader: WRTDataLoader,
              epochs: int,
              keylength: int,
              y_wm: np.ndarray = None,
              output_dir: str = None,
              min_val: float = 1.0,
              patience: int = 5,
              evaluate_every_n_batches: int = None,
              callbacks: List[WRTCallback] = None,
              device="cuda",
              **kwargs):
        """
        Train a model on the watermark data.
        :param train_loader The training data loader.
        :param valid_loader The validation data loader
        :param epochs Number of epochs to embed.
        :param keylength Number of watermarking keys to embed
        :param y_wm Secret message containing bits. Will be sampled randomly if none is provided. 
        :param output_dir (optional) Directory to save intermediate results and the trained model
        :param evaluate_every_n_batches (optional) Logging interval after which to log the watermark accuracy
        :param min_val Minimum wm accuracy for early stopping.
        :param patience Patience for early stopping.
        :param learning_rate_multiplier Multiplier for the learning rate during embedding.
        :param callbacks List of callbacks during embedding.
        :param device Device to train on.
        :return: watermarking keys as np.ndarray
        """
        if callbacks is None:
            callbacks = []

        # Generate the watermarking key.
        x_wm, y_wm = self.keygen(keylength=keylength,
                                 layer_idx=self.layer_idx,
                                 y_wm=y_wm)

        ce_loss = self.classifier.loss
        reg_loss = Uchida.RegularizedLoss(classifier=self.get_classifier(), layer_index=self.layer_idx,
                                          b=y_wm, X=x_wm, lmbda=self.lmbda)
        self.classifier.loss = reg_loss

        callbacks.append(EarlyStoppingWRTCallback(lambda: self.verify(x_wm, y_wm)[0], verbose=True,
                                                  patience=patience, check_every_n_batches=evaluate_every_n_batches))

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=device, num_epochs=epochs, callbacks=callbacks)
        history = trainer.fit()

        # Save the final model with all parameters necessary for validation of the watermark.
        if output_dir is not None:
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(history, fp=f)

            checkpoint = {
                'model': self.get_classifier().model.state_dict(),
                'optimizer': self.get_classifier().optimizer.state_dict(),
                'layer_idx': self.layer_idx,
                'x_wm': x_wm,
                'y_wm': y_wm
            }
            self.save(filename='best.pth', path=output_dir, checkpoint=checkpoint)

        self.classifier.loss = ce_loss
        return x_wm, y_wm

    def load(self, filename, path=None, **load_kwargs: dict):
        """ Loads parameters necessary for validating a watermark.
        """
        checkpoint = super().load(filename, path, **load_kwargs)

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])

        self.layer_idx = checkpoint['layer_idx']
        return checkpoint["x_wm"], checkpoint["y_wm"]

    def extract(self,
                x: np.ndarray,
                classifier=None,
                **kwargs):
        """
        Return the bit-error rate between the given vector and the
            vector extracted from layer weights
        For consistency with the rest of the interface, return instead the bit accuracy
            as a float between 0 and 1.
        :param x: np.ndarray; the watermarking key
        :param classifier: Classifier; if provided, extract the watermark from
                           the given classifier instead of this object's classifier
        :param kwargs: unused
        :return: int
        """
        if classifier is None:
            classifier = self.get_classifier()

        w = classifier.get_weights()[self.layer_idx]
        w = classifier.functional.numpy(w)
        if len(w.shape) == 4:
            # average the output channels
            w = np.mean(w, axis=0).reshape(-1)
        else:
            w = w.reshape(-1)

        print(w.shape, w.mean())
        #w = -w
        #print(w.shape, w.mean())

        b_extract = np.matmul(x, w)
        b_extract[b_extract >= 0] = 1
        b_extract[b_extract < 0] = 0
        return b_extract

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


######################################################################################################################
####################################### Used for running experiments. ################################################
@mlconfig.register
def wm_uchida(classifier: PyTorchClassifier, **kwargs):
    return Uchida(classifier=classifier, **kwargs)


@mlconfig.register
def wm_uchida_embed(defense: Uchida, **kwargs):
    return defense.embed(**kwargs), defense


@mlconfig.register
def wm_uchida_keygen(defense: Uchida,
                     keylengths: List[int],
                     **kwargs: dict):
    x_wm, y_wm = defense.keygen(keylength=max(keylengths))
    for n in keylengths:
        yield x_wm[:n], y_wm[:n]
