from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from typing import List

import mlconfig
import numpy as np

from wrt.classifiers import PyTorchClassifier
from wrt.classifiers.loss import Loss
from wrt.config import WRT_NUMPY_DTYPE
from wrt.defenses.watermark.watermark import Watermark
from wrt.training import callbacks as wrt_callbacks
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.trainer import Trainer


class Deepmarks(Watermark):
    """
    Implement the Deepmarks fingerprinting scheme.
    Because of limitations to the interface, this only handles embedding
        fingerprints in one model at a time
    https://arxiv.org/pdf/1804.03648.pdf
    """

    class DMLoss(Loss):
        """
        Loss used for fine-tuning a Deepmarks model
        """

        def __init__(self, classifier, layer_index, f, X, gamma=0.1):
            """
            Initiate an instance
            :param classifier: Classifier instance
            :param layer_index: index to embed
            :param f: fingerprint to embed
            :param X: secret random projection matrix
            :param gamma: embedding strength
            """
            super(Deepmarks.DMLoss, self).__init__(classifier)

            self._weights = classifier.get_weights()[layer_index]
            self._f_np = f
            self._X_np = X
            self._gamma = gamma

            self._f = None
            self._X = None

        def reduce_labels(self):
            return True

        def on_functional_change(self):
            self._f = self._functional.tensor(self._f_np)
            self._X = self._functional.tensor(self._X_np)

        def compute_loss(self, pred, true, x=None):
            if len(self._functional.shape(self._weights)) == 4:
                # average the output channels
                w = self._functional.mean(self._weights, axis=0)
                w = self._functional.reshape(w, -1)
            else:
                w = self._functional.reshape(self._weights, -1)

            ce_loss = self._functional.cross_entropy_loss(pred, true)
            fp_loss = self._functional.mse_loss(self._functional.matmul(self._X, w), self._f)

            # it's not clear if the paper does mean reduction on the reg loss
            fp_loss = fp_loss * self._functional.shape(self._f)[0]

            return ce_loss + self._gamma * fp_loss

    def __init__(self, classifier, layer_index=0, gamma=0.1, **kwargs):
        """
        Create an :class:`Deepmarks` instance.

        :param classifier: Model to train.
        :type classifier: :class:`.Classifier`
        :param fp_length: Length of the fingerprint to embed
        :param layer_index: Index of the layer to embed the fingerprint.
        :param gamma: float; strength of the regularization term
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier

        # embed the fingerprint into the first layer
        self.layer_index = layer_index
        self.gamma = gamma

    @staticmethod
    def get_name():
        return "Deepmarks"

    def keygen(self,
               keylength: int,
               layer_index: int = None,
               **kwargs):
        if layer_index is None:
            layer_index = self.layer_index

        w = self.classifier.get_weights()[layer_index]
        w = self.classifier.functional.numpy(w)
        if len(w.shape) == 4:
            # average the output channels
            w = np.mean(w, axis=0).reshape(-1)
        else:
            w = w.reshape(-1)

        # generate a random projection matrix
        x_wm = np.random.rand(keylength, w.shape[0]).astype(WRT_NUMPY_DTYPE)
        y_wm = self.extract(x_wm)
        return x_wm, y_wm

    def embed(self,
              train_loader: WRTDataLoader,
              valid_loader: WRTDataLoader,
              keylength: int,
              signature: np.ndarray = None,
              epochs: int = 5,
              patience: int = 5,
              check_every_n_batches: int = 100,
              output_dir: str = None,
              device="cuda",
              **kwargs):
        """
        Train a model on the watermark data.

        :param train_loader Training data loader
        :param valid_loader Loader for the validation data.
        :param key_expansion_factor: Number of keys to generate for the embedding relative to the keylength.
        :param keylength: Number of keys to embed into the model.
        :param signature (optional) The secret watermarking message (in bits). If None, one is generated randomly.
        :param epochs Number of epochs to use for trainings.
        :param patience Patience for early stopping on the wm acc.
        :param check_every_n_batches: Check early stopping every n batches.
        :param finetune_batches Number of epochs for fine-tuning
        :param output_dir: The output directory for logging the models and intermediate training progress.
        :param device: cuda or cpu
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :return: Watermark train set, watermark labels
        """
        if signature is None:
            signature = np.random.randint(0, 2, size=keylength).astype(np.float32)

        x_wm, _ = self.keygen(keylength)

        ce_loss = self.classifier.loss
        dm_loss = Deepmarks.DMLoss(self.classifier,
                                   layer_index=self.layer_index,
                                   f=signature,
                                   X=x_wm,
                                   gamma=self.gamma)
        self.classifier.loss = dm_loss

        # Early stopping on the training loss.
        callbacks = [wrt_callbacks.EarlyStoppingCallback(metric="train_loss",
                                                         better="smaller",
                                                         log_after_n_batches=check_every_n_batches,
                                                         patience=patience)]

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=device, num_epochs=epochs, callbacks=callbacks)
        history = trainer.fit()
        self.classifier.loss = ce_loss

        y_wm = self.extract(x_wm)
        if output_dir is not None:
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(history, fp=f)

            checkpoint = {
                'model': self.get_classifier().model.state_dict(),
                'optimizer': self.get_classifier().optimizer.state_dict(),
                'x_wm': x_wm,
                'y_wm': y_wm
            }
            self.save('best.pth', output_dir, checkpoint)

        return x_wm, y_wm

    def load(self, filename, path=None, **load_kwargs: dict):
        """ Loads parameters necessary for validating a watermark.
        """
        checkpoint = super().load(filename, path, **load_kwargs)

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])

        print(checkpoint.keys())

        return checkpoint["x_wm"], checkpoint["y_wm"]

    def extract(self, x, classifier=None, **kwargs):
        """
        Return the correlation score between the given fingerprint x and
            the embedded fingerprint. The correlation score is a number between
            -1 and 1. Zero means no correlation, and close to 1 means strong
            positive correlation. In the Deepmarks paper, the correlation score
            is determined between the embedded fingerprint and every column of an
            orthogonal matrix. Here, we just compute it between the embedded fingerprint
            and the given fingerprint. Empirically, a value >0.6 indicates a
            fingerprinted model
        :param x: np.ndarray; The random projection matrix.
        :param classifier: Classifier; if provided, extract the watermark from
                           the given classifier instead of this object's classifier
        :param kwargs: unused
        :return: int representing the correlation score
        """
        if classifier is None:
            classifier = self.classifier

        w = classifier.get_weights()[self.layer_index]
        w = classifier.functional.numpy(w)
        if len(w.shape) == 4:
            # average the output channels
            w = np.mean(w, axis=0).reshape(-1)
        else:
            w = w.reshape(-1)

        f_extract = np.matmul(x, w)
        f_extract = f_extract / np.linalg.norm(f_extract)
        return f_extract

    def verify(self,
               x: np.ndarray,
               y_wm: np.ndarray = None,
               classifier: PyTorchClassifier = None,
               **kwargs):
        """ Verification procedure that checks if the watermark accuracy is high enough.
        For DeepMarks it is a special case, because we compute the correlation as the dot product between the
        extracted message and secret watermarking key.

        :param y_wm: Not used.
        :param x Secret watermarking key
        :param classifier The classifier to verify.
        """
        if y_wm is None:  # Extract the source model's message.
            print("Extracting the source model's message.. ")
            y_wm = self.extract(x)

        f_extract = self.extract(x, classifier=classifier)
        corr = np.dot(y_wm, f_extract)

        print(f"True Correlation: {corr}")

        corr = np.clip(corr, 0, 1)
        return corr, corr > 0  # ToDo: Implement decision boundary.

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


@mlconfig.register
def wm_deepmarks(classifier, **kwargs):
    return Deepmarks(classifier=classifier, **kwargs)


@mlconfig.register
def wm_deepmarks_embed(defense: Deepmarks, **kwargs):
    return defense.embed(**kwargs), defense


@mlconfig.register
def wm_deepmarks_keygen(defense: Deepmarks, keylengths: List[int], **kwargs):
    for n in keylengths:
        x_wm, y_wm = defense.keygen(keylength=n, **kwargs)
        yield x_wm[:n], y_wm[:n]
