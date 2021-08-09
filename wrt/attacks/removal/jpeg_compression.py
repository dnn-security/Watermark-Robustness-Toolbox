"""
This module implements the White box attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable

from tqdm import tqdm
import numpy as np

from wrt.classifiers.classifier import Classifier
from wrt.attacks.attack import RemovalAttack
from wrt.exceptions import ClassifierError
from wrt.preprocessors import Preprocessor
from wrt.training.datasets import WRTDataLoader
import mlconfig

logger = logging.getLogger(__name__)


class JPEGCompression(RemovalAttack):
    """
    The attack preprocesses inputs by compressing them.
    """

    attack_params = RemovalAttack.attack_params

    class JPEGCompressionPreprocessor(Preprocessor):

        def __init__(self, quality: int,
                     normalize_fn: Callable[[np.ndarray], np.ndarray],
                     unnormalize_fn: Callable[[np.ndarray], np.ndarray]):
            super().__init__()
            self.quality = quality
            self.normalize_fn = normalize_fn
            self.unnormalize_fn = unnormalize_fn

        @property
        def apply_fit(self):
            return False

        @property
        def apply_predict(self):
            return True

        def __compress(self, x):
            from PIL import Image
            from io import BytesIO

            x = np.transpose(x, (0, 2, 3, 1))

            # Convert into uint8
            x = x * 255
            x = x.astype("uint8")

            # Set image mode
            if x.shape[-1] == 1:
                image_mode = "L"
            elif x.shape[-1] == 3:
                image_mode = "RGB"
            else:
                raise NotImplementedError("Currently only support `RGB` and `L` images.")

            # Prepare grayscale images for "L" mode
            if image_mode == "L":
                x = np.squeeze(x, axis=-1)

            # Compress one image at a time
            x_jpeg = x.copy()
            for idx in tqdm(range(x.shape[0]), desc="JPEG compression", disable=True):
                with BytesIO() as tmp_jpeg:
                    x_image = Image.fromarray(x[idx], mode=image_mode)
                    x_image.save(tmp_jpeg, format="jpeg", quality=self.quality)
                    x_jpeg[idx] = np.array(Image.open(tmp_jpeg))

            # Undo preparation grayscale images for "L" mode
            if image_mode == "L":
                x_jpeg = np.expand_dims(x_jpeg, axis=-1)

            x_jpeg = x_jpeg / 255.0
            x_jpeg = x_jpeg.astype(np.float32)
            x_jpeg = np.transpose(x_jpeg, (0, 3, 1, 2))
            return x_jpeg

        def __call__(self, x, y=None):
            """
            Perform data preprocessing and return preprocessed data as tuple.
            :param x: Dataset to be preprocessed.
            :param y: Labels to be preprocessed.
            :return: Preprocessed data.
            """
            x = self.unnormalize_fn(x)
            x_compressed = self.__compress(x)
            x_compressed = self.normalize_fn(x_compressed)
            return x_compressed, y

        def fit(self, x, y=None, **kwargs):
            pass

    def __init__(self, classifier, quality=90, **kwargs):
        """
        :param classifier: Classifier
        :param quality: int between 1 and 99; higher = better quality
        """
        super(JPEGCompression, self).__init__(classifier)

        if not isinstance(classifier, Classifier):
            raise ClassifierError(self.__class__, [Classifier], classifier)

        kwargs = {}
        JPEGCompression.set_params(self, **kwargs)

        self.quality = quality

    @classmethod
    def is_valid_classifier_type(cls, classifier):
        """
        Checks whether the classifier provided is a classifier which this class can perform an attack on
        :param classifier:
        :return:
        """
        return True if isinstance(classifier, Classifier) else False

    def remove(self,
               train_loader: WRTDataLoader,
               **kwargs):
        """Apply the attack

        :param train_loader The training dataset loader.
        """
        preprocessor = JPEGCompression.JPEGCompressionPreprocessor(self.quality,
                                                                   train_loader.normalize,
                                                                   train_loader.unnormalize)
        self.classifier.add_preprocessor(preprocessor, 'jpeg')

    def remove_generator(self, generator, **kwargs):
        """Attempt to remove the watermark

        :param generator: (inputs, target) data generator
        :return: An array holding the loss and accuracy
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

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
def jpeg_compression_attack(classifier, **kwargs):
    return JPEGCompression(classifier, **kwargs)


@mlconfig.register
def jpeg_compression_removal(attack: JPEGCompression,
                       train_loader,
                       **kwargs):
    return attack, attack.remove(train_loader=train_loader, **kwargs)