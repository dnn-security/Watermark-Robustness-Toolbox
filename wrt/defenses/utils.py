import torch
import numpy as np
from torch.utils import data

from wrt.preprocessors import Preprocessor


class NormalizingPreprocessor(Preprocessor):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        super().__init__()

    @property
    def apply_fit(self):
        return True

    @property
    def apply_predict(self):
        return True

    def __call__(self, x, y=None):
        """
        Perform data preprocessing and return preprocessed data as tuple.
        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        x_norm = x - self.mean
        x_norm = x_norm / self.std
        x_norm = x_norm.astype(np.float32)
        return x_norm, y

    def estimate_gradient(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        std = np.asarray(self.std, dtype=np.float32)
        gradient_back = gradient / std
        return gradient_back

    def fit(self, x, y=None, **kwargs):
        pass