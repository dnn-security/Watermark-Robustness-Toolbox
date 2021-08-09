from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List, Tuple

import mlconfig
import numpy as np
from torch.utils import data
import os
import json
import matplotlib.pyplot as plt

from wrt.training import callbacks as wrt_callbacks

from wrt.classifiers import PyTorchClassifier
from wrt.defenses.watermark.watermark import Watermark
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer


class Adi(Watermark):
    """
    Embedding a model independent watermark.
    | Paper link: https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf
    """

    def __init__(self,
                 classifier: PyTorchClassifier,
                 num_classes: int,
                 **kwargs):
        """
        Create an :class:`.Adi` instance.
        :param classifier: Model to embed the watermark.
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier
        self.num_classes = num_classes

    def visualize_key(self, x_wm: np.ndarray, output_dir: str = None):
        """ Visualizes the watermarking key.
        """
        idx = np.random.choice(np.arange(x_wm.shape[0]), size=9, replace=False)
        fig, _ = plt.subplots(nrows=3, ncols=3)
        fig.suptitle("Adi Watermarking Key")
        for j, i in enumerate(idx):
            plt.subplot(3, 3, j + 1)
            plt.axis('off')
            plt.imshow(x_wm[i].transpose((1, 2, 0)), aspect='auto')
        plt.subplots_adjust(hspace=0, wspace=0)
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, "wm_sample.png"))
        plt.show()

    @staticmethod
    def get_name():
        return "Adi"

    @staticmethod
    def keygen(wm_loader: data.DataLoader,
               num_classes: int,
               keylength: int,
               y_wm: np.ndarray = None) -> np.ndarray:
        """ Expects a data loader that outputs OOD images and returns a set of Adi trigger
         images. Randomly samples the labels if no message is given.
         @:param wm_loader The (unnormalized in range [0-1]) watermark dataset without labels.
        """
        x_wm = collect_n_samples(n=keylength, data_loader=wm_loader, has_labels=False)
        if y_wm is None:
            y_wm = np.random.randint(0, num_classes, size=x_wm.shape[0])
        return x_wm, y_wm

    def embed(self,
              train_loader: WRTDataLoader,
              valid_loader: WRTDataLoader,
              wm_loader: data.DataLoader,
              keylength: int,
              boost_factor: int = 10,
              epochs: int = 5,
              min_val: float = 1.0,
              patience=2,
              output_dir: str = None,
              check_every_n_batches: int = None,
              device="cuda",
              **kwargs):
        """
        Embeds the watermark of Adi et al. with early stopping on the watermark accuracy.

        :param train_loader: The (normalized) training dataset.
        :param valid_loader: The (normalized) validation dataset.
        :param wm_loader: The (unnormalized in range [0-1]) watermark dataset without labels.
        :param keylength: Number of keys to embed into the model.
        :param max_iter: The maximum number of iterations to embed the watermark.
        :param boost_factor: Repetition factor for the watermark.
        :param epochs: Number of epochs to embed.
        :param min_val: Minimum watermark accuracy to be considered 'embedded'
        :param patience: Number of consecutive times the watermark accuracy has to be on min_val to stop.
        :param output_dir: Output directory to save intermediate metrics and the models.
        :param check_every_n_batches: Number of batches after which the watermark accuracy is checked.
        :param device: cuda or cpu
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :return: watermark key and labels.
        """
        if len(kwargs.items()) > 0:
            print(f"[WARNING] Unused parameters: {kwargs}")

        # Sample and normalize the watermarking key.
        x_wm, y_wm = self.keygen(wm_loader, num_classes=self.num_classes, keylength=keylength)
        train_and_wm_loader = train_loader.add_numpy_data(x_wm, y_wm, boost_factor=boost_factor)

        self.visualize_key(x_wm, output_dir=output_dir)

        callbacks = [wrt_callbacks.EarlyStoppingWRTCallback(lambda: self.classifier.evaluate(x_wm, y_wm)[0],
                                                            check_every_n_batches=check_every_n_batches,
                                                            patience=patience,
                                                            mode='min'),
                     wrt_callbacks.DebugWRTCallback(lambda: self.verify(x_wm, y_wm)[0],
                                                    message="wm_acc",
                                                    check_every_n_batches=check_every_n_batches)]

        trainer = Trainer(model=self.get_classifier(), train_loader=train_and_wm_loader, valid_loader=valid_loader,
                          device=device, num_epochs=epochs, output_dir=output_dir, callbacks=callbacks,
                          save_data={"x_wm": x_wm, "y_wm": y_wm})
        history = trainer.fit()

        if output_dir is not None:
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(history, fp=f)

            checkpoint = {
                'model': self.get_classifier().model.state_dict(),
                'optimizer': self.get_classifier().optimizer.state_dict(),
                'x_wm': x_wm,
                'y_wm': y_wm
            }
            self.save('best.pth', path=output_dir, checkpoint=checkpoint)

        return x_wm, y_wm

    def load(self, filename, path=None, **load_kwargs: dict) -> Tuple[np.ndarray, np.ndarray]:
        """ Loads the watermark data necessary to validate the watermark.
        """
        checkpoint = super().load(filename, path, **load_kwargs)

        print(checkpoint.keys())

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint["x_wm"], checkpoint["y_wm"]

    def extract(self, x, classifier=None, **kwargs):
        """
        Extract a message from a classifier.

        :param x: Watermarking key.
        :param classifier The classifier to extract a message from.
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :return: Message from the source model.
        """
        if classifier is None:
            classifier = self.get_classifier()
        return classifier.predict(x, **kwargs)

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
def wm_adi(classifier, **kwargs):
    return Adi(classifier=classifier, **kwargs)


@mlconfig.register
def wm_adi_embed(defense: Adi, **kwargs):
    return defense.embed(**kwargs), defense


@mlconfig.register
def wm_adi_keygen(defense: Adi,
                  train_loader,
                  wm_loader: data.DataLoader,
                  num_classes: int,
                  keylengths: List[int],
                  **kwargs):
    x_wm, y_wm = defense.keygen(wm_loader, num_classes, max(keylengths))
    x_wm = train_loader.normalize(x_wm)
    for n in keylengths:
        yield x_wm[:n], y_wm[:n]
