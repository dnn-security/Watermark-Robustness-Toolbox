"""
    Embedding a watermark based on the frontier stitching algorithm.
    | Paper link: https://arxiv.org/pdf/1711.01894.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List

import mlconfig
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import os
import json

from wrt.training import callbacks as wrt_callbacks

from wrt.art_classes import FastGradientMethod
from wrt.classifiers import PyTorchClassifier
from wrt.defenses.utils import NormalizingPreprocessor
from wrt.defenses.watermark.watermark import Watermark
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer


class FrontierStitching(Watermark):
    """
    Embedding a Zero_Bit watermark.

    | Paper link: https://arxiv.org/pdf/1711.01894.pdf
    """

    def __init__(self,
                 classifier: PyTorchClassifier,
                 num_classes: int,
                 eps: float = 0.25,
                 **kwargs):
        """
        Create an :class:`Zero_Bit` instance.

        :param classifier: Model to embed the watermark.
        :param num_classes: Number of classes for the task.
        :param key_length: The length of the key.
        :param eps: Maximum perturbation (for range [0,1])
        :param lr_ratio: float; ratio of the original learning rate to fine-tune with
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier

        self.num_classes = num_classes
        self.eps = eps

    def visualize_key(self, x_wm: np.ndarray, output_dir: str = None):
        """ Visualizes the watermarking key.
        """
        idx = np.random.choice(np.arange(x_wm.shape[0]), size=9, replace=False)
        fig, _ = plt.subplots(nrows=3, ncols=3)
        fig.suptitle(f"{type(self).__name__} Watermarking Key")
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
        return "Frontier Stitching"

    @staticmethod
    def keygen(train_loader: WRTDataLoader,
               wm_loader: WRTDataLoader,
               classifier: PyTorchClassifier,
               keylength: int,
               eps: float,
               key_expansion_factor: int = 10,
               batch_size: int = 32) -> np.ndarray:
        """ Generates FGM adversarial examples with the ART toolbox.
        """
        x_train, y_train = collect_n_samples(n=key_expansion_factor * keylength, data_loader=wm_loader, verbose=False)

        # Add the preprocessor to normalize the data.
        preprocessor = NormalizingPreprocessor(mean=train_loader.mean, std=train_loader.std)
        classifier.add_preprocessor(preprocessor, "frontier_stitching_normalizer")

        # Labels before the adversarial attack.
        y_before = np.argmax(classifier.predict(x_train, verbose=True), axis=1)
        correct_idx, = np.where(y_train == y_before)
        print(f"Correctly predicted {len(correct_idx)}/{len(x_train)}. Using these to generate adversarial examples!")
        x_train, y_train, y_before = x_train[correct_idx], y_train[correct_idx], y_before[correct_idx]

        # Generate the adversarial examples
        fgm = FastGradientMethod(classifier=classifier,
                                 norm=np.inf,
                                 eps=eps,
                                 minimal=False,
                                 batch_size=batch_size)

        # Generate the adversarial examples.
        x_adv = fgm.generate(x_train)

        # Filter true and false adversaries.
        y_after = np.argmax(classifier.predict(np.clip(x_adv, 0, 1), verbose=True), axis=1)
        false_idx, = np.where(y_before == y_after)
        true_idx, = np.where(y_before != y_after)

        if (len(false_idx) < keylength // 2) or (len(true_idx) < keylength // 2):
            print(f"[WARNING] Not enough samples generated. True: {len(true_idx)}, False: {len(false_idx)}. Needed at "
                  f"least {keylength // 2} in each.")
        true_idx, false_idx = true_idx[:keylength // 2], false_idx[:keylength // 2]

        classifier.remove_preprocessor("frontier_stitching_normalizer")

        x_wm = np.vstack((x_adv[false_idx][:keylength // 2], x_adv[true_idx][:keylength // 2]))
        y_wm = np.hstack((y_train[false_idx][:keylength // 2], y_train[true_idx][:keylength // 2]))
        return x_wm, y_wm

    def embed(self,
              train_loader: WRTDataLoader,
              valid_loader: WRTDataLoader,
              wm_loader: WRTDataLoader,
              keylength: int,
              patience: int = 2,
              min_val: float = 1.0,
              epochs: int = 5,
              boost_factor: int = 10,
              decrease_lr_factor: float = 1.0,
              key_expansion_factor: int = 10,
              log_wm_acc_after_n_batches: int = 100,
              batch_size: int = 32,
              output_dir: str = None,
              device="cuda",
              **kwargs):
        """
        Embeds a model with the Frontier Stitching watermark.

        :param train_loader: The (normalized) training dataset.
        :param valid_loader: The (normalized) validation dataset. 
        :param wm_loader: The (unnormalized) data loader. Expects images in the range [0-1]. 
        :param keylength: Number of keys to embed into the model.
        :param boost_factor Number of repetitions of the watermark in the training data (to speed up embedding)
        :param decrease_lr_factor Decrease lr for the embedding by this factor.
        :param key_expansion_factor
        :param log_wm_acc_after_n_batches (optional) Logging interval after which to log the watermark accuracy
        :param patience: Patience for the early stopping.
        :param min_val: Minimum watermark accuracy to be considered 'embedded'
        :param epochs: Number of epochs to embed.
        :param batch_size: Batch size for the attack.
        :param output_dir (optional) Directory to save intermediate results and the trained model
        :param device: cuda or cpu
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :return: watermark key and labels.
        """
        self.classifier.lr /= decrease_lr_factor

        # Generate the watermarking keys.
        x_wm, y_wm = self.keygen(train_loader=train_loader,
                                 wm_loader=wm_loader,
                                 classifier=self.classifier,
                                 keylength=keylength,
                                 key_expansion_factor=key_expansion_factor,
                                 eps=self.eps,
                                 batch_size=batch_size)
        self.visualize_key(x_wm, output_dir=output_dir)

        train_and_wm_loader = train_loader.add_numpy_data(x_wm, y_wm, boost_factor=boost_factor)
        x_wm = train_loader.normalize(x_wm)

        callbacks = [wrt_callbacks.EarlyStoppingWRTCallback(lambda: self.classifier.evaluate(x_wm, y_wm)[0],
                                                            check_every_n_batches=log_wm_acc_after_n_batches,
                                                            patience=patience,
                                                            mode='min'),
                     wrt_callbacks.DebugWRTCallback(lambda: self.verify(x_wm, y_wm)[0],
                                                    message="wm_acc",
                                                    check_every_n_batches=log_wm_acc_after_n_batches)]

        trainer = Trainer(model=self.get_classifier(), train_loader=train_and_wm_loader, valid_loader=valid_loader,
                          device=device, num_epochs=epochs, callbacks=callbacks)
        history = trainer.fit()

        if output_dir is not None:
            # Overwrite the model with the highest validation accuracy with the 'final' model.
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(history, fp=f)

            checkpoint = {
                'model': self.get_classifier().model.state_dict(),
                'optimizer': self.get_classifier().optimizer.state_dict(),
                'x_wm': x_wm,
                'y_wm': y_wm
            }
            self.save('best.pth', path=output_dir, checkpoint=checkpoint)

        self.classifier.lr *= decrease_lr_factor
        return x_wm, y_wm

    def load(self, filename, path=None, **load_kwargs: dict):
        """ Loads the watermark data necessary to validate the watermark.
        """
        checkpoint = super().load(filename, path, **load_kwargs)

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint["x_wm"], checkpoint["y_wm"]

    def extract(self, x: np.ndarray, classifier: PyTorchClassifier = None, **kwargs):
        """ Extract a watermarking message from a given classifier. If no classifier is provided,
        uses the source model.
        """
        if classifier is None:
            classifier = self.get_classifier()
        return classifier.predict(x, **kwargs).argmax(1)

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
def wm_frontier_stitching(classifier, **kwargs):
    return FrontierStitching(classifier=classifier, **kwargs)


@mlconfig.register
def wm_frontier_stitching_embed(defense: FrontierStitching, **kwargs):
    return defense.embed(**kwargs), defense


@mlconfig.register
def wm_frontier_stitching_keygen(defense: FrontierStitching,
                                 train_loader: WRTDataLoader,
                                 wm_loader: WRTDataLoader,
                                 key_expansion_factor,
                                 keylengths: List[int],
                                 **kwargs):
    wm_x, wm_y = defense.keygen(train_loader=train_loader,
                                wm_loader=wm_loader,
                                classifier=defense.get_classifier(),
                                keylength=max(keylengths),
                                eps=defense.eps,
                                key_expansion_factor=key_expansion_factor)

    idx = np.arange(len(wm_x))
    np.random.shuffle(idx)
    for n in keylengths:
        yield wm_x[idx][:n], wm_y[idx][:n]
