"""
Implement the AsiaCCS watermarking scheme from: https://gzs715.github.io/pubs/WATERMARK_ASIACCS18.pdf
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List, Tuple, Union

import mlconfig
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from torch.utils import data

from wrt.training import callbacks as wrt_callbacks

from wrt.classifiers import PyTorchClassifier
from wrt.defenses.watermark.watermark import Watermark
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer


class Zhang(Watermark):
    """
    Base class for watermarks from this paper:
    https://gzs715.github.io/pubs/WATERMARK_ASIACCS18.pdf
    """

    def __init__(self, classifier: PyTorchClassifier, num_classes: int, **kwargs):
        """
        :param classifier: Model to train
        :param lr: float; learning rate to embed the watermark
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier
        self.num_classes = num_classes

    @staticmethod
    def get_name():
        return "Zhang"

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

    def keygen(self,
               data_loader: data.DataLoader,
               keylength: int,
               source_class: int = None,
               target_class: int = None,
               trigger_pattern: np.ndarray = None,
               verbose: bool = True) -> np.ndarray:
        """ Gets inputs from a source class and adds a trigger onto them (specified by the _paste function)
        """
        if source_class is None:
            source_class = np.random.randint(self.num_classes)
        if target_class is None:
            target_class = (np.random.randint(1, self.num_classes) + source_class) % self.num_classes

        x_wm, y_wm = collect_n_samples(n=keylength, data_loader=data_loader, class_label=source_class, verbose=verbose)
        x_wm, y_wm = self._paste(x_wm, trigger_pattern), np.tile(target_class, x_wm.shape[0])
        return x_wm, y_wm

    def _paste(self, x: np.ndarray, content: np.ndarray):
        """
        Embeds content into every array in x
        :param x: n + 1 dimensional input data, where the size of axis 0 is the number of samples
        :param content: n dimensional content data
        :return: New data with embedded content with the same shape as the input x
        """
        raise NotImplementedError

    def embed(self,
              train_loader: WRTDataLoader,
              valid_loader: WRTDataLoader,
              wm_loader: WRTDataLoader,
              source_class: int,
              target_class: int,
              epochs: int,
              keylength: int,
              boost_factor: int,
              output_dir: str = None,
              decrease_lr_factor: float = 1.0,
              trigger_pattern: np.ndarray = None,
              check_every_n_batches: int = None,
              patience: int = 2,
              device="cuda",
              **kwargs):
        """
        Train a model on the watermark data.
        @Note: We slightly adapt the embedding by repeating the watermarking key multiple times in the
        training dataset, which we refer to as 'boosting'. We equally boost the watermarking key and
        samples from the source class to prevent a degradation of the accuracy of the original samples.

        :param train_loader The training data loader.
        :param wm_loader The watermark data loader. It is expected to output unnormalized images (range [0-1])
        :param valid_loader The validation data loader
        :param source_class The source class from which to sample all watermarking keys.
        :param target_class New class for watermarking keys.
        :param epochs Number of epochs to embed.
        :param keylength Number of watermarking keys to embed
        :param boost_factor Number of repetitions of the watermark in the training data (to speed up embedding)
        :param output_dir (optional) Directory to save intermediate results and the trained model
        :param decrease_lr_factor Decrease learning rate by this factor.
        :param trigger_pattern (optional) Trigger to paste into the sampled elements for the watermarking key
        :param check_every_n_batches (optional) Logging interval after which to log the watermark accuracy
        :param patience: Patience for early stopping on the watermark accuracy
        :param min_val: Minimum watermark accuracy for early stopping
        :param device Device to train on.
        :return: watermarking keys as np.ndarray
        """
        if len(kwargs.items()) > 0:
            print(f"[WARNING] Unused parameters: {kwargs}")

        self.classifier.lr /= decrease_lr_factor

        # Sample and normalize the watermarking key.
        x_wm, y_wm = self.keygen(data_loader=wm_loader, source_class=source_class, target_class=target_class,
                                 trigger_pattern=trigger_pattern, keylength=keylength)
        self.visualize_key(x_wm, output_dir=output_dir)
        x_wm = train_loader.normalize(x_wm)

        # Sample from the source class and add boosted data to ensure high class test accuracy.
        x_source, y_source = collect_n_samples(n=np.inf, data_loader=wm_loader, class_label=source_class)
        x_source = train_loader.normalize(x_source)

        x_boost, y_boost = np.vstack((x_wm, x_source)), np.hstack((y_wm, y_source))
        train_and_wm_loader = train_loader.add_numpy_data(train_loader.unnormalize(x_boost), y_boost, boost_factor=boost_factor)

        callbacks = [wrt_callbacks.EarlyStoppingWRTCallback(lambda: self.classifier.evaluate(x_wm, y_wm)[0],
                                                            check_every_n_batches=check_every_n_batches,
                                                            patience=patience,
                                                            mode='min'),
                     wrt_callbacks.DebugWRTCallback(lambda: self.verify(x_wm, y_wm)[0],
                                                    message="wm_acc",
                                                    check_every_n_batches=check_every_n_batches),
                     wrt_callbacks.DebugWRTCallback(lambda: self.verify(x_source, y_source)[0],
                                                    message=f"Class {source_class} accuracy",
                                                    check_every_n_batches=check_every_n_batches)]

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
            self.save(filename='best.pth', path=output_dir, checkpoint=checkpoint)

        self.classifier.lr *= decrease_lr_factor
        return x_wm, y_wm

    def load(self, filename, path=None, **load_kwargs: dict) -> Tuple[np.ndarray, np.ndarray]:
        """ Loads the watermark data necessary to validate the watermark.
        """
        checkpoint = super().load(filename, path, **load_kwargs)

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint["x_wm"], checkpoint["y_wm"]

    def predict(self, x: np.ndarray, **kwargs):
        """
        Perform prediction using this classifier.

        :param x: Test set.
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :return: Predictions for test set.
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)

    def extract(self, x: np.ndarray, classifier: PyTorchClassifier = None, **kwargs):
        """ Extract a watermarking message from a given classifier. If no classifier is provided,
        uses the source model.
        """
        if classifier is None:
            classifier = self.get_classifier()

        return classifier.predict(x, **kwargs).argmax(1)


class ZhangContent(Zhang):
    """
    Content-based variant of Zhang watermarking scheme
    """

    def __init__(self, classifier: PyTorchClassifier,
                 size: int = None,
                 pos: Union[str, Tuple[int, int, int]] = None,
                 **kwargs):
        """ Content inserts some given content into the data.
        :param classifier The source model.
        """
        super().__init__(classifier, **kwargs)

        # Parse tuple from string.
        if type(pos) == str:
            self.pos = tuple(map(int, pos.replace(" ", "").strip()[1:-2].split(',')))
        else:
            self.pos = pos
        self.size = size

    @staticmethod
    def get_name():
        return "Content"

    def _paste(self, x, content):
        x = np.copy(x)
        pos = (0, 5, 5) if self.pos is None else self.pos

        if self.size is None:
            # embed the given content
            dims = content.shape
            x[:, pos[0]:pos[0] + dims[0], pos[1]:pos[1] + dims[1], pos[2]:pos[2] + dims[2]] = content
            return x
        else:
            # ignore the content and embed a white rectangle with the given size
            x[:, :, pos[1]:pos[1] + self.size // 2, pos[2]:pos[2] + self.size] = 1
            return x


class ZhangUnrelated(Zhang):
    """
    Unrelated-data variant of Zhang watermarking scheme
    """

    @staticmethod
    def get_name():
        return "Unrelated"

    def _paste(self, x, trigger_pattern=None):
        return x

    def keygen(self,
               data_loader: WRTDataLoader,
               keylength: int,
               verbose: bool = True,
               **kwargs) -> np.ndarray:
        """ Gets inputs from a source class and adds a trigger onto them (specified by the _paste function)
        """
        y_wm = np.random.randint(self.num_classes, size=keylength)
        x_wm, _ = collect_n_samples(n=keylength, data_loader=data_loader, verbose=verbose)
        return x_wm, y_wm


class ZhangNoise(Zhang):
    """
    Noise-based variant of Zhang watermarking scheme
    """

    def __init__(self, classifier, std=0.3, **kwargs):
        super().__init__(classifier, **kwargs)
        self.std = std

    @staticmethod
    def get_name():
        return "Noise"

    def _paste(self, x, trigger_pattern=None):
        x = np.copy(x)
        x += np.random.normal(0, self.std, size=x.shape)
        x = np.clip(x, 0, 1)
        return x


@mlconfig.register
def wm_unrelated(classifier, **kwargs):
    return ZhangUnrelated(classifier=classifier, **kwargs)


@mlconfig.register
def wm_unrelated_embed(defense: ZhangUnrelated, **kwargs):
    return defense.embed(**kwargs), defense


@mlconfig.register
def wm_unrelated_keygen(defense: ZhangUnrelated,
                        train_loader: WRTDataLoader,
                        wm_loader: data.DataLoader,
                        keylengths: List[int],
                        source_class: int = None,
                        target_class: int = None,
                        **kwargs: dict):
    x_wm, y_wm = defense.keygen(data_loader=wm_loader,
                                source_class=source_class,
                                target_class=target_class,
                                keylength=max(keylengths))
    x_wm = train_loader.normalize(x_wm)
    for n in keylengths:
        yield x_wm[:n], y_wm[:n]


@mlconfig.register
def wm_content(classifier, **kwargs):
    return ZhangContent(classifier=classifier, **kwargs)


@mlconfig.register
def wm_content_embed(defense: ZhangContent, config, **kwargs):
    return defense.embed(**kwargs), defense


@mlconfig.register
def wm_content_keygen(defense: ZhangContent,
                      train_loader: WRTDataLoader,
                      keylengths: List[int],
                      config,
                      source_class: int = None,
                      target_class: int = None,
                      **kwargs: dict):
    # Create a custom wm loader with the correct class
    if source_class is None:
        source_class = np.random.randint(defense.num_classes)

    wm_loader = config.wm_dataset(class_labels=source_class)

    x_wm, y_wm = defense.keygen(data_loader=wm_loader,
                                source_class=source_class,
                                target_class=target_class,
                                keylength=max(keylengths))
    x_wm = train_loader.normalize(x_wm)
    for n in keylengths:
        yield x_wm[:n], y_wm[:n]


@mlconfig.register
def wm_noise(classifier, **kwargs):
    return ZhangNoise(classifier=classifier, **kwargs)


@mlconfig.register
def wm_noise_embed(defense: ZhangNoise, **kwargs):
    return defense.embed(**kwargs), defense


@mlconfig.register
def wm_noise_keygen(defense: ZhangNoise,
                    train_loader: WRTDataLoader,
                    wm_loader: WRTDataLoader,
                    keylengths: List[int],
                    source_class: int = None,
                    target_class: int = None,
                    **kwargs: dict):
    x_wm, y_wm = defense.keygen(data_loader=wm_loader,
                                source_class=source_class,
                                target_class=target_class,
                                keylength=max(keylengths))
    x_wm = train_loader.normalize(x_wm)
    for n in keylengths:
        yield x_wm[:n], y_wm[:n]
