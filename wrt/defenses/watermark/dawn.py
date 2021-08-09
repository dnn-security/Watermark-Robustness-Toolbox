from __future__ import absolute_import, division, print_function, unicode_literals

import hmac
import os
import random
from typing import Tuple, List, Callable

import mlconfig

import numpy as np
import torch

from wrt.classifiers import PyTorchClassifier
from wrt.defenses.watermark.watermark import Watermark
from wrt.postprocessors import Postprocessor
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.trainer import Trainer
from wrt.training.utils import compute_accuracy


class DawnPostprocessor(Postprocessor):
    """
    The postprocessor used by DAWN. Take predictions from the classifier
    and decide whether to alter the predictions to create a trigger set.
    """

    # The trigger set needs to be global in case the classifier is deep-copied
    x_wm, y_wm = None, None

    def __init__(self, classifier, medians, ratio=0.01, activation_threshold=0.2,
                 embed_layer_index=-2, hmac_key=1):
        """
        Create an instance of this postprocessor
        :param classifier: Classifier
        :param medians: np.ndarray; the median activations of an internal layer, used to
                        smooth the decisions
        :param ratio: float; approximate percentage of predictions to alter
        :param embed_layer_index: int; the index of the internal layer used as the embedded space
        :param hmac_key: int; the secret key used by HMAC
        """
        super(DawnPostprocessor, self).__init__()

        self.classifier = classifier
        self.medians = medians.reshape(-1)
        self.decision_threshold = ratio * 2 ** 128
        self.embed_layer_index = embed_layer_index
        self.hmac_key = hmac_key.to_bytes(32, byteorder='little')
        self.activation_threshold = activation_threshold

        # when measuring test accuracy, set to False
        self.add_preds_to_trigger = True

    def __add_to_trigger(self, x, y):
        """
        Add the given data to the trigger set
        :param x: np.ndarray; input data
        :param y: np.ndarray; corresponding label
        :return: None
        """
        if type(x) is torch.Tensor:
            x = x.cpu().numpy()
        x = np.expand_dims(x, axis=0)
        y = np.eye(self.classifier.nb_classes())[np.argmax(y)]
        y = np.expand_dims(y, axis=0)

        if DawnPostprocessor.x_wm is None:
            DawnPostprocessor.x_wm = x
        else:
            DawnPostprocessor.x_wm = np.vstack([DawnPostprocessor.x_wm, x])

        if DawnPostprocessor.y_wm is None:
            DawnPostprocessor.y_wm = y
        else:
            DawnPostprocessor.y_wm = np.vstack([DawnPostprocessor.y_wm, y])

    def __compute_watermark_keys(self, x):
        """
        Two lists, with the first containing HMAC(K_w, x)[0..127] and
        the second containing HMAC(K_w, x)[128..255]
        :param x: np.ndarray; input data
        :return: (list of bytes, list of bytes)
        """
        # there's no way to leverage numpy to parallelize the hashing operations,
        # so just loop through the data
        lower, upper = [], []
        if type(x) is torch.Tensor:
            x = x.cpu().numpy()
        for data in x:
            data = np.expand_dims(data, axis=0)
            x_embed = self.classifier.get_all_activations(data)[self.embed_layer_index]
            x_embed = self.classifier.functional.numpy(x_embed.reshape(-1))
            x_smoothed = (x_embed > self.medians).astype(np.int32)

            h = hmac.new(self.hmac_key, digestmod='sha256')
            h.update(x_smoothed)
            digest = h.digest()
            lower.append(digest[:16])
            upper.append(digest[16:])

        return lower, upper

    def __shuffle(self, arr, seed):
        """
        Shuffle the given array using seeded Fisher-Yates. The
            returned array is different from the given array to
            ensure that there are no false positives
        :param arr: np.ndarray
        :param seed: int; seed for the random number generator
        :return: np.ndarray; shuffled array
        """
        if len(arr) <= 1:
            raise ValueError("Error: array to be shuffled must have more than 1 element")

        new_arr = arr.copy()
        random.seed(seed)
        while np.all(new_arr == arr):
            for i in range(len(arr) - 1, 0, -1):
                j = random.randint(0, i)
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
        return new_arr

    @property
    def apply_fit(self):
        return False

    @property
    def apply_predict(self):
        return True

    def __call__(self, preds, x=None):
        """
        Apply postprocessing to the given predictions
        :param preds: np.ndarray; model predictions
        :param x: np.ndarray; model inputs
        :return: np.ndarray; processed model predictions
        """
        new_preds = preds.copy()
        lower, upper = self.__compute_watermark_keys(x)
        for i, pred in enumerate(preds):
            if int.from_bytes(lower[i], byteorder='little') < self.decision_threshold:
                # find the predicted indices that exceed a threshold
                indices = pred > self.activation_threshold

                if np.count_nonzero(indices) <= 1:
                    # if there are not enough indices, instead take the top 3
                    indices = np.argpartition(pred, -3)[-3:]

                pred_shuffle = pred[indices]
                pred_shuffle = self.__shuffle(pred_shuffle, upper[i])
                new_preds[i][indices] = pred_shuffle

                if self.add_preds_to_trigger:
                    y = new_preds[i].copy()
                    self.__add_to_trigger(x[i], y)
        return new_preds

    def fit(self, preds, **kwargs):
        pass

    @staticmethod
    def get_trigger():
        """
        Get the trigger set
        :return: (np.ndarray, np.ndarray) tuple for (input_data, labels)
        """
        x_wm, y_wm = DawnPostprocessor.x_wm, DawnPostprocessor.y_wm
        # DawnPostprocessor.x_wm, DawnPostprocessor.y_wm = None, None
        return x_wm, y_wm

    @staticmethod
    def flush():
        DawnPostprocessor.x_wm, DawnPostprocessor.y_wm = None, None


class Dawn(Watermark):
    """
    Implement the DAWN remote API layer
    https://arxiv.org/pdf/1906.00830.pdf
    """

    def __init__(self, classifier, ratio=0.01, activation_threshold=0.2, embed_index=-2, normalize=None, **kwargs):
        """
        Create a :class:`Dawn` instance.

        :param classifier: Model to train adversarially.
        :type classifier: :class:`.Classifier`
        :param embed_index: int; the index of the internal layer used as the embedded space
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier

        self.ratio = ratio
        self.activation_threshold = activation_threshold
        self.embed_index = embed_index
        self.normalize = normalize
        self.postprocessor = None

    @staticmethod
    def get_name():
        return "Dawn"

    def initialize_median(self, train_loader: WRTDataLoader, max_batches=np.inf):
        # Compute the median activation
        activations = []
        for i, (x_batch, y_batch) in enumerate(train_loader):
            activations.append(self.classifier.functional.numpy(
                self.classifier.get_all_activations(x_batch)[self.embed_index]))
            if i > max_batches:
                break
        activations = np.vstack(activations)
        medians = np.median(activations, axis=0)
        return medians

    def keygen(self,
               keylength: int,
               train_loader: WRTDataLoader,
               device="cuda",
               **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        medians = self.initialize_median(train_loader, max_batches=3)
        signature = np.random.randint(low=0, high=999999999999999)

        postprocessor = DawnPostprocessor(self.classifier,
                                          medians,
                                          ratio=self.ratio,
                                          activation_threshold=self.activation_threshold,
                                          embed_layer_index=self.embed_index,
                                          hmac_key=signature)
        postprocessor.flush()   # Clear any watermark.

        for x, y in train_loader:
            preds: np.ndarray = self.get_classifier().predict(x.to(device))
            postprocessor(preds, x=x.numpy())

            x_wm, y_wm = postprocessor.get_trigger()
            if x_wm is not None:
                if x_wm.shape[0] >= keylength:
                    break

        print("Done generating DAWN key!")
        x_wm, y_wm = postprocessor.get_trigger()
        return x_wm[:keylength], y_wm[:keylength].argmax(1)

    @staticmethod
    def __load_model(model, optimizer, image_size, num_classes, pretrained_dir: str = None,
                     filename: str = 'best.pth'):
        """ Loads a (pretrained) source model from a directory and wraps it into a PyTorch classifier.
        """
        criterion = torch.nn.CrossEntropyLoss()

        if pretrained_dir:
            assert filename.endswith(".pth"), "Only '*.pth' are allowed for pretrained models"
            print(f"Loading a pretrained source model from '{pretrained_dir}'.")
            state_dict = torch.load(os.path.join(pretrained_dir, filename))
            model.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optimizer"])

        model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, image_size, image_size),
            nb_classes=num_classes
        )
        return model

    def embed(self,
              train_loader: WRTDataLoader,
              valid_loader: WRTDataLoader,
              create_train_loader_fn: Callable,
              create_surrogate_model_fn: Callable,
              create_surrogate_optimizer_fn: Callable,
              create_scheduler_fn: Callable,
              keylength: int,
              signature: int = None,
              epochs: int = 1,
              output_dir: str = None,
              device="cuda",
              **kwargs):
        """
        Attach a DAWN postprocessor to the classifier predictions.
        Creates a new surrogate model by retraining.

        :param train_loader Training data loader
        :param valid_loader Loader for the validation data.
        :param key_expansion_factor: Number of keys to generate for the embedding relative to the keylength.
        :param keylength: Number of keys to embed into the model.
        :param signature (optional) The secret watermarking message (in bits). If None, one is generated randomly.
        :param epochs Number of epochs to use for trainings.
        :param output_dir Output directory to save intermediary results
        :param device: cuda or cpu
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :return: Nothing
        """
        medians = self.initialize_median(train_loader, max_batches=20)
        signature = np.random.randint(low=0, high=999999999999999)

        postprocessor = DawnPostprocessor(self.classifier,
                                          medians,
                                          ratio=self.ratio,
                                          activation_threshold=self.activation_threshold,
                                          embed_layer_index=self.embed_index,
                                          hmac_key=signature)
        postprocessor.flush()  # Clear any existing watermark.

        self.classifier.add_postprocessor(postprocessor, name='dawn')
        self.postprocessor = postprocessor

        train_loader: WRTDataLoader = create_train_loader_fn(source_model=self.get_classifier(), train=True)

        surrogate_model: torch.nn.Sequential = create_surrogate_model_fn()
        optimizer = create_surrogate_optimizer_fn(surrogate_model.parameters())
        scheduler = create_scheduler_fn(optimizer)
        model: PyTorchClassifier = self.__load_model(surrogate_model, optimizer, self.classifier.input_shape[-1], self.classifier.nb_classes())
        self.classifier = model
        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=device, scheduler=scheduler, num_epochs=epochs)
        trainer.fit()

        x_wm, y_wm = postprocessor.x_wm[:keylength], postprocessor.y_wm[:keylength]

        if output_dir is not None:
            checkpoint = {
                'model': self.get_classifier().model.state_dict(),
                'optimizer': self.get_classifier().optimizer.state_dict(),
                'x_wm': x_wm,
                'y_wm': y_wm
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))
        print(x_wm.shape)
        print(y_wm.shape)

        return x_wm, y_wm

    def load(self, filename, path=None, **load_kwargs: dict) -> Tuple[np.ndarray, np.ndarray]:
        """ Loads the watermark data necessary to validate the watermark.
        """
        checkpoint = super().load(filename, path, **load_kwargs)

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint["x_wm"], checkpoint["y_wm"]

    def extract(self, x_wm, classifier=None, **kwargs):
        """
        Return the verification accuracy of the classifier on the
        DAWN postprocessor's trigger set
        :param x: Unused
        :param classifier: Classifier; if provided, extract the watermark from
                           the given classifier instead of this object's classifier
        :param kwargs: Other keyword arguments passed to classifier predict function
        :return: float representing the watermark error rate
        """
        return classifier.predict(x_wm).argmax(1)

    def verify(self,
               x: np.ndarray,
               y: np.ndarray = None,
               classifier: PyTorchClassifier = None,
               **kwargs) -> Tuple[float, bool]:
        if classifier is None:
            classifier = self.get_classifier()

        if y is None:
            print("Extracting the source model's message.")
            y = self.extract(x)  # Extract the source model's message.

        msg = self.extract(x, classifier=classifier)
        wm_acc = compute_accuracy(msg, y)
        return wm_acc, wm_acc > 0

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
        if self.postprocessor is None:
            raise ValueError("Error: embed must be called before predict")

        self.postprocessor.add_preds_to_trigger = False
        preds = self.classifier.predict(x, **kwargs)
        self.postprocessor.add_preds_to_trigger = True
        return preds


@mlconfig.register
def wm_dawn(classifier, **kwargs):
    return Dawn(classifier=classifier, **kwargs)


@mlconfig.register
def wm_dawn_embed(defense: Dawn, config, **kwargs):
    return defense.embed(create_train_loader_fn=config.dataset, create_surrogate_model_fn=config.surrogate_model,
                         create_surrogate_optimizer_fn=config.surrogate_optimizer, create_scheduler_fn=config.scheduler,
                         **kwargs), defense


@mlconfig.register
def wm_dawn_keygen(defense: Dawn, keylengths: List[int], **kwargs):
    x_wm, y_wm = defense.keygen(keylength=max(keylengths), **kwargs)
    for n in keylengths:
        yield x_wm[:n], y_wm[:n]

