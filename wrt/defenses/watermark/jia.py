from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from copy import deepcopy
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import mlconfig
import numpy as np
import torch
from tqdm import tqdm

from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import StopTrainingException, PyTorchClassifier
from wrt.classifiers.loss import Loss
from wrt.defenses.watermark.watermark import Watermark
from wrt.training import callbacks as wrt_callbacks
from wrt.training.callbacks import WRTCallback
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.metrics import Average, Accuracy


class Jia(Watermark):
    """
    Implement the Entangled Watermarks Embedding scheme.
    https://arxiv.org/pdf/2002.12200.pdf
    """

    class SNNLoss(Loss):
        """
        Loss class for computing layer-wise soft nearest neighbor loss
        """

        def __init__(self, classifier, temp_init, layer=None):
            super(Jia.SNNLoss, self).__init__(classifier)

            # more layers than we need
            self.temps = [temp_init] * 100

            self.layer = layer

        @staticmethod
        def snnl(functional, x, y, t):
            """
            Compute the soft nearest neighbor loss using the given data x, the
                labels y, and the temperature t
            :param functional: Functional
            :param x: Framework-specific tensor
            :param y: Framework-specific tensor
            :param t: Framework-specific int type or built-in int type
            :return: Framework-specific tensor
            """

            def pairwise_euclidean_distance(a, b):
                ba = functional.shape(a)[0]
                bb = functional.shape(b)[0]
                sqr_norm_a = functional.reshape(functional.sum(functional.pow(a, 2), axis=1), 1, ba)
                sqr_norm_b = functional.reshape(functional.sum(functional.pow(b, 2), axis=1), bb, 1)
                inner_prod = functional.matmul(b, functional.transpose(a))
                tile1 = functional.tile(sqr_norm_a, bb, 1)
                tile2 = functional.tile(sqr_norm_b, 1, ba)
                return tile1 + tile2 - 2 * inner_prod

            x = functional.reshape(x, functional.shape(x)[0], -1)

            distance = pairwise_euclidean_distance(x, x)
            exp_distance = functional.exp(-(distance / (t + 1e-7)))
            f = exp_distance - functional.tensor(np.eye(functional.shape(x)[0]))
            f[f >= 1] = 1
            f[f <= 0] = 0
            pick_probability = f / (1e-7 + functional.sum(f, axis=1, keep_dims=True))

            y_shape = functional.shape(y)
            same_label_mask = functional.equal(y, functional.reshape(y, y_shape[0], 1))

            masked_pick_probability = pick_probability * same_label_mask
            sum_masked_probability = functional.sum(masked_pick_probability, axis=1)

            return functional.mean(-functional.log(1e-7 + sum_masked_probability))

        def reduce_labels(self):
            return True

        def compute_loss(self, pred, true, x=None):
            if x is None:
                raise ValueError("Inputs must be provided to SNNLoss")

            losses = []
            normalizer = 0
            if self.layer is None:
                activations = self.classifier.get_all_activations(x)
                for i, activation in enumerate(activations):
                    # don't account the last layer
                    if i == len(activations) - 1:
                        break

                    loss = Jia.SNNLoss.snnl(
                        self._functional,
                        activation,
                        true,
                        100 / (self.temps[i] + 1e-7)
                    )
                    losses.append(loss)
            else:
                activation = self.classifier.get_all_activations(x)[self.layer]
                normalizer += activation.shape
                loss = Jia.SNNLoss.snnl(
                    self._functional,
                    activation,
                    true,
                    100 / (self.temps[0] + 1e-7)
                )
                losses.append(loss)

            return sum(losses) / normalizer

    class JiaLoss(Loss):
        """
        Loss used for embedding Entangled Watermarks
        """

        def __init__(self, classifier, target, temp_init, snnl_weight, layer=None):
            super(Jia.JiaLoss, self).__init__(classifier)

            self.target = None
            self.target_reduced = None
            self.target_np = target

            self.temp_init = temp_init
            self.temps = None

            self.snnl_weight = snnl_weight
            self.wm_labels = None

            self.layer = layer

        def initialize_temps(self):
            from torch.autograd import Variable
            self.temps = [Variable(self._functional.tensor(self.temp_init, diff=True)) for _ in range(100)]
            for temp in self.temps:
                temp.requires_grad = True

            self.classifier.optimizer.add_param_group({'params': self.temps, 'lr': 0.01, 'name': 'temps'})

        def exit(self):
            # remove temps from the optimizer params
            for i in reversed(range(len(self.classifier.optimizer.param_groups))):
                param_group = self.classifier.optimizer.param_groups[i]
                if 'name' in param_group and param_group['name'] == 'temps':
                    del self.classifier.optimizer.param_groups[i]

        def on_functional_change(self):
            self.target = self._functional.tensor(self.target_np)
            self.target_reduced = self._functional.tensor(np.argmax(self.target_np))
            self.initialize_temps()

        def reduce_labels(self):
            return True

        def set_snnl_weight(self, weight):
            """
            Set the weight of the SNNL
            :param weight: numeric type
            :return: None
            """
            self.snnl_weight = weight

        def set_wm_labels(self, labels):
            """
            Set the labels to be used for SNNL calculation
            :param labels: np.ndarray
            :return: None
            """
            self.wm_labels = self._functional.tensor(np.argmax(labels, axis=1))

        def compute_loss(self, pred, true, x=None):
            """
            Note: 'true' is cross-entropy loss labels. To pass SNNL labels,
                call set_wm_labels
            """
            if x is None:
                raise ValueError("Inputs must be provided to SNNLoss")

            losses = []
            n = 1  # Normalizing expression
            if self.snnl_weight > 0:
                mask = self._functional.equal(true, self.target_reduced, return_bool=True)
                x_snnl = x[mask]
                wm_labels = self.wm_labels[mask]

                if self.layer is None:
                    activations = self.classifier.get_all_activations(x_snnl)
                    n = len(activations)
                    for i, activation in enumerate(activations):
                        # don't account the last layer
                        if i == len(activations) - 1:
                            break

                        loss = Jia.SNNLoss.snnl(
                            self._functional,
                            activation,
                            wm_labels,
                            100 / (self.temps[i] + 1e-7)
                        )
                        losses.append(loss)
                else:
                    all_activations = self.classifier.get_all_activations(x_snnl)
                    activation = all_activations[self.layer]

                    loss = Jia.SNNLoss.snnl(
                        self._functional,
                        activation,
                        wm_labels,
                        100 / (self.temps[0] + 1e-7)
                    )
                    losses.append(loss)

            return self._functional.cross_entropy_loss(pred, true) - self.snnl_weight * sum(losses) / n

    def __init__(self, classifier, snnl_weight, num_classes: int, rate=2, layer=None, alternating_rate: int = 2,
                 pos: Union[str, Tuple[int, int, int]] = None, trigger_width: int = 5, trigger_height: int = 5,
                 **kwargs):
        """
        Create an :class:`Jia` instance.

        :param classifier: Source model.
        :param snnl_weight: float; weight of the SNN loss
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier
        self.snnl_weight = snnl_weight
        self.rate = rate
        self.layer = layer
        self.alternating_rate = alternating_rate
        self.num_classes = num_classes
        self.trigger_width = trigger_width
        self.trigger_height = trigger_height

        # Parse tuple from string.
        if type(pos) == str:
            self.pos = tuple(map(int, pos.replace(" ", "").strip()[1:-2].split(',')))
        else:
            self.pos = pos

        # For saving checkpoints, we memorize the best testing acc.
        self.best_acc = 0
        self.epoch = 0

    @staticmethod
    def get_name():
        return "Jia"

    def __compute_trigger_position_numpy(self, x, y, source, target, mean, std):
        """
        Compute the trigger position as is done in the paper
        :param x: np.ndarray; Train data
        :param y: np.ndarray; Train labels
        :param source: np.ndarray; one-hot encoded label for source class
        :param target: np.ndarray; one-hot encoded label for target class
        :return: 2-tuple; position to embed
        """
        mean = mean.reshape((1, 3, 1, 1))
        std = std.reshape((1, 3, 1, 1))
        x = ((x - mean) / std).astype(np.float32)

        source_indices = np.all(y == source, axis=1)
        target_indices = np.all(y == target, axis=1)

        x_source, y_source = x[source_indices], y[source_indices]
        x_target, y_target = x[target_indices], y[target_indices]

        x = np.vstack([x_source, x_target])
        y = np.vstack([y_source, y_target])

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x, y = x[indices], y[indices]

        ce_loss = self.classifier.loss
        loss_function = Jia.SNNLoss(self.classifier, 1e-1)
        self.classifier.loss = loss_function

        batch_size = 64
        losses = []
        for batch in range(x.shape[0] // batch_size):
            losses.append(self.classifier.loss_gradient(
                x[batch * batch_size:(batch + 1) * batch_size],
                y[batch * batch_size:(batch + 1) * batch_size]
            ))

        self.classifier.loss = ce_loss

        avg_loss = np.average(np.vstack(losses), axis=0)
        pos = np.unravel_index(np.argmax(avg_loss), shape=self.classifier.input_shape)

        return pos

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
               wm_loader: WRTDataLoader,
               keylength: int,
               source_class: int = None,
               target_class: int = None,
               verbose: bool = True):
        """ Embeds a secret trigger into the data.
        @mode: Return the watermark set for the training
        """
        if source_class is None:
            source_class = np.random.randint(self.num_classes)

        if target_class is None:
            target_class = (np.random.randint(1, self.num_classes) + source_class) % self.num_classes

        x_wm, _ = collect_n_samples(keylength, wm_loader, class_label=source_class, has_labels=True, verbose=verbose)
        y_wm = np.tile(target_class, x_wm.shape[0])

        pos = self.pos
        x_wm[:, :, pos[1]:pos[1] + self.trigger_width, pos[2]:pos[2] + self.trigger_height] = 1
        return x_wm, y_wm

    def _embed_step(self,
                    train_loader: WRTDataLoader,
                    jia_loss: JiaLoss,
                    wm_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    callbacks=None,
                    device: str = "cuda"):
        """ The embedding function for the Jia watermark (one epoch)
        """
        if callbacks is None:
            callbacks = []

        train_loss, wm_loss = Average(), Average()
        train_acc, wm_acc = Accuracy(), Accuracy()

        x_wm, y_ce_wm, y_snnl = wm_data
        batch_size = train_loader.batch_size
        with tqdm(train_loader) as train_loop:
            for batch_id, (x, y) in enumerate(train_loop):
                x, y = x.to(device), torch.eye(self.num_classes)[y].to(device)

                # Train on the batch of benign training data. Adjust batch norm every second layer.
                loss, output = self.classifier.fit_batch(x, y, eval_mode=(batch_id % self.alternating_rate == 0))

                # Save the metrics.
                train_loss.update(loss.item(), number=x.size(0))
                train_acc.update(output, y.argmax(dim=1))

                # Additionally train on a watermarking batch.
                if (batch_id % self.rate) == (self.rate - 1):
                    # Sample watermarking keys.
                    idx = np.arange(x_wm.shape[0])
                    np.random.shuffle(idx)
                    idx = idx[:batch_size]
                    x = torch.from_numpy(x_wm[idx]).to(device)
                    y_batch_ce = torch.from_numpy(y_ce_wm[idx]).to(device)
                    y_batch_snnl = y_snnl[idx]

                    # Train on the watermarking keys with the SNNLoss.
                    jia_loss.set_snnl_weight(self.snnl_weight)
                    jia_loss.set_wm_labels(y_batch_snnl)

                    loss, output = self.classifier.fit_batch(x, y_batch_ce, eval_mode=True)

                    jia_loss.set_snnl_weight(0)

                    # Save the metrics
                    wm_loss.update(loss.item(), number=x.size(0))
                    wm_acc.update(output, y_batch_ce.argmax(dim=1))

                for callback in callbacks:
                    callback.on_batch_end(batch_id)

                train_loop.set_description(f"{self.epoch}: Train Acc {train_acc}, Train Loss: {train_loss}, "
                                           f"WM Acc {wm_acc}, SNNL Loss {wm_loss}")
        return train_acc.value, train_loss.value, wm_acc.value, wm_loss.value

    def _keygen_embedding(self,
                          train_loader: WRTDataLoader,
                          wm_loader: WRTDataLoader,
                          wm_loader_target: WRTDataLoader,
                          embedding_keylength: int,
                          source_class: int,
                          target_class: int,
                          output_dir: str = None):
        """ Generates key for the embedding process
        """
        # Obtain all elements from the source class and relabel them by the target class.
        x_source, y_target = self.keygen(wm_loader=wm_loader,
                                         keylength=embedding_keylength,
                                         source_class=source_class,
                                         target_class=target_class)

        if output_dir is not None:
            self.visualize_key(x_source, output_dir=output_dir)

        # Load all elements from the target class.
        x_target, _ = collect_n_samples(embedding_keylength, wm_loader_target, class_label=target_class)

        # Expand the source and target labels for training.
        y_source = np.eye(self.num_classes)[np.tile(source_class, (x_source.shape[0]))]
        y_target = np.eye(self.num_classes)[np.tile(target_class, (x_target.shape[0]))]

        # Normalize the watermark data with the training dataset's parameters.
        x_source = train_loader.normalize(x_source)
        x_target = train_loader.normalize(x_target)

        # x_wm, y_ce_wm and y_snnl are the 'watermark data' used during training.
        x_wm = np.vstack([x_source, x_target])
        y_ce_wm = np.tile(y_target[0], (x_wm.shape[0], 1))
        y_snnl = np.vstack([y_source, y_target])
        return x_wm, y_ce_wm, y_snnl

    def embed(self,
              train_loader: WRTDataLoader,
              valid_loader: WRTDataLoader,
              wm_loader: WRTDataLoader,
              wm_loader_target: WRTDataLoader,
              source_class: int,
              target_class: int,
              epochs: int,
              keylength: int,
              embedding_keylength: int = np.inf,
              check_every_n_batches: int = None,
              patience: int = 2,
              max_rate_factor: int = 1,
              temp_init: int = 64,
              reduced_lr_rate: float = 1.0,
              boost_factor_source: int = 1.0,
              callbacks: List[WRTCallback] = None,
              output_dir: str = None,
              device: str = "cuda",
              **kwargs):
        """ Embeds a model with the Jia watermark.

        :param train_loader The training data loader.
        :param wm_loader The watermark data loader. It is expected to output unnormalized images (range [0-1]) of the
        source class.
        :param wm_loader_target A data loader for the target class. It is expected to output unnormalized images
        (range [0-1]) of the target class.
        :param valid_loader The validation data loader
        :param source_class The source class from which to sample all watermarking keys.
        :param target_class New class for watermarking keys.
        :param epochs Number of epochs to embed.
        :param keylength Number of watermarking keys to embed
        :param embedding_keylength Number of samples that are like a key to embed (only for Jia)
        :param check_every_n_batches Which batches to check for early stopping.
        :param patience Patience for early stopping.
        :param max_rate_factor Maximum factor by which the rate can grow.
        :param temp_init Initial value for the temperature.
        :param reduced_lr_rate Reduce learning rate times this factor for embedding. lr_new = lr / reduced_lr_rate
        :param callbacks List of callbacks during training
        :param output_dir Directory to save intermediate results and the model
        :param device Device to train on.
        """
        # Generate keys used during the training process.
        wm_data = self._keygen_embedding(train_loader, wm_loader, wm_loader_target,
                                         embedding_keylength, source_class, target_class, output_dir)
        x_wm, y_ce_wm, y_snnl = wm_data

        # Prepare the source model.
        ce_loss = self.classifier.loss
        jia_loss = Jia.JiaLoss(self.classifier,
                               np.eye(self.num_classes)[target_class],
                               temp_init=temp_init,
                               snnl_weight=0,
                               layer=self.layer)
        self.classifier.loss = jia_loss
        self.classifier.model.return_hidden_activations = True
        self.classifier.lr /= reduced_lr_rate

        # Sample from the source class and add boosted data to ensure high class test accuracy.
        x_source, y_source = collect_n_samples(n=np.inf, data_loader=wm_loader, class_label=source_class)

        train_loader = train_loader.add_numpy_data(x_source, y_source, boost_factor=boost_factor_source)
        x_source = train_loader.normalize(x_source)

        def eval_early_stopping():
            self.classifier.model.eval()
            self.classifier.loss = jia_loss
            return self.classifier.evaluate(x_wm[:keylength], y_ce_wm[:keylength])[0]

        # Early stopping on the watermarking loss.
        callbacks = [wrt_callbacks.EarlyStoppingWRTCallback(eval_early_stopping, # Watermarking loss.
                                                            check_every_n_batches=check_every_n_batches,
                                                            patience=patience,
                                                            mode='min'),
                     wrt_callbacks.DebugWRTCallback(lambda: self.verify(x_wm[:keylength], y_ce_wm[:keylength])[0],
                                                    message="wm_acc",
                                                    check_every_n_batches=check_every_n_batches),
                     wrt_callbacks.DebugWRTCallback(lambda: self.verify(x_source, y_source)[0],
                                                    message=f"Class {source_class} accuracy",
                                                    check_every_n_batches=check_every_n_batches)]

        history = {}
        for self.epoch in range(epochs):
            try:
                train_acc, train_loss, wm_acc, wm_loss = self._embed_step(train_loader=train_loader,
                                                                          jia_loss=jia_loss,
                                                                          wm_data=wm_data,
                                                                          callbacks=callbacks)
                # Manually save the history.
                history.setdefault("train_acc", []).append(train_acc)
                history.setdefault("train_loss", []).append(train_loss)
                history.setdefault("wm_acc", []).append(wm_acc)
                history.setdefault("wm_loss", []).append(wm_loss)

            except StopTrainingException:
                print("StopTrainingException has been raised!")
                break

            # Evaluate the test accuracy
            test_acc = evaluate_test_accuracy(self.get_classifier(), valid_loader)
            history.setdefault("test_acc", []).append(test_acc)
            print(f"Source Model Test Acc: {test_acc}")

        x_wm, y_wm = x_wm[:keylength], np.argmax(y_ce_wm[:keylength], axis=1)

        jia_loss.exit()
        self.classifier.loss = ce_loss
        self.classifier.lr *= reduced_lr_rate
        self.classifier.model.eval()

        # Save the best model.
        if output_dir is not None:
            # Save the history to file.
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(history, fp=f)

            tmp_optimizer = deepcopy(self.get_classifier().optimizer)

            # remove temps from the optimizer params
            try:
                for i in reversed(range(len(tmp_optimizer.param_groups))):
                    param_group = tmp_optimizer.param_groups[i]
                    if 'name' in param_group and param_group['name'] == 'temps':
                        del tmp_optimizer.param_groups[i]
            except:
                print("Error, could not remove temps from optimizer")
                pass

            checkpoint = {
                'model': self.get_classifier().model.state_dict(),
                'optimizer': tmp_optimizer.state_dict(),
                'x_wm': x_wm,
                'y_wm': y_wm,
            }
            self.save('best.pth', path=output_dir, checkpoint=checkpoint)

        return x_wm, y_wm

    def load(self, filename, path=None, **load_kwargs: dict):
        """ Loads parameters necessary for validating a watermark.
        """
        checkpoint = super().load(filename, path, **load_kwargs)

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint["x_wm"], checkpoint["y_wm"]

    def extract(self, x, classifier=None, **kwargs):
        if classifier is None:
            classifier = self.classifier
        return classifier.predict(x, **kwargs).argmax(1)

    def predict(self, x, **kwargs):
        """
        Perform prediction using the watermarked classifier.

        :param x: Test set.
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :return: Predictions for test set.
        :rtype: `np.ndarray`
        """
        return self.classifier.predict(x, **kwargs)


@mlconfig.register
def wm_jia(classifier: PyTorchClassifier, **kwargs):
    """ Creates a Jia watermarking instance.
    """
    return Jia(classifier, **kwargs)


@mlconfig.register
def wm_jia_embed(defense: Jia,
                 config,
                 **kwargs):
    """ Embeds a watermark into a pre-trained model.
    """
    wm_loader_target = config.wm_dataset_target()
    return defense.embed(wm_loader_target=wm_loader_target, **kwargs), defense


@mlconfig.register
def wm_jia_keygen(defense: Jia,
                  keylengths: List[int],
                  train_loader: WRTDataLoader,
                  config,
                  source_class: int = None,
                  target_class: int = None,
                  **kwargs):
    """
    Key generation for Jia requires a watermark loader that loads unnormalized images from the source class.
    """
    # Create a custom wm loader with the correct class
    if source_class is None:
        source_class = int(np.random.randint(0, defense.num_classes, size=1)[0])

    if target_class is None:
        target_class = (np.random.randint(1, defense.num_classes) + source_class) % defense.num_classes
    print(f"\nLoading source/target classes {source_class}, {target_class}")

    wm_loader = config.wm_dataset(class_labels=source_class)

    x_wm, y_wm = defense.keygen(wm_loader=wm_loader,
                                keylength=max(keylengths),
                                source_class=source_class,
                                target_class=target_class,
                                verbose=False)
    x_wm = train_loader.normalize(x_wm)

    for keylength in keylengths:
        yield x_wm[:keylength], y_wm[:keylength]
