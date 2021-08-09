"""
    Embedding a DeepSign watermark.

    | Paper link: https://arxiv.org/pdf/1804.00750.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from copy import deepcopy
from typing import List, Tuple, Callable

import os

import mlconfig
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from wrt.training import callbacks as wrt_callbacks

from wrt.classifiers import PyTorchClassifier, StopTrainingException
from wrt.defenses.watermark.watermark import Watermark
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.datasets.utils import collect_n_samples
from wrt.training.trainer import Trainer


class DeepSignWB(Watermark):
    """
    Embedding a DeepSign White-Box watermark.

    | Paper link: https://arxiv.org/pdf/1804.00750.pdf
    """

    def __init__(self, classifier,
                 num_classes: int,
                 create_target_class_loader: Callable[[int], WRTDataLoader],
                 num_gaussians: int,
                 layer_index=None,
                 layer_dim=None,
                 gamma0=1,
                 gamma1=500,
                 gamma2=1,
                 gamma3=0.01,
                 embedding_rate: int = 2,
                 mu_reg=None,
                 mu_lr: float = 1.0,
                 allow_caching: bool = True,
                 from_scratch=True,
                 separate_means=True,
                 **kwargs):
        """
        Create an :class:`DeepSignWB` instance.

        :param create_target_class_loader Callable; creates a data loader that only loads images from a target class.
        :param classifier: Model to embed to the watermark.
        :param target_class: target watermark class label.
        :param target_layer: target watermark layer name.
        :param layer_dim: target watermark layer dimension.
        :param gamma1: gamma for center loss.
        :param gamma2: gamma for watermark embedding loss.
        :param allow_caching Whether to cache the watermarking key.
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier

        if layer_index is None:
            raise ValueError('layer_index is not provided.')
        if layer_dim is None:
            raise ValueError('layer_dim is not provided.')

        self.create_target_class_loader = create_target_class_loader
        self.num_gaussians = num_gaussians
        self.layer_index = layer_index
        self.layer_dim = layer_dim
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.embedding_rate = embedding_rate
        self.mu_lr = mu_lr
        self.mu_reg = gamma1 if mu_reg is None else mu_reg
        self.from_scratch = from_scratch
        self.separate_means = separate_means
        self.num_classes = num_classes

        self.mu = None
        self.A_value = None
        self.b_value = None
        self.X_key = {}

    @staticmethod
    def get_name():
        return "Deepsign"

    @staticmethod
    def chunks(x, batch_size):
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size]

    def _get_mean_activation(self,
                             classifier: PyTorchClassifier,
                             x: np.ndarray,
                             layer: int,
                             batch_size: int):
        """ Gets the mean activation of the source model on input x for some layer
        """

        def get_activations(x: np.ndarray):
            model = classifier.model
            model = model.eval()

            if hasattr(model, 'return_hidden_activations'):
                model.return_hidden_activations = True

            x_batch = torch.from_numpy(x).to("cuda")

            model_outputs = model(x_batch)
            act = model_outputs[layer].view(x.shape[0], -1)

            if hasattr(model, 'return_hidden_activations'):
                model.return_hidden_activations = False

            # act = classifier.get_all_activations(x)[layer]
            return act.sum(axis=0).detach().cpu().numpy()

        # Split into batches
        target_activation = None
        for x_batch in self.chunks(x, batch_size):
            if target_activation is None:
                target_activation = get_activations(x_batch)
            else:
                target_activation += get_activations(x_batch)

        return (target_activation / x.shape[0]).reshape(-1)

    def get_activations(self,
                        classifier: PyTorchClassifier = None,
                        from_scratch: bool = False,
                        max_n=64,
                        batch_size=64,
                        verbose: bool = False) -> np.ndarray:
        """
        Initialize the centers. If from_scratch is True, then initialize them randomly. Otherwise,
        initialize them with the current mean activations

        :param classifier Classifier to obtain activations from.
        :param from_scratch (optional) Called only during training, when the model is not pretrained.
        :param num_gaussians (optional) Number of Gaussian distributions. Defaults to the number of classes.
        :param max_n Maximum number of samples to load per target class.
        :param verbose Whether to print a progress bar

        :return: mu The mean of the Gaussian mixture.
        """
        if classifier is None:
            classifier = self.get_classifier()

        if from_scratch:
            # Initialize mu randomly.
            return np.random.normal(0, 1, size=(self.num_gaussians, self.layer_dim)).astype(np.float32)

        # Initialize mu from the activation on some target classes.
        # For simplicity, we default to the first num_gaussian classes.
        mu = []
        for target_class in tqdm(range(self.num_gaussians), desc="Initializing GMM.", disable=not verbose):

            if str(target_class) in self.X_key:
                x = self.X_key[str(target_class)]
            else:
                # Load samples from the target class.
                target_class_loader: WRTDataLoader = self.create_target_class_loader(class_labels=target_class)
                x, _ = collect_n_samples(n=max_n,
                                         data_loader=target_class_loader,
                                         class_label=target_class,
                                         has_labels=True,
                                         verbose=False)
                self.X_key.setdefault(str(target_class), x)

            # Get their mean activation from the hidden layer.
            mu_i = self._get_mean_activation(classifier=classifier, x=x, layer=self.layer_index,
                                             batch_size=batch_size)
            mu.append(mu_i)
        return np.array(mu).astype(np.float32)

    def keygen(self,
               keylength: int,
               **kwargs):
        """
        :param keylength: Number of keys.
        :param M: Feature dimensionality.
        :param N: Number of bits per Gaussian.
        :return:
        """
        signature = np.random.randint(2, size=(self.num_gaussians, keylength)).astype(np.float32)
        A = np.random.normal(0, 1, (self.layer_dim, keylength)).astype(np.float32)
        return A, signature

    def embed(self,
              train_loader: WRTDataLoader,
              valid_loader: WRTDataLoader,
              keylength: int,
              epochs: int = 1,
              output_dir: str = None,
              check_every_n_batches: int = None,
              reduce_lr_by_factor: float = 1.0,
              verbose: bool = True,
              patience: int = 5,
              device="cuda",
              **kwargs):
        """
        Train a model on the watermark data. See class documentation for more information on the exact procedure.

        :param train_loader Training data loader
        :param valid_loader Loader for the validation data.
        :param keylength: Number of keys to embed into the model.
        :param epochs Number of epochs to use for trainings.
        :param output_dir: The output directory for logging the models and intermediate training progress.
        :param check_every_n_batches Callback for early stopping every n batches
        :param reduce_lr_by_factor Reduces the model's learning rate by this factor.
        :param verbose Whether to print statements for training.
        :param device: cuda or cpu
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :return: Watermark train set, watermark labels
        """
        if len(kwargs.items()) > 0:
            print(f"[WARNING] Unused params {kwargs}")

        if self.num_gaussians != 1:
            print(f"[WARNING] Parameter num_gaussians has value '{self.num_gaussians}', but has only been validated"
                  f"to work for a value of '1'.")

        # Rename parameters to comply with the paper.
        M = self.layer_dim  # Feature size of the selected layer.
        N = keylength  # Length of the embedded watermark.

        # Initialize the random projection matrix A^{M \times N}
        A = np.random.normal(0, 1, (M, N)).astype(np.float32)

        # Initialize the Gaussian mixture. (s \times M)
        mu = self.get_activations(from_scratch=self.from_scratch, max_n=max(64, train_loader.batch_size), verbose=verbose)

        # Initialize the signature. (s \times N)
        signature = np.random.randint(2, size=(self.num_gaussians, keylength)).astype(np.float32)

        self.A_value = torch.from_numpy(A).to(device)
        self.b_value = torch.from_numpy(signature).to(device)

        mu = torch.nn.Parameter(torch.from_numpy(mu).to(device), requires_grad=True)
        self.mu = mu

        # Initialize a trainer to evaluate the test accuracy.
        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          device=device, num_epochs=0)

        callbacks = [wrt_callbacks.DebugWRTCallback(lambda: self.verify(A, signature)[0],
                                                    message="wm_acc",
                                                    check_every_n_batches=check_every_n_batches),
                     wrt_callbacks.DebugWRTCallback(lambda: trainer.evaluate()[1].value,
                                                    check_every_n_batches=check_every_n_batches),
                     wrt_callbacks.EarlyStoppingWRTCallback(lambda: self.verify(A, signature)[0],
                                                            check_every_n_batches=check_every_n_batches,
                                                            patience=patience,
                                                            mode='max')]

        # Fine-Tune the model.
        self.classifier.lr /= reduce_lr_by_factor
        self._embedding_procedure(train_loader=train_loader, nb_epochs=epochs, mu=mu, gamma0=self.gamma0,
                                  gamma1=self.gamma1, num_gaussians=self.num_gaussians, gamma2=self.gamma2,
                                  gamma3=self.gamma3, separate_means=self.separate_means, callbacks=callbacks,
                                  device=device)
        self.classifier.lr *= reduce_lr_by_factor



        if output_dir is not None:
            checkpoint = {
                'model': self.get_classifier().model.state_dict(),
                'optimizer': self.get_classifier().optimizer.state_dict(),
                'X_key': self.X_key,    # The watermarking key (a dict of images for each Gaussian)
                'x_wm': self.A_value,   # The random projection matrix.
                'y_wm': signature,      # The signature provided by the user.
            }
            self.save('best.pth', path=output_dir, checkpoint=checkpoint)
        return A, signature

    def load(self, filename, path=None, **load_kwargs: dict) -> Tuple[np.ndarray, np.ndarray]:
        """ Loads the source model and params needed for verification
        """
        checkpoint = super().load(filename, path)

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])

        self.A_value = checkpoint["x_wm"]
        self.X_key = checkpoint["X_key"]

        # Ensure we return numpy data.
        x_wm, y_wm = checkpoint['x_wm'], checkpoint['y_wm']
        if isinstance(x_wm, torch.Tensor):
            x_wm = x_wm.cpu().numpy()
        if isinstance(y_wm, torch.Tensor):
            y_wm = y_wm.cpu().numpy()

        return x_wm, y_wm

    def _train_step(self, x: torch.Tensor, y: torch.Tensor,
                    model, optimizer, criterion, mu,
                    gamma0=0, gamma1=0, gamma2=0, gamma3=0, separate_means=False):
        model = model.train()
        # Always ensure that all activations are returned.
        if hasattr(model, 'return_hidden_activations'):
            model.return_hidden_activations = True

        optimizer.zero_grad()
        model_outputs = model(x)
        layer_activation = model_outputs[self.layer_index].view(-1, self.layer_dim)

        # Form the loss.
        loss0 = gamma0 * criterion(model_outputs[-1], y)

        loss1 = 0
        if gamma1 > 0:
            loss1 = gamma1 * DeepSignWB.cal_loss1(activ=layer_activation, mu=mu, targets=y,
                                                  num_features=self.layer_dim, separate_means=separate_means)

        loss2, acc2 = 0, 0
        if gamma2 > 0:
            loss2, acc2 = DeepSignWB.cal_loss2(mu=mu, A=self.A_value, b=self.b_value, targets=y)
            loss2 *= gamma2

        # regularize the centers (this was not explicitly stated in the original paper, but exists
        # in the decompiled binary.
        loss3 = 0
        if gamma3 > 0:
            loss3 = gamma3 * torch.sum(torch.abs(1 - torch.sum(mu ** 2, dim=1)))

        loss = loss0 + loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()

        if gamma2 > 0:
            # print("grad data", mu.grad.data)
            mu.data = mu.data - self.mu_lr * mu.grad.data

        return loss, [loss0, loss1, loss2, loss3], [acc2]

    def _embedding_procedure(self, train_loader: WRTDataLoader, num_gaussians: int, mu: torch.nn.Parameter,
                             gamma0: float = 1, gamma1=0.01, gamma2=0.01, gamma3=0, separate_means=True,
                             nb_epochs=20, callbacks: List = None, device="cuda"):
        """ Trains the model with the Deepsigns loss on the training data.
        """
        if callbacks is None:
            callbacks = []

        model: torch.nn.Sequential = self.get_classifier().model
        optimizer = self.get_classifier().optimizer
        criterion = self.get_classifier().loss

        # Create a subset of the training data containing only the Gaussian classes.
        subset_loader = train_loader

        # Start training
        for e in range(nb_epochs):
            with tqdm(subset_loader, desc="Embed DeepSigns") as train_loop:
                for batch_id, (x_batch, y_batch) in enumerate(train_loop):
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                    # Train on regular data.
                    _ = self._train_step(x_batch, y_batch, model, optimizer, criterion,
                                         mu=mu,
                                         gamma0=1.0, gamma1=0, gamma2=0, gamma3=0,
                                         separate_means=False)

                    # Embed specifically the watermarking key data.
                    if batch_id % self.embedding_rate == 0:
                        class_indices = np.arange(num_gaussians)
                        np.random.shuffle(class_indices)

                        # Take three random classes and embed.
                        for class_index in class_indices[:3]:
                            rnd_idx = np.random.choice(len(self.X_key[str(class_index)]),
                                                       size=train_loader.batch_size,
                                                       replace=False)

                            x = self.X_key[str(class_index)][rnd_idx]
                            y = np.array([class_index] * len(rnd_idx))
                            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

                            loss, losses, accs = self._train_step(x, y, model, optimizer, criterion, mu=mu,
                                                                  gamma0=gamma0, gamma1=gamma1, gamma2=gamma2,
                                                                  gamma3=gamma3,
                                                                  separate_means=separate_means)

                    # Compute the loss.
                    train_loop.set_description(f'Epoch {e + 1}/{nb_epochs} CE Loss: {losses[0]:.4f}, '
                                               f'Reg Loss 1: {losses[1]:.4f}, Reg Loss 2: {losses[2]:.5f}, '
                                               f'Reg Loss 3: {losses[3]:.4f}, (Train) wm acc: {accs[0]}')
                    try:
                        for callback in callbacks:
                            callback.on_batch_end(batch_id)
                    except StopTrainingException:
                        print("Raise StopTrainingException")
                        if hasattr(model, 'return_hidden_activations'):
                            model.return_hidden_activations = False
                        return
                try:
                    for callback in callbacks:
                        callback.on_epoch_end(e)
                except StopTrainingException:
                    print("Raise StopTrainingException")
                    break

        if hasattr(model, 'return_hidden_activations'):
            model.return_hidden_activations = False
        print("Done Training!")

    @staticmethod
    def cal_loss1(activ, mu, targets, num_features, separate_means):
        """ This loss makes sure that mu corresponds to the actual activations for that target class.
        """
        targets = targets.unsqueeze(dim=1).repeat(1, num_features)
        centers_batch = torch.gather(mu, 0, targets)  # If this fails, there are more targets than mu tensors.
        loss = torch.nn.MSELoss(reduction='sum')(activ, centers_batch)

        if separate_means:
            activ_norm = torch.sum(torch.pow(activ, 2), dim=1).view(1, activ.shape[0])
            mu_norm = torch.sum(torch.pow(mu, 2), dim=1).view(mu.shape[0], 1)
            inner_prod = torch.matmul(mu, torch.transpose(activ, dim0=0, dim1=1))
            activ_norm_repeat = activ_norm.repeat(mu.shape[0], 1)
            mu_norm_repeat = mu_norm.repeat(1, activ.shape[0])
            second_loss = torch.sum(activ_norm_repeat + mu_norm_repeat - 2 * inner_prod)
            second_loss = ((second_loss - loss) / mu.shape[0])

            loss = loss - second_loss

        return loss

    @staticmethod
    def cal_loss2(mu: torch.Tensor,
                  A: torch.Tensor,
                  b: torch.Tensor,
                  targets):
        counter, loss, acc = 0, 0, 0

        # Compute the key extraction matrix.
        G = torch.matmul(mu, A)
        G = torch.sigmoid(G)

        for k in range(b.shape[0]):
            if k in targets:
                counter += 1
                loss += F.binary_cross_entropy(G[k], b[k].float(), reduction='mean')
                acc += (((G[k] > 0.5) * 1 == b[k]) * 1).float().mean()

        return loss, (acc / counter)

    def extract(self,
                x: np.ndarray,
                classifier: PyTorchClassifier = None,
                batch_size: int = 64,
                max_n=64,
                device="cuda",
                **kwargs):
        """
        Extract a secret message from the model.

        :param x: Unused.
        :param classifier: Classifier; if provided, extract the watermark from
                           the given classifier instead of this object's classifier
        :param batch_size: Size of batches.
        :param max_n Maximum number of samples to estimate the mean.
        :param device Device for the forward pass.
        :param kwargs: Other parameters.
        :return: The secret message.
        """
        if classifier is None:
            classifier = self.classifier

        if x is None:
            x = self.A_value

        # Get the mean activations.
        classifier.model.eval()
        activ_centerK = self.get_activations(classifier=classifier, max_n=max_n, verbose=False)

        X_Ck = np.matmul(activ_centerK, x)
        X_Ck_sigmoid = torch.sigmoid(torch.from_numpy(X_Ck)).numpy()
        decode_wmark = (X_Ck_sigmoid > 0.5) * 1

        return decode_wmark  # shape sxN (num_gaussians, keylength)

    def verify(self,
               x: np.ndarray,
               y: np.ndarray = None,
               classifier: PyTorchClassifier = None,
               **kwargs) -> Tuple[float, bool]:
        """
        :param x: self.A_value (the random projection matrix)
        :param y: The message (keylength many bits)
        :param classifier: The classifier to verify.
        :param kwargs:
        :return:
        """

        extracted_watermarks: np.ndarray = self.extract(x, classifier=classifier)
        print(f"Verify: {y}, Extracted: {extracted_watermarks}")

        diff = 0
        for extracted_wm, target_wm in zip(y, extracted_watermarks):
            diff += np.abs(extracted_wm - target_wm)  # Hamming distance

        # Compute the average accuracy as 1-BER
        total = np.prod(extracted_watermarks.shape)
        acc = 1 - (np.sum(diff) / total)
        return acc, acc > 0

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
def wm_deepsignwb(classifier, config, **kwargs):
    create_target_class_loader = config.wm_dataset
    return DeepSignWB(classifier=classifier, create_target_class_loader=create_target_class_loader, **kwargs)


@mlconfig.register
def wm_deepsignwb_embed(defense: DeepSignWB, **kwargs):
    return defense.embed(**kwargs), defense


@mlconfig.register
def wm_deepsignwb_keygen(defense: DeepSignWB, keylengths: List[int], **kwargs):
    for n in keylengths:
        wm_x, wm_y = defense.keygen(keylength=n, **kwargs)
        yield wm_x, wm_y
