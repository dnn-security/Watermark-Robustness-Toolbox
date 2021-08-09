from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import mlconfig
import numpy as np
from torch.utils import data
from tqdm import tqdm

from wrt.art_classes.fast_gradient import FastGradientMethod
from wrt.classifiers import PyTorchClassifier, StopTrainingException
from wrt.classifiers.loss import Loss
from wrt.config import WRT_NUMPY_DTYPE, WRT_DATA_PATH
from wrt.defenses.utils import NormalizingPreprocessor
from wrt.defenses.watermark.watermark import Watermark
from wrt.training import WRTCallback
from wrt.training import callbacks as wrt_callbacks
from wrt.training.datasets.wrt_data_loader import WRTDataLoader
from wrt.training.utils import compute_accuracy


class Blackmarks(Watermark):
    """
    Implement the Blackmarks watermarking scheme
    https://arxiv.org/pdf/1904.00344.pdf
    """

    class BMLoss(Loss):
        """
        Loss used for fine-tuning in Blackmarks
        Called the "regularized loss" in the paper
        """

        def __init__(self, classifier: PyTorchClassifier,
                     encoding: np.ndarray):
            """
            :param classifier: Classifier instance
            """
            super(Blackmarks.BMLoss, self).__init__(classifier)

            self._f_np = encoding

            self._b = None
            self._f = None

        def reduce_labels(self):
            return True

        def on_functional_change(self):
            self._f = self._functional.tensor(self._f_np)

        def set_b(self, b):
            self._b = self._functional.tensor(b)

        def compute_loss(self, pred, true, x=None):
            def hamming_distance(x, y):
                """
                Returns the Hamming distance between x and y
                """
                return self._functional.sum(self._functional.abs(x - y))

            ce_loss = self._functional.cross_entropy_loss(pred, true)

            pred_softmax = self._functional.softmax(pred, axis=1)

            # decode the predictions
            predictions = self._functional.matmul(pred_softmax, self._f)

            # get the Hamming distance between b and the predictions
            ret = hamming_distance(predictions, self._b)

            # Fix a bug where the regularized loss completely overshadows the standard
            # cross-entropy loss; it turns out Pytorch averages the loss across the shape
            # of the output
            ret = ret / self._functional.shape(pred)[0]

            return ce_loss + ret

    def __init__(self,
                 classifier: PyTorchClassifier,
                 num_classes: int,
                 eps: float = 0.5,
                 lmbda: float = 0.1,
                 num_variants: int = 3,
                 dataset_name="cifar10",
                 compute_new_encoding: bool = False,
                 **kwargs):
        """
        Create an :class:`Blackmarks` instance.

        :param classifier: Model to train.
        :param lmbda: float; strength of the regularization loss
        """
        super().__init__(classifier, **kwargs)
        self.classifier = classifier
        self.lmbda = lmbda
        self.eps = eps
        self.num_variants = num_variants
        self.num_classes = num_classes
        self.compute_new_encoding = compute_new_encoding
        self.dataset_name = dataset_name

        self.encoding = None

    @staticmethod
    def get_name():
        return "Blackmarks"

    @staticmethod
    def _save_encoding(f):
        import os
        import pickle

        with open(os.path.join(WRT_DATA_PATH, 'blackmarks_encoding'), 'wb') as encoding_file:
            pickle.dump(f, encoding_file)

    def _try_load_encoding(self):
        import os
        import pickle

        if self.dataset_name == "cifar10":
            file = "blackmarks_encoding"
        elif self.dataset_name == "imagenet":
            file = "blackmarks_encoding_imagenet"
        else:
            print(f"No existing encoding found! for {self.dataset_name}")
            return

        try:
            with open(os.path.join(WRT_DATA_PATH, file), 'rb') as encoding_file:
                f = pickle.load(encoding_file)
                return f
        except FileNotFoundError:
            return None

    def compute_encoding(self,
                         data_loader: WRTDataLoader,
                         max_batches: int = 2000,
                         device="cuda"):
        """
        Compute the encoding scheme
        :param data_loader: The training data loader.
        :param max_batches Maximum number of batches to load.
        :param device Device to compute ford propagations.
        :return: 1-D np.ndarray with length equal to the number of classes. The
            array contains '0' at position i if class i encodes to bit0, otherwise
            if class i encodes to bit1, the array contains '1'
        """
        from sklearn.cluster import KMeans

        if not self.compute_new_encoding:
            f = self._try_load_encoding()
            if f is not None:
                print("Loaded an existing encoding!")
                self.encoding = f
                return f

        num_classes = self.classifier.nb_classes()
        # Collect one mean activation per class
        activations = np.zeros(shape=(num_classes, num_classes)).astype(np.float64)
        class_counter = np.ones(shape=num_classes)

        with tqdm(data_loader, desc="Computing encoding", total=min(max_batches, len(data_loader))) as train_loop:
            for batch_id, (x, y) in enumerate(train_loop):
                logits_batch = self.predict(x.to(device), batch_size=data_loader.batch_size)
                for class_label, logits in zip(y.to(device), logits_batch):
                    activations[class_label] += logits
                    class_counter[class_label] += 1

                if batch_id >= max_batches:
                    train_loop.close()
                    break

        mean_activations = np.divide(activations, class_counter, out=np.zeros_like(activations),
                                     where=class_counter != 0)
        kmeans = KMeans(n_clusters=2).fit(mean_activations)

        f = kmeans.labels_.astype(WRT_NUMPY_DTYPE)
        self._save_encoding(f)
        self.encoding = f

        return f

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
               wm_loader: data.DataLoader,
               train_loader: WRTDataLoader,
               keylength: int = 100,
               key_expansion_factor: int = 10,
               **kwargs):

        # encoding = np.random.randint(2, size=self.num_classes)
        try:
            self.encoding = self._try_load_encoding()
            print(f"Loaded an existing Blackmarks (shape {self.encoding.shape} encoding!")
        except:
            self.encoding = self.compute_encoding(train_loader)
            print("Computed a new Blackmarks encoding!")

        # Key generation requires running the entire embedding procedure.
        x_wm, y_wm, cluster_ids = self._generate_candidate_keys(wm_loader=wm_loader,
                                                                train_loader=train_loader,
                                                                encoding=self.encoding,
                                                                keylength=keylength,
                                                                key_expansion_factor=key_expansion_factor)
        x_wm, y_wm = train_loader.normalize(x_wm), y_wm

        # Filter only successful adversarial examples.
        y_pred: np.ndarray = self.predict(x_wm, batch_size=train_loader.batch_size)
        keep_indices = np.where(np.argmax(y_pred, axis=1) != y_wm)
        x_wm, y_wm, cluster_ids = x_wm[keep_indices], y_wm[keep_indices], cluster_ids[keep_indices]

        print(f"Successful adversarial examples: {x_wm.shape}")

        signature = np.random.randint(0, 2, keylength)
        zero_idx, ones_idx = np.where(cluster_ids == 0)[0].tolist(), np.where(cluster_ids == 1)[0].tolist()

        x_wm_final, y_wm_final = [], []
        for bit in signature:
            if bit == 0:
                i = zero_idx.pop()
            elif bit == 1:
                i = ones_idx.pop()
            else:
                raise ValueError

            x_wm_final.append(x_wm[i])
            y_wm_final.append(y_wm[i])
        x_wm_final, y_wm_final = np.asarray(x_wm_final), np.asarray(y_wm_final)
        print(x_wm_final.shape)
        return x_wm_final[:keylength], signature

    def _generate_candidate_keys(self,
                                 wm_loader: data.DataLoader,
                                 train_loader: WRTDataLoader,
                                 encoding: np.ndarray,
                                 keylength: int = 100,
                                 key_expansion_factor: int = 1,
                                 batch_size: int = 32):
        """
        Generate a watermark key for the given signature b

        :param wm_loader: The data loader from which to generate the watermark. Expects
        unnormalized images in the range [0-1].
        :param train_loader: The training data loader. Used only to get the preprocessing normalizer.
        :param encoding: np.ndarray with shape (num_classes,) for the encoding scheme
        :param keylength: Number of keys.
        :param key_expansion_factor Factor of candidate to actual keys.
        :param batch_size: Batch size used for generating the adversarial examples.
        :return: Watermark data, watermark labels
        """
        # Generate Adversarial Examples.
        adv_attack = FastGradientMethod(
            self.classifier,
            eps=self.eps,
            minimal=True,
            targeted=True,
            batch_size=batch_size
        )
        preprocessor = NormalizingPreprocessor(mean=train_loader.mean, std=train_loader.std)
        self.classifier.add_preprocessor(preprocessor, "blackmarks_normalizer")

        group0 = set(np.arange(self.num_classes)[encoding == 0])
        group1 = set(np.arange(self.num_classes)[encoding == 1])

        # print(f"Group0: {[cifar_classes[x] for x in group0]}, Group1: {[cifar_classes[x] for x in group1]}")

        total_samples_per_set = (key_expansion_factor * keylength) // 2
        x_wm0, x_wm1, y_wm0, y_wm1 = [], [], [], []
        with tqdm(desc="Blackmarks Keygen 1", total=2 * total_samples_per_set, disable=True) as pbar:
            for x_batch, y_batch in wm_loader:
                if y_batch.dim() > 1:
                    y_batch = y_batch.argmax(1)

                if (len(x_wm0) >= total_samples_per_set) and (len(x_wm1) >= total_samples_per_set):
                    print("Done with keygen1!")
                    break
                # We want to embed from group a into group b or vice-versa depending on the bit in the message.
                for x, y in zip(x_batch, y_batch):
                    if len(x_wm0) >= total_samples_per_set and len(x_wm1) >= total_samples_per_set:
                        break
                    # Map from group 1 to group 0
                    if len(x_wm1) < total_samples_per_set and y.item() in group1:
                        target_class = list(group0)[np.random.randint(len(group0))]
                        x_adv = adv_attack.generate(x=x[np.newaxis].cpu().numpy().astype(np.float32),
                                                    y=np.eye(self.num_classes)[target_class][np.newaxis],
                                                    axis=0)
                        x_wm1.append(x_adv)
                        y_wm1.append(y)  # Source class is the label
                    # Map from group 0 to group 1
                    elif len(x_wm0) < total_samples_per_set and y.item() in group0:
                        target_class = list(group1)[np.random.randint(len(group1))]
                        x_adv = adv_attack.generate(x=x[np.newaxis].cpu().numpy().astype(np.float32),
                                                    y=np.eye(self.num_classes)[target_class][np.newaxis],
                                                    axis=0)
                        x_wm0.append(x_adv)
                        y_wm0.append(y)  # Source class is the label
                    # Skipping otherwise.
                    pbar.n = len(x_wm0) + len(x_wm1)
                    pbar.refresh()
        self.classifier.remove_preprocessor("blackmarks_normalizer")

        x_wm = np.asarray(x_wm0 + x_wm1).squeeze().astype(np.float32)
        y_wm = np.asarray(y_wm0 + y_wm1).squeeze()
        print(f"Generated candidate key: x_wm {x_wm.shape}, y_wm, {y_wm.shape}")
        return x_wm, y_wm, np.array([0] * len(x_wm0) + [1] * len(x_wm1))

    def _embed_candidate_keys(self,
                              train_loader: WRTDataLoader,
                              epochs: int,
                              train_signature: np.ndarray,
                              x_wm: np.ndarray,
                              y_wm: np.ndarray,
                              callbacks: List[WRTCallback] = None,
                              device="cuda"):
        """ Embed a set of candidate keys.
        """
        # Embed with CE loss and Blackmarks loss and early stopping on the watermark accuracy.
        ce_loss = self.classifier.loss  # CE loss
        rate = int(1 / self.lmbda) + 1  # Ratio of training vs watermark batches.
        bm_loss = Blackmarks.BMLoss(self.classifier, encoding=self.encoding)

        if x_wm.shape[0] < train_loader.batch_size:
            print("[WARNING] Size of key smaller than batch size. ")

        for epoch in range(epochs):
            reg_loss, loss = 0, 0
            with tqdm(train_loader) as pbar:
                self.classifier.loss = ce_loss
                for batch_id, (x, y) in enumerate(pbar):
                    x, y = x.to(device), y.to(device)

                    loss, _ = self.classifier.fit_batch(x, y, eval_mode=(batch_id % 2 == 0))

                    if batch_id % rate == 0:
                        self.classifier.loss = bm_loss
                        b_indices = np.random.randint(0, high=len(train_signature),
                                                      size=train_loader.batch_size)
                        bm_loss.set_b(train_signature[b_indices])
                        x_wm_batch, y_wm_batch = x_wm[b_indices], y_wm[b_indices]
                        reg_loss, out = self.classifier.fit_batch(x_wm_batch, y_wm_batch, eval_mode=True)
                        self.classifier.loss = ce_loss
                    pbar.set_description(f"Epoch {epoch + 1}/{epochs} Loss: {loss:.4f}, Reg Loss: {reg_loss:.4f}")
                    try:
                        for callback in callbacks:
                            callback.on_batch_end(batch_id)
                    except StopTrainingException:
                        print("StopTrainingException raised!")
                        return
                try:
                    for callback in callbacks:
                        callback.on_epoch_end(epoch)
                except StopTrainingException:
                    print("StopTrainingException raised!")
                    return

    def _filter_candidate_keys(self,
                               train_loader: WRTDataLoader,
                               finetune_batches: int,
                               x_wm: np.ndarray,
                               y_wm: np.ndarray,
                               device: str = "cuda") -> np.ndarray:
        # Filter only successful candidates.
        y_pred: np.ndarray = self.predict(x_wm, batch_size=train_loader.batch_size)
        keep_indices = np.where(np.argmax(y_pred, axis=1) == y_wm)

        if finetune_batches > 0:
            # Train n variants of the prewatermarked source model.
            pbar_granularity = 1000
            total_batches = finetune_batches * self.num_variants

            with tqdm(desc="Fine-Tuning Blackmarks", total=pbar_granularity) as pbar:
                for n_variant in range(self.num_variants):
                    classifier_variant: PyTorchClassifier = copy.deepcopy(self.classifier)
                    classifier_variant.load("blackmarks-classifier-prewatermark")

                    # Fine-tune the classifier on some amount of batches
                    finetuning_batches_current = 0
                    batches_left = True
                    while batches_left:
                        for batch_id, (x_batch, y_batch) in enumerate(train_loader):
                            classifier_variant.fit_batch(x_batch.to(device), y_batch.to(device))
                            finetuning_batches_current += 1

                            progress = (n_variant * finetune_batches) + finetuning_batches_current
                            pbar.n = int(pbar_granularity * progress / total_batches)
                            pbar.refresh()

                            if finetuning_batches_current >= finetune_batches:
                                batches_left = False
                                break

                    # Predict the watermark labels.
                    y_pred: np.ndarray = classifier_variant.predict(x_wm, batch_size=train_loader.batch_size)
                    variant_keep_indices, = np.where(np.argmax(y_pred, axis=1) != y_wm)
                    keep_indices = np.intersect1d(keep_indices, variant_keep_indices)

        x_wm, y_wm = x_wm[keep_indices], y_wm[keep_indices]
        shuffle_idx = np.arange(len(x_wm))
        np.random.shuffle(shuffle_idx)
        x_wm, y_wm = x_wm[shuffle_idx], y_wm[shuffle_idx]

        return x_wm, y_wm

    def _sort_keys_to_encoding(self, x_wm, y_wm):
        # Sort examples into their groups.
        group0 = set(np.arange(self.num_classes)[self.encoding == 0])
        x_wm0, x_wm1, y_wm0, y_wm1 = [], [], [], []
        for x, y in zip(x_wm, y_wm):
            if y in group0:  # Moved from group1->group0 => bit0
                x_wm0.append(x)
                y_wm0.append(y)
            else:  # Moved from group0->group1 => bit1
                x_wm1.append(x)
                y_wm1.append(y)
        print(f"Sampled {len(x_wm0)} elements from wm0 and {len(x_wm1)} from wm1.")
        return (x_wm0, y_wm0), (x_wm1, y_wm1)

    @staticmethod
    def _compose_key(group0: Tuple[List, List],
                     group1: Tuple[List, List],
                     signature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_wm_final, y_wm_final = [], []
        for bit in tqdm(signature, "Keygen2"):
            if int(bit) == 0:
                if len(group0[0]) == 0:
                    print("Ran out of inputs for bit 0.. ")
                    break
                x_wm_final.append(group0[0].pop(0))
                y_wm_final.append(group0[1].pop(0))
            elif int(bit) == 1:
                if len(group1[0]) == 0:
                    print("Ran out of inputs for bit 1.. ")
                    break
                x_wm_final.append(group1[0].pop(0))
                y_wm_final.append(group1[1].pop(0))
            else:
                raise ValueError
        return np.asarray(x_wm_final).squeeze(), np.asarray(y_wm_final).squeeze()

    def embed(self,
              train_loader: WRTDataLoader,
              wm_loader: WRTDataLoader,
              keylength: int,
              key_expansion_factor: int = 1,
              signature: np.ndarray = None,
              epochs: int = 1,
              patience: int = 2,
              min_val: float = 1.0,
              finetune_batches: int = 0,
              decrease_lr_by_factor: float = 1.0,
              log_wm_acc_after_n_batches: int = 50,
              output_dir: str = None,
              callbacks: List[WRTCallback] = None,
              device="cuda",
              **kwargs):
        """
        Train a model on the watermark data. See class documentation for more information on the exact procedure.

        :param train_loader Training data loader
        :param wm_loader Loader for the watermark data. loads unnormalized images.
        :param key_expansion_factor: Number of keys to generate for the embedding relative to the keylength.
        :param keylength: Number of keys to embed into the model.
        :param signature (optional) The secret watermarking message (in bits). If None, one is generated randomly.
        :param epochs Number of epochs to use for trainings.
        :param patience Patience for early stopping on the wm acc.
        :param min_val For early stopping, minimum watermark accuracy
        :param log_wm_acc_after_n_batches: Check early stopping every n batches.
        :param output_dir Output directory to save intermediary results.
        :param finetune_batches Number of epochs for fine-tuning
        :param callbacks List of callbacks to call during the embedding.
        :param device: cuda or cpu
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        :return: Watermark train set, watermark labels
        """
        if callbacks is None:
            callbacks = []

        # Persist the pre-watermarked classifier for later.
        self.classifier.save("blackmarks-classifier-prewatermark")

        # Compute (and visualize) the secret candidate keys.
        self.encoding = self.compute_encoding(data_loader=train_loader)
        x_wm, y_wm, train_signature = self._generate_candidate_keys(wm_loader=wm_loader,
                                                                    train_loader=train_loader,
                                                                    encoding=self.encoding,
                                                                    key_expansion_factor=key_expansion_factor,
                                                                    keylength=keylength)
        self.visualize_key(x_wm, output_dir=output_dir)
        x_wm = train_loader.normalize(x_wm)

        # Early stopping on the watermarking loss.
        callbacks.extend([wrt_callbacks.EarlyStoppingWRTCallback(
            lambda: self.classifier.evaluate(x_wm, y_wm)[0],
            check_every_n_batches=log_wm_acc_after_n_batches,
            patience=patience,
            mode='min'),
            wrt_callbacks.DebugWRTCallback(lambda: self.verify(x_wm, train_signature)[0],
                                           message="wm_acc",
                                           check_every_n_batches=log_wm_acc_after_n_batches)])

        # Embed all candidate keys.
        self.classifier.lr /= decrease_lr_by_factor
        print(self.classifier.lr)
        self._embed_candidate_keys(train_loader, epochs=epochs, train_signature=train_signature, x_wm=x_wm, y_wm=y_wm,
                                   callbacks=callbacks)
        self.classifier.lr *= decrease_lr_by_factor

        # Filter candidates that are transferable or non-successful.
        x_wm, y_wm = self._filter_candidate_keys(train_loader, finetune_batches=finetune_batches, x_wm=x_wm, y_wm=y_wm)
        print(f"Generated {len(x_wm)} keys in total.. Now starting round two of the keygen.")

        # Sort keys into their group.
        group0, group1 = self._sort_keys_to_encoding(x_wm, y_wm)

        # If no signature is provided, we now sample a signature. This will be the part of the watermarking key.
        if signature is None:
            signature = np.random.randint(0, 2, size=keylength)

        # Compose the watermarking key from examples in a group that correspond to the signature's bit.
        x_wm, y_wm = self._compose_key(group0, group1, signature)

        # Save everything to the output dir.
        if output_dir is not None:
            checkpoint = {
                'model': self.get_classifier().model.state_dict(),
                'optimizer': self.get_classifier().optimizer.state_dict(),
                'encoding': self.encoding,
                'x_wm': x_wm,
                'y_wm': signature
            }
            self.save('best.pth', path=output_dir, checkpoint=checkpoint)

        # Save the watermarking keys and the watermarked model.
        if len(x_wm) < len(signature):
            print(f"[WARNING]: Wanted {keylength} keys, but only generated {len(x_wm)}. "
                  f"Consider increasing the key_expansion_factor")

        return x_wm, signature[:len(x_wm)]

    def load(self, filename, path=None, **load_kwargs: dict) -> Tuple[np.ndarray, np.ndarray]:
        """ Loads the watermark data necessary to validate the watermark.
        """
        checkpoint = super().load(filename, path, **load_kwargs)

        self.get_classifier().model.load_state_dict(checkpoint['model'])
        self.get_classifier().optimizer.load_state_dict(checkpoint['optimizer'])

        if 'encoding' in checkpoint.keys():
            self.encoding = checkpoint['encoding']
            print(f"Encoding shape: {self.encoding.shape}")
        else:
            self._try_load_encoding()

        return checkpoint["x_wm"], checkpoint["y_wm"]

    def _get_encoding(self):
        if self.encoding is None:
            self.encoding = self._try_load_encoding()
            if self.encoding is None:
                print("No encoding found!")
                raise ValueError
        return self.encoding

    def extract(self, x, classifier=None, **kwargs):
        """
        Compute the bit error rate between the predicted labels and the owner's signature.
        For consistency with the rest of the interface, return instead the bit accuracy
        as a float between 0 and 1.
        :param x: Watermark key
        :param classifier: Classifier; if provided, extract the watermark from
                           the given classifier instead of this object's classifier
        :param kwargs: keyword arguments passed to classifier.predict()
        :return: int
        """
        if classifier is None:
            classifier = self.classifier

        if self.encoding is None:
            self.encoding = self._try_load_encoding()
            if self.encoding is None:
                print("No encoding found!")
                raise ValueError

        y_pred = classifier.predict(x, **kwargs).argmax(1)
        decoded = self.encoding[y_pred]
        return decoded

    def verify(self,
               x: np.ndarray,
               y: np.ndarray = None,
               classifier: PyTorchClassifier = None,
               **kwargs) -> Tuple[float, bool]:
        """ Verifies whether the given classifier retains the watermark. Returns the watermark
        accuracy and whether it is higher than the decision threshold.

        :param x: The secret watermarking key.
        :param y The expected message.
        :param classifier The classifier to verify.
        :param kwargs: Other parameters for the extraction.
        :return A tuple of the watermark accuracy and whether it is larger than the decision threshold.
        """
        if classifier is None:
            classifier = self.get_classifier()

        classifier.model.eval()
        msg = self.extract(x, classifier=classifier, **kwargs)
        wm_acc = compute_accuracy(msg, y)

        return wm_acc, wm_acc > 0  # ToDo: Implement decision boundary as a second parameter

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
def wm_blackmarks(classifier, **kwargs):
    return Blackmarks(classifier=classifier, **kwargs)


@mlconfig.register
def wm_blackmarks_embed(defense: Blackmarks, wm_loader, config, **kwargs):
    wm_loader = config.wm_dataset()
    if "wm_data_requires_labels" in config.keys():
        if config.wm_data_requires_labels:
            wm_loader = config.wm_dataset(source_model=defense.get_classifier())

    return defense.embed(wm_loader=wm_loader, **kwargs), defense


@mlconfig.register
def wm_blackmarks_keygen(defense: Blackmarks, keylengths: List[int], **kwargs):
    wm_x, wm_y = defense.keygen(**kwargs)
    for n in keylengths:
        yield wm_x[:n], wm_y[:n]
