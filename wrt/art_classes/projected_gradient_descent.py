# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Projected Gradient Descent attack `ProjectedGradientDescent` as an iterative method in which,
after each iteration, the perturbation is projected on an lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.
| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.stats import truncnorm

from wrt.config import WRT_NUMPY_DTYPE
from wrt.classifiers.classifier import Classifier
from wrt.art_classes import EvasionAttack, FastGradientMethod
from wrt.training.utils import compute_success, get_labels_np_array, check_and_transform_label_format, random_sphere

logger = logging.getLogger(__name__)


class ProjectedGradientDescent(EvasionAttack):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.
    | Paper link: https://arxiv.org/abs/1706.06083
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
        "max_iter",
        "random_eps",
    ]

    def __init__(
        self,
        classifier,
        norm: int = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
    ):
        """
        Create a :class:`.ProjectedGradientDescent` instance.
        :param classifier: An trained estimator.
        :param norm: The norm of the adversarial perturbation supporting np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step
                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this
                           method with PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        super(ProjectedGradientDescent, self).__init__(classifier)

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.random_eps = random_eps

        ProjectedGradientDescent._check_params(self)

        self._attack = ProjectedGradientDescentPyTorch(
            classifier=classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
        )

    def generate(self, x, y=None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :return: An array holding the adversarial examples.
        """
        logger.info("Creating adversarial samples.")
        return self._attack.generate(x=x, y=y, **kwargs)

    def set_params(self, **kwargs) -> None:
        super().set_params(**kwargs)
        self._attack.set_params(**kwargs)

    def _check_params(self) -> None:
        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

        if self.eps <= 0:
            raise ValueError("The perturbation size `eps` has to be positive.")

        if self.eps_step <= 0:
            raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, (int, np.int)):
            raise TypeError("The number of random initialisations has to be of type integer")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if self.norm == np.inf and self.eps_step > self.eps:
            raise ValueError("The iteration step `eps_step` has to be smaller than the total attack `eps`.")

        if self.max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` has to be a positive integer.")


class ProjectedGradientDescentCommon(FastGradientMethod):
    """
    Common class for different variations of implementation of the Projected Gradient Descent attack. The attack is an
    iterative method in which, after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted data range). This is the
    attack proposed by Madry et al. for adversarial training.
    | Paper link: https://arxiv.org/abs/1706.06083
    """

    attack_params = FastGradientMethod.attack_params + ["max_iter", "random_eps"]

    def __init__(
        self,
        classifier: Classifier,
        norm: int = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
    ) -> None:
        """
        Create a :class:`.ProjectedGradientDescentCommon` instance.
        :param estimator: A trained classifier.
        :param norm: The norm of the adversarial perturbation supporting np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
            suggests this for FGSM based training to generalize across different epsilons. eps_step is
            modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
            is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        super(ProjectedGradientDescentCommon, self).__init__(
            classifier=classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            minimal=False,
        )
        self.max_iter = max_iter
        self.random_eps = random_eps
        ProjectedGradientDescentCommon._check_params(self)

        if self.random_eps:
            lower, upper = 0, eps
            mu, sigma = 0, (eps / 2)
            self.norm_dist = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    def _random_eps(self):
        """
        Check whether random eps is enabled, then scale eps and eps_step appropriately.
        """
        if self.random_eps:
            ratio = self.eps_step / self.eps
            self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            self.eps_step = ratio * self.eps

    def _set_targets(self, x, y, classifier_mixin=True):
        """
        Check and set up targets.
        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :type classifier_mixin: `bool`
        :return: The targets.
        :rtype: `np.ndarray`
        """
        if classifier_mixin:
            y = check_and_transform_label_format(y, self.classifier.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            if classifier_mixin:
                targets = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))
            else:
                targets = self.classifier.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets

    @staticmethod
    def _get_mask(x, classifier_mixin=True, **kwargs):
        """
        Get the mask from the kwargs.
        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :type classifier_mixin: `bool`
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: The mask.
        :rtype: `np.ndarray`
        """
        mask = kwargs.get("mask")

        if mask is not None:
            if classifier_mixin:
                # Ensure the mask is broadcastable
                if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape) :]:
                    raise ValueError("Mask shape must be broadcastable to input shape.")

            else:
                raise ValueError("Mask is only supported for classification.")

        return mask

    def _check_params(self) -> None:
        super(ProjectedGradientDescentCommon, self)._check_params()

        if self.eps_step > self.eps:
            raise ValueError("The iteration step `eps_step` has to be smaller than the total attack `eps`.")

        if self.max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` has to be a positive integer.")


class ProjectedGradientDescentPyTorch(ProjectedGradientDescentCommon):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.
    | Paper link: https://arxiv.org/abs/1706.06083
    """

    def __init__(
        self,
        classifier,
        norm: int = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
    ):
        """
        Create a :class:`.ProjectedGradientDescentPytorch` instance.
        :param classifier: An trained estimator.
        :type classifier: :class:`.BaseEstimator`
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step is
                           modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
                           is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        if (
            hasattr(classifier, "preprocessing")
            and (classifier.preprocessing is not None and classifier.preprocessing != (0, 1))
        ) or (
            hasattr(classifier, "preprocessing_defences")
            and (classifier.preprocessing_defences is not None and classifier.preprocessing_defences != [])
        ):
            raise NotImplementedError(
                "The framework-specific implementation currently does not apply preprocessing and "
                "preprocessing defences."
            )

        super(ProjectedGradientDescentPyTorch, self).__init__(
            classifier=classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
        )

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        # Check whether random eps is enabled
        self._random_eps()

        # Set up targets
        targets = self._set_targets(x, y)

        # Get the mask
        mask = self._get_mask(x, **kwargs)

        # Create dataset
        if mask is not None:
            # Here we need to make a distinction: if the masks are different for each input, we need to index
            # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
            if len(mask.shape) == len(x.shape):
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(WRT_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(WRT_NUMPY_DTYPE)),
                    torch.from_numpy(mask.astype(WRT_NUMPY_DTYPE)),
                )

            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(WRT_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(WRT_NUMPY_DTYPE)),
                    torch.from_numpy(np.array([mask.astype(WRT_NUMPY_DTYPE)] * x.shape[0])),
                )

        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(x.astype(WRT_NUMPY_DTYPE)), torch.from_numpy(targets.astype(WRT_NUMPY_DTYPE)),
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples
        adv_x_best = None
        rate_best = None

        for _ in range(max(1, self.num_random_init)):
            adv_x = x.astype(WRT_NUMPY_DTYPE)

            # Compute perturbation with batching
            for (batch_id, batch_all) in enumerate(data_loader):
                if mask is not None:
                    (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
                else:
                    (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                adv_x[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels, mask_batch)

            if self.num_random_init > 1:
                rate = 100 * compute_success(
                    self.classifier, x, targets, adv_x, self.targeted, batch_size=self.batch_size
                )
                if rate_best is None or rate > rate_best or adv_x_best is None:
                    rate_best = rate
                    adv_x_best = adv_x
            else:
                adv_x_best = adv_x

        logger.info(
            "Success rate of attack: %.2f%%",
            rate_best
            if rate_best is not None
            else 100 * compute_success(self.classifier, x, y, adv_x_best, self.targeted, batch_size=self.batch_size),
        )

        return adv_x_best

    def _generate_batch(self, x: "torch.Tensor", targets: "torch.Tensor", mask: "torch.Tensor") -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.
        :param x: An array with the original inputs.
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :return: Adversarial examples.
        """
        inputs = x.to(self.classifier.device)
        targets = targets.to(self.classifier.device)
        adv_x = inputs

        if mask is not None:
            mask = mask.to(self.classifier.device)

        for i_max_iter in range(self.max_iter):
            adv_x = self._compute_torch(
                adv_x, inputs, targets, mask, self.eps, self.eps_step, self.num_random_init > 0 and i_max_iter == 0,
            )

        return adv_x.cpu().detach().numpy()

    def _compute_perturbation(self, x: "torch.Tensor", y: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        """
        Compute perturbations.
        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :return: Perturbations.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.classifier.loss_gradient_framework(x, y) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == np.inf:
            grad = grad.sign()

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol)  # type: ignore

        assert x.shape == grad.shape

        if mask is None:
            return grad
        else:
            return grad * mask

    def _apply_perturbation(self, x: "torch.Tensor", perturbation: "torch.Tensor", eps_step: float) -> "torch.Tensor":
        """
        Apply perturbation on examples.
        :param x: Current adversarial examples.
        :param perturbation: Current perturbations.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        x = x + eps_step * perturbation

        if self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
            x = torch.max(
                torch.min(x, torch.tensor(clip_max).to(self.classifier.device)),
                torch.tensor(clip_min).to(self.classifier.device),
            )

        return x

    def _compute_torch(
        self,
        x: "torch.Tensor",
        x_init: "torch.Tensor",
        y: "torch.Tensor",
        mask: "torch.Tensor",
        eps: float,
        eps_step: float,
        random_init: bool,
    ) -> "torch.Tensor":
        """
        Compute adversarial examples for one iteration.
        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236).
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])

            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(WRT_NUMPY_DTYPE)
            random_perturbation = torch.from_numpy(random_perturbation).to(self.classifier.device)

            if mask is not None:
                random_perturbation = random_perturbation * mask

            x_adv = x + random_perturbation

            if self.classifier.clip_values is not None:
                clip_min, clip_max = self.classifier.clip_values
                x_adv = torch.max(
                    torch.min(x_adv, torch.tensor(clip_max).to(self.classifier.device)),
                    torch.tensor(clip_min).to(self.classifier.device),
                )

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation(x_adv, y, mask)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation(x_adv, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)

        # Recompute x_adv
        x_adv = perturbation + x_init

        return x_adv

    def _projection(self, values: "torch.Tensor", eps: float, norm_p: int) -> "torch.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.
        :param values: Values to clip.
        :param eps: Maximum norm allowed.
        :param norm_p: L_p norm to use for clipping supporting 1, 2 and `np.Inf`.
        :return: Values of `values` after projection.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)

        if norm_p == 2:
            values_tmp = values_tmp * torch.min(
                torch.tensor([1.0], dtype=torch.float32).to(self.classifier.device),
                eps / (torch.norm(values_tmp, p=2, dim=1) + tol),
            ).unsqueeze_(-1)

        elif norm_p == 1:
            values_tmp = values_tmp * torch.min(
                torch.tensor([1.0], dtype=torch.float32).to(self.classifier.device),
                eps / (torch.norm(values_tmp, p=1, dim=1) + tol),
            ).unsqueeze_(-1)

        elif norm_p == np.inf:
            values_tmp = values_tmp.sign() * torch.min(
                values_tmp.abs(), torch.tensor([eps], dtype=torch.float32).to(self.classifier.device)
            )

        else:
            raise NotImplementedError(
                "Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported."
            )

        values = values_tmp.reshape(values.shape)

        return values
