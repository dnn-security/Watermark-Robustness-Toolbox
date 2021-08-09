# MIT License
#
# Copyright (C) IBM Corporation 2018
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
This module implements the Fine-Tuning attack.

| Paper link: -
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc

import mlconfig
from torch.nn import Conv2d

from wrt.attacks.attack import RemovalAttack
from wrt.classifiers import Loss
from wrt.training.callbacks import EvaluateWmAccCallback, DebugWRTCallback
from wrt.attacks.util import evaluate_test_accuracy
from wrt.training.models.torch.classifier.resnet import Sequential
from wrt.training.trainer import Trainer


# original FineTuningAttack moved to FTALAttack
class FineTuningAttack(RemovalAttack):
    """
    The attack consists of fine-tuning a watermarked classifier on more target data.
    (Examplary Implementation)
    """

    class CELoss(Loss):
        """
        Cross-entropy loss with support for soft labels, since PyTorch cross-entropy loss
        requires labels to be argmax-encoded
        """

        def reduce_labels(self):
            return False

        def compute_loss(self, pred, true, x=None):
            import torch

            if len(true.shape) == 1:
                return torch.nn.functional.cross_entropy(pred, true)

            logprobs = torch.nn.functional.log_softmax(pred, dim=1)
            return -(true * logprobs).sum() / pred.shape[0]

    def __init__(
            self,
            classifier,
            num_classes,
            lr=0.001,
    ):
        """
        Create a :class:`.RemovalAttack` instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param lr: Learning rate for the fine-tuning process
        :type lr: `float`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(FineTuningAttack, self).__init__(classifier)

        self.lr = lr
        self.num_classes = num_classes

    @abc.abstractmethod
    def prepare_classifier(self, classifier, train_all_params_after_n_batches=None, target_layer=-1):
        """
        Modify the classifier for fine-tuning
        :param classifier: Classifier
        :return: Classifier
        """
        pass

    class CustomCallback:
        def __init__(self, model, train_all_params_after_n_batches=300):
            self.model = model
            self.train_all_params_after_n_batches = train_all_params_after_n_batches
            self.epoch = 1
            self.batch_no = 0

        def on_batch_end(self, b, **kwargs):
            if self.batch_no == self.train_all_params_after_n_batches:
                for params in self.model.parameters():
                    params.requires_grad = True
                print("Activating gradients for all parameters!")
            self.batch_no += 1

        def on_epoch_end(self, e, **kwargs):
            self.epoch += 1

    def remove(self,
               train_loader,
               epochs: int = 5,
               lr: float = None,
               train_all_params_after_n_batches: int = None,
               scheduler=None,
               valid_loader=None,
               output_dir=None,
               device="cuda",
               target_layer=-1,
               epsilon: float = 0.0,
               check_every_n_batches: int = None,
               wm_data=None,
               callbacks=None,
               **kwargs):
        """Attempt to remove the watermark
        :param train_loader The loader for the training data.
        :param epochs Number of epochs to train for.
        :param batch_size
        :param scheduler
        :param output_dir
        :param valid_loader
        :param wm_data
        :param callbacks
        :param device:
        """
        if callbacks is None:
            callbacks = []

        # Apply function to the classifier (such as resetting this layer's weights)
        self.prepare_classifier(self.get_classifier(),
                                train_all_params_after_n_batches=train_all_params_after_n_batches,
                                target_layer=target_layer)

        # Change to loss over soft labels.
        self.classifier.loss = FineTuningAttack.CELoss(self.classifier)

        if wm_data:
            print("Found wm data! Adding callback")
            callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data, log_after_n_batches=check_every_n_batches))

        callbacks.append(DebugWRTCallback(lambda: evaluate_test_accuracy(self.get_classifier(), valid_loader,
                                                                         limit_batches=50, verbose=False),
                                          message="test acc",
                                          check_every_n_batches=check_every_n_batches))

        if train_all_params_after_n_batches is not None:
            callbacks.append(FineTuningAttack.CustomCallback(self.classifier.model, train_all_params_after_n_batches))

        initial_lr = self.classifier.lr
        if lr is not None:
            self.classifier.lr = lr

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          scheduler=scheduler, device=device, num_epochs=epochs, epsilon=epsilon,
                          output_dir=output_dir, callbacks=callbacks)
        trainer.evaluate()
        history = trainer.fit()

        self.classifier.lr = initial_lr
        return history


class FTLLAttack(FineTuningAttack):

    def prepare_classifier(self, classifier, **kwargs):
        """
        Modify the classifier for fine-tuning
        :param classifier: Classifier
        :return: Classifier
        """
        layers = list(classifier.model.children())
        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                for param in layer.parameters():
                    param.requires_grad = False


class FTALAttack(FineTuningAttack):
    def prepare_classifier(self, classifier, **kwargs):
        """
        Modify the classifier for fine-tuning
        :param classifier: Classifier
        :return: Classifier
        """
        pass


class RTLLAttack(FineTuningAttack):

    def prepare_classifier(self, classifier, **kwargs):
        """
        Modify the classifier for fine-tuning
        :param classifier: Classifier
        :return: Classifier
        """
        layers = list(classifier.model.children())
        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                layer.reset_parameters()


class RTALAttack(FineTuningAttack):

    def prepare_classifier(self, classifier, target_layer=-1, train_all_params_after_n_batches=None, **kwargs):
        """
        Modify the classifier for fine-tuning
        :param classifier: Classifier
        :return: Classifier
        """
        if train_all_params_after_n_batches is not None:
            layers = list(classifier.model.children())
            for i, layer in enumerate(layers):
                if i < len(layers) - 1:
                    for param in layer.parameters():
                        param.requires_grad = False

        def get_last_reset_layer():
            last_layer = list(classifier.model.children())[target_layer]
            print(f"last layer is {type(last_layer)}")
            if type(last_layer) is Sequential:
                for l1 in reversed(list(last_layer.children())):
                    for l2 in reversed(list(l1.children())):
                        if hasattr(l2, 'reset_parameters'):
                            return l2
            return last_layer
        last_layer = get_last_reset_layer()

        print(f"Resetting layer '{target_layer}' of the classifier called '{last_layer}'!")
        last_layer.reset_parameters()


#################### Configuration functions callable through mlconfig

@mlconfig.register
def ftll_attack(config,
                **kwargs):
    return FTLLAttack(**kwargs)


@mlconfig.register
def ftll_removal(attack: FTLLAttack,
                 config,
                 **kwargs):
    scheduler = None
    if "scheduler" in config.keys():
        scheduler = config.scheduler(attack.get_classifier().optimizer)
    return attack, attack.remove(scheduler=scheduler, **kwargs)


@mlconfig.register
def ftal_attack(config,
                **kwargs):
    return FTALAttack(**kwargs)


@mlconfig.register
def ftal_removal(attack: FTALAttack,
                 config,
                 **kwargs):
    scheduler = None
    if "scheduler" in config.keys():
        scheduler = config.scheduler(attack.get_classifier().optimizer)
    return attack, attack.remove(scheduler=scheduler, **kwargs)


@mlconfig.register
def rtal_attack(config,
                **kwargs):
    return RTALAttack(**kwargs)


@mlconfig.register
def rtal_removal(attack: RTALAttack,
                 config,
                 layer_bounds=[],
                 layer_lrs=[],
                 **kwargs):
    classifier = attack.get_classifier()
    previous_layer_bound = 0
    if len(layer_bounds) > 0:
        params = []
        for next_layer_bound, layer_lr in zip(layer_bounds, layer_lrs):
            p = list(classifier.model.parameters())[previous_layer_bound:next_layer_bound]
            params.append({
                'params': p,
                'lr': layer_lr
            })
            print(f"Setting {len(p)} to lr {layer_lr}")
            previous_layer_bound = next_layer_bound
        optimizer = config.optimizer(params)
    else:
        optimizer = config.optimizer(classifier.model.parameters())
    attack.get_classifier()._optimizer = optimizer

    scheduler = None
    if "scheduler" in config.keys():
        scheduler = config.scheduler(attack.get_classifier().optimizer)

    return attack, attack.remove(scheduler=scheduler, **kwargs)


@mlconfig.register
def rtll_attack(config,
                **kwargs):
    return RTLLAttack(**kwargs)


@mlconfig.register
def rtll_removal(attack: RTLLAttack,
                 config,
                 **kwargs):
    scheduler = None
    if "scheduler" in config.keys():
        scheduler = config.scheduler(attack.get_classifier().optimizer)
    return attack, attack.remove(scheduler=scheduler, **kwargs)
