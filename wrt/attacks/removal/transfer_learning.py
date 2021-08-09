"""
This module implements the Transfer Learning attack
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import mlconfig
import numpy as np

from wrt.attacks.attack import RemovalAttack
from wrt.classifiers import Loss, PyTorchClassifier
from wrt.classifiers.classifier import Classifier
from wrt.training.callbacks import EvaluateWmAccCallback
from wrt.training.trainer import Trainer

logger = logging.getLogger(__name__)


class TransferLearningAttack(RemovalAttack):
    """
    The attack consists of replacing layers in a classifier and fine-tuning
    the new model.
    """

    attack_params = RemovalAttack.attack_params + ["epochs"]

    class CELoss(Loss):
        """
        Cross-entropy loss with soft labels, since PyTorch cross-entropy loss
        requires labels to be argmax-encoded
        """

        def reduce_labels(self):
            return False

        def compute_loss(self, pred, true, x=None):
            import torch
            if true.dim() == 1:
                true = torch.eye(pred.shape[-1])[true].to("cuda")

            logprobs = torch.nn.functional.log_softmax(pred, dim=1)
            return -(true * logprobs).sum() / pred.shape[0]

    def __init__(
            self,
            classifier,
            removed_layers,
            image_size,
            num_classes,
            parent_shape,
            preprocessor=None,
            optimizer=None,
            loss=None,
            freeze=False,
            lr=0.001,
            train_all_params_after_n_batches=300,
            **kwargs
    ):
        """
        Create a :class:`.TransferLearningAttack` instance.

        :param classifier: A trained classifier.
        :param removed_layers: int or list of str; names of the layers in the model to remove if list
                               strs, or the index of the first layer to remove if int
        :param input_shape: tuple; new input shape of the model
        :param num_classes: int; new number of classes
        :param optimizer: if specified, the optimizer to use to fine-tune
        :param loss: if specified, the loss to use
        :param freeze: bool; if True, only train the replaced layers; if False, fine-tune the whole model
        :param epochs: number of epochs to train for
        :param batch_size: batch size
        """
        super(TransferLearningAttack, self).__init__(classifier)

        self.image_size = image_size
        self.num_classes = num_classes
        self.lr = lr
        self.train_all_params_after_n_batches = train_all_params_after_n_batches

        self.classifier = self.__construct_new_classifier(
            model=classifier._model._model,
            removed_layers=removed_layers,
            input_shape=(3, image_size, image_size),
            num_classes=num_classes,
            parent_shape=parent_shape,
            preprocessor=preprocessor,
            optimizer=optimizer,
            loss=loss,
            freeze=freeze
        )

    @classmethod
    def is_valid_classifier_type(cls, classifier):
        """
        Checks whether the classifier provided is a classifier which this class can perform an attack on
        :param classifier:
        :return:
        """
        return True if isinstance(classifier, Classifier) else False

    def __construct_new_classifier(self, model, removed_layers, input_shape, num_classes, parent_shape,
                                   preprocessor=None, optimizer=None, loss=None, freeze=False):
        """
        Construct a new PyTorchClassifier object; see __init__ for parameter meanings
        """
        # TODO: find a way to do this generically

        import torch
        import torch.nn as nn
        import torch.optim as optim

        from wrt.classifiers import PyTorchClassifier

        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()

            def forward(self, x):
                return x

        class TransferModelByLayerName(nn.Module):
            def __init__(self, model, removed_layers, parent_input_size, preprocessor, num_classes):
                super().__init__()
                for layer_name in removed_layers:
                    setattr(model, layer_name, Identity())

                y = model(torch.randn(2, *input_shape).to(device))
                if isinstance(y, list):
                    y = y[-1]
                y = y.detach().clone().cpu().numpy()
                output_size = np.prod(y.shape) // 2

                self.output_size = output_size
                self.preprocessor = preprocessor
                self.upsample = nn.Upsample(size=parent_input_size, mode='bilinear', align_corners=False)
                self.model = model
                self.fc = nn.Linear(output_size, num_classes)

            def forward(self, x):
                x = self.upsample(x)
                if self.preprocessor is not None:
                    x = self.preprocessor(x)
                model_outputs = self.model(x)
                if isinstance(model_outputs, list):
                    x = self.fc(model_outputs[-1].view(-1, self.output_size))
                    return model_outputs + [x]
                else:
                    x = self.fc(model_outputs.view(-1, self.output_size))
                    return x

        class TransferModelByLayerIndex(nn.Module):
            def __init__(self, model, removed_layers, parent_input_size, preprocessor, num_classes):
                super().__init__()

                y = model(torch.randn(1, *input_shape).to(device))
                assert isinstance(y, list)
                y = y[removed_layers - 1].detach().clone().cpu().numpy()
                output_size = np.prod(y.shape)

                self.output_size = output_size
                self.preprocessor = preprocessor
                self.upsample = nn.Upsample(size=parent_input_size, mode='bilinear')
                self.model = model
                self.fc = nn.Linear(output_size, num_classes)
                self.removed_layer_index = removed_layers

            def forward(self, x):
                if self.preprocessor is not None:
                    x = self.preprocessor(x)
                x = self.upsample(x)
                model_outputs = self.model(x)
                x = self.fc(model_outputs[self.removed_layer_index - 1].view(-1, self.output_size))
                return model_outputs + [x]

        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            device = torch.device("cuda:{}".format(cuda_idx))
        model = model.to(device)

        if freeze:
            print("Freezing all layers ...")
            for params in model.parameters():
                params.requires_grad = False
            model.eval()

        if isinstance(removed_layers, list):
            new_model = TransferModelByLayerName(model, removed_layers, parent_shape, preprocessor, num_classes)
        elif isinstance(removed_layers, int):
            new_model = TransferModelByLayerIndex(model, removed_layers, parent_shape, preprocessor, num_classes)
        else:
            raise ValueError("Error: removed_layers must be int or list of strs")

        loss = loss if loss is not None else nn.CrossEntropyLoss()
        if optimizer is not None:
            print("Found existing optimizer, putting last layer's parameters. ")
            optimizer.add_param_group({'params': new_model.fc.parameters()})
        else:
            print("[WARNING] No optimizer provided. Creating a new one that is attached to the model. ")
            optimizer = optim.SGD(new_model.parameters(), lr=self.lr, momentum=0.9)

        return PyTorchClassifier(
            new_model,
            loss=loss,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=num_classes,
            clip_values=(0, 1)
        )

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
               scheduler=None,
               valid_loader=None,
               output_dir=None,
               device="cuda",
               wm_data=None,
               **kwargs):
        """Attempt to remove the watermark
        :param train_loader The loader for the training data.
        :param epochs Number of epochs to train for.
        :param batch_size
        :param scheduler
        :param output_dir
        :param valid_loader
        :param wm_data
        :param device:
        """
        # Change to loss over soft labels.
        self.classifier.loss = TransferLearningAttack.CELoss(self.classifier)
        callbacks = [self.CustomCallback(self.get_classifier().model,
                                         train_all_params_after_n_batches=self.train_all_params_after_n_batches)]

        if wm_data:
            print("Found wm data! Adding callback")
            callbacks.append(EvaluateWmAccCallback(self.classifier, wm_data))

        trainer = Trainer(model=self.get_classifier(), train_loader=train_loader, valid_loader=valid_loader,
                          scheduler=scheduler, device=device, num_epochs=epochs, callbacks=callbacks)
        trainer.evaluate()
        return trainer.fit()


@mlconfig.register
def transfer_learning_attack(classifier: PyTorchClassifier, layer_bounds, layer_lrs, config, preprocessor, **kwargs):
    previous_layer_bound = 0
    if len(layer_bounds) > 0:
        params = []
        for next_layer_bound, layer_lr in zip(layer_bounds, layer_lrs):
            params.append({
                'params': list(classifier._model.parameters())[previous_layer_bound:next_layer_bound], 'lr': layer_lr
            })
            previous_layer_bound = next_layer_bound
        optimizer = config.optimizer(params)
    else:
        optimizer = config.optimizer(classifier.model.parameters())

    # ToDo: Make the preprocessor part of the config.
    if preprocessor == "imagenet":
        print("[WARNING] Hardcoded Preprocessor only valid for ImageNet.")
        import torch
        std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).astype(np.float32).reshape((1, 3, 1, 1))).to("cuda")
        preprocessor = lambda x: x * 255 * std
    elif preprocessor == "cifar":
        import torch
        std = torch.from_numpy(np.array([0.247, 0.243, 0.261]).astype(np.float32).reshape((1, 3, 1, 1))).to("cuda")
        preprocessor = lambda x: x * 255 * std
    else:
        preprocessor = None

    # Compile all arguments.
    full_args = {
        'optimizer': optimizer,
        'preprocessor': preprocessor,
        **kwargs
    }
    return TransferLearningAttack(classifier, **full_args)


@mlconfig.register
def transfer_learning_removal(attack: TransferLearningAttack,
                              train_loader,
                              valid_loader,
                              config,
                              output_dir=None,
                              **kwargs):
    optimizer = attack.get_classifier().optimizer
    return attack, attack.remove(train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 scheduler=config.scheduler(optimizer),
                                 output_dir=output_dir,
                                 **kwargs)

@mlconfig.register
def transfer_learning_transfer_set_removal(attack: TransferLearningAttack,
                              source_model,
                              train_loader,  # Replace the train_loader with the transfer set.
                              valid_loader,
                              config,
                              output_dir=None,
                              **kwargs):
    transfer_dataset = config.transfer_dataset(source_model=source_model)
    optimizer = attack.get_classifier().optimizer
    return attack, attack.remove(train_loader=transfer_dataset,
                                 valid_loader=valid_loader,
                                 scheduler=config.scheduler(optimizer),
                                 output_dir=output_dir,
                                 **kwargs)

