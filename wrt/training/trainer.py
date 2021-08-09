import json
import os
from abc import ABCMeta, abstractmethod
from typing import Callable, Tuple

import mlconfig
import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from .metrics import Accuracy, Average
from ..classifiers import PyTorchClassifier, StopTrainingException


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


@mlconfig.register
class Trainer(AbstractTrainer):

    def __init__(self, model: PyTorchClassifier, train_loader: data.DataLoader, device: torch.device,
                 num_epochs: int, valid_loader: data.DataLoader = None, output_dir: str = None,
                 scheduler=None, callbacks=None, disable_scheduler=False, save_data: dict = None,
                 train_fn: Callable[[AbstractTrainer], Tuple] = None, disable_progress: bool = False,
                 epsilon: float = 0.0, train_all_features: bool = False):
        self.model = model
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.callbacks = callbacks if callbacks else []
        self.disable_scheduler = disable_scheduler or (scheduler is None)
        self.train_fn = self.train if train_fn is None else train_fn
        self.history = {}
        self.save_data = save_data
        self.disable_progress = disable_progress
        self.epsilon = epsilon
        self.train_all_features = train_all_features

        if self.save_data is None:
            self.save_data = {}

        self.stop_training_exception = False
        self.epoch = 1
        self.best_acc = 0

    def set_label_smoothing(self, epsilon):
        """ Activates label smoothing with a random uniform distribution and a weight factor epsilon.
        Given K classes and some one-hot encoded label y:

        y' = (1-epsilon) * y + (epsilon) * 1/K
        """
        assert 0 <= epsilon <= 1.0, f"Epsilon must be within 0 and 1! '{epsilon}' is not allowed. "
        self.epsilon = epsilon

    def fit(self):
        self.stop_training_exception = False
        for self.epoch in range(1, self.num_epochs + 1):
            print(f"Learning Rate: {self.model.lr}")
            train_loss, train_acc = self.train_fn()

            valid_loss, valid_acc = self.evaluate()
            if not self.disable_scheduler:  # Let callbacks regulate the learning rate.
                print("Scheduler Step")
                try:
                    self.scheduler.step()
                except:
                    # Add support for dynamic schedulers.
                    self.scheduler.step(valid_loss.value)

            if self.output_dir is not None:
                self.save_checkpoint(os.path.join(self.output_dir, 'checkpoint.pth'))
                if valid_acc > self.best_acc:
                    f = os.path.join(self.output_dir, 'best.pth')
                    print(f"## Trainer saving best checkpoint val_acc={self.best_acc} to {f}!")
                    self.best_acc = valid_acc.value
                    self.save_checkpoint(f)

            # Stop training if an exception has been raised.
            if self.stop_training_exception:
                return self.history

            # Execute callbacks at the end of an epoch.
            for callback in self.callbacks:
                try:
                    output = callback.on_epoch_end(self.epoch, train_loss=train_loss, train_acc=train_acc,
                                                   valid_loss=valid_loss, valid_acc=valid_acc)
                except StopTrainingException:
                    return self.history

                if output is not None:
                    self.history.setdefault(str(callback), []).append(output)

            self.history.setdefault("train_acc", []).append(train_acc.value)
            self.history.setdefault("train_loss", []).append(train_loss.value)
            self.history.setdefault("val_acc", []).append(valid_acc.value)
            self.history.setdefault("val_loss", []).append(valid_loss.value)

        return self.history

    def train(self):
        train_loss = Average()
        train_acc = Accuracy()

        self.model.model.train()

        train_loader = tqdm(self.train_loader, desc='Train', disable=self.disable_progress)
        for batch_id, (x, y) in enumerate(train_loader):

            x = x.to(self.device)
            y = y.to(self.device)

            if self.epsilon > 0:
                # Apply label smoothing.
                if len(y.shape) == 1:
                    y = torch.eye(self.model.nb_classes())[y]
                uniform_label = torch.ones_like(y) / self.model.nb_classes()
                y = (1-self.epsilon)*y + self.epsilon * uniform_label
                y = y.to(self.device)

            pre_state = None
            if self.train_all_features and hasattr(self.model.model, 'return_hidden_activations'):
                pre_state = self.model.model.return_hidden_activations
                self.model.model.return_hidden_activations = True
            loss, output = self.model.fit_batch(x, y, all_features=self.train_all_features)
            if pre_state is not None:
                self.model.model.return_hidden_activations = pre_state

            if len(y.shape) > 1:
                y = y.argmax(dim=1)

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(output, y)

            self._update_history(prefix="batch_", train_loss=train_loss.value, train_acc=train_acc.value)
            train_loader.set_postfix_str(
                f'Epoch [{self.epoch}/{self.num_epochs}] train loss: {train_loss}, train acc: {train_acc}.')

            try:
                for callback in self.callbacks:
                    value = callback.on_batch_end(batch_id, train_loss=train_loss.value, train_acc=train_acc.value)
                    if value is not None:
                        data = {
                            str(callback): value
                        }
                        self._update_history(prefix="batch_", **data)
            except StopTrainingException:
                print("StopTrainingException has been raised!")
                self.stop_training_exception = True
                train_loader.close()
                break

        return train_loss, train_acc

    def evaluate(self):
        valid_loss = Average()
        valid_acc = Accuracy()

        self.model.model.eval()

        with torch.no_grad():
            valid_loader = tqdm(self.valid_loader, desc='Validate')
            for x, y in valid_loader:
                x = x.to(self.device)
                output = self.model.predict(x)

                output = torch.from_numpy(output)
                loss = F.cross_entropy(output, y)

                valid_loss.update(loss.item(), number=x.size(0))
                valid_acc.update(output, y)
                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')
        print(f"Valid acc: {valid_acc}")
        return valid_loss, valid_acc

    def _update_history(self, prefix="", **kwargs):
        for key, item in kwargs.items():
            self.history.setdefault(prefix + key, []).append(item)

    def save_checkpoint(self, f):
        checkpoint = {
            'model': self.model.model.state_dict(),
            'optimizer': self.model.optimizer.state_dict(),
            'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_acc': self.best_acc,
            'history': self.history,
            **self.save_data
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
