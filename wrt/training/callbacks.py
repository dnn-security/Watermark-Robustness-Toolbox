import abc
from typing import Callable, Union

import numpy as np

from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import StopTrainingException, PyTorchClassifier
from wrt.training.utils import compute_accuracy


class WRTCallback:

    @abc.abstractmethod
    def on_batch_end(self, batch_id, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def on_epoch_end(self, epoch, **kwargs):
        raise NotImplementedError


class StepCallback(WRTCallback):
    """ Callback that steps the epoch for the fixed learning rate scheduler
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_batch_end(self, b, **kwargs):
        pass

    def on_epoch_end(self, e, **kwargs):
        self.scheduler.step()


class DebugWRTCallback(WRTCallback):
    """ Calls a function during training and prints its output (if any)
    """
    def __init__(self,
                 debug_fn: Callable[[], Union[None, float]],
                 message: str = None,
                 check_every_n_batches: int = None):
        self.debug_fn = debug_fn
        self.check_every_n_batches = check_every_n_batches
        self.message = message

        self.batch_counter = 0

    def on_batch_end(self, batch_id, **kwargs):
        if self.check_every_n_batches is not None:
            self.batch_counter += 1
            if (self.batch_counter % self.check_every_n_batches) == (self.check_every_n_batches-1):
                value = self.debug_fn()
                if self.message is not None:
                    print(f"[Debug] {self.message}: {value:.4f}")
                return value

    def on_epoch_end(self, epoch, **kwargs):
        if self.check_every_n_batches is None:
            value = self.debug_fn()
            if self.message is not None:
                print(f"[Debug] {self.message}: {value:.4f}")
            return value


class EarlyStoppingWRTCallback(WRTCallback):
    """ Early Stopping callback with a callable function to measure some metric. """
    def __init__(self,
                 measure_value_fn: Callable[[], float],
                 patience: int = 1,
                 min_delta: float = 0.0,
                 mode: str = 'max',
                 check_every_n_batches: int = None,
                 verbose: bool = True):
        self.measure_value_fn = measure_value_fn
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.check_every_n_batches = check_every_n_batches
        self.verbose = verbose

        self.current_patience = 0
        assert mode in ['min', 'max']
        if mode == 'max':
            self.best_metric = 0
        else:
            self.best_metric = np.inf

    def _is_improvement(self, metric):
        if self.mode == 'max':
            return metric > (self.best_metric + self.min_delta)
        else:
            return (metric + self.min_delta) < self.best_metric

    def _apply_step(self):
        metric = self.measure_value_fn()

        # Check for improvement.
        if self._is_improvement(metric):
            self.best_metric = metric
            self.current_patience = 0
        else:
            self.current_patience += 1

        if self.verbose:
            print(f"\n[Early Stopping] Metric (current, best): {metric:.4f}/{self.best_metric:.4f} Patience {self.current_patience}/{self.patience}")

        # Raise Exception to notify training stop.
        if self.current_patience >= self.patience:
            raise StopTrainingException

    def on_batch_end(self, batch_id, **kwargs):
        if self.check_every_n_batches is not None:
            if (batch_id % self.check_every_n_batches) == self.check_every_n_batches-1:
                self._apply_step()

    def on_epoch_end(self, epoch, **kwargs):
        if self.check_every_n_batches is None:
            self._apply_step()


class EarlyStoppingCallback(WRTCallback):
    def __init__(self,
                 metric: str,
                 log_after_n_batches: int = None,
                 patience: int = 1,
                 better="larger",
                 verbose=True):
        """ Callback for early stopping
        """
        self.metric = metric
        self.patience = patience
        self.log_after_n_batches = log_after_n_batches
        self.better = better

        assert better in ["larger", "smaller"], "Parameter 'better' must be either 'smaller' or 'larger'"

        self.patience_counter = 0
        self.best_val = np.inf
        self.verbose = verbose

    def _apply(self, val):
        if self.better == "larger":
            if val >= self.best_val:
                self.best_val = val
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        if self.better == "smaller":
            if val <= self.best_val:
                self.best_val = val
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        if self.verbose:
            print(f"{self.metric}: current: {val}, best: {self.best_val}, patience: {self.patience_counter}/{self.patience}")
        if self.patience_counter >= self.patience:
            raise StopTrainingException

    def on_batch_end(self, batch, **kwargs):
        if self.log_after_n_batches is not None and ((batch % self.log_after_n_batches) == 0):
            if self.metric in kwargs.keys():
                val = kwargs[self.metric]
                return self._apply(val)

    def on_epoch_end(self, epoch, **kwargs):
        if self.log_after_n_batches is None:
            if self.metric in kwargs.keys():
                val = kwargs[self.metric]
                return self._apply(val)


class EarlyStoppingWmCallback:
    def __init__(self, classifier: PyTorchClassifier,
                 wm_data,
                 patience: int = 1,
                 min_val: float = 1.0,
                 log_after_n_batches: int = None,
                 verbose=True):
        self.classifier = classifier
        self.defense_instance, self.x_wm, self.y_wm = wm_data
        self.patience = patience
        self.min_val = min_val
        self.log_after_n_batches = log_after_n_batches
        self.verbose = verbose

        self.patience_counter = 0

    def _apply(self):
        wm_acc = self.defense_instance.verify(self.x_wm, self.y_wm, classifier=self.classifier)[0]
        # wm_acc = compute_accuracy(msg, self.y_wm)
        if wm_acc >= self.min_val:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
        if self.verbose:
            print(f"wm acc: {wm_acc}/{self.min_val}, patience: {self.patience_counter}/{self.patience}")
        if self.patience_counter >= self.patience:
            raise StopTrainingException

    def on_batch_end(self, batch, **kwargs):
        if self.log_after_n_batches is not None and ((batch % self.log_after_n_batches) == 0):
            return self._apply()

    def on_epoch_end(self, epoch, **kwargs):
        if self.log_after_n_batches is None:
            return self._apply()
        else:
            return self.defense_instance.verify(self.x_wm, self.y_wm, classifier=self.classifier)[0]


class EvaluateWmAccCallback:
    def __init__(self, classifier: PyTorchClassifier,
                 wm_data,
                 log_after_n_batches: int = None):
        """ Evaluates the watermark accuracy after n_batches and at the end of every epoch
        @param classifier The Model for which to measure the watermark accuracy
        @:param wm_data The watermarking data.
        @:param log_after_n_batches Invokes watermark accuracy evaluation every n batches.
        """
        self.defense_instance, self.x_wm, self.y_wm = wm_data
        self.classifier = classifier
        self.log_after_n_batches = log_after_n_batches

    def eval_wm_acc(self):
        return self.defense_instance.verify(self.x_wm, self.y_wm, classifier=self.classifier)[0]

    def on_batch_end(self, b, **kwargs):
        if self.log_after_n_batches is not None and (b % self.log_after_n_batches) == 0:
            wm_acc = self.eval_wm_acc()
            print(f"Batch {b+1}: Wm Acc: {wm_acc * 100:.4f}%")

    def on_epoch_end(self, e, **kwargs):
        if self.log_after_n_batches is None:
            wm_acc = self.eval_wm_acc()
            print(f"Epoch {e}: Wm Acc: {wm_acc * 100:.4f}%")
            return wm_acc

    def __str__(self):
        return "wm_acc"


class EvaluateAccCallback:
    def __init__(self, classifier, val_data, log_after_n_batches=600, logger=None):
        self.classifier = classifier
        self.val_data = val_data
        self.logger = logger
        self.log_after_n_batches = log_after_n_batches

    def get_test_acc(self):
        if type(self.val_data) == tuple:
            predictions = self.classifier.predict(self.val_data[0])
            metrics = compute_accuracy(predictions, self.val_data[1])[0]
        else:
            metrics = evaluate_test_accuracy(self.classifier, self.val_data, batch_size=32, limit_batches=50)
        return metrics

    def on_batch_end(self, b, **kwargs):
        if ((b % (self.log_after_n_batches+1)) == self.log_after_n_batches) and self.logger:
            metrics = self.get_test_acc()
            self.logger.info(f"Batch {b+1}: Wm Acc: {metrics * 100:.4f}%")

    def on_epoch_end(self, e, **kwargs):
        if self.logger:
            metrics = self.get_test_acc()
            self.logger.info(f"Epoch {e}: Test Acc: {metrics * 100:.4f}%")


class ReduceStepAndEarlyStopCallback:
    """
    Callback that passes on a loss value to a dynamic learning rate scheduler,
    and stops training once the learning rate plateaus x times
    """

    def __init__(self, scheduler, stop_lr=1e-5):
        self.scheduler = scheduler
        self.stop_lr = stop_lr

        print("Added ReduceStepAndEarlyStopCallback!")

    def on_batch_end(self, b, **kwargs):
        pass

    def on_epoch_end(self, e, loss=None, lr=None, **kwargs):
        self.scheduler.step(loss)
        print("stepping scheduler...")
        if abs(lr - self.stop_lr) < 1e-8:
            raise StopTrainingException
