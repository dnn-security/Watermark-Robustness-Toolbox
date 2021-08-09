import numpy as np
from tqdm import tqdm

from wrt.classifiers import PyTorchClassifier
from wrt.training.datasets.wrt_data_loader import WRTDataLoader


def evaluate_test_accuracy(classifier: PyTorchClassifier,
                           val_data: WRTDataLoader,
                           batch_size: int = 64,
                           limit_batches: int = np.inf,
                           device="cuda",
                           verbose: bool = True):
    """ Evaluates the test accuracy of a classifier on some validation dataset
    :param classifier The classifier whose test accuracy should be measured
    :param val_data The validation dataset loader
    :param batch_size The batch size for the prediction
    :param limit_batches Whether to only predict a subset of batches
    :param verbose Whether to show a progress bar
    """
    classifier.model.eval()

    corrects, counter = 0, 0
    val_loop = tqdm(val_data, disable=not verbose, total=min(limit_batches, len(val_data)))

    for batch_id, (x, y) in enumerate(val_loop):
        x = x.to(device)
        if batch_id >= limit_batches:
            break

        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        outputs: np.ndarray = classifier.predict(x, batch_size=batch_size)

        corrects += len(np.where(np.argmax(outputs, axis=1) == y.cpu().numpy())[0])
        counter += len(outputs)
        val_loop.set_description('Validation ({:.4f})'.format(corrects/counter))
    return corrects / counter


def soft_argmax(y, axis):
    """ Argmax that also allows with hard labels.
    """
    if len(y.shape) > 1:
        return np.argmax(y, axis)
    return y


def to_onehot(y, n_classes):
    """ Convert hard to soft labels if necessary.
    """
    if np.isscalar(y[0]):
        return np.eye(n_classes)[y].astype(np.float64)
    return y
