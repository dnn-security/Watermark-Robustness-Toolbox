""" This script trains null models given a configuration file (see configs) """

import argparse

import mlconfig
import torch

import os

from tqdm import tqdm

from wrt.utils import reserve_gpu

import numpy as np

# Registers all hooks. Do not remove.
from wrt.classifiers import PyTorchClassifier
from wrt.defenses import Watermark
from wrt.training.utils import compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wm_config', type=str, default='configs/imagenet/wm_configs/jia.yaml',
                        help="Path to config file for the watermarking scheme.")
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--best', action='store_true')
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--pretrained_dir", default="outputs/imagenet/wm/jia/00013_jia")
    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes, pretrained_dir: str = None,
                 best=False, load_optimizer=False):
    """ Loads a source model from a directory and wraps it into a pytorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    if pretrained_dir:
        print(f"Loading source model from '{pretrained_dir}'.")
        for file in os.listdir(pretrained_dir):
            if best:
                if file.endswith(".pth"):
                    model.load_state_dict(torch.load(os.path.join(pretrained_dir, file))["model"])
                    print(f"Loaded model '{file}'")
            elif file.endswith(".model"):
                model.load_state_dict(torch.load(os.path.join(pretrained_dir, file)))
                print(f"Loaded model '{file}'")

            if load_optimizer and file.endswith(".optimizer"):
                optimizer.load_state_dict(torch.load(os.path.join(pretrained_dir, file)))
                print(f"Loaded optimizer '{file}'.")

    model = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model


def evaluate_test_accuracy(predictor, val_data, learning_phase=False, batch_size=32, verbose=True, limit_batches=np.inf):
    accs = []
    val_loop = tqdm(enumerate(val_data), disable=not verbose, total=min(limit_batches, len(val_data)))
    for i, (x_batch, y_batch) in val_loop:
        if i >= limit_batches:
            break
        if len(accs) > 0:
            val_loop.set_description('Validation ({:.4f})'.format(sum(accs) / len(accs)))
        x_batch = x_batch.detach().clone().cpu().numpy()
        y_batch = y_batch.detach().clone().cpu().numpy()
        if len(y_batch.shape) > 1:
            y_batch = np.argmax(y_batch, axis=1)
        with torch.no_grad():
            accs.append(compute_accuracy(predictor.predict(x_batch, batch_size=batch_size, learning_phase=learning_phase), y_batch)[0])
    return sum(accs) / len(accs)


def compute_metrics(defense_instance, x_wm, y_wm, test_loader):
    source_model = defense_instance.get_classifier()

    test_acc = evaluate_test_accuracy(source_model, test_loader, limit_batches=50, learning_phase=False)
    wm_acc = compute_accuracy(source_model.predict(x_wm, learning_phase=True), y_wm)[0]
    return {
        "wm_acc": wm_acc,
        "test_acc": test_acc
    }


def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    if args.gpu is None:
    args.gpu = reserve_gpu()  # Automatically selects the least busy GPU.
    elif not torch.cuda.is_available():
    print("CUDA is not available. Running on CPU.")

    defense_config = mlconfig.load(args.wm_config)
    print(defense_config)

    source_model = defense_config.source_model()
    optimizer = defense_config.optimizer(source_model.parameters())
    # source_model.override_learning_phase = True

    source_model: PyTorchClassifier = __load_model(source_model,
                                                   optimizer,
                                                   best=True,
                                                   load_optimizer=True,
                                                   image_size=defense_config.source_model.image_size,
                                                   num_classes=defense_config.source_model.num_classes,
                                                   pretrained_dir=args.pretrained_dir)

    valid_loader = defense_config.predict_dataset(train=False)

    # Create the defense instance
    defense: Watermark = defense_config.wm_scheme(source_model)

    keys = np.load(os.path.join(args.pretrained_dir, "secret_key.npz"))
    x_wm, y_wm = keys["x_wm"], keys["y_wm"]

    # Outputs relevant for saving.
    metrics: dict = compute_metrics(defense, x_wm, y_wm, valid_loader)

    print(metrics)


if __name__ == "__main__":
    main()
