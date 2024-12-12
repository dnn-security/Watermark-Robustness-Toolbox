""" This script trains null models given a configuration file (see configs) """

import argparse
import json
import os
import time
from datetime import datetime
from shutil import copyfile

import mlconfig
import torch

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import wrt.training

from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import PyTorchClassifier
from wrt.defenses import Watermark
from wrt.utils import reserve_gpu, get_max_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wm_config', type=str, default='configs/cifar10/wm_configs/dawn1.yaml',
                        help="Path to config file for the watermarking scheme.")
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument("--filename", type=str, default="best.pth", help="Filepath to the pretrained model.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--pretrained_dir", default="outputs/cifar10/wm/pretrained/resnet/00000_pretrained")
    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes, pretrained_dir: str = None,
                 filename: str = 'best.pth'):
    """ Loads a (pretrained) source model from a directory and wraps it into a PyTorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    if pretrained_dir:
        assert filename.endswith(".pth"), "Only '*.pth' are allowed for pretrained models"
        print(f"Loading a pretrained source model from '{pretrained_dir}'.")
        state_dict = torch.load(os.path.join(pretrained_dir, filename))
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])

    model = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model


def compute_metrics(defense_instance, x_wm, y_wm, test_loader):
    """ Computes the test and watermark accuracy.
    """
    test_acc = evaluate_test_accuracy(classifier=defense_instance.get_classifier(),
                                      val_data=test_loader)
    wm_acc = defense_instance.verify(x_wm, y_wm)[0]
    return {
        "wm_acc": float(wm_acc),
        "test_acc": float(test_acc)
    }


def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    reserve_gpu(args.gpu)

    defense_config = mlconfig.load(args.wm_config)
    print(defense_config)

    # Create the output folder.
    if not os.path.exists(defense_config.output_dir):
        os.makedirs(defense_config.output_dir)
    output_dir = os.path.join(defense_config.output_dir,
                              f"{get_max_index(defense_config.output_dir, suffix=defense_config.name).zfill(5)}_"
                              f"{defense_config.name}")
    os.makedirs(output_dir, exist_ok=False)
    print(f"===========> Creating directory '{output_dir}'")

    # Save the cmd line arguments.
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # Copy the config (.yaml) file.
    path, filename = os.path.split(args.wm_config)
    copyfile(args.wm_config, os.path.join(output_dir, filename))

    source_model: torch.nn.Sequential = mlconfig.instantiate(defense_config.source_model)
    optimizer = mlconfig.instantiate(defense_config.optimizer, source_model.parameters())

    source_model: PyTorchClassifier = __load_model(source_model,
                                                   optimizer,
                                                   image_size=defense_config.source_model.image_size,
                                                   num_classes=defense_config.source_model.num_classes,
                                                   filename=args.filename,
                                                   pretrained_dir=args.pretrained_dir)
    # Load the training and testing data.
    train_loader = mlconfig.instantiate(defense_config.dataset, train=True)
    valid_loader = mlconfig.instantiate(defense_config.dataset, train=False)

    # Optionally load a dataset to load watermarking images from.
    wm_loader = None
    if "wm_dataset" in dict(defense_config).keys():
        wm_loader = mlconfig.instantiate(defense_config.wm_dataset)
        print(f"Instantiated watermark loader (with {len(wm_loader)} batches): {wm_loader}")

    source_test_acc_before_attack = evaluate_test_accuracy(source_model, valid_loader)
    print(f"Source model test acc (before): {source_test_acc_before_attack}")

    # Create the defense instance with the pretrained source model. Note: The source model is copied here.
    defense: Watermark = mlconfig.instantiate(defense_config.wm_scheme, source_model, config=defense_config)

    # Save this configuration.
    from omegaconf import OmegaConf
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        config = {
            "timestamp": str(datetime.now()),
            "defense_config": OmegaConf.to_container(defense_config, resolve=True),
            "args": vars(args)
        }
        json.dump(config, f)

    # Embed the watermark. Note that all inputs are copied here.
    # We assume the defense stores the model and all auxiliary information in the output directory.
    start_time = time.time()
    (x_wm, y_wm), defense = mlconfig.instantiate(defense_config.embed, 
                                                 defense=defense,
                                                 train_loader=train_loader,
                                                 valid_loader=valid_loader,
                                                 wm_loader=wm_loader,
                                                 config=defense_config,
                                                 output_dir=output_dir)
    end_time = time.time()
    total_elapsed = end_time - start_time

    # Compute the outputs.
    metrics: dict = compute_metrics(defense, x_wm, y_wm, valid_loader)

    print("Source model test acc: {}".format(metrics["test_acc"]))
    print("Source model wm acc: {}".format(metrics["wm_acc"]))

    # Save the final metrics (if available)
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump({**metrics, "time": total_elapsed}, f)


if __name__ == "__main__":
    main()
