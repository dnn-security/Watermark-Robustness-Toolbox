""" This script trains null models given a configuration file (see configs) """

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from shutil import copyfile

import mlconfig
import numpy as np
import torch

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import wrt.training
from wrt.attacks import RemovalAttack

from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import PyTorchClassifier
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets.cifar10 import cifar_classes
from wrt.utils import reserve_gpu, get_max_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack_config', type=str,
                        default='configs/imagenet/attack_configs/label_smoothing.yaml',
                        help="Path to config file for the attack.")
    parser.add_argument('-w', "--wm_dir", type=str,
                        default="outputs/imagenet/wm/jia/00000_jia",
                        help="Path to the directory with the watermarking files. "
                             "This scripts expects a 'best.pth' and one '*.yaml' file "
                             "to exist in this dir.")
    parser.add_argument('-r', "--resume", type=str,
                        default=None,
                        help="Path to checkpoint to continue the attack. ")
    parser.set_defaults(true_labels=False, help="Whether to use ground-truth labels.")
    parser.add_argument('--true_labels', action='store_true')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")

    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes, defense_filename: str = None):
    """ Loads a source model from a directory and wraps it into a pytorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    # Load defense model from a saved state, if available.
    # We allow loading the optimizer, as it only loads states that the attacker could tune themselves (E.g. learning rate)
    if defense_filename is not None:
        pretrained_data = torch.load(defense_filename)
        model.load_state_dict(pretrained_data["model"])
        try:
            optimizer.load_state_dict(pretrained_data["optimizer"])
        except:
            print("Optimizer could not be loaded. ")
            pass

        print(f"Loaded model and optimizer from '{defense_filename}'.")

    model = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model


def file_with_suffix_exists(dirname, suffix, not_contains="", raise_error=False):
    for file in os.listdir(dirname):
        if file.endswith(suffix) and (not not_contains in file or len(not_contains) == 0):
            return os.path.abspath(os.path.join(dirname, file))
    if raise_error:
        raise FileNotFoundError(f"No file found with suffix '{suffix}' in '{dirname}")
    return False


def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    reserve_gpu(args.gpu)
    device = "cuda"

    # Discover the '*.yaml' config file and the 'best.pth' file.
    defense_yaml = file_with_suffix_exists(dirname=args.wm_dir, suffix=".yaml")
    pth_file = file_with_suffix_exists(dirname=args.wm_dir, suffix="best.pth")

    if not defense_yaml or not pth_file:
        raise FileNotFoundError(defense_yaml)

    defense_config = mlconfig.load(defense_yaml)
    print(defense_config)

    attack_config = mlconfig.load(args.attack_config)
    print(attack_config)

    # Create output folder.
    if not os.path.exists(attack_config.output_dir):
        os.makedirs(attack_config.output_dir)
    output_dir = os.path.join(attack_config.output_dir,
                              f"{get_max_index(attack_config.output_dir, suffix=attack_config.create.name).zfill(5)}_"
                              f"{attack_config.create.name}_{defense_config.wm_scheme.name}")
    os.makedirs(output_dir)
    print(f"======> Logging outputs to '{os.path.abspath(output_dir)}'")
    print(f"Saving outputs? {args.save}")

    # Copy the config (.yaml) files.
    for config_path in [args.attack_config, defense_yaml]:
        path, filename = os.path.split(config_path)
        copyfile(config_path, os.path.join(output_dir, filename))

    # Save the cmd line arguments.
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    model_basedir, model_filename = os.path.split(pth_file)

    source_model = defense_config.source_model()
    source_model = source_model.to(device)
    optimizer = defense_config.optimizer(source_model.parameters())
    source_model = __load_model(source_model, optimizer,
                                image_size=defense_config.source_model.image_size,
                                num_classes=defense_config.source_model.num_classes,
                                defense_filename=pth_file)

    defense = defense_config.wm_scheme(classifier=source_model, optimizer=optimizer, config=defense_config)
    x_wm, y_wm = defense.load(filename=model_filename, path=model_basedir)

    print(y_wm)

    use_gt = args.true_labels or ("true_labels" in attack_config.keys() and attack_config.true_labels)
    print(f"Using ground truth labels? {use_gt}")
    if use_gt:
        print("Using ground-truth labels ..")
        train_loader = attack_config.dataset(train=True)
        valid_loader = attack_config.dataset(train=False)
    else:
        print("Using predicted labels ..")
        train_loader = attack_config.dataset(source_model=source_model, train=True)
        valid_loader = attack_config.dataset(source_model=source_model, train=False)

    source_test_acc_before_attack = evaluate_test_accuracy(source_model, valid_loader)
    print(f"Source model test acc: {source_test_acc_before_attack:.4f}")
    source_wm_acc = defense.verify(x_wm, y_wm, classifier=source_model)[0]
    print(f"Source model wm acc: {source_wm_acc:.4f}")

    if "surrogate_model" in attack_config.keys():
        surrogate_model = attack_config.surrogate_model()
        optimizer = attack_config.optimizer(surrogate_model.parameters())
        surrogate_model = __load_model(surrogate_model, optimizer,
                                       image_size=attack_config.surrogate_model.image_size,
                                       num_classes=attack_config.surrogate_model.num_classes)
    else:
        surrogate_model = deepcopy(source_model)

    if args.resume is not None:
        print(f"Resuming from checkpoint '{args.resume}' ... ")
        pretrained_data = torch.load(args.resume)
        surrogate_model.model.load_state_dict(pretrained_data["model"])
        try:
            surrogate_model.optimizer.load_state_dict(pretrained_data["optimizer"])
        except:
            pass

    surrogate_test_acc_before_attack, surrogate_wm_acc_before_attack = -1, -1
    try:
        surrogate_test_acc_before_attack = evaluate_test_accuracy(surrogate_model, valid_loader)
        print(f"Surrogate model test acc: {surrogate_test_acc_before_attack:.4f}")
        surrogate_wm_acc_before_attack = defense.verify(x_wm, y_wm, classifier=surrogate_model)[0]
        print(f"Surrogate model wm acc: {surrogate_wm_acc_before_attack:.4f}")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Could not extract watermark accuracy from the surrogate model ... Continuing ..")

    attack: RemovalAttack = attack_config.create(classifier=surrogate_model, config=attack_config)

    # Run the removal. We still need wrappers to conform to the old interface.
    start = time.time()
    attack, train_metric = attack_config.remove(attack=attack,
                                                source_model=source_model,
                                                train_loader=train_loader,
                                                valid_loader=valid_loader,
                                                config=attack_config,
                                                output_dir=output_dir,
                                                wm_data=(defense, x_wm, y_wm))
    end = time.time()
    execution_time = end - start

    surrogate_model = attack.get_classifier()
    surrogate_test_acc_after_attack = evaluate_test_accuracy(surrogate_model, valid_loader)
    print(f"Surrogate model test acc: {surrogate_test_acc_after_attack:.4f}")
    surrogate_wm_acc_after_attack = defense.verify(x_wm, y_wm, classifier=surrogate_model)[0]
    print(f"Surrogate model wm acc: {surrogate_wm_acc_after_attack:.4f}")

    if args.save:
        with open(os.path.join(output_dir, 'result.json'), "w") as f:
            json.dump({
                "test_acc_before": surrogate_test_acc_before_attack,
                "wm_acc_before": surrogate_wm_acc_before_attack,
                "test_acc_after": surrogate_test_acc_after_attack,
                "wm_acc_after": surrogate_wm_acc_after_attack,
                "time": execution_time
                       }, f)

        if train_metric is None:
            train_metric = {}

        # Save the model and the watermarking key.
        checkpoint = {
            "model": surrogate_model.model.state_dict(),
            "optimizer": surrogate_model.optimizer.state_dict(),
            "x_wm": x_wm,
            "y_wm": y_wm
        }
        torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))

        # Save the training metrics.
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            all_metrics = {
                "source_test_acc": source_test_acc_before_attack,
                "source_wm_acc": source_wm_acc,
                "surr_test_acc_before": surrogate_test_acc_before_attack,
                "surr_wm_acc_before": surrogate_wm_acc_before_attack,
                **train_metric
            }
            json.dump(all_metrics, f)
        print(f"Successfully saved data to '{os.path.abspath(output_dir)}'")


if __name__ == "__main__":
    main()
