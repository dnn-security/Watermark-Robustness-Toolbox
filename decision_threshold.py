"""
This script computes a decision threshold for a given watermarking scheme.
"""

import argparse
import json
from shutil import copyfile
from typing import List

import mlconfig
import os
import torch
import torchvision
import numpy as np

from scipy.stats import norm
from tqdm import tqdm

from wrt.utils import reserve_gpu, get_max_index
from wrt.classifiers import PyTorchClassifier
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/imagenet/decision_threshold/blackmarks.yaml',
                        help="Path to config file. Determines null models and key paths.")
    parser.add_argument('--p_value', type=float, default=0.05, help="The p-value of the experiment.")
    parser.add_argument('--n_keys', type=int, default=100, help="Number of keys that are sampled.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")

    return parser.parse_args()


@mlconfig.register
def load_null_models_resnet(create_model_fn, paths: List[str], image_size, num_classes, device="cuda"):
    """ Loads null models from a given path
    """
    model = create_model_fn()
    for model_path in paths:
        model.load_state_dict(torch.load(model_path)["model"])

        criterion = torch.nn.CrossEntropyLoss()
        wrapper = PyTorchClassifier(
            model=model.to(device),
            clip_values=(0, 1),
            loss=criterion,
            optimizer=None,
            input_shape=(3, image_size, image_size),
            nb_classes=num_classes
        )
        yield wrapper


@mlconfig.register
def load_pretrained_model(source_model: PyTorchClassifier, filename: str):
    checkpoint = torch.load(filename)
    source_model.model.load_state_dict(checkpoint["model"])
    source_model.optimizer.load_state_dict(checkpoint["optimizer"])
    return source_model


@mlconfig.register
def load_null_models_imagenet(device="cuda", image_size=224, num_classes=1000, **kwargs):
    models = [torchvision.models.resnet18,
              torchvision.models.resnet34,
              torchvision.models.resnet50,
              torchvision.models.resnet101,
              torchvision.models.resnet152,
              torchvision.models.wide_resnet50_2,
              torchvision.models.wide_resnet101_2,
              torchvision.models.vgg11,
              torchvision.models.vgg13,
              torchvision.models.vgg16,
              torchvision.models.vgg19,
              torchvision.models.squeezenet1_0,
              torchvision.models.densenet121,
              torchvision.models.densenet161,
              torchvision.models.googlenet,
              torchvision.models.alexnet,
              torchvision.models.resnext50_32x4d,
              torchvision.models.inception_v3,
              torchvision.models.mobilenet_v2]  # 20 models.

    for model in models:
        criterion = torch.nn.CrossEntropyLoss()
        wrapper = PyTorchClassifier(
            model=model(pretrained=True).to(device),
            loss=criterion,
            optimizer=None,
            input_shape=(3, image_size, image_size),
            nb_classes=num_classes
        )
        yield wrapper


def wrap_model(model, optimizer, image_size, num_classes, device="cuda"):
    criterion = torch.nn.CrossEntropyLoss()
    wrapper = PyTorchClassifier(
        model=model.to(device),
        optimizer=optimizer,
        clip_values=(0, 1),
        loss=criterion,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return wrapper


def __compute_decision_threshold(wm_accs, p_value=0.05):
    # Compute the mean and std.
    mean, std = np.mean(wm_accs), np.std(wm_accs)

    if std == 0:
        print("[WARNING] Too little data. Never saw a positive example. Setting mean=0, std=1")
        mean, std = 1, 0

    x = np.linspace(0, 20 * mean, 1000)
    y = norm.cdf(x, loc=mean, scale=std)

    if len(np.where(y >= 1 - p_value)[0]) == 0:
        print(f"[ERROR] CDF never reaches {1 - p_value}")
        return float(np.min(x))

    return float(x[np.where(y >= 1 - p_value)[0]][0])


def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    reserve_gpu(args.gpu)
    config = mlconfig.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)
    print(config)

    # Load the datasets.
    valid_loader = config.dataset(train=False)
    train_loader = config.dataset(train=True)
    wm_loader = None
    if "wm_dataset" in dict(config).keys():
        wm_loader = config.wm_dataset()
        print(f"Instantiated watermark loader (with {len(wm_loader)} batches): {wm_loader}")

    base_dir = config.output_dir
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    # Create an output directory.
    idx = get_max_index(data_dir=base_dir, suffix=config.name)
    output_dir = os.path.join(base_dir, idx.zfill(5) + "_" + config.name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        raise FileExistsError
    print(f"Output dir: '{output_dir}'")

    # Save the cmd line arguments.
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # Copy the config (.yaml) file.
    path, filename = os.path.split(args.config)
    copyfile(args.config, os.path.join(output_dir, filename))

    # Load the source model.
    source_model = config.source_model_architecture()
    source_model = wrap_model(source_model,
                              config.optimizer(source_model.parameters()),
                              config.source_model_architecture.image_size,
                              config.source_model_architecture.num_classes)

    # Load from pretrained state.
    if "load_pretrained_source_model" in dict(config).keys():
        source_model = config.load_pretrained_source_model(source_model)

    # Define keylengths (for plotting the graph)
    keylengths = np.arange(10, 101, 10)

    result_dict = {}
    key_dict = {}  # Cache the keys.
    previous_model = source_model
    for i, null_model in enumerate(
            tqdm(config.null_models(create_model_fn=config.null_model_architecture, device=device),
                 desc="Computing decision threshold")):
        # Instantiate the watermarking scheme.
        defense_instance = config.wm_scheme(classifier=previous_model, config=config)

        REUSE_KEYS = True  # Reuse keys or generate new keys for every null model?
        for k in range(args.n_keys):  # Sample n different keys.
            if REUSE_KEYS:
                if k not in key_dict.keys():
                    for i, (x_wm, y_wm) in enumerate(config.secret_keys(keylengths=keylengths,
                                                                        classifier=null_model,
                                                                        train_loader=train_loader,
                                                                        wm_loader=wm_loader,
                                                                        valid_loader=valid_loader,
                                                                        defense=defense_instance,
                                                                        config=config)):
                        keylen = int(keylengths[i])
                        key_dict.setdefault(k, {})[keylen] = x_wm, y_wm
                    print(f"Done computing key '{k}'!", end="\r")
            else:
                for i, (x_wm, y_wm) in enumerate(config.secret_keys(keylengths=keylengths,
                                                                    classifier=null_model,
                                                                    train_loader=train_loader,
                                                                    wm_loader=wm_loader,
                                                                    valid_loader=valid_loader,
                                                                    defense=defense_instance,
                                                                    config=config)):
                    keylen = int(keylengths[i])
                    key_dict.setdefault(k, {})[keylen] = x_wm, y_wm

            for keylen, (x_wm, y_wm) in key_dict[k].items():
                wm_acc = defense_instance.verify(x_wm, y_wm, classifier=null_model)[0]
                result_dict.setdefault(keylen, []).append(int(wm_acc * keylen))
        previous_model = null_model

    # Compute a decision threshold on the two variables (keylength and null models).
    x, y = [], []
    for keylen, wm_accs in result_dict.items():
        x.append(keylen)
        y.append(__compute_decision_threshold(wm_accs, p_value=args.p_value))

    y_norm = []
    for x_i, y_i in zip(x, y):
        y_norm.append(y_i / x_i)

    experiment_name = args.config.split("/")[-1].split(".yaml")[0].title()

    plt.subplot(1, 2, 1)
    plt.plot(x, y_norm, label=f"{experiment_name} (p={args.p_value})", marker="x")

    plt.title(f"Keylength vs Decision Threshold (n={args.n_keys})")
    plt.xlabel("Keylength")
    plt.ylabel("Decision Threshold")
    # plt.ylim(0, max(y) + 0.1)
    plt.vlines(100, 0, max(y_norm) + 0.1, linestyle="--", label="N=100")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(max(y))
    plt.grid()
    plt.plot(x, y, label=f"{experiment_name} (p={args.p_value})", marker="x")
    full_path = os.path.join(output_dir, "decision_threshold.png")
    plt.savefig(full_path)
    plt.show()

    file = os.path.join(output_dir, f"decision_threshold_{args.p_value}.json")
    with open(file, "w") as f:
        json.dump({
            "name": experiment_name,
            "x": x,
            "y": y
        }, f)
    print(f"Saved results to '{os.path.abspath(file)}'.")


if __name__ == "__main__":
    main()
