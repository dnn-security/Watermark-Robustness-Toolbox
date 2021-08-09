"""
This script can be used to verify a source model with a given key.
Provide a '*.pth' file with a source model and a watermarking key field and this script calls the verification
function of the defense.

"""

import argparse
import json
import os

import mlconfig
import torch

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import wrt.training
from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers import PyTorchClassifier
from wrt.defenses import Watermark
from wrt.utils import reserve_gpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wm_config', type=str,
                        default='outputs/imagenet/wm/jia/00000_jia/jia.yaml',
                        help="Path to config file (*.yaml) for the watermarking scheme.")
    parser.add_argument('-f', "--wm_file", type=str,
                        default='outputs/imagenet/wm/jia/00000_jia/best.pth',
                        help="Filepath to the defense.")
    parser.add_argument('-a', '--atk_config', type=str,
                        default='outputs/imagenet/attacks/retraining/00014_retraining_attack_wm_jia/retraining.yaml',
                        help="Path to config file (*.yaml) for the attack file.")
    parser.add_argument('-d', '--atk_file', type=str,
                        default='outputs/imagenet/attacks/retraining/00014_retraining_attack_wm_jia/checkpoint.pth',
                        help="Filepath to the attack model.")
    parser.add_argument('-o', "--output_filename", type=str, default="")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
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
                                      val_data=test_loader,
                                      limit_batches=1)
    try:
        wm_acc = defense_instance.verify(x_wm, y_wm)[0]
    except:
        print("Error reading WM accuracy!")
        wm_acc = -1
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
    atk_config = mlconfig.load(args.atk_config)
    print(defense_config)

    source_model: torch.nn.Sequential = defense_config.source_model()
    optimizer = defense_config.optimizer(source_model.parameters())
    source_model: PyTorchClassifier = __load_model(source_model,
                                                   optimizer,
                                                   image_size=defense_config.source_model.image_size,
                                                   num_classes=defense_config.source_model.num_classes,
                                                   filename=args.wm_file)

    if "surrogate_model" in atk_config.keys():
        surrogate_model: torch.nn.Sequential = atk_config.surrogate_model()
        optimizer = atk_config.optimizer(surrogate_model.parameters())
    else:
        surrogate_model: torch.nn.Sequential = defense_config.source_model()
        optimizer = defense_config.optimizer(surrogate_model.parameters())

    base, head = os.path.split(args.atk_file)
    surrogate_model: PyTorchClassifier = __load_model(surrogate_model,
                                                      optimizer,
                                                      image_size=defense_config.source_model.image_size,
                                                      num_classes=defense_config.source_model.num_classes,
                                                      pretrained_dir=base,
                                                      filename=head)

    valid_loader = defense_config.dataset(train=False)

    # Load up the defense instance. Note that the source model is copied here.
    print(f"Loading model from '{args.wm_file}'")

    defense: Watermark = defense_config.wm_scheme(source_model, config=defense_config)
    path, file = os.path.split(args.wm_file)
    x_wm, y_wm = defense.load(filename=file, path=path)

    # Compute the outputs.
    metrics: dict = compute_metrics(defense, x_wm, y_wm, valid_loader)
    print("Source model test acc: {}".format(metrics["test_acc"]))
    print("Source model wm acc: {}".format(metrics["wm_acc"]))


    defense.classifier = surrogate_model
    # Compute the outputs.
    metrics: dict = compute_metrics(defense, x_wm, y_wm, valid_loader)
    print("Surrogate model test acc: {}".format(metrics["test_acc"]))
    print("Surrogate model wm acc: {}".format(metrics["wm_acc"]))

    # numbers = []
    # for batch_id, (x, y) in enumerate(valid_loader):
    #    numbers.append(torch.topk(torch.from_numpy(defense.get_classifier().predict(x.cuda())), k=2))

    # import numpy as np
    # print(numbers)
    # print(len(np.where(np.array(numbers) == 4)[0]))

    base, head = os.path.split(args.atk_file)
    if args.output_filename is not None and len(args.output_filename) > 0:
        outpath = os.path.join(base, args.output_filename)
        with open(outpath, "w") as f:
            json.dump(metrics, f)
        print(f"Saved file at '{outpath}'!")

    try:
        print(f"Extracted: '{defense.extract(x_wm)}'")
        print(f"Target: '{y_wm}'")
    except:
        pass


if __name__ == "__main__":
    main()
