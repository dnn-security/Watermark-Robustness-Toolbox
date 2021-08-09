""" This script plots the test accuracy in relation to the amount of training data
 for CIFAR-10 and ImageNet. """

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from visualization.utils import dataset_to_str


def parse_args():
    """Parses cmd arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "cifar10", "imagenet"],
                        help="Datasets for which the graphs should be generated.")
    parser.add_argument("--output_dir", type=str,
                        default="../outputs/visualization/lineplots/",
                        help="Directory to store the `.png` file(s). ")
    return parser.parse_args()

# Hardcoded path to the csv documents.
FILEPATHS = {
    "cifar10": "../data/experiment_results/cifar_data_ablation.csv",
    "imagenet": "../data/experiment_results/imagenet_data_ablation.csv"
}

def parse_data(dataset: str) -> dict:
    """ Parses the data and prepares it for plotting.
    Skips data point with too low model acc.
    """
    return pd.read_csv(FILEPATHS[dataset])


def plot(df, dataset, savefig: str = None):
    plt.title(f"Attacker Training Data Size vs Test Accuracy ({dataset_to_str(dataset)})")
    source_acc, max_y = 0, 0
    for index, row in df.iterrows():
        label = row[0]
        y = row[1:].to_numpy()
        x = [int(x) for x in list(df.columns.values)[1:]]
        if max(y) > max_y:
            max_y = max(y)
        if label == "Source Model":
            plt.hlines(y[0], 0, max(x), label=label, linestyles="--")
            plt.hlines(y[0]-5, 0, max(x), label="Stealing Loss Threshold (5%)", linestyles="dotted")
            source_acc = y[0]
        else:
            plt.plot(x, y, marker='x', label=label)

    plt.fill_between([0, max(x)], source_acc - 5, min(max_y, 100), alpha=0.2)
    plt.grid()
    plt.ylabel("Test Accuracy")
    plt.xlabel("Amount of (Unlabeled) Training Data")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()



def main():
    args = parse_args()

    # Fill the dataset.
    if args.dataset == "all":
        datasets = ["cifar10", "imagenet"]
    else:
        datasets = [args.dataset]

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # One graphic per dataset.
    for dataset in datasets:
        data = parse_data(dataset)
        plot(data, dataset=dataset, savefig=os.path.join(args.output_dir, f"{dataset}_lineplot_training_data.pdf"))


if __name__ == "__main__":
    main()
