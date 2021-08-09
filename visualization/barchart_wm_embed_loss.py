import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from visualization import defense_to_color,  get_defense_category
from visualization import BLACKBOX_BACKDOOR, WHITEBOX_BACKDOOR, PARAMETER_EMBEDDING, POSTPROCESSING
from visualization import DataParser
from visualization.utils import compute_alignment, sort_lists, dataset_to_str

import pandas as pd

def parse_args():
    """Parses cmd arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="all",
                        choices=["all", "cifar10", "imagenet"],
                        help="Datasets for which the graphs should be generated. ")
    parser.add_argument("--output_dir", type=str,
                        default="../outputs/visualization/barcharts/",
                        help="Directory to store the `.pdf` file(s). ")
    parser.add_argument("--filename", type=str,
                        default="embed_loss_barchart.pdf")
    return parser.parse_args()

FILEPATH = {
    "cifar10": "../data/experiment_results/cifar_embed_loss.csv",
    "imagenet": "../data/experiment_results/imagenet_embed_loss.csv"
}


def parse_data(dataset: str) -> dict:
    """ Parses the data and prepares it for plotting.
    """
    frame = pd.read_csv(FILEPATH[dataset])

    data = {"dataset": dataset}
    for _, row in frame.iterrows():
        label, (base_acc, marked_acc, time) = row[0], row[1:]
        data.setdefault(row[0], {}).setdefault("embed_losses", float(base_acc) - float(marked_acc))
    return data


def plot_watermark_embedding_loss(data, savefig=None, verbose=True):
    """ Plots the embedding loss (loss in test accuracy relative to the null model) into a bar chart.
    """
    dataset = data.pop("dataset")

    # Fill the data.
    x, y, xerr, colors = [], [], [], []
    for defense, values in data.items():
        x.append(defense)
        y.append((np.mean(values["embed_losses"])))
        xerr.append(np.std(values["embed_losses"]))
        colors.append(defense_to_color[get_defense_category(defense)])

    # Enforce ordering.
    sort_idx = compute_alignment(x, defense_to_color)
    x, y, xerr, colors = sort_lists(sort_idx, x, y, xerr, colors)

    # Plot the graph.
    plt.barh(x, y, align='center', alpha=0.8, capsize=5, color=colors)
    plt.grid()
    plt.xlabel("Embedding Loss")
    plt.title('Watermark Embedding Loss ({})'.format(dataset_to_str(dataset)))

    # Custom legend
    legend_elements = [Line2D([0], [0], color=defense_to_color[BLACKBOX_BACKDOOR], lw=4, label=BLACKBOX_BACKDOOR),
                       Line2D([0], [0], color=defense_to_color[WHITEBOX_BACKDOOR], lw=4, label=WHITEBOX_BACKDOOR),
                       Line2D([0], [0], color=defense_to_color[PARAMETER_EMBEDDING], lw=4,
                              label=PARAMETER_EMBEDDING),
                       Line2D([0], [0], color=defense_to_color[POSTPROCESSING], lw=4,
                              label=POSTPROCESSING)]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()

    # Savefig to abs path if desired.
    if savefig:
        plt.savefig(savefig)

    # Show fig if desired.
    if verbose:
        plt.show()
    else:
        plt.clf()


def main():
    args = parse_args()

    # Fill the dataset.
    if args.dataset == "all":
        datasets = ["cifar10", "imagenet"]
    else:
        datasets = [args.dataset]

    # One graphic per dataset.
    for dataset in datasets:
        filename = os.path.join(args.output_dir, f"{dataset}_{args.filename}")
        data = parse_data(dataset)
        plot_watermark_embedding_loss(data, savefig=filename)

        print(f"Saved as '{filename}'")


if __name__ == "__main__":
    main()
