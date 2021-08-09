import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from visualization import DataParser, defense_to_color, get_defense_category
from visualization import PARAMETER_EMBEDDING, BLACKBOX_BACKDOOR, WHITEBOX_BACKDOOR, POSTPROCESSING
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
                        default="embed_time_barchart.pdf")
    return parser.parse_args()

FILEPATH = {
    "cifar10": "../data/experiment_results/cifar_runtime_defense.csv",
    "imagenet": "../data/experiment_results/imagenet_runtime_defense.csv"
}

normalizer = {
    "cifar10": 60,
    "imagenet": 3600
}

xlabel = {
    "cifar10": "Embedding Time (in Minutes)",
    "imagenet": "Embedding Time (in Hours)"
}


def parse_data(dataset: str) -> dict:
    """ Parses the data and prepares it for plotting.
    """
    frame = pd.read_csv(FILEPATH[dataset])

    data = {"dataset": dataset}
    for _, row in frame.iterrows():
        label, time = row[0], row[1]
        time = np.mean([float(x) for x in str(time).split(";")])
        data.setdefault(row[0], {}).setdefault("runtime", float(time))
    return data


def plot_watermark_embedding_time(data, savefig=None, verbose=True):
    """ Plots the embedding loss (loss in test accuracy relative to the null model) into a bar chart.
    """
    dataset = data.pop("dataset")

    # Fill the data.
    x, y, yerr, colors = [], [], [], []
    for defense, values in data.items():
        x.append(defense)
        y.append(np.mean(values["runtime"]) / normalizer[dataset])
        yerr.append(np.std(values["runtime"]) / normalizer[dataset])
        colors.append(defense_to_color[get_defense_category(defense)])

    # Enforce ordering.
    sort_idx = compute_alignment(x, defense_to_color)
    x, y, yerr, colors = sort_lists(sort_idx, x, y, yerr, colors)

    # Plot the graph.
    plt.barh(x, y, align='center', alpha=0.8, capsize=5, color=colors)
    plt.grid()
    plt.xlabel(xlabel[dataset])
    plt.title('Watermark Embedding Times ({})'.format(dataset_to_str(dataset)))
    plt.tight_layout()

    # Custom legend
    legend_elements = [Line2D([0], [0], color=defense_to_color[BLACKBOX_BACKDOOR], lw=4, label=BLACKBOX_BACKDOOR),
                       Line2D([0], [0], color=defense_to_color[WHITEBOX_BACKDOOR], lw=4, label=WHITEBOX_BACKDOOR),
                       Line2D([0], [0], color=defense_to_color[PARAMETER_EMBEDDING], lw=4,
                              label=PARAMETER_EMBEDDING),
                       Line2D([0], [0], color=defense_to_color[POSTPROCESSING], lw=4,
                              label=POSTPROCESSING)]
    plt.legend(handles=legend_elements, loc='lower right')

    if savefig:
        plt.savefig(savefig)

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
        data = parse_data(dataset)
        plot_watermark_embedding_time(data, savefig=os.path.join(args.output_dir, f"{dataset}_{args.filename}"))


if __name__ == "__main__":
    main()
