import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from visualization import DataParser, attack_to_color, get_attack_category
from visualization import INPUT_PREPROCESSING, MODEL_MODIFICATION, MODEL_EXTRACTION
from visualization.utils import dataset_to_str, sort_lists, compute_alignment


# hardcode null_model_training_time in hours.
null_training_time = {
    "cifar10": 3480 / 3600,
    "imagenet": 356400 / 3600
}


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
                        default="attack_time_barchart.pdf")
    return parser.parse_args()


def parse_data(dataset: str) -> dict:
    """ Parses the data and prepares it for plotting.
    """
    data_master = DataParser.get(dataset)

    data = {"dataset": dataset}
    for attack, runtime in data_master.get_attack_times().items():
        data.setdefault(attack, {}).setdefault("runtime", runtime)
    print(data)
    return data


def plot_attack_time(data, savefig=None, verbose=True):
    """ Plots the embedding loss (loss in test accuracy relative to the null model) into a bar chart.
    """
    dataset = data.pop("dataset")

    # Fill the data.
    x, y, yerr, colors, categories = [], [], [], [], []
    for attack, values in data.items():
        # Do not track attacks with zero runtime (but warn about them)
        if np.mean(values["runtime"]) == 0. or get_attack_category(attack) == INPUT_PREPROCESSING:
            print("[WARNING] On '{}' the attack '{}' has runtime '{:.4f}'. Skipping.".format(dataset, attack,
                                                                                     np.mean(values["runtime"])))
            continue
        x.append(attack)
        y.append(np.mean(values["runtime"]) / 3600)
        yerr.append(np.std(values["runtime"]) / 3600)
        colors.append(attack_to_color[get_attack_category(attack)])
        categories.append(get_attack_category(attack))

    # Print out mean values per category .
    times = dict()
    for attack, time, category in zip(x, y, categories):
        times.setdefault(category, []).append(time)
    for category, time in times.items():
        print(category, np.mean(time))

    # Enforce ordering.
    sort_idx = compute_alignment(x, attack_to_color)
    x, y, yerr, colors = sort_lists(sort_idx, x, y, yerr, colors)

    # Plot the graph.
    plt.barh(x, y, color=colors, align='center', alpha=0.8, capsize=5)
    plt.grid()
    plt.title('Attack Runtimes ({})'.format(dataset_to_str(dataset)))
    plt.xlabel("Attack Runtime (In Hours)")
    #plt.xscale('log')
    plt.tight_layout()

    plt.vlines(null_training_time[dataset], -1, len(y)-1, linestyles="--", label="Training an Unmarked Model", color="black")

    # Custom legend
    legend_elements = [Line2D([0], [0], color=attack_to_color[MODEL_MODIFICATION], lw=4, label=MODEL_MODIFICATION),
                       Line2D([0], [0], color=attack_to_color[MODEL_EXTRACTION], lw=4, label=MODEL_EXTRACTION),
                       Line2D([0], [0], color="black", ls="--", lw=1, label="Training of an Unmarked Model"),
                       ]
    plt.legend(handles=legend_elements, loc='upper right')

    if savefig:
        plt.savefig(savefig)

    if verbose:
        plt.show()
    else:
        plt.clf()


def main():
    args = parse_args()

    print(args)

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
        plot_attack_time(data, savefig=os.path.join(args.output_dir, f"{dataset}_{args.filename}"))


if __name__ == "__main__":
    main()




