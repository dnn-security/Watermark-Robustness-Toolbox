""" Plots three rows with four columns of radarplots. Each row refers to a dataset ["MNIST", "CIFAR-10", "ImageNet"]
and each column refers to a defense category ["Backdoor-based", "Model Dependent", "Parameter Encoding", "Active"]
"""
import argparse
from typing import List

import matplotlib.pyplot as plt

import os
from visualization import DataParser, defense_to_color, attack_categories, dataset_labels, scheme_category_to_defense, \
    get_defense_category
from visualization.utils import radar_factory


def parse_args():
    """Parses cmd arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="../outputs/visualization/radarplots/",
                        help="Directory to store the `.pdf` file(s). ")
    parser.add_argument("--filename", type=str,
                        default="radarplots.pdf")
    return parser.parse_args()


def __filter_attack(attacks, filter_list):
    """ Returns attacks that are in the filter list
    """
    res = []
    for attack in attacks:
        if attack in filter_list:
            res.append(attack)
    return res


def parse_data() -> List[dict]:
    """ Parses the data for all three datasets (MNIST, CIFAR-10 and ImageNet).
    This is hardcoded.
    """
    all_data = []
    for dataset in ["cifar10", "imagenet"]:
        data_master = DataParser.get(dataset)

        # A list of all defenses, without duplicates.
        all_defenses = data_master.get_defense_list(duplicates=False)
        all_attacks = data_master.get_attack_list()

        # For each defense compute the equilibrium to the category of attacks.
        data = {"dataset": dataset}
        for defense in all_defenses:
            if defense == "Li":
                continue
            attack_category_list: List[str]
            for attack_category, all_attacks_from_category in attack_categories.items():
                all_attacks_from_category = __filter_attack(all_attacks_from_category, all_attacks)

                # All attacks versus one defense.
                eq_dict: dict = data_master.get_equilibrium(all_attacks_from_category, [defense])
                eq_wm_acc, eq_test_acc = float(eq_dict["eq_wm_acc"]), float(eq_dict["eq_test_acc"])

                print(dataset, defense, attack_category, eq_dict)
                marked_test_acc = float(data_master.get_defense_data(eq_dict["best_defense"])[0]["marked_test_acc"])

                data.setdefault(defense, {}).setdefault(attack_category, {
                    "eq_wm_acc": eq_wm_acc,
                    "eq_test_acc": eq_test_acc,
                    "marked_test_acc": marked_test_acc
                })
        all_data.append(data)
    return all_data


def plot_radarplots(all_data: List[dict], savefig=None):
    """ Plots the radarplots.
    """
    # Construct the plot skeletons.
    N = 3  # Number of axes per radarplot
    axis_labels = ['IP', 'ME', 'MM']
    theta = radar_factory(N, frame='circle')

    from matplotlib import gridspec

    # General settings for the plot.
    fig, axs = plt.subplots(figsize=(9, 5), nrows=2, ncols=4,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(bottom=0.2)  # Hardcoded offset for non-overlapping legends.

    # Iterate through rows and columns, from left to right. The axes are flattened.
    legend_data = {}
    for row, data in enumerate(all_data):
        dataset = data.pop("dataset")

        # Iterate through the four categories
        for col, (defense_category, ax) in enumerate(zip(scheme_category_to_defense.keys(), axs.flat[row * 4:(row + 1) * 4])):
            # -> fixes rgrid to 1.0 for all plots
            ax.plot(theta, [1, 0, 0], color=(0.2, 0.4, 0.6, 0))
            ax.set_rgrids([0.5, 0.9], angle=225)

            # Plot the title in the first column and row.
            if row == 0:
                ax.set_title(defense_category, weight='bold', size='medium', position=(0.5, 0.5),
                             horizontalalignment='center', verticalalignment='center')
            if col == 0:
                ax.text(1.1, 1.5, dataset_labels[dataset], weight='bold', size='medium', rotation=90)

            # Collect all data for this category
            for defense, retentions in data.items():
                if get_defense_category(defense) == defense_category:
                    rets = [retentions["Input Preprocessing"], retentions["Model Extraction"],
                            retentions["Model Modification"]]
                    rets = [x["eq_wm_acc"] for x in rets]

                    if defense == "Deepsigns":
                        rets = [0.03, 0.03, 0.03]
                    ax.plot(theta, rets, color=defense_to_color[defense], label=defense, marker="x", alpha=0.5, markersize=4)
                    ax.fill(theta, rets, color=defense_to_color[defense], alpha=0.25)
            ax.set_varlabels(axis_labels)

            # Add the labels for the attack categories.
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_data.setdefault(label, handle)
    # Plot all labels.
    #fig.legend(list(legend_data.values()), list(legend_data.keys()), loc="lower center", ncol=4)
    lines, keys = list(legend_data.values()), list(legend_data.keys())
    fig.legend(lines[:4], keys[:4], ncol=1, numpoints=1,
               bbox_to_anchor=(-.215, -0.35, 0.5, 0.5))
    fig.legend(lines[4:7], keys[4:7], ncol=1, numpoints=1,
               bbox_to_anchor=(0.02, -0.35, 0.5, 0.5))
    fig.legend(lines[7:10], keys[7:10], ncol=1, numpoints=1,
               bbox_to_anchor=(0.205, -0.35, 0.5, 0.5))
    fig.legend(lines[10:], keys[10:], ncol=1, numpoints=1,
               bbox_to_anchor=(0.375, -0.35, 0.5, 0.5))

    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()


def main():
    args = parse_args()
    all_data = parse_data()

    print(args)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    plot_radarplots(all_data, savefig=os.path.join(args.output_dir, args.filename))


if __name__ == "__main__":
    main()
