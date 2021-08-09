""" This script shows the fastest successful removal attack for each scheme """

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from visualization import DataParser, defense_to_color, get_defense_category, \
    BLACKBOX_BACKDOOR, WHITEBOX_BACKDOOR, PARAMETER_EMBEDDING, POSTPROCESSING, INPUT_PREPROCESSING, get_attack_category
from visualization.utils import dataset_to_str, sort_lists, compute_alignment


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
                        default="fastest_attack_barchart.pdf")
    return parser.parse_args()


def parse_data(dataset: str) -> dict:
    """ Parses the data and prepares it for plotting.
    """
    data_master = DataParser.get(dataset)

    data = {"dataset": dataset}
    for attack, runtime in data_master.get_attack_times().items():
        data.setdefault("attacks", {}).setdefault(attack, {}).setdefault("runtime", runtime)

    # get the best defense config against all attacks.
    all_attacks = data_master.get_attack_list()
    best_defense_config = {}
    for defense in data_master.get_defense_list(duplicates=False):
        # All attacks versus one defense.
        eq_dict: dict = data_master.get_equilibrium(all_attacks, [defense])
        best_defense_config.setdefault(defense, eq_dict["best_defense_name"])

    # Get the successful attacks against each 'best' defense config.
    for defense in data_master.get_defense_list(duplicates=False):
        defense_data = data_master.get_defense_data(defense)[0]
        for attack in all_attacks:
            eq_dict: dict = data_master.get_equilibrium([attack], [defense])
            eq_wm_acc, eq_test_acc = float(eq_dict["eq_wm_acc"]), float(eq_dict["eq_test_acc"])

            if eq_test_acc >= float(defense_data["unmarked_test_acc"]) - 0.05 and float(eq_wm_acc) <= data_master.base_value:
                # Successful attack.
                data.setdefault("defenses", {}).setdefault(defense, []).append(attack)
    print(data)
    return data


def plot_fastest_attack(data, savefig=None, verbose=True):
    """ Plots the fastest attack out of all successful attacks in a barchart.
    """
    dataset = data.pop("dataset")

    # Find the fastest attack for each defense.
    fastest_attacks = {}
    for defense in data["defenses"].keys():
        best_attack, best_runtime = "", np.inf

        if defense.lower().strip() == "dawn":
            best_attack = "Smooth Retraining"
            best_runtime = 0
        else:
            for attack in data["defenses"][defense]:
                if data["attacks"][attack]["runtime"] < best_runtime:
                    best_attack = attack
                    if get_attack_category(best_attack) == INPUT_PREPROCESSING:
                        best_runtime = 0
                    else:
                        best_runtime = data["attacks"][attack]["runtime"]
        fastest_attacks.setdefault(defense, {"attack": best_attack, "runtime": best_runtime})

    # Manual adjustenemtns
    if dataset == "imagenet":
        fastest_attacks["DAWN"] = {"attack": "Smooth Retraining", "runtime": 0}
    fastest_attacks["Deepsigns"] = {"attack": "Feature Permutation", "runtime": 0}

    print(fastest_attacks)

    # Fill the data.
    x, y, colors = [], [], []
    for defense, values in fastest_attacks.items():
        x.append(defense)
        y.append(values["runtime"] / 3600)
        colors.append(defense_to_color[get_defense_category(defense)])

    plt.xlim(0, max(y) + 0.5*max(y))

    # Enforce ordering.
    sort_idx = compute_alignment(x, defense_to_color)
    x, y, colors = sort_lists(sort_idx, x, y, colors)

    # Add the text of the fastest attack.
    for i, v in enumerate(y):
        plt.text(0, i-0.125, str(fastest_attacks[x[i]]["attack"]), color='black')

    # Plot the graph.
    plt.barh(x, y, align='center', alpha=0.8, capsize=5, color=colors)
    #   plt.grid()
    plt.xlabel("Time in Hours")
    plt.title('Fastest Attacks to Remove a Watermark ({})'.format(dataset_to_str(dataset)))

    def split(input_string, delimiter=" "):
        final = ""
        splits = input_string.split(delimiter)
        for x in splits[:-1]:
            final += x + "\n"
        final += splits[-1]
        return final
    l1, l2, l3 = split(BLACKBOX_BACKDOOR), split(WHITEBOX_BACKDOOR), split(PARAMETER_EMBEDDING)

    # Custom legend
    legend_elements = [Line2D([0], [0], color=defense_to_color[BLACKBOX_BACKDOOR], lw=4, label=l1),
                       Line2D([0], [0], color=defense_to_color[WHITEBOX_BACKDOOR], lw=4, label=l2),
                       Line2D([0], [0], color=defense_to_color[PARAMETER_EMBEDDING], lw=4, label=l3),
                       Line2D([0], [0], color=defense_to_color[POSTPROCESSING], lw=4, label=POSTPROCESSING)]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.grid()
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
        plot_fastest_attack(data, savefig=os.path.join(args.output_dir, f"{dataset}_{args.filename}"))


if __name__ == "__main__":
    main()
