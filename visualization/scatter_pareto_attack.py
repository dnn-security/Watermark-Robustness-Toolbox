import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from visualization import get_attack_category, attack_to_color, dataset_labels
from visualization import DataParser


def parse_args():
    """Parses cmd arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="all",
                        choices=["all", "cifar10", "imagenet"],
                        help="Datasets for which the graphs should be generated. ")
    parser.add_argument("--xval", type=str,
                        default="stealing_loss",
                        choices=["stealing_loss", "test_acc"],
                        help="Which x-axis value to plot. Stealing loss takes the difference in test accuracy between"
                             "the source and surrogate model. Test acc takes test accuracy difference to null model.")
    parser.add_argument("--output_dir", type=str,
                        default="../outputs/visualization/pareto/",
                        help="Directory to store the `.png` file(s). ")
    return parser.parse_args()


def parse_data(dataset: str, max_stealing_loss=0.05) -> dict:
    """ Parses the data and prepares it for plotting.
    Skips data point with too low model acc.
    """
    data_master = DataParser.get(dataset)

    # A list of all defenses, without duplicates.
    all_defenses = data_master.get_defense_list(duplicates=False)

    data = {"dataset": dataset}
    attack: str
    for attack in data_master.get_attack_list():
        # All attacks versus one defense.
        eq_dict: dict = data_master.get_equilibrium([attack], all_defenses)
        eq_wm_acc, eq_test_acc = float(eq_dict["eq_wm_acc"]), float(eq_dict["eq_test_acc"])

        # Obtain marked test acc for computing the stealing loss.
        marked_test_acc = float(data_master.get_defense_data(eq_dict["best_defense"])[0]["marked_test_acc"])
        eq_steal_loss = marked_test_acc - eq_test_acc
        data.setdefault(attack, dict(eq_wm_acc=eq_wm_acc, eq_test_acc=eq_test_acc,
                                 eq_steal_loss=eq_steal_loss))
    return data


def __find_pareto_points(x_vals, retentions, orientation="high"):
    """ Returns a subset of points in sorted order (for test acc) that mark the pareto line
    @:param orientation: Either "high" or "low", depending on which values are better.
    """
    assert orientation in ["high", "low"], f"Value '{orientation}' for finding pareto points is invalid. "

    # The return array for all connected pareto plots.
    pareto_x, pareto_y = [], []

    # Sort the x-axis values according to the orientation (see doc).
    sort_idx = np.argsort(x_vals)
    if orientation == "low":
        sort_idx = sort_idx[::-1]

    curr_ret = 0.5
    for idx in sort_idx:
        if retentions[idx] <= curr_ret and x_vals[idx] <= 0.05:
            pareto_x.append(x_vals[idx])
            pareto_y.append(retentions[idx])
            curr_ret = retentions[idx]
    return pareto_x, pareto_y


def plot_annotations_non_overlapping(x, y, labels, bold, text_vertical_offset=0.045):
    """ Plot labels without overlaps by clustering labels that are close.
    """
    x, y, labels, bold = np.array(x), np.array(y), np.array(labels), np.array(bold)
    xy = [[x[i], y[i]] for i in range(len(x))]
    db = DBSCAN(eps=0.01, min_samples=1).fit(xy)

    for cluster_id in range(np.max(db.labels_) + 1):
        idx, = np.where(db.labels_ == cluster_id)

        # Sort by y_value
        sort_idx = np.argsort(y[idx])
        for i, (xp, yp, label, is_bold) in enumerate(zip(x[idx][sort_idx], y[idx][sort_idx], labels[idx][sort_idx],
                                                         bold[idx][sort_idx])):
            plt.annotate(label, (xp, yp + 0.002 + i * text_vertical_offset), weight='bold' if is_bold else None)


def plot_scatter_pareto_attack(data, xval, savefig=None, verbose=True):
    """ Plots the embedding loss (loss in test accuracy relative to the null model) into a bar chart.
    """
    assert xval in ["stealing_loss", "test_acc"], f"Value '{xval}' invalid for xval."

    dataset = data.pop("dataset")

    # Fill the data.
    x, y, labels, colors = [], [], [], []
    for attack, values in data.items():
        if xval == "stealing_loss":
            x.append(values["eq_steal_loss"])
        else:
            x.append(values["eq_test_acc"])
        y.append(values["eq_wm_acc"])
        labels.append(attack)
        colors.append(attack_to_color[get_attack_category(attack)])

    # Draw decision boundary
    plt.hlines(0.5, 0, 0.05, linestyles="--", label="Decision Threshold")
    plt.vlines(0.05, 0, 1.0, linestyles="dotted", label="Stealing Loss Boundary")

    plt.ylim(0, 0.52)
    plt.xlim(0, 0.055)

    # Draw the pareto front.
    orientation = "low" if xval == "test_acc" else "high"
    pareto_x, pareto_y = __find_pareto_points(x, y, orientation=orientation)
    plt.plot(pareto_x, pareto_y, label="Pareto Frontier")

    bold = [(x[i] in pareto_x) and (y[i] in pareto_y) for i in range(len(x))]
    bold_idx, = np.where(bold)
    x, y, labels, bold = np.asarray(x), np.asarray(y), np.asarray(labels), np.asarray(bold)

    # Plot the data points with text annotations.
    plt.scatter(x[bold_idx], y[bold_idx], alpha=0.8, color=[colors[i] for i in bold_idx])

    if len(bold_idx) > 0:
        plot_annotations_non_overlapping(x[bold_idx], y[bold_idx], labels[bold_idx], bold=bold[bold_idx],
                                     text_vertical_offset=0.05)

    plt.grid()
    plt.xlabel("Test Accuracy" if xval == "test_acc" else "Stealing Loss")
    plt.ylabel("Watermark Accuracy")
    plt.title('Pareto Frontier: Single Attack versus All Schemes ({})'.format(dataset_labels[dataset]))
    plt.legend(loc="lower right")
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

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
        plot_scatter_pareto_attack(data, xval=args.xval, savefig=os.path.join(args.output_dir,
                                                                              f"{dataset}_scatter_pareto_attack.pdf"))


if __name__ == "__main__":
    main()
