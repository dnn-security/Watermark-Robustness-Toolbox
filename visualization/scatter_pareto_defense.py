import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from visualization import defense_to_color, get_defense_category, dataset_labels
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


def parse_data(dataset: str) -> dict:
    """ Parses the data and prepares it for plotting.
    """
    data_master = DataParser.get(dataset)

    # A list of all attacks, without duplicates.
    all_attacks = data_master.get_attack_list()

    print(all_attacks)

    data = {"dataset": dataset}
    defense: str
    for defense in data_master.get_defense_list(duplicates=False):
        # All attacks versus one defense.
        eq_dict: dict = data_master.get_equilibrium(all_attacks, [defense])
        eq_wm_acc, eq_test_acc = float(eq_dict["eq_wm_acc"]), float(eq_dict["eq_test_acc"])

        print(eq_dict)

        # Obtain marked test acc for computing the stealing loss.
        marked_test_acc = float(data_master.get_defense_data(defense)[0]["marked_test_acc"])
        data.setdefault(defense, dict(eq_wm_acc=eq_wm_acc, eq_test_acc=eq_test_acc,
                                      eq_steal_loss=(marked_test_acc - eq_test_acc)))
    return data


def __find_pareto_points(test_acc, retentions, orientation="high"):
    """ Returns a subset of points in sorted order (for test acc) that mark the pareto line
    @:param orientation: Either "high" or "low", depending on which values are better.
    """
    assert orientation in ["high", "low"], f"Value '{orientation}' for finding pareto points is invalid. "

    # The return array for all connected pareto plots.
    pareto_x, pareto_y = [], []

    # Sort the x-axis values according to the orientation (see doc).
    sort_idx = np.argsort(test_acc)
    if orientation == "low":
        sort_idx = sort_idx[::-1]

    curr_ret = 0
    for idx in sort_idx:
        if retentions[idx] >= curr_ret:
            pareto_x.append(test_acc[idx])
            pareto_y.append(retentions[idx])
            curr_ret = retentions[idx]
    return pareto_x, pareto_y


def plot_annotations_non_overlapping(x, y, labels, bold, text_vertical_offset=-0.012,
                                     text_horizontal_offset=-.008, custom_offsets: dict = {}):
    """ Plot labels without overlaps by clustering labels that are close.
    """
    x, y, labels, bold = np.array(x), np.array(y), np.array(labels), np.array(bold)
    xy = [[x[i], y[i]] for i in range(len(x))]
    db = DBSCAN(eps=0.005, min_samples=1).fit(xy)

    for cluster_id in range(np.max(db.labels_) + 1):
        idx, = np.where(db.labels_ == cluster_id)

        if len(idx) == 1:
            x_offset = [0]
        else:
            x_offset = [text_horizontal_offset]

        # Sort by y_value
        sort_idx = np.argsort(y[idx])
        x_cluster = None
        for i, (xp, yp, label, is_bold) in enumerate(zip(x[idx][sort_idx], y[idx][sort_idx], labels[idx][sort_idx],
                                                         bold[idx][sort_idx])):
            cox, coy = custom_offsets.setdefault(label, (0,0))
            if is_bold:
                plt.annotate(label, (xp+cox, yp+coy), weight='bold' if is_bold else None)
            else:
                xo, yo = x_offset[i % len(x_offset)], text_vertical_offset
                #if x_cluster is None:
                x_cluster = xp
                plt.annotate(label, (x_cluster+xo+cox, coy+yp + 0.002 + i * yo), weight='bold' if is_bold else None)


def is_xy_contained(x, y, points_x, points_y):
    for point_x, point_y in zip(points_x, points_y):
        if x == point_x and y == point_y:
            return True
    return False


def plot_scatter_pareto_defense(data, xval, savefig=None, verbose=True):
    """ Plots the embedding loss (loss in test accuracy relative to the null model) into a bar chart.
    """
    assert xval in ["stealing_loss", "test_acc"], f"Value '{xval}' invalid for xval."

    dataset = data.pop("dataset")

    # Fill the data.
    x, y, labels, colors = [], [], [], []
    for defense, values in data.items():
        if xval == "stealing_loss":
            x.append(values["eq_steal_loss"])
        else:
            x.append(values["eq_test_acc"])
        y.append(values["eq_wm_acc"])
        labels.append(defense)
        colors.append(defense_to_color[get_defense_category(defense)])

    # Draw decision boundary
    plt.hlines(0.5, min(x), max(x)+0.01, linestyles="--", label="Decision Threshold")

    # Draw the pareto front.
    orientation = "high" if xval == "test_acc" else "low"
    pareto_x, pareto_y = __find_pareto_points(x, y, orientation=orientation)
    plt.plot(pareto_x, pareto_y, label="Pareto Frontier")

    bold = [is_xy_contained(x[i], y[i], pareto_x, pareto_y) for i in range(len(x))]
    print(bold, labels)
    print(x,y)

    # Plot the data points with text annotations.
    plt.scatter(x, y, alpha=0.8, color=colors)

    if dataset == "cifar10":
        text_vertical_offset, text_horizontal_offset = 0,  0
        custom_offsets = {
            "Adi": (-.0003, -0.025),
            "Blackmarks": (-0.0012, 0.01),
            "Unrelated": (-.001, -0.02),
            "Jia": (-.0003, 0.012),
            "DAWN": (-0.0007, 0.018),
            "Uchida": (-0.0008, 0.01),
            "Noise": (-0.0005, 0.01),
            "Frontier Stitching": (-0.0027, -0.026),
            "Content": (0.0001, 0),
            "Deepsigns": (-0.0012, 0.01)
        }
    elif dataset == "imagenet":
        text_vertical_offset, text_horizontal_offset = 0,  0
        custom_offsets = {
            "Adi": (-.001, 0.012),
            "DAWN": (-.004, -0.01),
            "Content": (-0.002, -0.03),
            "Blackmarks": (0, -0.025),
            "Unrelated": (0.0002,-0.017),
            "Noise": (-.0012,0.012),
            "Uchida": (0, 0.004),
            "Jia": (0.0005, -0.02),
            "Frontier Stitching": (-.004, 0.005),
            "Deepsigns": (0.0, 0.005),
            "Deepmarks": (-0.003, 0.007)
        }
    else:
        raise ValueError

    plot_annotations_non_overlapping(x, y, labels, bold=bold, text_horizontal_offset=text_horizontal_offset,
                                     text_vertical_offset=text_vertical_offset, custom_offsets=custom_offsets)

    #xmin, xmax = min(min(x), 0), max(max(x), 0.05)
    #plt.xlim(xmin, xmax)

    plt.grid()
    plt.xlabel("Test Accuracy" if xval == "test_acc" else "Stealing Loss")
    plt.ylabel("Watermark Accuracy")
    plt.title('Pareto Frontier: Single Scheme versus All Attacks ({})'.format(dataset_labels[dataset]))
    plt.legend(loc="lower right")
    plt.tight_layout()

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
        datasets = ["cifar10"]
    else:
        datasets = [args.dataset]

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # One graphic per dataset.
    for dataset in datasets:
        data = parse_data(dataset)
        plot_scatter_pareto_defense(data, xval=args.xval,
                                    savefig=os.path.join(args.output_dir, f"{dataset}_scatter_pareto_defense.pdf"))


if __name__ == "__main__":
    main()
