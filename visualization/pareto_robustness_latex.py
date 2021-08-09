import argparse

import numpy as np

from visualization import DataParser


def parse_args():
    """Parses cmd arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="all",
                        choices=["all", "cifar", "imagenet"],
                        help="Datasets for which the graphs should be generated. ")
    parser.add_argument("--xval", type=str,
                        default="stealing_loss",
                        choices=["stealing_loss", "test_acc"],
                        help="Which x-axis value to plot. Stealing loss takes the difference in test accuracy between"
                             "the source and surrogate model. Test acc takes test accuracy difference to null model.")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/visualization/pareto/",
                        help="Directory to store the `.png` file(s). ")
    return parser.parse_args()


def parse_data(dataset: str) -> dict:
    """ Parses the data and prepares it for plotting.
    """
    data_master = DataParser.get(dataset)

    # A list of all attacks, without duplicates.
    all_attacks = data_master.get_attack_list()

    data = {}
    defense: str
    for defense in data_master.get_defense_list(duplicates=False):
        # All attacks versus one defense.
        eq_dict: dict = data_master.get_equilibrium(all_attacks, [defense], return_best_defense=True)
        data.setdefault(defense, eq_dict["best_defense_full"])
    return data


def convert_to_latex(data_dict):
    attacks = ["RTLL", "RTAL", "Fine Pruning", "Neural Pruning", "Neural Laundering", "Input Reconstruction",
               "Input Noising", "Knockoff", "Retraining", "Transfer Learning", "Cross Architecture Retraining"]

    res_dict = {}
    for defense, pd in data_dict.items():
        vals_to_list = pd.values.tolist()
        marked_model_acc = float(vals_to_list[2][1].split("/")[1])
        print(defense, marked_model_acc)
        for i, row in enumerate(vals_to_list[3:]):
            if row[0].strip() in attacks:
                surr_acc = float(row[1].split("/")[1])
                if marked_model_acc-surr_acc <= 0.05:
                    wm_acc = float(row[1].split("/")[0])
                    if defense == "Jia":
                        print(f"Jia vs {row[0]}: {wm_acc}")
                    res_dict.setdefault(defense, {}).setdefault(row[0], []).append(wm_acc)

    for defense, values in res_dict.items():
        for attack, wm_accs in values.items():
            if np.min(wm_accs) <= 0.17:
                print(f"Defense {defense} not robust against {attack}")

def main():
    args = parse_args()

    # Fill the dataset.
    if args.dataset == "all":
        datasets = ["imagenet"]
    else:
        datasets = [args.dataset]

    # One graphic per dataset.
    for dataset in datasets:
        data = parse_data(dataset)
        convert_to_latex(data)


if __name__ == "__main__":
    main()
