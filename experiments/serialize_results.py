""" This script serializes an attack output folder and captures the result as a .csv file.

It expects the following folder structure:

- base_dir
    - 00000_*
        - *.pth
        - *.yaml
        - result.json
    - 00001_*
        - *.pth
        - *.yaml
        - result.json
"""

import argparse
import csv
import json
import os

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import time
import sys
import hashlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--base_dir", default="../outputs/cifar10/attacks/",
                        help="Directory of an attack where all results are listed."
                             "the attack.")
    return parser.parse_args()


def file_with_suffix_exists(dirname, suffix, not_contains="", raise_error=False):
    for file in os.listdir(dirname):
        if file.endswith(suffix) and (not not_contains in file or len(not_contains) == 0):
            return os.path.abspath(os.path.join(dirname, file))
    if raise_error:
        raise FileNotFoundError(f"No file found with suffix '{suffix}' in '{dirname}")
    return False


def parse_data(base_dir, name):
    results = {}
    print(f"Parse args {base_dir}")

    folders = os.listdir(base_dir)
    folders.sort()

    for folder in folders:
        folder = os.path.join(base_dir, folder)

        if not os.path.isdir(folder):
            continue

        config_filename = file_with_suffix_exists(folder, suffix=".yaml", not_contains=name.split("_")[0])
        result_filename = file_with_suffix_exists(folder, suffix="result.json")
        try:
            if config_filename and result_filename:
                key = os.path.split(config_filename)[1]

                with open(result_filename, "r") as f:
                    data = json.load(f)
                if key in results.keys():
                    print(f"[WARNING] Duplicate result for '{key}' in folder '{folder}'")

                results.setdefault(key, data)
        except Exception as e:
            print(f"ERROR loading folder '{folder}': {e}")
            pass
    return results


def to_csv(data: dict, output_filename):
    all_keys = list(data.keys())
    all_keys.sort()

    with open(output_filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')

        key_row, value_row = [], []
        for key in all_keys:
            value = data[key]
            try:
                v = f"{value['wm_acc_after']:.4f}/{value['test_acc_after']:.4f}/{value['time']:.1f}"
                value_row.append(v)
                key_row.append(key)
            except Exception as e:
                print(f"Could not write value to file! {e}")
                pass

        spamwriter.writerow(key_row)
        spamwriter.writerow(value_row)
    print(f"Saved file '{output_filename}' to csv!")
    return os.path.split(output_filename)[1]


def main():
    args = parse_args()
    print(args)

    output_dir = os.path.join(args.base_dir, "zz_results_csv")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    files = []
    for attack_folder in os.listdir(args.base_dir):
        basedir = os.path.join(args.base_dir, attack_folder)
        if not os.path.isdir(basedir):
            continue
        base, name = os.path.split(basedir)
        data = parse_data(basedir, name=name)
        files.append(to_csv(data, output_filename=os.path.join(output_dir, f"{name}_summary.csv")))
    print(f"Wrote to the following files: {files}")


if __name__ == "__main__":
    main()
