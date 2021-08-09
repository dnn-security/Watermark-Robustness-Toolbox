""" This script runs the "steal.py" script on a list of watermarked models.
It expects a directory as with the following structure:

####
- base_dir
    - scheme1
        - 00000_scheme1
            - scheme1.yaml
            - ...
            - best.pth
    - scheme2
        - 00000_scheme2
            - scheme2.yaml
            - ...
            - best.pth
    - scheme3
        - ...
####

The script uses the specified attack against schemes found in this folder.
"""

import argparse
import hashlib
import os

import mlconfig
import numpy as np

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import time
from math import ceil
from multiprocessing import Process


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--attack_config', type=str,
                        default='configs/imagenet/attack_configs/combined_attack.yaml',
                        help="Path to config file for the attack.")
    parser.add_argument("-m", "--base_dir", default="../outputs/imagenet/wm/",
                        help="Directory where all data from the watermarking scheme"
                             " can be found required to run the attack.")
    parser.add_argument("-n", "--num_processes", type=int, default=1, help="Number of concurrent processes.")
    parser.add_argument('-l', '--list', nargs='*', help='GPUs to run on.', required=False)
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    return parser.parse_args()


def file_with_suffix_exists(dirname, suffix, not_contains="", raise_error=False):
    for file in os.listdir(dirname):
        if file.endswith(suffix) and (not not_contains in file or len(not_contains) == 0):
            return os.path.abspath(os.path.join(dirname, file))
    if raise_error:
        raise FileNotFoundError(f"No file found with suffix '{suffix}' in '{dirname}")
    return False


def run_tasks(attack_config, wm_paths, index, GPUS):
    for wm_path in wm_paths:
        print(f"[Process {index}] Stealing '{wm_path}' ..")
        cmd = f"cd .. && python steal.py -a {attack_config} -w {wm_path} --gpu {GPUS[index % len(GPUS)]} --save"
        print(cmd)
        os.system(cmd)
        print(f"[Process {index}] Done stealing '{wm_path}'!")


def hash_file(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    args = parse_args()
    print(args)

    # Check the output directory for existing defense file hashes to avoid duplicates.
    attack_config = mlconfig.load(os.path.join("..", args.attack_config))
    attack_outdir = os.path.join("..", attack_config.output_dir)

    ''' We expect the following structure.
    # <base folder>
        - 00000_<attack>_<defense>
            - <attack>.yaml
            - result.json       
            - <defense>.yaml    <-- Compute the file hash if result.json exists. 
        - 00001_<attack>_<defense>
            ...
    '''
    existing_file_hashes = []   # Hashes for all defense files in the output directory.
    if os.path.isdir(attack_outdir):
        for folder in os.listdir(attack_outdir):
            full_path = os.path.join(attack_outdir, folder)
            defense_config_file = file_with_suffix_exists(full_path, suffix=".yaml", not_contains=attack_config.name)
            # Check if result.json and <defense>.yaml exist.
            if os.path.exists(os.path.join(full_path, 'result.json')) and defense_config_file:
                existing_file_hashes.append((os.path.split(defense_config_file)[1], hash_file(defense_config_file)))

    # Discover and validate models.
    wm_dirs = []
    for scheme_path in os.listdir(args.base_dir):
        scheme_path = os.path.join(args.base_dir, scheme_path)  # e.g., Adi, Zhang, ...
        for model_dir in os.listdir(scheme_path):
            full_path = os.path.join(scheme_path, model_dir)  # e.g., 00000_adi, 00000_zhang, ...

            if not os.path.isdir(full_path):
                continue

            pretrained_file = file_with_suffix_exists(full_path, '.pth')
            config_file = file_with_suffix_exists(full_path, '.yaml')

            # Assert that you can find '*.pth' and a '*.yaml' file in the input directory.
            if pretrained_file and config_file:
                hash = hash_file(config_file)
                if hash in [x[1] for x in existing_file_hashes]:
                    print(f"Skipping {config_file} because it already exists in the output ...")
                else:
                    print(f"Adding ({os.path.split(config_file)[1]}, {hash}) because it does not exists in the output ...")
                    wm_dirs.append(os.path.abspath(full_path))

    print(f"Queuing experiments for {[os.path.split(x)[1] for x in wm_dirs]} ...")
    shuffle_idx = np.arange(len(wm_dirs))
    np.random.shuffle(shuffle_idx)
    wm_dirs = [wm_dirs[i] for i in shuffle_idx]

    n = len(wm_dirs)
    num_processes = args.num_processes
    # assert num_processes <= len(GPUS), f"Not enough GPUs to run {num_processes} (found {len(GPUS)} GPUs)"
    print(f"Working on {n} files with {num_processes} processes!")

    if args.list is None:
        # Hardcoded set of GPUs to choose from.
        GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        GPUS = args.list

    processes = []
    for m in range(num_processes):
        size = int(ceil(n / num_processes))
        p = Process(target=run_tasks,
                    args=(args.attack_config,
                          wm_dirs[size * m:size * (m + 1)],
                          m,
                          GPUS))
        p.daemon = True  # Process dies with its parent.
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("Finished!")


if __name__ == "__main__":
    main()
