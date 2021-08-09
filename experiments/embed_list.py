""" This script runs the "embed.py" script on a list of watermarking configurations
"""

import argparse
import os

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import time
from math import ceil
from multiprocessing import Process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wm_configs', type=str, default='../experiments/configs/cifar10_embed',
                        help="Path to config file for the watermarking scheme.")
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument("--pretrained_dir", default="outputs/cifar10/wm/pretrained/resnet/00000_pretrained")
    parser.add_argument("--filename", type=str, default="best.pth", help="Filepath to the pretrained model.")
    parser.add_argument("-n", "--num_processes", type=int, default=1, help="Number of concurrent processes.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use. Defaults to GPU with least memory.")
    return parser.parse_args()


# Hardcoded GPUs to run these experiments on.
GPUS = [2, 3]


def run_tasks(wm_configs, pretrained_dir, filename, index):
    for wm_config in wm_configs:
        print(f"[Process {index}] Embedding {wm_config} ..")
        os.system(f"cd .. && python embed.py -w {wm_config} --gpu {GPUS[index]} --pretrained_dir {pretrained_dir} "
                  f"--filename {filename}")
        print(f"[Process {index}] Done embedding {wm_config}!")


def main():
    args = parse_args()

    all_configs = [os.path.abspath(os.path.join(args.wm_configs, x)) for x in os.listdir(args.wm_configs)]
    n = len(all_configs)
    num_processes = args.num_processes

    processes = []
    for m in range(num_processes):
        size = int(ceil(n / num_processes))
        p = Process(target=run_tasks,
                    args=(all_configs[size * m:size * (m + 1)], args.pretrained_dir, args.filename, m))
        p.daemon = True  # Process dies with its parent.
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("Finished!")


if __name__ == "__main__":
    main()
