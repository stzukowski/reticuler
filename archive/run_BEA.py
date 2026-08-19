#!/usr/bin/env -S conda run -n reticuler --no-capture-output python
"""Template for running batches of Backward Evolution Algorithm (BEA) experiments
in parallel, natively in Python.

For each (ETA_ORIGINAL, SHIELDS_ORIGINAL) pair, clips the corresponding
forward-growth tree (copied from SOURCE_DIRECTORY) to CLIP_HEIGHT, then runs
the BEA once per eta in ETA_ORIGINAL + ETA_TO_ADD (in tenths), all etas
across all ETA_ORIGINALS/SHIELDS_ORIGINALS
sharing one process pool bounded by MAX_PARALLEL concurrent (via a semaphore).

Copy this file into a fresh experiment working directory and edit the
BEA SETTINGS section below.

Run with:
( ./run_BEA.py & )
(the shebang above assumes a conda env named "reticuler" — edit "-n reticuler" if yours is named differently)

Kill all with:
pkill -f '^ret_back'
"""

import sys
from pathlib import Path

# send stdout/stderr to <script_name>.log
sys.stdout = sys.stderr = open(Path(__file__).with_suffix(".log"), "w", buffering=1)

import multiprocessing as mp
import os
import shutil
import numpy as np

from reticuler.user_interface import clip_ret, runner

MAX_PARALLEL = 20
################ BEA SETTINGS ################
DS = 0.01  # spatial step
BACK_FORTH_STEPS_THRESH = 1  # n steps backward, then n forward, then compare graphs

# Where the original forward-growth trees live, relative to $HOME.
SOURCE_DIRECTORY = (
    "pro/rete/archive_loc/misc/4_crit_shields_param/1_/0_growth/"
)
CLIP_HEIGHT = 2.5

ETA_ORIGINAL_LIST = [15]  # in tenths, so 15 = eta 1.5
ETA_TO_ADD = range(-5, 26)  # in tenths, eta_back_list = eta_original + ETA_TO_ADD
SHIELDS_ORIGINAL_LIST = [0, 50, 100, 150, 200]  # in thousandths, so 50 = shields 0.05
SHIELDS_BACK_LIST = [0, 50, 100, 150, 200]  # in thousandths, so 50 = shields 0.05

def create_exp_dir_original_tree(source_file):
    """Create exp_dir, copy original_tree.json and clip it to CLIP_HEIGHT"""
    # create exp_dir
    exp_dir = f"original_{source_file}"
    Path(exp_dir).mkdir(exist_ok=True)
    # copy original_tree
    source_json = ( Path.home() / SOURCE_DIRECTORY / f"{source_file}.json" )
    shutil.copy(source_json, Path(exp_dir) / "original_tree.json")
    # clip original_tree
    original_tree = f"{exp_dir}/original_tree"
    clip_ret.main(argv=[original_tree, "-H", str(CLIP_HEIGHT), "-out", original_tree])

def prepare_eta_back_scan(source_file, output_prefix, eta_back_range, trimmer_params):
    """Prepare a scan over eta_back_range.

    Builds the EXPERIMENTS list 
    (source_file, output_prefix, params, remaining) 
    of tasks to run BEA on original_tree over eta_back_range.
    """
    experiments = []
    # counter to know when to plot BEA_results
    remaining = mp.Value("i", len(eta_back_range))
    for eta_back in eta_back_range:
        params = {
            "input_file": f"original_{source_file}/original_tree",
            "output_file": f"original_{source_file}/{output_prefix}eta_{eta_back:02d}",
            "BEA_params": {"back_forth_steps_thresh": BACK_FORTH_STEPS_THRESH},
            "trimmer_params": {
                "eta": eta_back / 10,
                "ds": DS,
                **trimmer_params,
            },
        }
        experiments.append((source_file, output_prefix, params, remaining))
    return experiments

EXPERIMENTS = []
for shields_original in SHIELDS_ORIGINAL_LIST:
    for eta_original in ETA_ORIGINAL_LIST:
        source_file = f"eta{eta_original:02d}_shields{shields_original:03d}"
        create_exp_dir_original_tree(source_file)

        eta_back_range = eta_original + np.array(ETA_TO_ADD)
        for shields_back in SHIELDS_BACK_LIST:
            output_prefix = f"shields{shields_back:03d}/"
            Path(f"original_{source_file}/{output_prefix}").mkdir(exist_ok=True)
            EXPERIMENTS.append(prepare_eta_back_scan(source_file, output_prefix, eta_back_range, {"crit_shields_param": shields_back / 1000}))


def main():
    runner.copy_reticuler_temp()
    runner.run_batch(MAX_PARALLEL, EXPERIMENTS, is_backward=True)

    os._exit(0)


if __name__ == "__main__":
    main()
