#!/usr/bin/env -S conda run -n reticuler --no-capture-output python
"""Template for running batches of Backward Evolution Algorithm (BEA) experiments
in parallel, natively in Python.

For each (ETA_ORIGINAL, SHIELDS_ORIGINAL) pair, clips the corresponding
forward-growth tree (copied from SOURCE_DIRECTORY) to CLIP_HEIGHT, then runs
the BEA once per eta in ETA_ORIGINAL + ETA_TO_ADD (in tenths), all etas 
across all ETA_ORIGINALS/SHIELDS_ORIGINALS
sharing one process pool bounded by MAX_PARALLEL concurrent (via a semaphore)
-- same effect as run.sh's wait_till-based throttling.

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

from reticuler.user_interface import clip_ret, runner

################ BEA SETTINGS ################
DS = 0.01  # spatial step
BACK_FORTH_STEPS_THRESH = 1  # n steps backward, then n forward, then compare graphs

CLIP_HEIGHT = 2.5

# Where the original forward-growth trees (etaXX.json) live, relative to $HOME.
SOURCE_DIRECTORY = (
    "pro/rete/archive_loc/misc/4_crit_shields_param/0_3_growth/"
)

ETA_ORIGINALS = [15]  # e.g. list(range(10, 61, 5)); in tenths, so 15 = eta 1.5
ETA_TO_ADD = range(-10, 11)  # in tenths
SHIELDS_ORIGINALS = [0, 0.2, 0.4, 0.6]

MAX_PARALLEL = 20


def prepare_eta_original(eta_original, shields_original):
    """Set up eta_original<NN>_shields<NN>/ (clip the source tree into it) and
    build the list of BEA experiment param dicts to run against that source
    tree -- one shields<j_shields>/ subdirectory per j_shields in
    SHIELDS_ORIGINALS (tried as the trimmer's crit_shields_param, independent
    of which shields value grew the tree), each holding every j_eta (from
    ETA_ORIGINAL + ETA_TO_ADD) trimmed with that j_shields."""
    shields_tag = f"{round(shields_original * 10):02d}"
    exp_dir = f"original_eta{eta_original:02d}_shields{shields_tag}"
    Path(exp_dir).mkdir(exist_ok=True)

    source_json = (
        Path.home()
        / SOURCE_DIRECTORY
        / f"eta{eta_original:02d}_shields{shields_tag}.json"
    )
    shutil.copy(source_json, Path(exp_dir) / "original_tree.json")

    original_tree = f"{exp_dir}/original_tree"
    clip_ret.main(argv=[original_tree, "-H", str(CLIP_HEIGHT), "-out", original_tree])

    experiments = []
    for j_shields in SHIELDS_ORIGINALS:
        shields_dir = f"{exp_dir}/shields{j_shields}"
        Path(shields_dir).mkdir(exist_ok=True)
        for eta_to_add in ETA_TO_ADD:
            j_eta = eta_original + eta_to_add
            experiments.append(
                {
                    "input_file": original_tree,
                    "output_file": f"{shields_dir}/eta{j_eta:02d}_shields{j_shields}",
                    "BEA_params": {"back_forth_steps_thresh": BACK_FORTH_STEPS_THRESH},
                    "trimmer_params": {
                        "eta": j_eta / 10,
                        "ds": DS,
                        "crit_shields_param": j_shields,
                    },
                }
            )
    return experiments


def main():
    runner.copy_reticuler_temp()

    experiments = [
        exp
        for eta_original in ETA_ORIGINALS
        for shields_original in SHIELDS_ORIGINALS
        for exp in prepare_eta_original(eta_original, shields_original)
    ]

    output_files = ", ".join(
        exp.get("output_file") or exp.get("input_file") for exp in experiments
    )
    print(f"Starting {len(experiments)} BEA experiment(s): {output_files}")

    # One process per experiment (never reused), bounded to MAX_PARALLEL concurrent.
    sem = mp.Semaphore(MAX_PARALLEL)
    for params in experiments:
        sem.acquire()
        p = mp.Process(
            target=runner.run_bounded, args=(runner.run_experiment_back, params, sem)
        )
        p.start()

    # Exit once everything is launched.
    os._exit(0)


if __name__ == "__main__":
    main()
