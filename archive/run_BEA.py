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

import logging
import multiprocessing as mp
import os
import shutil
import time

from reticuler.user_interface import clip_ret, plot_ret, runner

################ BEA SETTINGS ################
DS = 0.01  # spatial step
BACK_FORTH_STEPS_THRESH = 1  # n steps backward, then n forward, then compare graphs

CLIP_HEIGHT = 2.5

# Where the original forward-growth trees live, relative to $HOME.
SOURCE_DIRECTORY = (
    "pro/rete/archive_loc/misc/4_crit_shields_param/1_/0_growth/"
)

ETA_ORIGINALS = [15]  # in tenths, so 15 = eta 1.5
ETA_TO_ADD = range(-5, 26)  # in tenths
SHIELDS_ORIGINALS = [0, 0.05, 0.1, 0.15, 0.2]

MAX_PARALLEL = 20


def prepare_eta_original(eta_original, shields_original):
    """Set up directories and params for a pair (eta_original, shields_original).

    1. Copy original_tree.json and clip it to CLIP_HEIGHT.
    2. Build the list (exp_dir, shields_dir, params, remaining) of tasks to 
    run BEA on original_tree:
    - one shields<j_shields>/ subdirectory per j_shields in SHIELDS_ORIGINALS
    - each holding every j_eta (from ETA_ORIGINAL + ETA_TO_ADD)
    - ``remaining`` is a shared "etas still running in this shields_dir" counter,
    for `run_experiment_and_maybe_plot` to plot once it hits zero
    """
    shields_tag = f"{round(shields_original * 1000):03d}"
    exp_dir = f"original_eta{eta_original:02d}_shields{shields_tag}"
    Path(exp_dir).mkdir(exist_ok=True)

    # copy original_tree
    source_json = (
        Path.home()
        / SOURCE_DIRECTORY
        / f"eta{eta_original:02d}_shields{shields_tag}.json"
    )
    shutil.copy(source_json, Path(exp_dir) / "original_tree.json")

    # clip original_tree
    original_tree = f"{exp_dir}/original_tree"
    clip_ret.main(argv=[original_tree, "-H", str(CLIP_HEIGHT), "-out", original_tree])

    # subdir for each shields_back, with its own tasks + "etas remaining" counter
    tasks = []
    for shields_back in SHIELDS_ORIGINALS:
        shields_dir = f"{exp_dir}/shields{shields_back}"
        Path(shields_dir).mkdir(exist_ok=True)
        remaining = mp.Value("i", len(ETA_TO_ADD))
        for eta_to_add in ETA_TO_ADD:
            j_eta_back = eta_original + eta_to_add
            params = {
                "input_file": original_tree,
                "output_file": f"{shields_dir}/eta{j_eta_back:02d}",
                "BEA_params": {"back_forth_steps_thresh": BACK_FORTH_STEPS_THRESH},
                "trimmer_params": {
                    "eta": j_eta_back / 10,
                    "ds": DS,
                    "crit_shields_param": shields_back,
                },
            }
            tasks.append((exp_dir, shields_dir, params, remaining))
    return tasks


def plot_shields_dir(exp_dir, shields_dir):
    """Plot the BEA eta* scan for a finished shields_dir, via plot_ret's -back
    (`original_tree.json` is expected in `exp_dir`)."""
    cwd0 = os.getcwd()
    os.chdir(exp_dir)
    try:
        shields_prefix = os.path.relpath(shields_dir, exp_dir) + os.sep
        plot_ret.plot_back(shields_prefix)
    except Exception:
        logging.getLogger("reticuler").exception("Plotting failed for %s", shields_dir)
    finally:
        os.chdir(cwd0)


def run_experiment_and_maybe_plot(exp_dir, shields_dir, params, remaining, sem):
    """Run one eta_back experiment, then -- if it happens to be the last one
    to finish for its shields_dir -- plot that shields_dir's BEA scan.

    ``remaining`` is a shared counter (one per shields_dir group, created in
    main()): whichever worker decrements it to zero is the one that plots.
    """
    runner.run_bounded(runner.run_experiment_back, params, sem)

    with remaining.get_lock():
        remaining.value -= 1
        is_last = remaining.value == 0

    if is_last:
        plot_shields_dir(exp_dir, shields_dir)


def main():
    runner.copy_reticuler_temp()

    # One globally-ordered task list (group by group), built directly from
    # prepare_eta_original's own flat task lists.
    tasks = [
        task
        for eta_original in ETA_ORIGINALS
        for shields_original in SHIELDS_ORIGINALS
        for task in prepare_eta_original(eta_original, shields_original)
    ]

    # One process per eta (never reused), bounded to MAX_PARALLEL concurrent.
    # main() is the only thing calling sem.acquire(), so launch order is
    # exactly the order of `tasks` above.
    sem = mp.Semaphore(MAX_PARALLEL)
    for exp_dir, shields_dir, params, remaining in tasks:
        sem.acquire()
        p = mp.Process(
            target=run_experiment_and_maybe_plot,
            args=(exp_dir, shields_dir, params, remaining, sem),
        )
        p.start()
        time.sleep(0.001)

    # Exit once everything is launched; worker processes are not daemonic, so
    # they keep running independently.
    os._exit(0)


if __name__ == "__main__":
    main()
