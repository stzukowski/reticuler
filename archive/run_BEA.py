"""Template for running batches of Backward Evolution Algorithm (BEA) experiments
in parallel, natively in Python.

For each ETA_ORIGINAL, clips the corresponding forward-growth tree (copied from
SOURCE_DIRECTORY) to CLIP_HEIGHT, then runs the BEA once per eta in
ETA_ORIGINAL + ETA_TO_ADD (in tenths, matching the old run.sh convention),
all etas across all ETA_ORIGINALS sharing one multiprocessing.Pool bounded by
MAX_PARALLEL -- same effect as run.sh's wait_till-based throttling.

Copy this file into a fresh experiment working directory and edit the
BEA SETTINGS section below.

Run with:
( python run_BEA.py > run_BEA.log 2>&1 & )
"""

import multiprocessing as mp
import shutil
from pathlib import Path

import reticuler
from reticuler.user_interface import clip_ret, runner

################ BEA SETTINGS ################
DS = 0.01  # spatial step
BACK_FORTH_STEPS_THRESH = 1  # n steps backward, then n forward, then compare graphs

CLIP_HEIGHT = 2.5

# Where the original forward-growth trees (etaXX.json) live, relative to $HOME.
SOURCE_DIRECTORY = (
    "reticuler/archive/papers/2022SciRep/1_experiments/1_growth/2_Laplace"
)

ETA_ORIGINALS = [15]  # e.g. list(range(10, 61, 5)); in tenths, so 15 = eta 1.5
ETA_TO_ADD = range(-10, 11)  # in tenths

MAX_PARALLEL = 10


def prepare_eta_original(eta_original):
    """Set up eta_original<NN>/ (clip the source tree into it) and build the
    list of BEA experiment param dicts to run for that eta_original."""
    exp_dir = f"eta_original{eta_original:02d}"
    Path(exp_dir).mkdir(exist_ok=True)

    source_json = Path.home() / SOURCE_DIRECTORY / f"eta{eta_original:02d}.json"
    shutil.copy(source_json, Path(exp_dir) / "original_tree.json")

    original_tree = f"{exp_dir}/original_tree"
    clip_ret.main(argv=[original_tree, "-H", str(CLIP_HEIGHT), "-out", original_tree])

    return [
        {
            "input_file": original_tree,
            "output_file": f"{exp_dir}/eta{j_eta:02d}",
            "BEA_params": {"back_forth_steps_thresh": BACK_FORTH_STEPS_THRESH},
            "trimmer_params": {"eta": j_eta / 10, "ds": DS},
        }
        for eta_to_add in ETA_TO_ADD
        for j_eta in [eta_original + eta_to_add]
    ]


def main():
    experiment_dir = Path.cwd()
    snapshot_dir = experiment_dir / "reticuler_temp"
    shutil.copytree(
        Path(reticuler.__file__).parent,
        snapshot_dir,
        ignore=runner.ignore_non_py,
        dirs_exist_ok=True,
    )

    experiments = [
        exp
        for eta_original in ETA_ORIGINALS
        for exp in prepare_eta_original(eta_original)
    ]

    output_files = ", ".join(exp["output_file"] for exp in experiments)
    print(f"Starting {len(experiments)} BEA experiment(s): {output_files}")

    with mp.Pool(
        processes=min(MAX_PARALLEL, len(experiments)), maxtasksperchild=1
    ) as pool:
        pool.map(runner.run_experiment_back, experiments)


if __name__ == "__main__":
    main()
