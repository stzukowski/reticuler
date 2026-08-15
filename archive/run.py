#!/usr/bin/env -S conda run -n reticuler --no-capture-output python
"""Template for running batches of reticuler experiments in parallel.

Define a parameter sweep as plain Python
dicts, run each experiment in its own process (bounded to `MAX_PARALLEL`
concurrent via a semaphore), with each process's log going to its own
`<output_file>.log` file and the reticuler source snapshot copied into
`reticuler_temp/` for reproducibility.

Copy this file into a fresh experiment working directory and edit the
GROWTH SETTINGS section below.

Run with:
( ./run.py & )
(the shebang above assumes a conda env named "reticuler" — edit "-n reticuler" if yours is named differently)

Kill all with:
pkill -f '^ret'
"""

import sys
from pathlib import Path

# send stdout/stderr to <script_name>.log
sys.stdout = sys.stderr = open(Path(__file__).with_suffix(".log"), "w", buffering=1)

import multiprocessing as mp
import os

from reticuler.user_interface import runner

################ GROWTH SETTINGS ################
IS_SCRIPT_SAVED = 0
IS_RECONNECTING = 0
# If you want to continue simulations (instead of creating new) 
# replace "output_file" by "input_file" in EXPERIMENTS below.

# equation can have 2 values:
# 0 - Laplace equation
# 1 - Poisson equation
EQUATION = 0
PDE_SOLVER = "FreeFEM_ThinFingers"

# initial_condition:
INITIAL_CONDITION = 100
BOX_HEIGHT = 50
BOX_WIDTH = 2
SEEDS_X = [1.5]
INITIAL_LENGTHS = [0.01]

DS = 0.01  # spatial step
MAX_APPROXIMATION_STEP = 3  # for ModifiedEulerMethod

# growth limit (when to stop the simulation)
# 0: number of steps, 1: height, 2: network length, 3: evolution time
GROWTH_THRESH_TYPE = 1
GROWTH_THRESH = 2.5  # Poisson: 13 for pictures, 9 for the BEA

# bifurcationType:
# 0 - no bifurcation
# 1 - bifurcation when a1>bifurcation_treshold
# 2 - bifurcation when a3/a1<bifurcation_treshold
BIFURCATION_TYPE = 1

ETAS = [1.5]  # e.g. [5, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60]

MAX_PARALLEL = 5


# One params dict per ETAS value.
EXPERIMENTS = [
    {
        "output_file": f"eta{int(10*ETA):02d}",
        "growth_params": {
            "growth_thresh_type": GROWTH_THRESH_TYPE,
            "growth_thresh": GROWTH_THRESH,
        },
        "initial_condition": INITIAL_CONDITION,
        "kwargs_box": {
            "seeds_x": SEEDS_X,
            "initial_lengths": INITIAL_LENGTHS,
            "height": BOX_HEIGHT,
            "width": BOX_WIDTH,
        },
        "pde_solver": PDE_SOLVER,
        "pde_solver_params": {
            "equation": EQUATION,
            "eta": ETA,
            "ds": DS,
            "bifurcation_type": BIFURCATION_TYPE,
            "is_script_saved": IS_SCRIPT_SAVED,
        },
        "extender_params": {
            "is_reconnecting": IS_RECONNECTING,
            "max_approximation_step": MAX_APPROXIMATION_STEP,
        },
    }
    for ETA in ETAS
]


def main():
    runner.copy_reticuler_temp()

    output_files = ", ".join(
        exp.get("output_file") or exp.get("input_file") for exp in EXPERIMENTS
    )
    print(f"Starting {len(EXPERIMENTS)} experiment(s): {output_files}")

    # One process per experiment (never reused), bounded to MAX_PARALLEL concurrent.
    sem = mp.Semaphore(MAX_PARALLEL)
    for params in EXPERIMENTS:
        sem.acquire()
        p = mp.Process(target=runner.run_bounded, args=(runner.run_experiment, params, sem))
        p.start()

    # Exit once everything is launched.
    os._exit(0)


if __name__ == "__main__":
    main()
