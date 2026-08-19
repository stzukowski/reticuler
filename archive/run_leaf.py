#!/bin/bash
"exec" "conda" "run" "-n" "reticuler" "--no-capture-output" "python" "$0" "$@"

"""Template for running batches of reticuler experiments in parallel.

Copy this file into a fresh experiment working directory and run with:
( ./run.py & )
(the shebang above assumes a conda env named "reticuler";
edit "-n reticuler" if yours is named differently)

Kill all with:
pkill -f '^ret'
"""

import sys
from pathlib import Path

# send stdout/stderr to <script_name>.log
sys.stdout = sys.stderr = open(Path(__file__).with_suffix(".log"), "w", buffering=1)

import os
import numpy as np

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
PDE_SOLVER = "FreeFEM_ThinFingers_Boundary"

# initial_condition:
INITIAL_CONDITION = 351
RADIUS = 1
INITIAL_LENGTHS = [0.1]
SEEDS_PHI = 1

DS = 0.005  # spatial step
MAX_APPROXIMATION_STEP = 1  # for ModifiedEulerMethod

# growth limit (when to stop the simulation)
# 0: number of steps, 1: height, 2: network length, 3: evolution time
GROWTH_THRESH_TYPE = 1
GROWTH_THRESH = 10  # Poisson: 13 for pictures, 9 for the BEA

# bifurcationType:
# 0 - no bifurcation
# 1 - bifurcation when a1>bifurcation_treshold
# 2 - bifurcation when a3/a1<bifurcation_treshold
BIFURCATION_TYPE = 0

MAX_PARALLEL = 10

# morpher parameters
V_RIM = 1
NSEEDS_LIST = [2, 3, 5, 8]
SIGMOID_RATE_LIST = [1, 2, 5, 10]
SIGMOID_SHIFT_LIST = [0.5, 1, 2, 3]

# One params dict per ETAS value.
EXPERIMENTS = [
    {
        "output_file": f"sig_nseeds{NSEEDS}_sigshift{SIGMOID_SHIFT}_sigrate{SIGMOID_RATE}",
        "growth_params": {
            "growth_thresh_type": GROWTH_THRESH_TYPE,
            "growth_thresh": GROWTH_THRESH,
        },
        "initial_condition": INITIAL_CONDITION,
        "kwargs_box": {
            "seeds_phi": SEEDS_PHI,
            "initial_lengths": INITIAL_LENGTHS,
            "radius": RADIUS,
            "angular_width": 2 * np.pi / NSEEDS,
        },
        "pde_solver": PDE_SOLVER,
        "pde_solver_params": {
            "equation": EQUATION,
            "ds": DS,
            "bifurcation_type": BIFURCATION_TYPE,
            "is_script_saved": IS_SCRIPT_SAVED,
            "v_rim": 1,
            "response_func": "linear_sigmoid",
            "response_func_params": {
                "sig_shift": SIGMOID_SHIFT,
                "sig_rate": SIGMOID_RATE,
            },
        },
        "extender_params": {
            "is_reconnecting": IS_RECONNECTING,
            "max_approximation_step": MAX_APPROXIMATION_STEP,
        },
    }
    for SIGMOID_SHIFT in SIGMOID_SHIFT_LIST
    for SIGMOID_RATE in SIGMOID_RATE_LIST
    for NSEEDS in NSEEDS_LIST
]


def main():
    runner.copy_reticuler_temp()
    runner.run_batch(MAX_PARALLEL, EXPERIMENTS)

    # Exit once everything is launched.
    os._exit(0)


if __name__ == "__main__":
    main()
