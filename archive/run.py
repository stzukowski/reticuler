"""Template for running batches of reticuler experiments in parallel.

Replaces the old run.sh workflow: define a parameter sweep as plain Python
dicts, run each experiment in its own worker process via
multiprocessing.Pool (bounded by `MAX_PARALLEL`), with each worker's log
going to its own `<output_file>.log` file and the reticuler source snapshot
copied into `reticuler_temp/` for reproducibility, same as run.sh did.

Copy this file into a fresh experiment working directory and edit the
GROWTH SETTINGS section below.

Run with:
( python run.py > run.log 2>&1 & )
"""

import multiprocessing as mp
import shutil
from pathlib import Path

import reticuler
from reticuler.user_interface import runner

################ GROWTH SETTINGS ################
IS_SCRIPT_SAVED = 0
IS_RECONNECTING = 0
IS_CONTINUED = 0

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
GROWTH_THRESH_TYPE = (
    1  # 0: number of steps, 1: height, 2: network length, 3: evolution time
)
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
        "continued": IS_CONTINUED,
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
    experiment_dir = Path.cwd()
    snapshot_dir = experiment_dir / "reticuler_temp"
    shutil.copytree(
        Path(reticuler.__file__).parent,
        snapshot_dir,
        ignore=runner.ignore_non_py,
        dirs_exist_ok=True,
    )

    output_files = ", ".join(exp["output_file"] for exp in EXPERIMENTS)
    print(f"Starting {len(EXPERIMENTS)} experiment(s): {output_files}")

    with mp.Pool(
        processes=min(MAX_PARALLEL, len(EXPERIMENTS)), maxtasksperchild=1
    ) as pool:
        pool.map(runner.run_experiment, EXPERIMENTS)


if __name__ == "__main__":
    main()
