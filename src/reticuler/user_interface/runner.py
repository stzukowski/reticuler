"""Helpers for running batches of reticuler experiments in parallel.

Workers are expected to run under Pool(..., maxtasksperchild=1), so each
worker process handles exactly one experiment and is then discarded -- the
FileHandler added below never needs to be removed, since it dies with the
process it's attached to.

Functions:
    run_experiment
    run_experiment_back
    ignore_non_py

"""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from reticuler.system import System
from reticuler.backward_evolution.system_back import BackwardSystem
from reticuler.user_interface import graphics


def _add_file_handler(log_path):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    root_logger.addHandler(handler)


def run_experiment(params):
    """Worker entry point: configure per-process file logging, then construct
    (or import) and evolve a System."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}  Starting experiment: {params['output_file']}", flush=True)

    suffix = "_cont" if params.get("continued") else ""
    _add_file_handler(f"{params['output_file']}{suffix}.log")

    try:
        if params.get("continued"):
            system = System.import_json(input_file=params["output_file"])
            if params.get("output_file_override"):
                system.exp_name = params["output_file_override"]
        else:
            system = System.construct(params)

        system.evolve()

        if params.get("final_plot", True):
            fig, ax = plt.subplots()
            graphics.plot_tree(system=system, ax=ax)
            fig.savefig(system.exp_name + ".jpg", bbox_inches="tight", dpi=300)
    except Exception:
        logging.getLogger("reticuler").exception(
            "Experiment %s failed", params["output_file"]
        )
        raise


def run_experiment_back(params):
    """Worker entry point: configure per-process file logging, then construct
    (or import) and run a BackwardSystem."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}  Starting BEA experiment: {params['output_file']}", flush=True)

    suffix = "_cont" if params.get("continuation_file") else ""
    _add_file_handler(f"{params['output_file']}{suffix}.log")

    try:
        backward_system = BackwardSystem.construct(params)
        backward_system.run_BEA()
    except Exception:
        logging.getLogger("reticuler").exception(
            "BEA experiment %s failed", params["output_file"]
        )
        raise


def ignore_non_py(directory, names):
    """shutil.copytree ignore callback: keep only .py files (and dirs to recurse into)."""
    ignored = []
    for name in names:
        if (Path(directory) / name).is_dir():
            if name == "__pycache__" or name.startswith("."):
                ignored.append(name)
        elif not name.endswith(".py"):
            ignored.append(name)
    return ignored
