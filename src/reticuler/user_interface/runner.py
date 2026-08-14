"""Helpers for running batches of reticuler experiments in parallel.

Workers are expected to run under Pool(..., maxtasksperchild=1), so each
worker process handles exactly one experiment and is then discarded -- the
FileHandler added below never needs to be removed, since it dies with the
process it's attached to.

Functions:
    build_argv
    run_experiment
    build_argv_back
    run_experiment_back
    ignore_non_py

"""

import json
import logging
from datetime import datetime
from pathlib import Path

from reticuler.user_interface import reticulate, reticulate_back


def _add_file_handler(log_path):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    root_logger.addHandler(handler)


def build_argv(params):
    """Translate an experiment's parameter dict into a reticulate CLI-style argv list."""
    if params.get("continued"):
        argv = ["-in", params["output_file"]]
        if params.get("output_file_override"):
            argv += ["-out", params["output_file_override"]]
    else:
        argv = [
            "-out",
            params["output_file"],
            "--growth_params",
            json.dumps(params["growth_params"]),
            "-ic",
            str(params["initial_condition"]),
            "--kwargs_box",
            json.dumps(params["kwargs_box"]),
            "--pde_solver",
            params.get("pde_solver"),
            "--pde_solver_params",
            json.dumps(params["pde_solver_params"]),
            "--extender_params",
            json.dumps(params.get("extender_params")),
        ]
    argv.append("--final_plot")
    return argv


def run_experiment(params):
    """Worker entry point: configure per-process file logging, then run reticulate.main()."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}  Starting experiment: {params['output_file']}", flush=True)

    suffix = "_cont" if params.get("continued") else ""
    _add_file_handler(f"{params['output_file']}{suffix}.log")

    try:
        reticulate.main(argv=build_argv(params))
    except Exception:
        logging.getLogger("reticuler").exception(
            "Experiment %s failed", params["output_file"]
        )
        raise


def build_argv_back(params):
    """Translate a BEA experiment's parameter dict into a reticulate_back CLI-style argv list."""
    argv = [params["input_file"], "-out", params["output_file"]]
    if params.get("continuation_file"):
        argv += ["-cont", params["continuation_file"]]
    else:
        argv += [
            "--BEA_params",
            json.dumps(params["BEA_params"]),
            "--trimmer_params",
            json.dumps(params["trimmer_params"]),
        ]
    return argv


def run_experiment_back(params):
    """Worker entry point: configure per-process file logging, then run reticulate_back.main()."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}  Starting BEA experiment: {params['output_file']}", flush=True)

    suffix = "_cont" if params.get("continuation_file") else ""
    _add_file_handler(f"{params['output_file']}{suffix}.log")

    try:
        reticulate_back.main(argv=build_argv_back(params))
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
