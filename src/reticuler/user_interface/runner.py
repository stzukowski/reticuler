"""Helpers for running system/system_back in CLI and batches of reticuler experiments in parallel.

Functions:
    run_experiment
    run_experiment_back
    run_batch
    copy_reticuler_temp

"""

import logging
import multiprocessing as mp
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from setproctitle import setproctitle

import reticuler
from reticuler.system import System
from reticuler.backward_evolution.system_back import BackwardSystem
from reticuler.user_interface import graphics, plot_ret


def _log_to_file(log_path):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    root_logger.addHandler(handler)

    # route warnings.warn() (e.g. numpy) through the same handler
    logging.captureWarnings(True)


def run_experiment(params, log_to_file=True):
    """Construct (or import) and evolve a System, optionally saving a final
    plot. If log_to_file, attach a per-process FileHandler."""
    name = params.get("output_file") or params.get("input_file")
    setproctitle(f"ret:{name}")  # change name of the process in Linux ps

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}  Starting experiment: {name}", flush=True)

    # switch on logging to file
    if log_to_file:
        suffix = "_cont" if params.get("input_file") and not params.get("output_file") else ""
        _log_to_file(f"{name}{suffix}.log") # set logs to file

    # run growth and plot at the end
    try:
        if params.get("input_file"):
            system = System.import_json(input_file=params["input_file"])
            if params.get("output_file"):
                system.exp_name = params["output_file"]
        else:
            system = System.construct(params)

        system.evolve()

        if params.get("final_plot", True):
            fig, ax = plt.subplots()
            graphics.plot_tree(system=system, ax=ax)
            fig.savefig(system.exp_name + ".jpg", bbox_inches="tight", dpi=300)
    except Exception:
        logging.getLogger("reticuler").exception("Experiment %s failed", name)
        raise


def run_experiment_back(params, log_to_file=True):
    """Construct (or import) and run a BackwardSystem's BEA. If log_to_file,
    also attach a per-process FileHandler."""
    name = params.get("output_file") or params.get("input_file")
    setproctitle(f"ret_back:{name}")  # change name of the process in Linux ps

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}  Starting BEA experiment: {name}", flush=True)

    # switch on logging to file
    if log_to_file:
        suffix = "_cont" if params.get("continuation_file") else ""
        _log_to_file(f"{name}{suffix}.log")

    # run BEA
    try:
        backward_system = BackwardSystem.construct(params)
        backward_system.run_BEA()
    except Exception:
        logging.getLogger("reticuler").exception("BEA experiment %s failed", name)
        raise


def _run_bounded(target, params, sem):
    """Run ``target(params)`` (``run_experiment``/``run_experiment_back``), then
    release ``sem`` so a caller throttling other processes with a
    multiprocessing.Semaphore can start the next one."""
    name = params.get("output_file") or params.get("input_file")
    try:
        target(params)
    except Exception:
        print(f"Experiment {name} failed.", flush=True)
    finally:
        sem.release()


def _reap_finished(procs):
    """Join and close any of ``procs`` that have exited, returning the still-running ones."""
    still_running = []
    for p in procs:
        if p.exitcode is None:
            still_running.append(p)
        else:
            p.join()
            p.close()
    return still_running


def _plot_shields_dir(source_file, output_prefix):
    """Plot the BEA analysis for a finished output_prefix, via plot_ret's -back."""
    cwd0 = os.getcwd()
    os.chdir(f"original_{source_file}")
    try:
        plot_ret.plot_back(output_prefix)
    except Exception:
        logging.getLogger("reticuler").exception("Plotting failed for %s", f"original_{source_file}/{output_prefix}")
    finally:
        os.chdir(cwd0)


def _run_experiment_back_and_maybe_plot(source_file, output_prefix, params, remaining, sem):
    """Run one eta_back experiment, then -- if it's the last one for its
    output_prefix -- plot that output_prefix's BEA scan.

    ``remaining`` is a shared counter (one per output_prefix group): whichever
    worker decrements it to zero is the one that plots.
    """
    _run_bounded(run_experiment_back, params, sem)

    with remaining.get_lock():
        remaining.value -= 1
        is_last = remaining.value == 0

    if is_last:
        _plot_shields_dir(source_file, output_prefix)


def run_batch(max_parallel, experiments, is_backward=False):
    """Launch one ``mp.Process`` per item in ``experiments`` (never reused),
    bounded to ``max_parallel`` concurrent.

    Each item is a params dict run via ``run_experiment``.

    If ``is_backward``, each item is a ``(source_file, output_prefix, params,
    remaining)`` tuple run via ``run_experiment_back`` (see
    ``_run_experiment_back_and_maybe_plot``).
    """
    running = []
    sem = mp.Semaphore(max_parallel)
    for item in experiments:
        sem.acquire()
        running = _reap_finished(running)  # sem freed up -> something just exited
        if is_backward:
            target, args = _run_experiment_back_and_maybe_plot, (*item, sem)
        else:
            target, args = _run_bounded, (run_experiment, item, sem)
        p = mp.Process(target=target, args=args)
        p.start()
        running.append(p)
        time.sleep(0.001)


def copy_reticuler_temp(experiment_dir=None):
    """Copy the reticuler source (.py files only) into
    ``<experiment_dir>/reticuler_temp`` (default: cwd), for reproducibility."""
    experiment_dir = Path.cwd() if experiment_dir is None else Path(experiment_dir)
    shutil.copytree(
        Path(reticuler.__file__).parent,
        experiment_dir / "reticuler_temp",
        ignore=_ignore_non_py,
        dirs_exist_ok=True,
    )


def _ignore_non_py(directory, names):
    """shutil.copytree ignore callback: keep only .py files (and dirs to recurse into)."""
    ignored = []
    for name in names:
        if (Path(directory) / name).is_dir():
            if name == "__pycache__" or name.startswith("."):
                ignored.append(name)
        elif not name.endswith(".py"):
            ignored.append(name)
    return ignored
