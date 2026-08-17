r"""Post-processing and plotting of Backward Evolution Algorithm results.

Classes:
    BEAPostProcessor(exp_name, system, eta_range, eta_original=None, bif_type=1)

"""

import numpy as np
import matplotlib.pyplot as plt

from reticuler.backward_evolution.system_back import BackwardSystem
from reticuler.user_interface import graphics
from reticuler.utilities.misc import integerise


def _calculate_quantiles(data, q=0.25):
    """Calculate quantiles and width of a given data.

    Parameters
    -------
    data : array
    q : float, default 0.25

    Returns
    -------
    array
        A 1-7 array with:
        - quantiles of data of order 50%, q, 1-q
        - quantiles of abs(data) of order 50%, q, 1-q
        - width: (quant. of order q) - (quantiles of order 1-q)
        [q50, q32, q68, q50(abs), q32(abs), q68(abs), width]

    """
    q32 = np.quantile(data, q)
    q50 = np.quantile(data, 0.5)
    q68 = np.quantile(data, 1 - q)
    q32_abs = np.quantile(abs(data), q)
    q50_abs = np.quantile(abs(data), 0.5)
    q68_abs = np.quantile(abs(data), 1 - q)
    width = q68 - q32
    return np.array([q50, q32, q68, q50_abs, q32_abs, q68_abs, width])


def _norm_metric(metric_results):
    metric_results = metric_results - np.nanmin(metric_results)
    metric_results = metric_results / np.nanmax(metric_results)
    return np.copy(metric_results)


class BEAPostProcessor:
    """Gather, process, and plot BEA metrics across a range of eta* values.

    Expects ``exp_name + "eta{eta:02d}_back"`` BackwardSystem dumps (one per
    eta in ``eta_range``), as produced by `reticulate_back`, to exist on disk.

    Parameters
    -------
    exp_name : str
        Path/prefix shared by the BackwardSystem dumps; used as the figure title.
    system : System
        The original (forward) system the backward runs started from.
    eta_range : array
        Range of eta* values that were used for the backward runs.
    eta_original : float, default None
        eta of the original (forward) growth; marked with a vertical line in
        plots if given. If None, no such line is drawn.
    bif_type : {1, 2}, default 1
        Which bifurcation quantity (1: a1, 2: a3/a1) to highlight and use
        in the combined normalized-metric average.

    Attributes
    -------
    results, results_bif, results_branches_sep, results_bif_sep : array
        Populated by `gather()`.

    """

    def __init__(self, exp_name, system, eta_range, eta_original=None, bif_type=1):
        self.exp_name = exp_name
        self.system = system
        self.eta_range = eta_range
        self.eta_original = eta_original
        self.bif_type = bif_type

        self.results = None
        self.results_bif = None
        self.results_branches_sep = None
        self.results_bif_sep = None

    def gather(self, q=0.25):
        """Gather BEA metrics (flux coefficients, overshoot, angular
        deflection, length mismatch) across `eta_range`, populating
        `results`, `results_bif`, `results_branches_sep`, `results_bif_sep`.

        Parameters
        -------
        q : float, default 0.25
            Quantile order passed to `_calculate_quantiles`.

        """
        system = self.system
        eta_range = self.eta_range

        n_bifs = len(np.unique(system.network.branch_connectivity[:, 0]))
        results = np.zeros((3, eta_range.shape[0], 7))
        results_bif = np.zeros((3, eta_range.shape[0], 7))
        results_bif_sep = np.zeros((n_bifs, eta_range.shape[0], 5))
        results_branches_sep = np.zeros(
            (len(system.network.branches) - 1, eta_range.shape[0], 6)
        )
        for i, eta in enumerate(eta_range):
            file_name = self.exp_name + "eta{eta:02d}_back".format(eta=int(np.round(eta * 10)))
            backward_system = BackwardSystem.import_json(input_file=file_name, system=system)

            a1a2a3_coefficients = np.empty((0, 3), dtype=float)
            overshoot = []
            angular_deflection = []
            for jj, b_b in enumerate(backward_system.backward_branches[1:]):
                if len(b_b.steps) > 1:
                    a1a2a3_coefficients = np.vstack((a1a2a3_coefficients, b_b.flux_info))
                    overshoot = np.append(overshoot, b_b.overshoot)
                    angular_deflection = np.append(angular_deflection, b_b.angular_deflection)

                    results_branches_sep[jj, i, 1:4] = np.mean(b_b.flux_info, axis=0)
                    results_branches_sep[jj, i, 4] = np.mean(overshoot)
                    results_branches_sep[jj, i, 5] = np.mean(angular_deflection)
                    results_branches_sep[jj, i, 0] = b_b.ID
                else:
                    results_branches_sep[jj] = np.nan

            mask = backward_system.backward_bifurcations.flags == 3
            a1a2a3_coefficients_bif = backward_system.backward_bifurcations.flux_info
            length_mismatch = backward_system.backward_bifurcations.length_mismatch
            if sum(mask) < len(mask):
                print("eta={:.2f}, not all bifurcations reached...".format(eta))
                results_bif[:, i, :] = np.nan
                results[:, i, :] = np.nan
                results_bif_sep[:, i, :] = np.nan
            else:
                results_bif[0, i, :] = _calculate_quantiles(length_mismatch[mask])
                results_bif[1, i, :] = _calculate_quantiles(a1a2a3_coefficients_bif[mask, 0])
                results_bif[2, i, :] = _calculate_quantiles(
                    a1a2a3_coefficients_bif[mask, 2] / a1a2a3_coefficients_bif[mask, 0]
                )

                results[0, i, :] = _calculate_quantiles(
                    a1a2a3_coefficients[:, 1] / a1a2a3_coefficients[:, 0] ** 2, q
                )
                results[1, i, :] = _calculate_quantiles(overshoot, q)
                results[2, i, :] = _calculate_quantiles(angular_deflection, q)

                results_bif_sep[:, i, 1:4] = a1a2a3_coefficients_bif
                results_bif_sep[:, i, 4] = length_mismatch
                results_bif_sep[:, i, 0] = backward_system.backward_bifurcations.mother_IDs

        self.results = results
        self.results_bif = results_bif
        self.results_branches_sep = results_branches_sep
        self.results_bif_sep = results_bif_sep

    def normalized_metric_average(self):
        """Combine BEA metrics into a single normalized metric, log10-scaled
        and min-max normalized before averaging.

        Returns
        -------
        array
            The combined, normalized metric (one value per entry of `eta_range`).

        """
        if self.results is None:
            raise ValueError("No results gathered yet. Call gather() first.")
        results, results_bif = self.results, self.results_bif

        results_all_norm = []
        # local symmetry
        res = results[0]
        for i in [0, 6]:  # median, width
            results_all_norm.append(_norm_metric(np.log10(np.abs(res[:, i]))))

        # overshoot, angular deflection
        for res in results[1:]:
            for i in [3, 6]:  # median(abs), width
                results_all_norm.append(_norm_metric(np.log10(res[:, i])))

        # len. mismatch
        res = np.copy(results_bif[0])
        for i in [0, 6]:  # median(abs), width
            results_all_norm.append(_norm_metric(np.log10(res[:, i])))

        # bif IQR
        res = results_bif[1] if self.bif_type == 1 else results_bif[2]
        for i in [6]:  # width
            results_all_norm.append(_norm_metric(np.log10(res[:, i])))

        results_all_norm = np.array(results_all_norm)
        results_all_avg = np.mean(results_all_norm, axis=0)
        return _norm_metric(results_all_avg)

    def plot(self):
        """Full eta* scan summary plot: tree overview with per-branch/
        per-bifurcation markers, plus aggregated quantile/IQR panels for BEA
        metrics (length mismatch, a1, a3/a1, local symmetry a2/a1^2,
        overshoot, angular deflection) and a combined normalized-metric curve.

        Returns
        -------
        fig : Figure
        axes0 : array of Axes

        """
        if self.results is None:
            self.gather()

        #######################################################################################
        fig, axes0 = plt.subplots(nrows=7, ncols=3, figsize=(7, 8))  # 17.8/golden
        fig.subplots_adjust(wspace=0.5)

        # GENERAL SETTINGS
        titles = [
            ["$l_{0,i}$", "$a_{1,i}$", "$a_{3,i}/a_{1,i}$"],
            ["Q$(l_0)$", "Q$(a_1)$", "Q$(a_3/a_1)$"],
            ["IQR$(l_0)$", "IQR$(a_1)$", r"IQR$(a_3/a_1)$"],
            ["$a_{2,i}/a^2_{1,i}$", r"$\Delta d_i$", r"$\alpha_i$"],
            ["Q$(a_2/a_1^2)$", r"Q$(\Delta d)$", r"Q$(\alpha)$"],
            ["M$(|a_2/a_1^2|)$", r"M$(|\Delta d|)$", r"M$(|\alpha|)$"],
            ["IQR$(a_2/a_1^2)$", r"IQR$(\Delta d)$", r"IQR$(\alpha)$"],
        ]
        for i, axes1 in enumerate(axes0):
            for j, ax in enumerate(axes1):
                ax.set_ylabel(titles[i][j])
                if self.eta_original is not None:
                    ax.axvline(x=self.eta_original, color="0.5", linewidth=0.7, linestyle="--")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if i != 6:
                    ax.xaxis.set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                else:
                    ax.set_xlabel(r"$\eta^*$")
                    ax.xaxis.set_label_coords(1.08, 0.08)
                    ax.set_xticks(
                        np.arange(self.eta_range.min(), self.eta_range.max(), 0.5),
                        [integerise(e) for e in np.arange(self.eta_range.min(), self.eta_range.max(), 0.5)],
                    )
                if (i == 0 and j == 0) or (i == 1 and j == 0) or i == 2 or i == 5 or i == 6:
                    ax.set_yscale("log")
                else:
                    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

        ##########   TREE   ##########
        ax_tree = plt.axes([-0.35, 0.4, 0.3, 0.6])  # left, bottom, width, height
        graphics.plot_tree(self.system, ax_tree)

        ##########   BIFS SEPARATED RESULTS   ##########
        axes = axes0[0]
        for i in range(self.results_bif_sep.shape[0]):
            b_ID = int(self.results_bif_sep[i, 0, 0])
            ax_tree.scatter(
                *self.system.network.branches[int(self.results_bif_sep[i, 0, 0])].points[-1],
                s=10,
                marker="^",
                zorder=3,
                edgecolor="k",
                linewidth=0.2,
                label=b_ID,
            )
            axes[0].plot(self.eta_range, self.results_bif_sep[i, :, 4], label=b_ID)
            axes[1].plot(self.eta_range, self.results_bif_sep[i, :, 1], label=b_ID)
            axes[2].plot(
                self.eta_range,
                self.results_bif_sep[i, :, 3] / self.results_bif_sep[i, :, 1],
                label=b_ID,
            )
        axes[1].axhline(0.8, c="0.5", linewidth=0.7, linestyle="--")

        ##########   BIFS   ##########
        axes = axes0[1:3]
        # LENGTH MISMATCH QUANTILES
        axes[0, 0].plot(self.eta_range, self.results_bif[0][:, 0:3], "-", ms=5)
        axes[0, 0].fill_between(
            self.eta_range, self.results_bif[0][:, 1], self.results_bif[0][:, 2], alpha=0.3
        )
        # LENGTH MISMATCH MEDIAN
        axes[1, 0].plot(self.eta_range, self.results_bif[0][:, 3], "-", ms=5, label="M")
        # LENGTH MISMATCH IQR
        axes[1, 0].plot(self.eta_range, self.results_bif[0][:, 6] / 50, "-", ms=5, label="IQR/50")

        # BIFURCATION QUANTILES a1
        if self.bif_type == 1:
            # plot vertical line at a default bif_thresh value
            axes[0, 1].axhline(y=0.8, color="0.5", linewidth=0.5)
        axes[0, 1].plot(self.eta_range, self.results_bif[1][:, 0:3], "-")
        axes[0, 1].fill_between(
            self.eta_range, self.results_bif[1][:, 1], self.results_bif[1][:, 2], alpha=0.3
        )
        # BIFURCATION IQR
        axes[1, 1].plot(self.eta_range, self.results_bif[1][:, 6], label="IQR")

        # BIFURCATION QUANTILES a3/a1
        if self.bif_type == 2:
            # plot vertical line at a default bif_thresh value
            axes[0, 1].axhline(y=-0.1, color="0.5", linewidth=0.5)
        axes[0, 2].plot(self.eta_range, self.results_bif[2][:, 0:3])
        axes[0, 2].fill_between(
            self.eta_range, self.results_bif[2][:, 1], self.results_bif[2][:, 2], alpha=0.3
        )
        # BIFURCATION IQR
        axes[1, 2].plot(self.eta_range, self.results_bif[2][:, 6], label="IQR")

        [ax.legend(frameon=False, fontsize=5) for ax in [axes[1, 0]]]

        ##########   LOCAL SYMMETRY, OVERSHOOT, ANGULAR DEVIATION    ##########

        ########## BRANCHES (SEPARATED RESULTS) ##########
        axes = axes0[3]
        for i in range(self.results_branches_sep.shape[0]):
            if ~np.isnan(self.results_branches_sep[i, 0, 0]):
                b_ID = int(self.results_branches_sep[i, 0, 0])
                pts = self.system.network.branches[b_ID].points
                pt = pts[len(pts) // 2]
                ax_tree.scatter(*pt, s=10, zorder=3, edgecolor="k", linewidth=0.2, label=b_ID)
                axes[0].plot(
                    self.eta_range,
                    self.results_branches_sep[i, :, 2] / self.results_branches_sep[i, :, 1] ** 2,
                    label=b_ID,
                )
                axes[1].plot(self.eta_range, self.results_branches_sep[i, :, 4], label=b_ID)
                axes[2].plot(self.eta_range, self.results_branches_sep[i, :, 5], label=b_ID)
        [ax.axhline(0, c="0.5", linewidth=0.7, linestyle="-") for ax in axes]

        ########## BRANCHES (TOGETHER) ##########
        axes = axes0[4:]
        for i, measure_results in enumerate(self.results):
            # plot DATA
            if self.eta_original is not None:
                axes[0, i].axvline(x=self.eta_original, color="0.5", linewidth=0.7, linestyle="--")
            axes[0, i].axhline(y=0, color="0.5", linewidth=0.5)
            axes[0, i].plot(
                self.eta_range,
                measure_results[:, 0:3],
                label=["Median", r"Quantile 32\%", r"Quantile 68\%"],
            )
            axes[0, i].fill_between(
                self.eta_range, measure_results[:, 1], measure_results[:, 2], alpha=0.3
            )

            # plot DATA MEDIAN (log)
            if self.eta_original is not None:
                axes[1, i].axvline(x=self.eta_original, color="0.5", linewidth=0.7, linestyle="--")
            axes[1, i].plot(self.eta_range, measure_results[:, 3])

            # plot IQR (log)
            if self.eta_original is not None:
                axes[2, i].axvline(x=self.eta_original, color="0.5", linewidth=0.7, linestyle="--")
            axes[2, i].plot(self.eta_range, measure_results[:, 6])

        ##########   ALL COMBINED    ##########
        results_all_avg_norm = self.normalized_metric_average()

        ax_plus = plt.axes([-0.35, 0.25, 0.3, 0.2])  # left, bottom, width, height
        ax_plus.plot(self.eta_range, results_all_avg_norm, color="0", lw=2)
        ax_plus.set_title("Average of\nnormalized metrics")
        ax_plus.set_ylabel("Normalized value")
        ax_plus.set_xlabel(r"$\eta^*$")
        ax_plus.xaxis.set_label_coords(1.08, 0.04)
        if self.eta_original is not None:
            ax_plus.axvline(x=self.eta_original, color="0.5", linewidth=0.7, linestyle="--")
        ax_plus.set_yticks([0, 1], ["0", "1"])
        xticks = np.arange(self.eta_range.min(), self.eta_range.max() + 0.01, 0.5)
        ax_plus.set_xticks(
            xticks,
            [integerise(e) for e in np.arange(self.eta_range.min(), self.eta_range.max() + 0.01, 0.5)],
        )

        # colouring background
        for ax in [
            axes0[2, 0],
            axes0[2, 1] if self.bif_type == 1 else axes0[2, 2],
            *axes0[5],
            *axes0[6],
            ax_plus,
        ]:
            ax.add_artist(ax.patch)
            ax.patch.set_zorder(-1)
            ax.set_facecolor("#f0f0f0")

        fig.align_ylabels()
        fig.suptitle(self.exp_name, y=1, fontsize=12)
        fig.tight_layout()

        return fig, axes0
