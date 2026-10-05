"""
figures.py - generate all thesis figures from the results.
Saves PNGs into the figures/ folder.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import iter_records, bandpass, FS, PPG, ABP, ECG, find_part_files
from scipy.signal import find_peaks

RED = "#c53030"; BLUE = "#2b6cb0"; GREEN = "#2f855a"; ORANGE = "#c05621"


def fig_raw_signals(data_dir, outdir, log=print):
    """Plot PPG/ABP/ECG from the first real record - the raw data figure."""
    files = find_part_files(data_dir)
    if not files:
        return
    rec = None
    for rid, data in iter_records(files[0]):
        if data.shape[0] > FS * 20:
            rec = data
            break
    if rec is None:
        return
    n = FS * 8
    t = np.arange(n) / FS
    labels = ["PPG (column 1)", "ABP - blood pressure (column 2)", "ECG (column 3)"]
    colors = [BLUE, RED, GREEN]
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(3):
        ax[i].plot(t, rec[:n, i], lw=0.9, color=colors[i])
        ax[i].set_ylabel(labels[i], fontsize=9)
        ax[i].grid(alpha=0.3)
    ax[-1].set_xlabel("time (seconds)")
    ax[0].set_title("Raw physiological signals from one patient record")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_raw_signals.png"), dpi=150)
    plt.close(fig)
    log("  figure: raw signals")


def fig_bp_targets(df, outdir, log=print):
    """Distribution of systolic and diastolic BP in the dataset."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["sbp"], bins=60, color=RED, alpha=0.8)
    ax[0].set_title("Systolic BP distribution"); ax[0].set_xlabel("mmHg")
    ax[0].set_ylabel("beats")
    ax[1].hist(df["dbp"], bins=60, color=BLUE, alpha=0.8)
    ax[1].set_title("Diastolic BP distribution"); ax[1].set_xlabel("mmHg")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_bp_distribution.png"), dpi=150)
    plt.close(fig)
    log("  figure: BP distributions")


def fig_baseline(baseline_df, outdir, log=print):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, target in enumerate(["SBP", "DBP"]):
        sub = baseline_df[baseline_df.target == target]
        ax[j].bar(sub.model, sub.std_err, color=BLUE, alpha=0.8)
        ax[j].axhline(8, color=RED, ls="--", label="AAMI limit")
        ax[j].set_title(f"{target}: baseline std error by model")
        ax[j].set_ylabel("std of error (mmHg)")
        ax[j].tick_params(axis="x", rotation=20)
        ax[j].grid(axis="y", alpha=0.3); ax[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_baseline_models.png"), dpi=150)
    plt.close(fig)
    log("  figure: baseline model comparison")


def fig_personalization(pers_df, outdir, log=print):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, target in enumerate(["SBP", "DBP"]):
        for method in ["OFFSET", "BAYES"]:
            sub = pers_df[(pers_df.target == target) & (pers_df.method == method)]
            sub = sub.sort_values("N")
            lw = 2.6 if method == "BAYES" else 1.3
            ax[j].plot(sub.N, sub.std_err, "-o", lw=lw, label=method, ms=4)
        ax[j].axhline(8, color=RED, ls="--", label="AAMI limit")
        ax[j].set_title(f"{target}: error vs calibration readings")
        ax[j].set_xlabel("calibration readings (N)")
        ax[j].set_ylabel("std of error (mmHg)")
        ax[j].grid(alpha=0.3); ax[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_personalization.png"), dpi=150)
    plt.close(fig)
    log("  figure: personalization curves")


def fig_real_examples(examples_df, outdir, log=print):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(examples_df))
    w = 0.25
    for j, lbl in enumerate(["SBP", "DBP"]):
        ax[j].bar(x - w, examples_df[f"true_{lbl}"], w, label="true", color="black", alpha=0.7)
        ax[j].bar(x, examples_df[f"pred_{lbl}"], w, label="population", color=RED, alpha=0.7)
        ax[j].bar(x + w, examples_df[f"personalized_{lbl}"], w, label="personalized", color=GREEN, alpha=0.8)
        ax[j].set_title(f"{lbl}: true vs predicted (sample patients)")
        ax[j].set_ylabel("mmHg"); ax[j].set_xticks(x)
        ax[j].set_xticklabels(examples_df["patient"], rotation=45, fontsize=7)
        ax[j].grid(axis="y", alpha=0.3); ax[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_real_examples.png"), dpi=150)
    plt.close(fig)
    log("  figure: real patient examples")
