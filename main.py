#!/usr/bin/env python3
"""
============================================================================
 CUFFLESS BLOOD PRESSURE ESTIMATION - COMPLETE PIPELINE
 Run everything with one command:   python main.py
============================================================================

This single file runs the entire analysis end to end:
  1. reads every Part_*.mat file in the data/ folder (auto-detects Part_5 etc.)
  2. extracts per-beat features + ground-truth blood pressure
  3. builds one pooled, quality-controlled dataset
  4. trains five models (OLS, Ridge, Random Forest, Bayesian, Neural Net)
  5. runs per-person personalization (offset + Bayesian)
  6. runs robustness checks (5 splits + time-separated calibration)
  7. produces real true-vs-predicted BP examples
  8. saves ALL tables to results/ and ALL figures to figures/

Nothing to run one-by-one. If you add a Part_5.mat later, just rerun.

SETUP (first time only), in this folder:
    python -m pip install numpy pandas scipy scikit-learn h5py matplotlib
Then:
    python main.py

Put your .mat files in the data/ folder before running.
============================================================================
"""

import os
import sys
import time

# make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

LOGLINES = []
def log(msg):
    print(msg, flush=True)
    LOGLINES.append(str(msg))


def banner(t):
    log("\n" + "=" * 70)
    log(f"  {t}")
    log("=" * 70)


def main():
    t0 = time.time()
    banner("CUFFLESS BLOOD PRESSURE - FULL PIPELINE")

    import pipeline as P
    import modeling as M
    import figures as F

    # ---- STEP 1-3: build dataset from all Part files ----
    banner("STEP 1  Building dataset from all Part_*.mat files")
    try:
        df, per_file = P.build_dataset(DATA, log=log)
    except FileNotFoundError as e:
        log(f"\nERROR: {e}")
        log("Place your Part_1.mat ... Part_N.mat in the 'data' folder and rerun.")
        return

    df.to_csv(os.path.join(RESULTS, "bp_training_data.csv"), index=False)
    n_patients = df["record"].nunique()
    log(f"\nTOTAL: {len(df)} beats from {n_patients} patients across "
        f"{len(per_file)} file(s)")
    log(f"Saved results/bp_training_data.csv")

    # dataset summary table
    summ = pd.DataFrame(per_file).T
    summ.to_csv(os.path.join(RESULTS, "dataset_summary.csv"))
    with open(os.path.join(RESULTS, "dataset_overview.txt"), "w") as fh:
        fh.write(f"Total beats: {len(df)}\nTotal patients: {n_patients}\n")
        fh.write(f"Files: {len(per_file)}\n\n")
        fh.write(f"SBP mean {df.sbp.mean():.1f}  range {df.sbp.min():.0f}-{df.sbp.max():.0f}  std {df.sbp.std():.1f}\n")
        fh.write(f"DBP mean {df.dbp.mean():.1f}  range {df.dbp.min():.0f}-{df.dbp.max():.0f}  std {df.dbp.std():.1f}\n")

    # ---- STEP 4: baseline models ----
    banner("STEP 2  Baseline models (5 models x SBP/DBP, patient-split, vs AAMI)")
    baseline_df = M.baseline(df, log=log)
    baseline_df.to_csv(os.path.join(RESULTS, "baseline_models.csv"), index=False)
    log("\n" + baseline_df.to_string(index=False))
    log("\nSaved results/baseline_models.csv")

    # ---- STEP 5: personalization ----
    banner("STEP 3  Personalization (offset + Bayesian, calibration sweep)")
    pers_df = M.personalization(df, log=log)
    pers_df.to_csv(os.path.join(RESULTS, "personalization.csv"), index=False)
    # print a compact view
    for target in ["DBP", "SBP"]:
        log(f"\n  {target}:")
        sub = pers_df[pers_df.target == target].pivot(
            index="N", columns="method", values="std_err")
        log(sub.to_string())
    log("\nSaved results/personalization.csv")

    # ---- STEP 6: robustness ----
    banner("STEP 4  Robustness (5 splits + time-separated calibration)")
    robust_df = M.robustness(df, log=log)
    robust_df.to_csv(os.path.join(RESULTS, "robustness.csv"), index=False)
    log("\n" + robust_df.to_string(index=False))
    log("\nSaved results/robustness.csv")

    # ---- STEP 7: real examples ----
    banner("STEP 5  Real true-vs-predicted BP examples")
    examples_df = M.real_examples(df, log=log)
    examples_df.to_csv(os.path.join(RESULTS, "real_examples.csv"), index=False)
    # readable clinical-format BP block
    log("\n  Blood pressure readings (systolic/diastolic mmHg):")
    log(f"    {'patient':<14}{'TRUE BP':>10}{'population':>13}{'personalized':>15}")
    log("    " + "-" * 50)
    for _, r in examples_df.iterrows():
        pid = str(r["patient"]).split(":")[-1]
        tb = f"{r['true_SBP']:.0f}/{r['true_DBP']:.0f}"
        pb = f"{r['pred_SBP']:.0f}/{r['pred_DBP']:.0f}"
        xb = f"{r['personalized_SBP']:.0f}/{r['personalized_DBP']:.0f}"
        log(f"    {pid:<14}{tb:>10}{pb:>13}{xb:>15}")
    log("\n    (personalized should sit close to TRUE; population is the")
    log("     un-calibrated model, usually further off)")
    log("\nSaved results/real_examples.csv")

    # ---- STEP 8: figures ----
    banner("STEP 6  Generating figures")
    F.fig_raw_signals(DATA, FIGURES, log=log)
    F.fig_bp_targets(df, FIGURES, log=log)
    F.fig_baseline(baseline_df, FIGURES, log=log)
    F.fig_personalization(pers_df, FIGURES, log=log)
    F.fig_real_examples(examples_df, FIGURES, log=log)
    log("\nAll figures saved to figures/")

    # ---- done ----
    dt = time.time() - t0
    banner("COMPLETE")
    log(f"Everything ran in {dt/60:.1f} minutes.")
    log(f"Results  -> {RESULTS}")
    log(f"Figures  -> {FIGURES}")
    log("\nKey files to show your supervisor:")
    log("  results/baseline_models.csv     (five-model comparison)")
    log("  results/personalization.csv     (the main result)")
    log("  results/robustness.csv          (proof it holds up)")
    log("  results/real_examples.csv       (true vs predicted, real units)")
    log("  figures/*.png                   (all plots)")

    with open(os.path.join(RESULTS, "run_log.txt"), "w") as fh:
        fh.write("\n".join(LOGLINES))
    log("\nFull run log saved to results/run_log.txt")


if __name__ == "__main__":
    main()
