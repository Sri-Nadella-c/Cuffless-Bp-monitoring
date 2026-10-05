# Cuffless Blood Pressure Estimation — Complete Pipeline

Run the entire thesis analysis with **one command**. No running scripts one by one.

---

## What it does

`main.py` runs everything end to end and saves every result:

1. Reads **every** `Part_*.mat` file in `data/` (auto-detects `Part_5.mat` etc. if you add more)
2. Extracts per-beat features (heart rate, PPG amplitude, rise time) + true BP (from the ABP waveform)
3. Builds one pooled, quality-controlled dataset
4. Trains five models (OLS, Ridge, Random Forest, Bayesian, Neural Net) for systolic and diastolic BP
5. Runs per-person personalization (offset + Bayesian) across calibration sizes
6. Runs robustness checks (5 random splits + time-separated calibration)
7. Produces true-vs-predicted BP examples in real units
8. Saves all tables to `results/` and all figures to `figures/`

---

## How to run

**First time only — install the libraries** (in this folder):

```
python -m pip install -r requirements.txt
```

(The `python -m pip` form avoids Python-version mix-ups on Windows.)

**Put your data in place:** copy your `.mat` files into the `data/` folder, so it looks like:

```
data/Part_1.mat
data/Part_2.mat
data/Part_3.mat
data/Part_4.mat
```

**Run everything:**

```
python main.py
```

That's it. It prints progress as it goes and, when done, tells you exactly which files to open.

---

## Where the results appear

**`results/`**
- `bp_training_data.csv` — the full pooled feature table
- `dataset_overview.txt` / `dataset_summary.csv` — beats, patients, per-file breakdown
- `baseline_models.csv` — five-model comparison vs AAMI
- `personalization.csv` — the main result (error vs calibration readings)
- `robustness.csv` — proof the result holds across splits and time-separation
- `real_examples.csv` — true vs predicted BP, real patients, real units
- `run_log.txt` — full log of the run

**`figures/`**
- `fig_raw_signals.png` — PPG / ABP / ECG from a real record
- `fig_bp_distribution.png` — systolic & diastolic distributions
- `fig_baseline_models.png` — model comparison
- `fig_personalization.png` — the headline curves
- `fig_real_examples.png` — true vs predicted per patient

---

## Adding more data later

If a `Part_5.mat` (or more) arrives, just drop it in `data/` and run `python main.py` again.
The pipeline finds all `Part_*.mat` files automatically — no code changes needed.

---

## If something goes wrong

- **"No Part_*.mat files found"** — your `.mat` files aren't in the `data/` folder, or aren't named `Part_something.mat`.
- **"No module named ..."** — run the install command above with `python -m pip`.
- **It takes a few minutes** — that's normal; the Random Forest and the full extraction across all patients take time. The progress messages tell you it's working, not frozen.

---

## Honesty notes (kept visible on purpose)

- Results are only meaningful on **real** measured data. This code never fabricates data — if the `data/` folder is empty it stops and tells you.
- Diastolic BP reaching the AAMI standard with few calibration readings is the strong result; systolic does **not** reach AAMI from these features. Both are reported honestly.
- Pulse transit time (PTT) was tested and excluded because it failed within-patient validation — the pipeline deliberately does not use it.

---

## Folder layout

```
cuffless_project/
├── main.py              <- run this
├── requirements.txt
├── README.md
├── data/                <- put Part_*.mat files here
├── results/             <- tables appear here
├── figures/             <- plots appear here
└── src/
    ├── pipeline.py      <- data reading, feature extraction, dataset building
    ├── modeling.py      <- models, personalization, robustness, evaluation
    └── figures.py       <- all plots
```
