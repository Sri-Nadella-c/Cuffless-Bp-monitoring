"""
modeling.py - baseline models, personalization, robustness, evaluation.
Called by main.py after the dataset is built.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from pipeline import FEATURES, AAMI_MEAN, AAMI_STD

SEED = 42
CALIB_SIZES = [0, 1, 2, 5, 10, 20, 50]


def patient_split(df, seed=SEED, test_size=0.25):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["record"]))
    return df.iloc[tr], df.iloc[te]


def _models():
    return {
        "OLS": (LinearRegression(), True),
        "Ridge": (Ridge(alpha=1.0), True),
        "RandomForest": (RandomForestRegressor(
            n_estimators=120, min_samples_leaf=20, random_state=SEED, n_jobs=-1), False),
        "Bayesian": (BayesianRidge(), True),
        "NeuralNet": (MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=400,
                                   early_stopping=True, random_state=SEED), True),
    }


def baseline(df, log=print):
    """Five models x {SBP,DBP}, patient-split, vs AAMI. Returns a DataFrame."""
    df_tr, df_te = patient_split(df)
    rows = []
    for target in ["sbp", "dbp"]:
        Xtr, Xte = df_tr[FEATURES].values, df_te[FEATURES].values
        ytr, yte = df_tr[target].values, df_te[target].values
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        naive_std = (yte - ytr.mean()).std()

        for name, (model, scaled) in _models().items():
            Xa, Xb = (Xtr_s, Xte_s) if scaled else (Xtr, Xte)
            model.fit(Xa, ytr)
            err = model.predict(Xb) - yte
            me, sd, mae = err.mean(), err.std(), np.abs(err).mean()
            rows.append(dict(target=target.upper(), model=name,
                             mean_err=round(me, 3), std_err=round(sd, 3),
                             mae=round(mae, 3),
                             aami_gap=round(sd - AAMI_STD, 3),
                             vs_naive_pct=round((1 - sd / naive_std) * 100, 1),
                             meets_aami=(abs(me) <= AAMI_MEAN and sd <= AAMI_STD)))
    out = pd.DataFrame(rows)
    log("  baseline models trained (5 models x SBP/DBP)")
    return out


def personalization(df, log=print):
    """Offset + Bayesian personalization across calibration sizes.
    Returns a tidy DataFrame of std-error per (target, method, N)."""
    df_tr, df_te = patient_split(df, test_size=0.30)
    rows = []
    for target in ["sbp", "dbp"]:
        sc = StandardScaler().fit(df_tr[FEATURES].values)
        m = BayesianRidge().fit(sc.transform(df_tr[FEATURES].values),
                                df_tr[target].values)
        # bayesian shrinkage parameters
        pred_tr = m.predict(sc.transform(df_tr[FEATURES].values))
        sigma_obs2 = np.var(df_tr[target].values - pred_tr)
        sigma_prior2 = max(np.var(
            df_tr.groupby("record")[target].mean() - df_tr[target].mean()), 1e-6)

        errs = {("OFFSET", n): [] for n in CALIB_SIZES}
        errs.update({("BAYES", n): [] for n in CALIB_SIZES})

        for rid, g in df_te.groupby("record"):
            if len(g) < 80:
                continue
            g = g.sort_index()
            pred = m.predict(sc.transform(g[FEATURES].values))
            y = g[target].values
            resid = y - pred
            for n in CALIB_SIZES:
                if n == 0:
                    errs[("OFFSET", n)].extend((pred - y).tolist())
                    errs[("BAYES", n)].extend((pred - y).tolist())
                    continue
                if len(g) <= n + 20:
                    continue
                mr = resid[:n].mean()
                errs[("OFFSET", n)].extend((pred[n:] + mr - y[n:]).tolist())
                w = (n / sigma_obs2) / (n / sigma_obs2 + 1 / sigma_prior2)
                errs[("BAYES", n)].extend((pred[n:] + w * mr - y[n:]).tolist())

        for (method, n), e in errs.items():
            if e:
                e = np.array(e)
                rows.append(dict(target=target.upper(), method=method, N=n,
                                 mean_err=round(e.mean(), 3),
                                 std_err=round(e.std(), 3),
                                 mae=round(np.abs(e).mean(), 3),
                                 meets_aami=(abs(e.mean()) <= AAMI_MEAN
                                             and e.std() <= AAMI_STD)))
    log("  personalization computed (offset + Bayesian, all calibration sizes)")
    return pd.DataFrame(rows)


def robustness(df, log=print):
    """5 splits + time-separated calibration, for both targets."""
    rows = []
    for target in ["dbp", "sbp"]:
        # test 1: five splits, adjacent calibration
        for seed in [0, 1, 2, 3, 4]:
            me, sd, mae, n = _one_personalized_split(df, target, seed, "adjacent")
            rows.append(dict(target=target.upper(), test="split_stability",
                             seed=seed, mean_err=round(me, 3), std_err=round(sd, 3),
                             mae=round(mae, 3),
                             meets_aami=(abs(me) <= AAMI_MEAN and sd <= AAMI_STD)))
        # test 2: time-separated
        me, sd, mae, n = _one_personalized_split(df, target, 0, "separated")
        rows.append(dict(target=target.upper(), test="time_separated",
                         seed=0, mean_err=round(me, 3), std_err=round(sd, 3),
                         mae=round(mae, 3),
                         meets_aami=(abs(me) <= AAMI_MEAN and sd <= AAMI_STD)))
    log("  robustness checks done (5 splits + time-separation)")
    return pd.DataFrame(rows)


def _one_personalized_split(df, target, seed, mode, n_calib=10):
    df_tr, df_te = patient_split(df, seed=seed, test_size=0.30)
    sc = StandardScaler().fit(df_tr[FEATURES].values)
    m = BayesianRidge().fit(sc.transform(df_tr[FEATURES].values),
                            df_tr[target].values)
    errs = []
    for rid, g in df_te.groupby("record"):
        if len(g) < 120:
            continue
        g = g.sort_index()
        pred = m.predict(sc.transform(g[FEATURES].values))
        y = g[target].values
        if mode == "adjacent":
            ci, ti = slice(0, n_calib), slice(n_calib, None)
        else:
            ci = slice(0, n_calib)
            ti = slice(int(len(g) * 0.66), None)
        offset = np.mean(y[ci] - pred[ci])
        errs.extend((pred[ti] + offset - y[ti]).tolist())
    errs = np.array(errs)
    if len(errs) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return errs.mean(), errs.std(), np.abs(errs).mean(), len(errs)


def real_examples(df, n_patients=8, n_calib=10, log=print):
    """True vs predicted BP in real units for sample patients."""
    df_tr, df_te = patient_split(df, test_size=0.30)
    scs, ms = {}, {}
    for t in ["sbp", "dbp"]:
        scs[t] = StandardScaler().fit(df_tr[FEATURES].values)
        ms[t] = BayesianRidge().fit(scs[t].transform(df_tr[FEATURES].values),
                                    df_tr[t].values)
    patients = [r for r, g in df_te.groupby("record") if len(g) > n_calib + 30]
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(patients, size=min(n_patients, len(patients)), replace=False)
    rows = []
    for rid in chosen:
        g = df_te[df_te.record == rid].sort_index()
        rec = dict(patient=str(rid))
        for t, lbl in [("sbp", "SBP"), ("dbp", "DBP")]:
            pred = ms[t].predict(scs[t].transform(g[FEATURES].values))
            y = g[t].values
            off = np.mean(y[:n_calib] - pred[:n_calib])
            rec[f"true_{lbl}"] = round(float(y[n_calib:].mean()), 1)
            rec[f"pred_{lbl}"] = round(float(pred[n_calib:].mean()), 1)
            rec[f"personalized_{lbl}"] = round(float((pred[n_calib:] + off).mean()), 1)
        rows.append(rec)
    log("  real true-vs-predicted examples generated")
    # add readable "120/80" clinical-format columns
    df_out = pd.DataFrame(rows)
    if len(df_out):
        df_out["true_BP"] = (df_out["true_SBP"].round().astype(int).astype(str)
                             + "/" + df_out["true_DBP"].round().astype(int).astype(str))
        df_out["population_BP"] = (df_out["pred_SBP"].round().astype(int).astype(str)
                             + "/" + df_out["pred_DBP"].round().astype(int).astype(str))
        df_out["personalized_BP"] = (df_out["personalized_SBP"].round().astype(int).astype(str)
                             + "/" + df_out["personalized_DBP"].round().astype(int).astype(str))
    return df_out
