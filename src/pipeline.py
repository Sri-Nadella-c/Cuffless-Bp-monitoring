"""
pipeline.py - all the shared logic for the cuffless BP project.

This module holds every function main.py needs: reading the .mat files,
extracting features, building the dataset, training models, personalising,
and evaluating. main.py just calls these in order.

Channel order (confirmed): column 0 = PPG, 1 = ABP, 2 = ECG. Sampling 125 Hz.
"""

import glob
import os
import numpy as np
import pandas as pd
import h5py
from scipy.signal import find_peaks, butter, filtfilt

# ----------------------------------------------------------------------
# constants - all thresholds explicit so they can be quoted in the thesis
# ----------------------------------------------------------------------
FS = 125
PPG, ABP, ECG = 0, 1, 2
MIN_SECONDS = 30
MIN_SBP_SPREAD = 20
SBP_LO, SBP_HI = 70, 200
DBP_LO, DBP_HI = 40, 120
HR_LO, HR_HI = 45, 130
AAMI_MEAN, AAMI_STD = 5.0, 8.0
FEATURES = ["heart_rate", "ppg_amplitude", "ppg_rise_time"]


def bandpass(sig, lo, hi, order=3):
    nyq = FS / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def find_part_files(data_dir):
    """Find Part_*.mat files, sorted. Auto-detects Part_5 etc. if added later."""
    files = glob.glob(os.path.join(data_dir, "Part_*.mat"))
    # sort by the number in the filename
    def keynum(f):
        base = os.path.basename(f)
        digits = "".join(c for c in base if c.isdigit())
        return int(digits) if digits else 0
    return sorted(files, key=keynum)


def iter_records(path):
    """Yield (record_id, Nx3 array) from one .mat file."""
    with h5py.File(path, "r") as f:
        if "#refs#" not in f:
            return
        refs = f["#refs#"]
        for k in refs.keys():
            d = refs[k]
            if isinstance(d, h5py.Dataset) and d.ndim == 2 and 3 in d.shape:
                data = np.array(d)
                if data.shape[0] == 3:
                    data = data.T
                yield k, data


def sbp_spread(abp):
    peaks, _ = find_peaks(abp, distance=int(0.4 * FS), height=np.median(abp))
    if len(peaks) < 5:
        return 0
    s = abp[peaks]
    s = s[(s > SBP_LO) & (s < SBP_HI)]
    return (s.max() - s.min()) if len(s) > 5 else 0


def extract_beats(data, record_id, source_file):
    """Extract per-beat features + ground-truth BP from one record."""
    ppg = bandpass(data[:, PPG], 0.5, 8)
    ecg = bandpass(data[:, ECG], 5, 15)
    abp = data[:, ABP]

    ecg_peaks, _ = find_peaks(ecg, distance=int(0.4 * FS), height=np.std(ecg) * 2)
    ppg_peaks, _ = find_peaks(ppg, distance=int(0.4 * FS))

    rows = []
    for i in range(1, len(ecg_peaks)):
        r_prev, r_now = ecg_peaks[i - 1], ecg_peaks[i]
        rr = (r_now - r_prev) / FS
        if not (0.4 < rr < 1.5):
            continue
        hr = 60.0 / rr

        pk = ppg_peaks[(ppg_peaks > r_prev) & (ppg_peaks < r_now)]
        if len(pk) == 0:
            continue
        peak = pk[0]

        # precise foot via max 2nd derivative on the upstroke
        seg = ppg[r_prev:peak]
        if len(seg) < 4:
            continue
        d2 = np.diff(seg, 2)
        if len(d2) == 0:
            continue
        foot = r_prev + int(np.argmax(d2))
        if foot <= r_prev or foot >= peak:
            continue

        amplitude = ppg[peak] - ppg[foot]
        rise_time = (peak - foot) / FS
        if amplitude <= 0:
            continue

        beat_abp = abp[r_prev:r_now]
        if len(beat_abp) == 0:
            continue
        sbp, dbp = beat_abp.max(), beat_abp.min()

        # quality control
        if not (SBP_LO < sbp < SBP_HI):
            continue
        if not (DBP_LO < dbp < DBP_HI):
            continue
        if dbp >= sbp:
            continue
        if not (HR_LO <= hr <= HR_HI):
            continue
        if not np.isfinite([hr, amplitude, rise_time]).all():
            continue

        rows.append(dict(heart_rate=hr, ppg_amplitude=amplitude,
                         ppg_rise_time=rise_time, sbp=sbp, dbp=dbp,
                         record=f"{source_file}:{record_id}"))
    return rows


def build_dataset(data_dir, log=print):
    """Process every Part file into one pooled, quality-controlled table."""
    files = find_part_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No Part_*.mat files found in {data_dir}. "
            "Place your .mat files there and rerun.")

    log(f"Found {len(files)} data file(s): {[os.path.basename(f) for f in files]}")
    all_rows = []
    per_file = {}

    for path in files:
        fname = os.path.basename(path)
        kept, skipped = 0, 0
        file_rows = []
        for rid, data in iter_records(path):
            if data.shape[0] < FS * MIN_SECONDS:
                skipped += 1
                continue
            if sbp_spread(data[:, ABP]) <= MIN_SBP_SPREAD:
                skipped += 1
                continue
            beats = extract_beats(data, rid, fname)
            if beats:
                file_rows.extend(beats)
                kept += 1
            else:
                skipped += 1
        all_rows.extend(file_rows)
        per_file[fname] = dict(records_kept=kept, records_skipped=skipped,
                               beats=len(file_rows))
        log(f"  {fname}: kept {kept} records, {len(file_rows)} beats "
            f"(skipped {skipped})")

    df = pd.DataFrame(all_rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, per_file
