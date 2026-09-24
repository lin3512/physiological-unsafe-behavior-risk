"""Input validation, recording-level splitting, windowing, and sequence assembly."""
from __future__ import annotations

import math
import pandas as pd
import numpy as np

from features import CHANNELS, FEATURES_PER_CHANNEL, cwt_denoise, extract_window_features


def validate_input(df: pd.DataFrame) -> None:
    required = {"participant", "recording_id", "timestamp_s", "label", *CHANNELS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    labels = set(pd.to_numeric(df.label, errors="coerce").dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError("label must contain only 0 (Normal) and 1 (Unsafe)")


def split_recordings(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed=42):
    """Stratify recording units by whether they contain any unsafe samples."""
    from sklearn.model_selection import train_test_split
    keys = df[["participant", "recording_id"]].drop_duplicates().reset_index(drop=True)
    key_tuples = list(map(tuple, keys.to_numpy()))
    labels = [int(df[(df.participant == p) & (df.recording_id == r)].label.max()) for p, r in key_tuples]
    train_ratio, val_ratio, test_ratio = ratios
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split ratios must sum to 1")
    try:
        train, remain, _, y_rem = train_test_split(
            key_tuples, labels, test_size=val_ratio + test_ratio,
            random_state=seed, stratify=labels)
        val, test = train_test_split(remain, test_size=test_ratio/(val_ratio+test_ratio),
                                     random_state=seed, stratify=y_rem)
    except ValueError as e:
        raise ValueError("Not enough recording units/classes for stratified 8:1:1 split; "
                         "collect more recording units or provide preassigned splits.") from e

    def subset(selected):
        idx = pd.MultiIndex.from_frame(df[["participant", "recording_id"]])
        selected_idx = pd.MultiIndex.from_tuples(selected, names=["participant", "recording_id"])
        return df.loc[idx.isin(selected_idx)].copy()

    return subset(train), subset(val), subset(test)


def make_windows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Window synchronized rows; label Unsafe if any timestamp is Unsafe."""
    fs_ref = cfg["sampling_rate_hz"]["ecg"]
    win_n, step_n = int(cfg["window_seconds"] * fs_ref), int(cfg["step_seconds"] * fs_ref)
    rows = []
    for (participant, rec), group in df.groupby(["participant", "recording_id"], sort=False):
        group = group.sort_values("timestamp_s")
        values = {}
        for ch in CHANNELS:
            arr = pd.to_numeric(group[ch], errors="coerce").interpolate(limit_direction="both").to_numpy(float)
            values[ch] = cwt_denoise(arr, fs_ref, ch, cfg)
        lab = pd.to_numeric(group.label, errors="coerce").fillna(0).to_numpy(int)
        ts = group.timestamp_s.to_numpy(float)
        for start in range(0, len(group) - win_n + 1, step_n):
            stop = start + win_n
            row = {"participant": participant, "recording_id": rec,
                   "window_start_s": float(ts[start]), "label": int(lab[start:stop].max())}
            for ch in CHANNELS:
                feats = extract_window_features(values[ch][start:stop], cfg["sampling_rate_hz"][ch])
                row.update({f"{ch}_{name}": v for name, v in zip(FEATURES_PER_CHANNEL, feats)})
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No complete windows produced; check sampling rate and recording lengths")
    return out


def participant_baselines(train: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Per-person baseline bins; expects optional is_rest=1 rows."""
    baselines = {}
    for ch in CHANNELS:
        feature = f"{ch}_mean"
        if "is_rest" not in raw.columns:
            source = train.groupby("participant")[feature]
        else:
            rest_ids = raw.loc[raw.is_rest.astype(int) == 1,
                               ["participant", "recording_id"]].drop_duplicates()
            baseline = train.merge(rest_ids, on=["participant", "recording_id"], how="inner")
            source = baseline.groupby("participant")[feature]
        means = source.mean().to_dict()
        stds = source.std().replace(0, np.nan).to_dict()
        fallback_mean = float(train[feature].mean())
        fallback_std = float(train[feature].std() or 1.0)
        baselines[ch] = {
            p: (float(means.get(p, fallback_mean)), float(stds.get(p, fallback_std) or fallback_std))
            for p in train.participant.unique()
        }
    return baselines


def discretize_windows(frames: list[pd.DataFrame], baseline: dict) -> list[list[list[str]]]:
    """Encode each channel mean as participant-relative low/medium/high items."""
    transformed = []
    for frame in frames:
        items_by_row = []
        for row in frame.itertuples(index=False):
            items = []
            for ch in CHANNELS:
                mu, sd = baseline[ch][row.participant]
                value = getattr(row, f"{ch}_mean")
                level = "low" if value < mu - .5*sd else "high" if value > mu + .5*sd else "medium"
                items.append(f"{ch.upper()}_{level}")
            items.append("Unsafe" if row.label else "Normal")
            items_by_row.append(items)
        transformed.append(items_by_row)
    return transformed


def make_sequences(windows: pd.DataFrame, x: np.ndarray, length: int):
    """Use preceding windows plus current; target is the final window's label."""
    xs, ys, meta = [], [], []
    indexed = windows.reset_index(drop=True)
    for _, ids in indexed.groupby(["participant", "recording_id"], sort=False).groups.items():
        ids = list(ids)
        for end in range(length-1, len(ids)):
            span = ids[end-length+1:end+1]
            if any(not np.isclose(indexed.loc[b, "window_start_s"] - indexed.loc[a, "window_start_s"], 30)
                   for a, b in zip(span, span[1:])):
                continue
            xs.append(x[span]); ys.append(int(indexed.loc[span[-1], "label"]))
            meta.append((indexed.loc[span[-1], "participant"], indexed.loc[span[-1], "recording_id"],
                         indexed.loc[span[-1], "window_start_s"]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), meta
