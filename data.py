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
    """Create time-based windows and resample each channel to its native rate.

    The input CSV has one synchronized row per timestamp.  Windows are defined
    on that shared time axis, while each channel is interpolated onto the
    sampling rate reported for that channel before CWT and feature extraction.
    This preserves the configured per-channel sampling rates without treating
    a repeated/upsampled slow channel as if it were sampled at the ECG rate.
    """
    window_s = float(cfg["window_seconds"])
    step_s = float(cfg["step_seconds"])
    rows = []
    for (participant, rec), group in df.groupby(["participant", "recording_id"], sort=False):
        group = group.sort_values("timestamp_s")
        ts = pd.to_numeric(group["timestamp_s"], errors="coerce").to_numpy(float)
        if len(ts) < 2 or not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
            raise ValueError(f"timestamps must be finite and strictly increasing for recording {rec!r}")
        source_cadence = float(np.median(np.diff(ts)))
        channel_values = {}
        for ch in CHANNELS:
            arr = pd.to_numeric(group[ch], errors="coerce").interpolate(limit_direction="both").to_numpy(float)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"channel {ch!r} has no finite values for recording {rec!r}")
            channel_values[ch] = arr
        lab = pd.to_numeric(group.label, errors="coerce").fillna(0).to_numpy(int)
        start = float(ts[0])
        last_start = float(ts[0] + (ts[-1] - ts[0] + source_cadence) - window_s)
        while start <= last_start + 1e-9:
            row = {"participant": participant, "recording_id": rec,
                   "window_start_s": start,
                   "label": int(lab[(ts >= start) & (ts < start + window_s)].max())}
            for ch in CHANNELS:
                fs = float(cfg["sampling_rate_hz"][ch])
                target_ts = start + np.arange(int(round(window_s * fs)), dtype=float) / fs
                sampled = np.interp(target_ts, ts, channel_values[ch])
                sampled = cwt_denoise(sampled, fs, ch, cfg)
                feats = extract_window_features(sampled, fs)
                row.update({f"{ch}_{name}": v for name, v in zip(FEATURES_PER_CHANNEL, feats)})
            rows.append(row)
            start += step_s
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No complete windows produced; check sampling rate and recording lengths")
    return out


def participant_baselines(train: pd.DataFrame, raw: pd.DataFrame, cfg: dict,
                          baseline_seconds: float = 300.0) -> dict:
    """Compute per-person baselines from marked raw resting samples only.

    The baseline is the participant's pre-shift five-minute rest period. A
    missing or shorter marked period is an error instead of silently falling
    back to task-period training data.
    """
    if "is_rest" not in raw.columns:
        raise ValueError("Input must include is_rest=1 rows for the five-minute resting baseline")
    rest_mask = pd.to_numeric(raw["is_rest"], errors="coerce").fillna(0).astype(int).eq(1)
    rest = raw.loc[rest_mask].copy()
    participants = list(train["participant"].drop_duplicates())
    if not participants:
        raise ValueError("No training participants available for baseline estimation")
    baselines = {}
    missing = [p for p in participants if not (rest["participant"] == p).any()]
    if missing:
        raise ValueError(f"Missing resting baseline rows for participants: {missing}")
    baseline_samples = {}
    for participant in participants:
        person_rest = rest.loc[rest["participant"] == participant]
        qualifying = []
        for _, group in person_rest.groupby("recording_id", sort=False):
            group = group.sort_values("timestamp_s")
            pts = pd.to_numeric(group["timestamp_s"], errors="coerce").to_numpy(float)
            if len(pts) < 2 or not np.all(np.isfinite(pts)) or np.any(np.diff(pts) <= 0):
                continue
            deltas = np.diff(pts)
            cadence = float(np.median(deltas))
            if cadence <= 0:
                continue
            # Do not join separate rest episodes merely because their local
            # timestamps sort together. A run may tolerate modest jitter.
            breaks = np.flatnonzero(deltas > 1.5 * cadence) + 1
            for run in np.split(np.arange(len(group)), breaks):
                if len(run) < 2:
                    continue
                run_group = group.iloc[run]
                run_ts = pts[run]
                duration = float(run_ts[-1] - run_ts[0] + cadence)
                if duration + 1e-6 >= baseline_seconds:
                    qualifying.append((run_ts[0], run_group, run_ts, cadence))
        if not qualifying:
            raise ValueError(
                f"No continuous {baseline_seconds:.0f}-second is_rest=1 baseline for "
                f"participant {participant!r}")
        # Use the earliest qualifying rest segment, and exactly the requested
        # five-minute interval, not all rest/task windows in the recording.
        _, group, pts, _ = min(qualifying, key=lambda item: item[0])
        baseline_samples[participant] = {}
        for ch in CHANNELS:
            fs = float(cfg["sampling_rate_hz"][ch])
            target_ts = pts[0] + np.arange(int(round(baseline_seconds * fs)), dtype=float) / fs
            source = pd.to_numeric(group[ch], errors="coerce").to_numpy(float)
            valid = np.isfinite(source) & np.isfinite(pts)
            if valid.sum() < 2 or target_ts[-1] > pts[valid][-1] + 1e-6:
                raise ValueError(f"Insufficient resting {ch} samples for participant {participant!r}")
            baseline_samples[participant][ch] = np.interp(target_ts, pts[valid], source[valid])
    for ch in CHANNELS:
        baselines[ch] = {}
        for participant in participants:
            vals = baseline_samples[participant][ch]
            sd = float(np.std(vals, ddof=1))
            baselines[ch][participant] = (float(np.mean(vals)), sd if sd > 0 else 1.0)
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


def make_sequences(windows: pd.DataFrame, x: np.ndarray, length: int,
                   step_seconds: float = 30.0):
    """Use preceding windows plus current; target is the final window's label."""
    xs, ys, meta = [], [], []
    indexed = windows.reset_index(drop=True)
    for _, ids in indexed.groupby(["participant", "recording_id"], sort=False).groups.items():
        ids = list(ids)
        for end in range(length-1, len(ids)):
            span = ids[end-length+1:end+1]
            if any(not np.isclose(indexed.loc[b, "window_start_s"] - indexed.loc[a, "window_start_s"], step_seconds)
                   for a, b in zip(span, span[1:])):
                continue
            xs.append(x[span]); ys.append(int(indexed.loc[span[-1], "label"]))
            meta.append((indexed.loc[span[-1], "participant"], indexed.loc[span[-1], "recording_id"],
                         indexed.loc[span[-1], "window_start_s"]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), meta
