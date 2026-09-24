"""CWT preprocessing, statistical features, and feature scaling/weighting."""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["ecg", "eda", "bp", "spo2", "skt"]
FEATURES_PER_CHANNEL = ["mean", "std", "diff_mean", "centroid_hz", "rms_freq_hz"]


def cwt_denoise(x: np.ndarray, fs: float, channel: str, cfg: dict) -> np.ndarray:
    """Apply configurable CWT scale-energy attenuation to ECG or EDA."""
    if not cfg.get("cwt_enabled", True) or channel not in ("ecg", "eda"):
        return x
    import pywt
    low, high = cfg["cwt_ranges_hz"][channel]
    wavelet = cfg["cwt_wavelet"]
    freqs = np.geomspace(low, min(high, fs * 0.49), int(cfg["cwt_scales"]))
    scales = pywt.frequency2scale(wavelet, freqs / fs)
    pad = min(len(x)-1, 128) if len(x) > 1 else 0
    padded = np.pad(np.asarray(x, dtype=float), (pad, pad), mode="reflect") if len(x) > 1 else x
    coeff, _ = pywt.cwt(padded, scales, wavelet, sampling_period=1/fs)
    mag = np.abs(coeff)
    sigma = np.median(np.abs(mag - np.median(mag))) / 0.6745
    threshold = cfg.get("cwt_denoise_threshold_scale", 1.0) * sigma * np.sqrt(2*np.log(max(len(padded), 2)))
    shrunk = np.maximum(mag - threshold, 0.0)
    energy = np.sqrt(np.mean(shrunk**2, axis=0))
    original = np.sqrt(np.mean(mag**2, axis=0))
    gain = np.divide(energy, original, out=np.ones_like(energy), where=original > 1e-12)
    filtered = padded * np.clip(gain, 0.0, 1.0)
    return filtered[pad:pad+len(x)]


def extract_window_features(x: np.ndarray, fs: float) -> list[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [np.nan] * 5
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    diff_mean = float(np.mean(np.diff(x)))
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    denom = float(spectrum.sum())
    if denom <= 1e-20:
        centroid = rms_freq = 0.0
    else:
        centroid = float(np.sum(freqs * spectrum) / denom)
        rms_freq = float(np.sqrt(np.sum((freqs**2) * spectrum) / denom))
    return [mean, std, diff_mean, centroid, rms_freq]


def make_feature_matrix(windows: pd.DataFrame, weights: dict, scaler=None, fit=False):
    from sklearn.preprocessing import MinMaxScaler
    cols = [f"{ch}_{f}" for ch in CHANNELS for f in FEATURES_PER_CHANNEL]
    x = windows[cols].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median()).fillna(0).to_numpy(float)
    if scaler is None:
        scaler = MinMaxScaler()
    x = scaler.fit_transform(x) if fit else scaler.transform(x)
    channel_weights = np.repeat([weights[ch] for ch in CHANNELS], len(FEATURES_PER_CHANNEL))
    return (x * channel_weights).astype(np.float32), scaler
