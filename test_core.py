import json
import unittest

import numpy as np
import pandas as pd

from association import apriori_rules, channel_weights
from data import make_windows, participant_baselines
from features import extract_window_features


class CoreTests(unittest.TestCase):
    def test_feature_definition(self):
        x = np.ones(128)
        feats = extract_window_features(x, 150)
        self.assertEqual(len(feats), 5)
        self.assertAlmostEqual(feats[0], 1.0)
        self.assertAlmostEqual(feats[1], 0.0)
        self.assertAlmostEqual(feats[2], 0.0)
        self.assertEqual(feats[3], 0.0)
        self.assertEqual(feats[4], 0.0)

    def test_apriori_and_channel_weights(self):
        tx = []
        for _ in range(60): tx.append(["EDA_high", "Unsafe"])
        for _ in range(20): tx.append(["EDA_low", "Normal"])
        for _ in range(20): tx.append(["EDA_medium", "Normal"])
        cfg = {"min_support": .08, "min_confidence_single": .6,
               "min_confidence_multi": .75, "min_lift": 1.5}
        rules = apriori_rules(tx, cfg)
        weights = channel_weights(rules)
        self.assertAlmostEqual(weights["eda"], 1.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_multirate_time_windows_and_boundary(self):
        fs_in = 20
        ts = np.arange(0, 10, 1 / fs_in)
        frame = pd.DataFrame({
            "participant": "p1", "recording_id": "task", "timestamp_s": ts,
            "label": np.zeros(len(ts), dtype=int),
            **{channel: np.sin(2 * np.pi * ts) for channel in ("ecg", "eda", "bp", "spo2", "skt")},
        })
        cfg = {
            "window_seconds": 10, "step_seconds": 5,
            "sampling_rate_hz": {"ecg": 150, "eda": 20, "bp": 10, "spo2": 20, "skt": 20},
            "cwt_enabled": False,
        }
        windows = make_windows(frame, cfg)
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows.iloc[0]["ecg_centroid_hz"], windows.iloc[0]["bp_centroid_hz"], delta=.2)

    def test_baseline_uses_exact_marked_five_minutes_and_rejects_short(self):
        ts = np.arange(0, 360, .5)
        rest = pd.DataFrame({"participant": "p1", "recording_id": "rest", "timestamp_s": ts,
                             "is_rest": 1, **{ch: ts for ch in ("ecg", "eda", "bp", "spo2", "skt")}})
        rest["label"] = 0
        task = rest.iloc[:10].copy()
        task["recording_id"] = "task"
        train_windows = pd.DataFrame({"participant": ["p1"], **{f"{ch}_mean": [999.0] for ch in ("ecg", "eda", "bp", "spo2", "skt")}})
        cfg = {"sampling_rate_hz": {ch: 2 for ch in ("ecg", "eda", "bp", "spo2", "skt")}}
        baseline = participant_baselines(train_windows, pd.concat([rest, task]), cfg)
        self.assertAlmostEqual(baseline["ecg"]["p1"][0], 149.75, places=2)
        self.assertNotAlmostEqual(baseline["ecg"]["p1"][0], 999.0)
        with self.assertRaisesRegex(ValueError, "continuous 300-second"):
            participant_baselines(train_windows, pd.concat([rest.iloc[:400], task]), cfg)


if __name__ == "__main__":
    unittest.main()
