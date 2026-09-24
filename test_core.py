import json
import unittest

import numpy as np

from association import apriori_rules, channel_weights
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


if __name__ == "__main__":
    unittest.main()
