"""Command-line entry point for physiological risk analysis."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from association import apriori_rules, channel_weights
from data import (discretize_windows, make_sequences, make_windows,
                  participant_baselines, split_recordings, validate_input)
from evaluate import evaluate, select_recall_threshold
from features import CHANNELS, FEATURES_PER_CHANNEL, make_feature_matrix
from model import predict_probabilities, train_transformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["seed"])
    raw = pd.read_csv(args.input)
    validate_input(raw)

    # Split original recording units before creating overlapping windows.
    train_raw, val_raw, test_raw = split_recordings(raw, cfg["split"], cfg["seed"])
    train, val, test = [make_windows(part, cfg) for part in (train_raw, val_raw, test_raw)]

    baselines = participant_baselines(train, train_raw)
    training_transactions = discretize_windows([train], baselines)[0]
    rules = apriori_rules(training_transactions, cfg)
    weights = channel_weights(rules)

    x_train, scaler = make_feature_matrix(train, weights, fit=True)
    x_val, _ = make_feature_matrix(val, weights, scaler)
    x_test, _ = make_feature_matrix(test, weights, scaler)
    seq_train, y_train, _ = make_sequences(train, x_train, cfg["sequence_length"])
    seq_val, y_val, _ = make_sequences(val, x_val, cfg["sequence_length"])
    seq_test, y_test, test_meta = make_sequences(test, x_test, cfg["sequence_length"])
    if min(len(seq_train), len(seq_val), len(seq_test)) == 0:
        raise ValueError("Not enough sequential windows in one or more splits")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_transformer(seq_train, y_train, seq_val, y_val, cfg, device)
    val_prob = predict_probabilities(model, seq_val, device)
    validation_threshold = select_recall_threshold(y_val, val_prob)
    threshold = validation_threshold if args.select_threshold else float(cfg["unsafe_threshold"])
    test_prob = predict_probabilities(model, seq_test, device)

    result = {
        "threshold": threshold,
        "validation_selected_threshold": validation_threshold,
        "channel_weights": weights,
        "train_windows": len(train), "val_windows": len(val), "test_windows": len(test),
        "sequence_counts": {"train": len(seq_train), "val": len(seq_val), "test": len(seq_test)},
        "test": evaluate(y_test, test_prob, threshold),
        "rules": rules,
        "data_split": "recording-level stratified 80/10/10",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.external_input:
        external_raw = pd.read_csv(args.external_input)
        validate_input(external_raw)
        external_windows = make_windows(external_raw, cfg)
        external_x, _ = make_feature_matrix(external_windows, weights, scaler)
        external_seq, external_y, external_meta = make_sequences(
            external_windows, external_x, cfg["sequence_length"])
        if not len(external_seq):
            raise ValueError("External input did not produce any length-17 sequences")
        external_prob = predict_probabilities(model, external_seq, device)
        result["external"] = {
            "windows": len(external_windows), "sequences": len(external_seq),
            "participants": sorted(map(str, external_windows.participant.unique())),
            "metrics": evaluate(external_y, external_prob, threshold),
        }
        np.savez_compressed(
            output.with_name(output.stem + ".external_predictions.npz"),
            y_true=external_y, probability=external_prob,
            participant=np.array([item[0] for item in external_meta], dtype=str),
            recording_id=np.array([item[1] for item in external_meta], dtype=str),
            window_start_s=np.array([item[2] for item in external_meta], dtype=float),
        )

    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({
        "model_state": model.state_dict(), "scaler": scaler, "weights": weights,
        "config": cfg, "threshold": threshold,
        "feature_order": [f"{ch}_{feature}" for ch in CHANNELS for feature in FEATURES_PER_CHANNEL],
    }, output.with_suffix(".checkpoint.pt"))
    np.savez_compressed(
        output.with_suffix(".predictions.npz"), y_true=y_test, probability=test_prob,
        participant=np.array([item[0] for item in test_meta], dtype=str),
        recording_id=np.array([item[1] for item in test_meta], dtype=str),
        window_start_s=np.array([item[2] for item in test_meta], dtype=float),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Long-form synchronized samples CSV")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output", default="results/metrics.json")
    parser.add_argument("--external-input", help="Separate external cohort CSV evaluated with the fixed trained model")
    parser.add_argument("--select-threshold", action="store_true",
                        help="Select a recall-oriented threshold on validation data")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
