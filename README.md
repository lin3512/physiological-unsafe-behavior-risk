# Physiological Unsafe-Behavior Risk Analysis

A configurable Python application for processing synchronized physiological recordings, estimating channel weights from training data, training a temporal classifier, and evaluating unsafe-behavior risk.

## Project files

- `main.py` runs the end-to-end workflow.
- `data.py` validates input, splits recordings, creates windows and sequences, and derives participant baselines.
- `features.py` preprocesses signals and calculates statistical and frequency-domain features.
- `association.py` mines training-set associations and calculates channel weights.
- `model.py` defines and trains the Transformer classifier.
- `evaluate.py` calculates evaluation metrics and optionally selects a threshold using validation data.
- `config.json` contains model and analysis settings.

## Requirements

Python 3.10 or newer is recommended. Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Input data

Provide a CSV with one synchronized row per timestamp and these columns:

| Column | Description |
| --- | --- |
| `participant` | Participant identifier |
| `recording_id` | Continuous recording identifier; recordings are kept together when splitting data |
| `timestamp_s` | Time in seconds within the recording |
| `ecg`, `eda`, `bp`, `spo2`, `skt` | Physiological channel values on the same timeline |
| `label` | `0` for Normal or `1` for Unsafe |
| `is_rest` | Optional flag (`1`) for baseline/rest rows |

All five signal columns must already be synchronized to the same rows. Resample or align slower channels before creating the input CSV. Each recording must contain enough samples to form complete windows. The stratified 80/10/10 split also needs enough independent recordings in both label groups.

## Run

From this directory:

```powershell
python main.py --input ..\data\synchronized_samples.csv --output results\metrics.json
```

The output path receives a JSON metrics report, a model checkpoint (`.checkpoint.pt`), and prediction arrays (`.predictions.npz`). To evaluate a separate synchronized dataset with the model trained during the same run, add `--external-input`:

```powershell
python main.py --input ..\data\training_samples.csv --external-input ..\data\external_samples.csv --output results\metrics.json
```

By default, predictions use the `unsafe_threshold` in `config.json`. Add `--select-threshold` to select a recall-oriented threshold on the validation split, subject to the specificity constraint implemented in `evaluate.py`.

## Configuration and data assumptions

Edit `config.json` to change sampling rates, window size and stride, sequence length, model dimensions, training settings, CWT settings, association-rule thresholds, or the decision threshold. The current pipeline uses 60-second windows with a 30-second stride and sequences of 17 windows. A sequence is labeled by its final window.

When `is_rest` is supplied, participant baselines are estimated from marked rest windows. Otherwise, training-window statistics are used. The CSV format is expected to contain already synchronized channels; the application does not align independent raw sensor streams.

ECG and EDA use the configurable CWT scale-energy attenuation implemented in `features.py`. It is an explicit preprocessing option and can be disabled with `cwt_enabled: false`. Tune and validate preprocessing against the intended sensor data before interpreting results.

## Tests

Run the included core tests from this directory:

```powershell
python -m unittest discover -v
```

## Data and responsible use

No participant data is included. Add only data that you are authorized to use, and remove identifiable or sensitive information before sharing datasets. The program reports model metrics; it is not a substitute for operational safety procedures or expert review.
