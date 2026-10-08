# Freight Rate Prediction (Spotter AI Assessment)

Predicts the posted rate (USD) for 12,000 loads in November–December 2025 from 48,000 loads dated January–October 2025, and produces the fixed December prediction chart.

## Approach in brief

`quote_signal` tracks the true rate per mile closely in some months and is mostly noise in others. In training it is clean in Jan, Feb, Mar, Jun and Sep, degraded in Apr, May, Jul and Oct, and about half-degraded in Aug. Validation (Nov–Dec) looks half-degraded too.

The submitted model is a **two-stage, regime-aware model**:

1. **Stage 1:** Ridge regression on log(rate per mile) using lane, equipment, weight, distance and calendar features. It never sees `quote_signal`.
2. **Stage 2:** gradient boosting on the stage-1 residual. It gets `quote_signal` plus agreement features: the row's gap to the stage-1 estimate, the share of that week's loads where the two agree within 10%, and the week's median gap. These use features only, so they work on validation weeks that have no labels.

The December chart input has only seven columns (no `quote_signal`, no coordinates), so the December predictions come from the stage-1-style Ridge model on base features. The chart does not use `quote_signal`.

## Results (cross-validation only)

Final validation metrics are calculated by Spotter after submission.

| Model | Leave-one-month-out MAE, 10 months | MAE on regime-matched months* |
|---|---|---|
| Ridge, no `quote_signal` | 74.1 | 55.4 |
| Hybrid Ridge + boosting, with `quote_signal` | 55.4 | 54.7 |
| **Two-stage (submitted)** | **33.6** | **36.6** |

\*Months whose `quote_signal` agreement is within 0.15 of validation's 0.47: Apr, May, Jul, Aug, Oct.

Week-blocked check (hold out one week, keep the rest of the month in training):

| Month | Ridge | Two-stage |
|---|---|---|
| Aug (closest to validation) | 47.1 | 46.6 |
| Oct | 43.1 | 32.3 |

In August the two models are effectively tied. Expect validation MAE between roughly 32 and 47.

## Repository layout

| File | Purpose |
|---|---|
| `train_predict.py` | Cleaning, features, cross-validation, model selection, final fit, predictions |
| `eda.py` | Exploratory data analysis (prints a summary, saves `eda_outputs/eda_overview.png`) |
| `regime_check.py` | Leave-one-week-out check inside a month: two-stage vs Ridge |
| `august_diagnosis.py` | Splits held-out rows by true `quote_signal` quality; shows the headroom left |
| `score.py` | Provided by Spotter: validates the submission files and draws the December chart |
| `validation_predictions.csv` | Submission file (`load_id,predicted_rate`, 12,000 rows) |
| `december_predictions.csv` | December input with `predicted_rate` filled |
| `cv_results.csv`, `run_summary.json` | Per-fold metrics and the choices the run made |

## Setup

Python 3.10 or newer.

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the provided files in `data/` (or `Data/`; both are found automatically):
`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`.

## Run

```bash
python eda.py                 # optional: data exploration
python train_predict.py       # cleans, cross-validates, fits, writes the two prediction files
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The last command validates both files and writes the December chart to `scorer_results/candidate_december.png`.

Optional diagnostics:

```bash
python regime_check.py --months 2025-08 2025-10
python august_diagnosis.py --months 2025-08 2025-10
```

### Options for `train_predict.py`

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | auto (`data`, `Data`) | Folder containing the CSVs |
| `--out-dir` | `.` | Where outputs are written |
| `--cv-mode` | `lomo` | `lomo` = leave-one-month-out; `forward` = expanding-window time CV |
| `--cv-folds`, `--gap-months` | 3, 1 | Used by `--cv-mode forward` only; ignored in `lomo` |
| `--val-features` | `auto` | Force `base`, `quote` or `full` for validation predictions |
| `--regime-tol` | 0.15 | Select models on months whose agreement is within this of validation's |
| `--calendar` | `weekly` | `weekly` = weekday and day-of-month only; `full` adds month/week/day-of-year |
| `--models`, `--variants` | all | Comma-separated subsets to evaluate |

## Data-quality handling

| Issue found | Count | Treatment |
|---|---|---|
| Negative weights (sign flips) | 292 | Absolute value |
| Missing weights | 300 | Equipment median (valid range 1,000–60,000 lb) |
| Rate-per-mile outliers (max $14.1/mile vs median $2.15) | 677 | Dropped: robust z-score > 6 within equipment × distance decile |
| Duplicates, invalid dates, rates or distances | 0 | None found; checks stay in the pipeline |
| Cities in validation unseen in training (12.1% of rows) | 8 cities | Coordinates taken from validation rows (features, not labels) |
| Missing weights in validation (1.4%) | | Filled from training equipment median; validation rows are never dropped |

After cleaning, 47,323 training rows remain.

## Validation design

- Validation starts after training ends, and no training row is in December, so every split holds out whole blocks of time. No random split.
- **Leave-one-month-out** (10 folds) tests every `quote_signal` regime. It also trains on later months, so it is a regime-matched check, not a forecast.
- **Model selection** uses the months whose `quote_signal` agreement resembles validation's.
- **Week-blocked check** holds out one week and drops its two neighbours, so August's regime can be tested with the rest of August in training.
- **Leakage controls:** row-dropping uses training data only; fold coordinates come from the fold's training rows; weekly agreement never uses the label; calendar features exclude month, week and day-of-year, which cannot extrapolate to unseen months.

## Known limitations

- Only cross-validation numbers are available; Spotter computes the final score.
- In half-degraded months `quote_signal` is noisy even on its good rows (about 5% mean error), so the gain over Ridge is small there. A perfect good/bad gate would add about 1 MAE in August.
- The December chart uses weekday as a single numeric feature in the Ridge model, which adds a small weekly saw-tooth (about $3 on $820).
- Random seeds are fixed (`SEED = 42`); results should reproduce on the same library versions.
