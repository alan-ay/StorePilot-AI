# Forecast evaluation

This benchmark answers a narrow question: how well does each method forecast
recorded sales when it can only use history available at an earlier date?
It is separate from the model's internal one-day validation shown in Data &
settings. Neither score is a direct measurement of inventory savings.

## Run from the workbench

Select **Forecast evaluation / 预测评估** in the sidebar, choose the number and
spacing of historical cutoffs, then run the evaluation. The selected store is
used; **All stores** evaluates the imported stores together with one global
model. The result is kept in the session and translated when the language changes.
Changing data, store or evaluation settings hides results that no longer match.

The dashboard defaults to three cutoffs seven days apart and horizons of 1, 7 and
30 days. This fits the bundled 140-day demo. Longer windows therefore overlap;
multiple forecasts can target the same observed day. These are descriptive scores,
not independent samples for significance testing.

The full ZIP contains `report.md`, `manifest.json`, `summary.csv`, `predictions.csv`,
`fold_scores.csv`, `sku_scores.csv` and `excluded.csv`. CSV field names and model
identifiers stay stable for analysis. The report and exclusion reasons follow the selected language.
Store data stays local; no external model service is called.

## Run from the terminal

From the repository, with the virtual environment activated:

```bash
python -m storepilot.evaluate --output evaluation_output
```

The command uses seed 42, 210 synthetic days ending on 2026-09-18, and three
cutoffs 30 days apart. These test windows do not overlap. The longer history and
spacing differ deliberately from the dashboard defaults. The generated directory
is ignored by Git; the checked-in [demo report](forecast-evaluation-demo.md) records
one run without including bulk prediction files.

Use your own CSV files with the same schemas as dashboard imports:

```bash
python -m storepilot.evaluate \
  --sales /path/to/sales.csv \
  --products /path/to/products.csv \
  --inventory /path/to/inventory.csv \
  --horizons 1 7 30 --origins 3 --step-days 30 \
  --output evaluation_output/store
```

All three files are required together. IDs are read as text to retain leading
zeros. Product and inventory files are validated for consistency; current stock
levels are not forecast features. Historical price records are recommended: the
usual import fallback fills missing sales prices from the current product catalog.
That fallback cannot reconstruct historical prices.

Use `--language zh` for a Chinese report. `--min-train-days`, `--seed`,
`--demo-days` and `--demo-end-date` are also configurable; see `--help`.
Output files in the chosen directory are replaced when rerunning the command.

## Protocol

1. The latest cutoff is the final sales date minus the longest requested horizon.
   Earlier cutoffs move back by the configured spacing.
2. Each store-product sales series must have consecutive daily records and at
   least 70 training days before the first cutoff. Late-starting or incomplete
   series are excluded explicitly, using the same cohort for every method and
   cutoff. Inventory-only items with no sales series are not forecast. All sales
   series must end on the same date, as required by the normal data validator.
3. A fresh copy of the current gradient-boosting forecaster is fitted at each
   cutoff. No later sales, prices or flags enter training. Multi-step predictions
   feed back earlier predictions, rather than future observed sales. Its existing
   stockout correction is applied only inside the training history. Future
   promotion is zero and relative price is one, as in the workbench forecast.
4. **Repeat last week** repeats the last seven observed daily sales values for
   every future week. **28-day average** predicts the last 28 observed days' mean
   on every future day. Neither baseline reads future sales.
5. Every method is scored against observed sales on days 1 through H after each
   cutoff. A seven-day result covers all seven days, not just the seventh day.

Required history is `min_train_days + max(horizons) + (origins - 1) * step_days`:
114 days for dashboard defaults, 160 for command-line defaults. Runs do not
silently reduce the number of cutoffs when history is insufficient.

## Metrics

For each forecast row, error is `predicted_units - actual_units`.

| Field | Meaning |
| --- | --- |
| `mae` | Mean absolute daily error, in units. Lower is better. |
| `wape` | Sum of absolute daily errors divided by sum of observed units. Stored as a fraction in CSV; shown as a percentage in the UI/report. Undefined when all observed units are zero. |
| `bias` | Mean signed daily error. Positive values mean overprediction. |
| `window_mae` | First total each store-product's predictions and actuals within each cutoff's window, then average the absolute differences. In units per store-product window. Daily errors can cancel within a window. |
| `non_stockout_mae` | Daily MAE only where `stockout == 0`. Undefined if every scored row is flagged. |
| `scored_rows` | Number of daily predictions scored for that method and horizon. |
| `stockout_rows` | Number of scored predictions whose observed day is flagged as a stockout. |

Summary daily metrics give each forecast row equal weight. Fold scores separate
cutoffs; SKU scores separate store-product series. No average of per-SKU percentage
errors is used. Missing metric values are blank in CSV and shown as a dash in the
report. The dataset hash, settings, model parameters, cutoff dates and software
versions are recorded in the manifest. Keep the original input files privately to
reproduce an imported-data experiment; they are not embedded in the report.
Match the manifest's package versions when reproducing a published result; the
local environment used for the demo report can differ from the Python 3.12 release
constraints. Small numerical differences across platforms are possible.

## What this establishes

The [recorded demo experiment](forecast-evaluation-demo.md) demonstrates that the
evaluation workflow runs reproducibly on the generator's sales patterns. It does
not establish performance on independent retail data. Before making that claim,
evaluate a suitably licensed real dataset and retain a final period that was not
used to tune the model or choose evaluation settings.

Observed sales understate demand during stockouts. The learned model's adjusted
training target and the baselines' observed-sales targets are different design
choices, so results should be interpreted with that distinction in mind. If an
upload omits stockout flags, they default to zero; this does not prove that every
day had adequate stock. The non-stockout metric does not recover lost demand.

This milestone evaluates point forecasts. It does not test prediction-interval
calibration, choose a new model automatically, retrain from written owner feedback,
or simulate purchasing costs. No production forecasts, policies, stock counts or
purchase records are changed by running an evaluation.
