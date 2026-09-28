# StorePilot forecast evaluation report

Recorded locally on 2026-09-20 with the software versions listed in the manifest.
Reproduce from this repository with:

```bash
python -m storepilot.evaluate --output evaluation_output
```

The 28-day average had the lowest daily MAE at 1 and 30 days. Gradient boosting
had the lowest daily MAE at 7 days, while the average still had the lowest error
in total seven-day demand. For 30-day totals, the average's MAE was 35.708 units
versus 68.914 for gradient boosting. Daily accuracy and ordering-window accuracy
can therefore favour different methods. This run does not support assuming that
the more complex model is best at every horizon.

These observations describe one synthetic experiment with three historical
cutoffs. They are not claims about real-store performance or statistical
significance. See [the evaluation guide](EVALUATION.md) for the full protocol.

Data source: Demo data. Dataset ID: 9012f6b54684be4e63ee54d6.
Sales dates: 2026-02-21 to 2026-09-18. Evaluated 8 store-product series; excluded 0.
Historical cutoffs: 2026-06-20, 2026-07-20, 2026-08-19.

## Method and metrics
Each model uses only records available at its cutoff. Gradient boosting forecasts recursively, one day at a time; the baselines repeat the last week or use a fixed trailing 28-day mean.
Gradient boosting retains the app's stockout-adjusted training target; both baselines use observed historical sales. Future promotions and prices use the app's default assumptions.
Each horizon scores days 1 through H after the cutoff. Daily MAE measures absolute daily errors; Total demand MAE measures absolute errors in each store-product's total sales over that window.
WAPE is the sum of absolute errors divided by total observed sales; it is undefined when observed sales are all zero. Positive bias means overprediction.
Targets are recorded sales, not estimated demand. Stockouts can suppress sales, so MAE for days without a stockout flag is shown separately; this does not establish accuracy for latent demand.
Series with missing daily records or insufficient training history are explicitly excluded. All three models use the same products and evaluation dates.

## Results

| Model | Horizon (days) | Daily MAE | WAPE (%) | Daily bias | Total demand MAE | Non-stockout MAE | Scored rows | Stockout rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient boosting | 1 | 4.363 | 22.138 | 0.848 | 4.363 | 3.881 | 24 | 1 |
| 28-day average | 1 | 3.571 | 18.121 | 0.646 | 3.571 | 3.121 | 24 | 1 |
| Repeat last week | 1 | 4.000 | 20.296 | 1.917 | 4.000 | 3.522 | 24 | 1 |
| Gradient boosting | 7 | 3.911 | 19.202 | 0.145 | 14.546 | 3.867 | 168 | 4 |
| 28-day average | 7 | 4.121 | 20.230 | -0.015 | 13.646 | 4.079 | 168 | 4 |
| Repeat last week | 7 | 4.845 | 23.787 | -0.298 | 15.667 | 4.750 | 168 | 4 |
| Gradient boosting | 30 | 4.713 | 23.142 | -0.216 | 68.914 | 4.611 | 720 | 28 |
| 28-day average | 30 | 4.393 | 21.575 | -0.010 | 35.708 | 4.247 | 720 | 28 |
| Repeat last week | 30 | 5.703 | 28.004 | -0.250 | 68.750 | 5.594 | 720 | 28 |

## Limitations
This experiment uses synthetic demo data. It does not establish forecast accuracy or business benefits for a real store.
This comparison does not automatically change the app's model, purchasing preferences, inventory or purchase records.
Results measure historical sales forecast errors, not inventory cost savings. This report does not evaluate interval coverage or implement automatic model selection.

## Reproducibility
The manifest below is also included as manifest.json in the full experiment download. That download contains individual predictions, scores by cutoff and product, and excluded series for auditing.

```json
{
  "protocol": "rolling-origin-v1",
  "config": {
    "horizons": [
      1,
      7,
      30
    ],
    "origins": 3,
    "step_days": 30,
    "min_train_days": 70,
    "seed": 42
  },
  "dataset_id": "9012f6b54684be4e63ee54d6",
  "sales_start": "2026-02-21",
  "sales_end": "2026-09-18",
  "evaluated_series": 8,
  "excluded_series": 0,
  "overlapping_windows": false,
  "folds": [
    {
      "origin": "2026-06-20",
      "training_rows": 960,
      "training_start": "2026-02-21",
      "test_end": "2026-07-20"
    },
    {
      "origin": "2026-07-20",
      "training_rows": 1200,
      "training_start": "2026-02-21",
      "test_end": "2026-08-19"
    },
    {
      "origin": "2026-08-19",
      "training_rows": 1440,
      "training_start": "2026-02-21",
      "test_end": "2026-09-18"
    }
  ],
  "model_parameters": {
    "categorical_features": "from_dtype",
    "early_stopping": "auto",
    "interaction_cst": null,
    "l2_regularization": 1.0,
    "learning_rate": 0.06,
    "loss": "squared_error",
    "max_bins": 255,
    "max_depth": null,
    "max_features": 1.0,
    "max_iter": 220,
    "max_leaf_nodes": 24,
    "min_samples_leaf": 20,
    "monotonic_cst": null,
    "n_iter_no_change": 10,
    "quantile": null,
    "random_state": 42,
    "scoring": "loss",
    "tol": 1e-07,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": false
  },
  "versions": {
    "python": "3.14.5",
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "scikit-learn": "1.8.0",
    "streamlit": "1.63.0"
  },
  "demo_generation": {
    "catalog_version": "household-single-store-v2",
    "days": 210,
    "end_date": "2026-09-18",
    "seed": 42
  }
}
```
