import io
import json
from datetime import date
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from storepilot.data import RetailData, generate_demo_data
from storepilot.evaluate import main
from storepilot.evaluation import (
    EvaluationConfig,
    evaluate_forecasts,
    evaluation_archive,
    evaluation_report,
    score_predictions,
)
from storepilot.i18n import ValidationError


def sample(days=140):
    data = generate_demo_data(days, seed=7, end_date=date(2026, 9, 18))
    return RetailData(
        data.sales[data.sales.sku.isin(["SKU001", "SKU002"])].copy(),
        data.products[data.products.sku.isin(["SKU001", "SKU002"])].copy(),
        data.inventory[data.inventory.sku.isin(["SKU001", "SKU002"])].copy(),
    )


@pytest.fixture
def fit_history(monkeypatch):
    history = []

    class Forecaster:
        def __init__(self, random_state):
            self.model = SimpleNamespace(get_params=lambda: {"random_state": random_state})

        def fit(self, sales):
            history.append(sales.copy())
            self.sales = sales
            return self

        def predict(self, horizon):
            return pd.DataFrame(
                [
                    {
                        "store_id": store,
                        "sku": sku,
                        "date": self.sales.date.max() + pd.Timedelta(days=day),
                        "predicted_units": 0.0,
                    }
                    for store, sku in self.sales[["store_id", "sku"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                    for day in range(1, horizon + 1)
                ]
            )

    monkeypatch.setattr("storepilot.evaluation.AdaptiveDemandForecaster", Forecaster)
    return history


def test_metrics_measure_daily_and_total_demand_separately():
    frame = pd.DataFrame(
        {
            "model": ["baseline"] * 3,
            "store_id": ["s"] * 3,
            "sku": ["p"] * 3,
            "origin": [pd.Timestamp("2026-01-01")] * 3,
            "lead_day": [1, 2, 3],
            "actual_units": [10, 0, 10],
            "predicted_units": [12, 3, 5],
            "stockout": [0, 1, 0],
        }
    )
    scores = score_predictions(frame, (1, 3)).set_index("horizon_days")
    assert scores.loc[1, "mae"] == 2
    assert scores.loc[3, "mae"] == pytest.approx(10 / 3)
    assert scores.loc[3, "wape"] == 0.5
    assert scores.loc[3, "window_mae"] == 0  # Daily errors cancel in the window total.
    assert scores.loc[3, "bias"] == 0
    assert scores.loc[3, "non_stockout_mae"] == 3.5
    assert scores.loc[3, "stockout_rows"] == 1
    zero = frame.assign(actual_units=0, stockout=1)
    undefined = score_predictions(zero, (3,)).iloc[0]
    assert np.isnan(undefined.wape)
    assert np.isnan(undefined.non_stockout_mae)


def test_cutoffs_and_baselines_use_only_available_history(fit_history):
    data = sample()
    before = data.sales.copy(deep=True)
    result = evaluate_forecasts(data)
    assert len(fit_history) == 3
    assert len(result.predictions) == 3 * 3 * 2 * 30
    for train, fold in zip(fit_history, result.manifest["folds"], strict=True):
        cutoff = pd.Timestamp(fold["origin"])
        assert train.date.max() == cutoff
        pd.testing.assert_frame_equal(
            train.sort_values(["date", "sku"]).reset_index(drop=True),
            data.sales[data.sales.date <= cutoff]
            .sort_values(["date", "sku"])
            .reset_index(drop=True),
        )
        for sku in ["SKU001", "SKU002"]:
            units = train[train.sku == sku].sort_values("date").units
            forecast = result.predictions[
                result.predictions.origin.eq(cutoff) & result.predictions.sku.eq(sku)
            ]
            np.testing.assert_array_equal(
                forecast[forecast.model == "seasonal_naive"].predicted_units,
                np.resize(units.tail(7).to_numpy(), 30),
            )
            assert (
                forecast[forecast.model == "moving_average_28"].predicted_units
                == units.tail(28).mean()
            ).all()
    pd.testing.assert_frame_equal(before, data.sales)
    assert result.manifest["overlapping_windows"]
    assert set(result.scores.model) == {"seasonal_naive", "moving_average_28", "gradient_boosting"}


def test_actual_model_predictions_do_not_change_when_future_observations_change():
    data = sample(100)
    config = EvaluationConfig(horizons=(1, 7), origins=1)
    original = evaluate_forecasts(data, config)
    altered = RetailData(data.sales.copy(), data.products.copy(), data.inventory.copy())
    future = altered.sales.date > pd.Timestamp(original.manifest["folds"][0]["origin"])
    altered.sales.loc[future, "units"] += 500
    altered.sales.loc[future, "price"] *= 3
    altered.sales.loc[future, "promotion"] = 1
    altered.sales.loc[future, "stockout"] = 1
    changed = evaluate_forecasts(altered, config)
    columns = ["origin", "date", "model", "store_id", "sku", "predicted_units"]
    pd.testing.assert_frame_equal(original.predictions[columns], changed.predictions[columns])
    assert not original.scores.equals(changed.scores)
    assert original.manifest["dataset_id"] != changed.manifest["dataset_id"]
    assert not original.manifest["overlapping_windows"]


@pytest.mark.parametrize("missing", ["gap", "late_start"])
def test_incomplete_series_are_reported_and_all_models_use_same_cohort(fit_history, missing):
    data = sample()
    early = data.sales.date < data.sales.date.max() - pd.Timedelta(days=60)
    bad = data.sales.sku.eq("SKU002")
    if missing == "gap":
        early = data.sales.date.eq(data.sales.date.max() - pd.Timedelta(days=5))
    data.sales = data.sales[~(early & bad)]
    result = evaluate_forecasts(data)
    assert result.excluded.sku.tolist() == ["SKU002"]
    assert result.manifest["evaluated_series"] == 1
    assert set(result.predictions.sku) == {"SKU001"}
    assert all(set(train.sku) == {"SKU001"} for train in fit_history)
    expected = "缺少每日记录" if missing == "gap" else "训练历史不足"
    assert expected in result.excluded.iloc[0].reason
    with ZipFile(io.BytesIO(evaluation_archive(result, "演示数据", "en"))) as archive:
        exported = pd.read_csv(io.BytesIO(archive.read("excluded.csv")))
        reason = exported.iloc[0].reason
        assert "missing" in reason if missing == "gap" else "Insufficient" in reason


def test_insufficient_history_does_not_silently_reduce_folds(fit_history):
    with pytest.raises(ValidationError, match="114"):
        evaluate_forecasts(sample(100))
    assert fit_history == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizons": ()},
        {"horizons": (7, 7)},
        {"horizons": (True,)},
        {"horizons": (1.5,)},
        {"horizons": (91,)},
        {"origins": 0},
        {"origins": True},
        {"step_days": 0},
        {"step_days": 1.5},
        {"min_train_days": 69},
        {"seed": -1},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValidationError):
        EvaluationConfig(**kwargs).validate()


def test_report_and_archive_include_auditable_results_in_both_languages(fit_history):
    result = evaluate_forecasts(sample())
    english = evaluation_report(result, "演示数据", "en")
    assert "synthetic demo data" in english
    assert "windows overlap" in english
    assert "Total demand MAE" in english
    assert "库存" not in english
    chinese = evaluation_report(result, "演示数据", "zh")
    assert "窗口总量 MAE" in chinese
    assert result.manifest["dataset_id"] in chinese
    with ZipFile(io.BytesIO(evaluation_archive(result, "演示数据", "en"))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["source"] == "Demo data"
        assert manifest["config"]["horizons"] == [1, 7, 30]
        assert manifest["versions"]["scikit-learn"]
        assert archive.read("report.md").decode() == english
        raw = pd.read_csv(io.BytesIO(archive.read("predictions.csv")))
        assert len(raw) == len(result.predictions)
        # Recompute the published summary from exported individual forecasts.
        recomputed = score_predictions(raw, (1, 7, 30))
        pd.testing.assert_frame_equal(recomputed, result.scores)


def test_zero_sales_export_leaves_undefined_metrics_blank(fit_history):
    data = sample()
    data.sales["units"] = 0.0
    data.sales["stockout"] = 1
    result = evaluate_forecasts(data)
    with ZipFile(io.BytesIO(evaluation_archive(result, "演示数据", "en"))) as archive:
        summary = pd.read_csv(io.BytesIO(archive.read("summary.csv")))
        assert summary.wape.isna().all()
        assert summary.non_stockout_mae.isna().all()
        assert summary.mae.eq(0).all()


def test_cli_creates_reproducible_bundle(tmp_path, fit_history):
    main(["--output", str(tmp_path), "--origins", "1", "--demo-days", "100"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["demo_generation"]["end_date"] == "2026-09-18"
    assert manifest["sales_end"] == "2026-09-18"
    assert manifest["config"]["seed"] == 42
    assert manifest["config"]["step_days"] == 30
    assert manifest["source"] == "Demo data"
    assert {p.name for p in tmp_path.iterdir()} == {
        "report.md",
        "manifest.json",
        "summary.csv",
        "predictions.csv",
        "fold_scores.csv",
        "sku_scores.csv",
        "excluded.csv",
    }


def test_cli_requires_complete_import_group(tmp_path, fit_history):
    with pytest.raises(SystemExit) as exc:
        main(["--sales", "sales.csv", "--output", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())
    assert not fit_history


def test_cli_reads_imported_identifiers_as_text(tmp_path, fit_history):
    data = sample(100)
    paths = []
    for name, frame in [
        ("sales", data.sales),
        ("products", data.products),
        ("inventory", data.inventory),
    ]:
        frame = frame.copy()
        frame["sku"] = frame.sku.map({"SKU001": "001", "SKU002": "002"})
        if "store_id" in frame:
            frame["store_id"] = "0001"
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.extend([f"--{name}", str(path)])
    output = tmp_path / "results"
    main([*paths, "--origins", "1", "--output", str(output)])
    assert set(fit_history[0].sku) == {"001", "002"}
    assert set(fit_history[0].store_id) == {"0001"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"] == "Store data"
    assert "demo_generation" not in manifest
