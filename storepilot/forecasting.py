from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FEATURE_COLUMNS = [
    "store_code",
    "sku_code",
    "day_of_week",
    "week_of_year",
    "month",
    "time_index",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_7",
    "rolling_28",
    "promotion",
    "relative_price",
]


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    wape: float
    interval_coverage: float
    validation_rows: int
    model_name: str = "HistGradientBoostingRegressor"


class AdaptiveDemandForecaster:
    """Global tabular time-series model with iterative multi-step forecasting."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: HistGradientBoostingRegressor | None = None
        self.history: pd.DataFrame | None = None
        self.metrics: ForecastMetrics | None = None
        self.residual_quantiles = (-2.0, 2.0)
        self.store_codes: dict[str, int] = {}
        self.sku_codes: dict[str, int] = {}
        self.base_prices: dict[str, float] = {}

    @staticmethod
    def _daily_grid(sales: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for (store_id, sku), group in sales.groupby(["store_id", "sku"], sort=False):
            group = group.sort_values("date").copy()
            dates = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
            group = group.set_index("date").reindex(dates)
            group.index.name = "date"
            group["store_id"] = store_id
            group["sku"] = sku
            group["units"] = group["units"].fillna(0.0)
            group["promotion"] = group["promotion"].fillna(0).astype(int)
            group["stockout"] = group["stockout"].fillna(0).astype(int)
            group["price"] = group["price"].ffill().bfill()
            frames.append(group.reset_index())
        return pd.concat(frames, ignore_index=True)

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["store_code"] = result["store_id"].map(self.store_codes).fillna(-1)
        result["sku_code"] = result["sku"].map(self.sku_codes).fillna(-1)
        return result

    def _features(self, history: pd.DataFrame) -> pd.DataFrame:
        frame = history.sort_values(["store_id", "sku", "date"]).copy()
        group = frame.groupby(["store_id", "sku"], sort=False)["demand_target"]
        frame["lag_1"] = group.shift(1)
        frame["lag_7"] = group.shift(7)
        frame["lag_14"] = group.shift(14)
        frame["rolling_7"] = group.transform(lambda values: values.shift(1).rolling(7).mean())
        frame["rolling_28"] = group.transform(lambda values: values.shift(1).rolling(28).mean())
        frame["day_of_week"] = frame["date"].dt.dayofweek
        frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
        frame["month"] = frame["date"].dt.month
        origin = frame["date"].min()
        frame["time_index"] = (frame["date"] - origin).dt.days
        frame["relative_price"] = frame.apply(
            lambda row: (
                row["price"] / max(self.base_prices.get(str(row["sku"]), row["price"]), 0.01)
            ),
            axis=1,
        )
        return self._encode(frame)

    def fit(self, sales: pd.DataFrame) -> AdaptiveDemandForecaster:
        history = self._daily_grid(sales)
        self.store_codes = {
            value: index
            for index, value in enumerate(sorted(history["store_id"].astype(str).unique()))
        }
        self.sku_codes = {
            value: index for index, value in enumerate(sorted(history["sku"].astype(str).unique()))
        }
        self.base_prices = history.groupby("sku")["price"].median().to_dict()

        # Observed sales during a stockout are a lower bound on true demand.
        rolling_reference = history.groupby(["store_id", "sku"])["units"].transform(
            lambda values: values.shift(1).rolling(14, min_periods=3).median()
        )
        history["demand_target"] = np.where(
            history["stockout"].eq(1),
            np.maximum(history["units"], rolling_reference.fillna(history["units"])),
            history["units"],
        ).astype(float)
        self.history = history
        features = self._features(history).dropna(subset=FEATURE_COLUMNS)
        if len(features) < 40:
            raise ValueError("有效历史数据不足：至少需要约 6 周的每日销售数据")

        cutoff = features["date"].max() - pd.Timedelta(days=13)
        train = features[features["date"] < cutoff]
        valid = features[features["date"] >= cutoff]
        if len(train) < 30 or valid.empty:
            dates = sorted(features["date"].unique())
            cutoff = dates[max(1, int(len(dates) * 0.8))]
            train, valid = features[features["date"] < cutoff], features[features["date"] >= cutoff]
        if len(train) < 30 or len(valid) < 7:
            raise ValueError("可验证的历史不足，建议导入至少 70 天的每日销量")

        self.model = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=220,
            max_leaf_nodes=24,
            l2_regularization=1.0,
            random_state=self.random_state,
        )
        self.model.fit(train[FEATURE_COLUMNS], train["demand_target"])
        valid_prediction = np.maximum(0, self.model.predict(valid[FEATURE_COLUMNS]))
        residuals = valid["demand_target"].to_numpy() - valid_prediction
        self.residual_quantiles = (
            min(0.0, float(np.quantile(residuals, 0.10))),
            max(0.0, float(np.quantile(residuals, 0.90))),
        )
        actual = valid["demand_target"].to_numpy()
        lower = np.maximum(0, valid_prediction + self.residual_quantiles[0])
        upper = np.maximum(lower, valid_prediction + self.residual_quantiles[1])
        self.metrics = ForecastMetrics(
            mae=float(mean_absolute_error(actual, valid_prediction)),
            wape=float(np.abs(actual - valid_prediction).sum() / max(actual.sum(), 1.0)),
            interval_coverage=float(np.mean((actual >= lower) & (actual <= upper))),
            validation_rows=len(valid),
        )
        # Refit on all available actuals: calling fit again is the adaptive update path.
        self.model.fit(features[FEATURE_COLUMNS], features["demand_target"])
        return self

    def predict(self, horizon_days: int = 30) -> pd.DataFrame:
        if self.model is None or self.history is None:
            raise RuntimeError("请先调用 fit()")
        if not 1 <= horizon_days <= 90:
            raise ValueError("预测周期必须在 1 到 90 天之间")

        working = self.history.copy()
        predictions: list[dict] = []
        groups = working[["store_id", "sku"]].drop_duplicates().to_records(index=False)
        origin = working["date"].min()
        last_date = working["date"].max()

        for step in range(1, horizon_days + 1):
            forecast_date = last_date + pd.Timedelta(days=step)
            new_rows: list[dict] = []
            for store_id, sku in groups:
                group = working[(working["store_id"] == store_id) & (working["sku"] == sku)]
                values = group["demand_target"].to_numpy(dtype=float)
                price = self.base_prices[str(sku)]
                feature_row = pd.DataFrame(
                    [
                        {
                            "store_code": self.store_codes[str(store_id)],
                            "sku_code": self.sku_codes[str(sku)],
                            "day_of_week": forecast_date.dayofweek,
                            "week_of_year": int(forecast_date.isocalendar().week),
                            "month": forecast_date.month,
                            "time_index": (forecast_date - origin).days,
                            "lag_1": values[-1],
                            "lag_7": values[-7] if len(values) >= 7 else values[-1],
                            "lag_14": values[-14] if len(values) >= 14 else values[-1],
                            "rolling_7": values[-7:].mean(),
                            "rolling_28": values[-28:].mean(),
                            "promotion": 0,
                            "relative_price": 1.0,
                        }
                    ]
                )
                point = float(max(0, self.model.predict(feature_row[FEATURE_COLUMNS])[0]))
                lower = float(max(0, point + self.residual_quantiles[0]))
                upper = float(max(lower, point + self.residual_quantiles[1]))
                predictions.append(
                    {
                        "date": forecast_date,
                        "store_id": str(store_id),
                        "sku": str(sku),
                        "predicted_units": point,
                        "lower_units": lower,
                        "upper_units": upper,
                    }
                )
                new_rows.append(
                    {
                        "date": forecast_date,
                        "store_id": str(store_id),
                        "sku": str(sku),
                        "units": point,
                        "demand_target": point,
                        "promotion": 0,
                        "stockout": 0,
                        "price": price,
                    }
                )
            working = pd.concat([working, pd.DataFrame(new_rows)], ignore_index=True)

        return pd.DataFrame(predictions)

    def trend_summary(self, forecast: pd.DataFrame) -> pd.DataFrame:
        if self.history is None:
            raise RuntimeError("请先调用 fit()")
        recent = (
            self.history.sort_values("date")
            .groupby(["store_id", "sku"], as_index=False)
            .tail(28)
            .groupby(["store_id", "sku"], as_index=False)["demand_target"]
            .mean()
            .rename(columns={"demand_target": "historical_daily_avg"})
        )
        future = (
            forecast.sort_values("date")
            .groupby(["store_id", "sku"], as_index=False)
            .head(7)
            .groupby(["store_id", "sku"], as_index=False)["predicted_units"]
            .mean()
            .rename(columns={"predicted_units": "forecast_daily_avg"})
        )
        result = recent.merge(future, on=["store_id", "sku"])
        result["trend_pct"] = (
            100
            * (result["forecast_daily_avg"] - result["historical_daily_avg"])
            / result["historical_daily_avg"].clip(lower=0.5)
        )
        result["trend"] = np.select(
            [result["trend_pct"] >= 10, result["trend_pct"] <= -10],
            ["增长", "下降"],
            default="稳定",
        )
        return result
