from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from storepilot.config import StrategyConfig
from storepilot.data import generate_demo_data, validate_data
from storepilot.forecasting import AdaptiveDemandForecaster
from storepilot.optimization import InventoryOptimizer


def test_short_shared_history_raises_validation_error():
    data = generate_demo_data(days=29, end_date=date(2026, 9, 5))
    # Enough rows across stores, but only one date remains after the 28-day lag.
    one_store = data.sales[data.sales.store_id == "STORE-01"]
    sales = pd.concat([one_store.assign(store_id=f"S{i}") for i in range(5)])
    with pytest.raises(ValueError, match="历史不足"):
        AdaptiveDemandForecaster().fit(sales)


@pytest.mark.parametrize("period", [float("nan"), float("inf"), 1.5, 0])
def test_invalid_review_period_is_rejected(period):
    with pytest.raises(ValueError):
        replace(StrategyConfig(), review_period_days=period).validate()


def test_zero_demand_does_not_create_purchase_or_stockout_warning():
    data = generate_demo_data(days=1)
    forecast = pd.DataFrame(
        {
            "date": pd.date_range("2026-09-06", periods=30),
            "store_id": "STORE-01",
            "sku": "SKU001",
            "predicted_units": 0.0,
            "lower_units": 0.0,
            "upper_units": 0.0,
        }
    )
    inventory = data.inventory.head(1).assign(on_hand=0, on_order=0)
    result = InventoryOptimizer().recommend(forecast, data.products, inventory, StrategyConfig())
    assert result.iloc[0].suggested_order_qty == 0
    assert result.iloc[0].status == "正常"
    assert result.iloc[0].daily_demand == 0


def test_budget_can_buy_exactly_affordable_case():
    recommendations = pd.DataFrame(
        [
            {
                "priority_score": 2,
                "purchase_cost": 0.1,
                "unit_cost": 0.1,
                "suggested_order_qty": 1,
                "case_pack": 1,
                "min_order_qty": 1,
            },
            {
                "priority_score": 1,
                "purchase_cost": 1.0,
                "unit_cost": 0.1,
                "suggested_order_qty": 10,
                "case_pack": 1,
                "min_order_qty": 1,
            },
        ]
    )
    result = InventoryOptimizer._apply_budget(recommendations, 0.3)
    assert result.iloc[1].suggested_order_qty == 2
    assert result.purchase_cost.sum() == pytest.approx(0.3)


def test_mixed_timezone_dates_raise_validation_error():
    data = generate_demo_data(days=3)
    data.sales["date"] = data.sales.date.astype(str)
    data.sales.loc[0, "date"] += "T00:00:00+08:00"
    with pytest.raises(ValueError):
        validate_data(data)


def test_forecast_horizon_rejects_fractional_values():
    model = AdaptiveDemandForecaster()
    model.model = object()
    model.history = pd.DataFrame()
    with pytest.raises(ValueError):
        model.predict(1.5)
