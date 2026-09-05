from __future__ import annotations

import pandas as pd

from .config import ScenarioConfig


def apply_scenario(forecast: pd.DataFrame, scenario: ScenarioConfig) -> pd.DataFrame:
    """Return a separate scenario forecast while preserving base predictions."""

    result = forecast.copy()
    multiplier = scenario.demand_multiplier()
    result["base_predicted_units"] = result["predicted_units"]
    for column in ("predicted_units", "lower_units", "upper_units"):
        result[column] = result[column] * multiplier
    result["scenario_multiplier"] = multiplier
    result["price_change_pct"] = scenario.price_change_pct
    result["promotion_lift_pct"] = scenario.promotion_lift_pct
    result["traffic_change_pct"] = scenario.traffic_change_pct
    result["holiday_lift_pct"] = scenario.holiday_lift_pct
    return result
