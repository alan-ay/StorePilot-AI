from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ScenarioConfig, StrategyConfig
from .data import RetailData, validate_data
from .forecasting import AdaptiveDemandForecaster, ForecastMetrics
from .optimization import InventoryOptimizer, build_purchase_orders
from .reporting import generate_chinese_report
from .scenarios import apply_scenario


@dataclass
class PipelineResult:
    base_forecast: pd.DataFrame
    adjusted_forecast: pd.DataFrame
    trends: pd.DataFrame
    recommendations: pd.DataFrame
    transfers: pd.DataFrame
    purchase_orders: pd.DataFrame
    report: str
    metrics: ForecastMetrics


class StorePilotPipeline:
    def __init__(self) -> None:
        self.forecaster = AdaptiveDemandForecaster()
        self.optimizer = InventoryOptimizer()

    def run(
        self,
        data: RetailData,
        strategy: StrategyConfig | None = None,
        scenario: ScenarioConfig | None = None,
        horizon_days: int = 30,
    ) -> PipelineResult:
        data = validate_data(data)
        strategy = strategy or StrategyConfig.preset("平衡")
        scenario = scenario or ScenarioConfig()
        strategy.validate()
        scenario.validate()
        if not 1 <= horizon_days <= 90 or horizon_days % 1:
            raise ValueError("预测周期必须为 1 到 90 天之间的整数")
        forecast_days = max(
            horizon_days,
            30,
            int(data.products.lead_time_days.max())
            + scenario.supplier_delay_days
            + strategy.review_period_days,
        )
        if forecast_days > 90:
            raise ValueError("交货与补货周期合计不能超过 90 天")
        self.forecaster.fit(data.sales)
        base_forecast = self.forecaster.predict(int(forecast_days))
        adjusted = apply_scenario(base_forecast, scenario)
        trends = self.forecaster.trend_summary(base_forecast)
        recommendations = self.optimizer.recommend(
            adjusted, data.products, data.inventory, strategy, scenario
        )
        transfers = self.optimizer.suggest_transfers(recommendations)
        purchase_orders = build_purchase_orders(recommendations)
        report = generate_chinese_report(
            data.sales, trends, recommendations, transfers, strategy.name
        )
        assert self.forecaster.metrics is not None
        return PipelineResult(
            base_forecast=base_forecast,
            adjusted_forecast=adjusted,
            trends=trends,
            recommendations=recommendations,
            transfers=transfers,
            purchase_orders=purchase_orders,
            report=report,
            metrics=self.forecaster.metrics,
        )
