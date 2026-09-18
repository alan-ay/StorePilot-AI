from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class StrategyConfig:
    """Business preferences used by the decision engine, not the forecaster."""

    name: str = "平衡"
    service_level: float = 0.95
    review_period_days: int = 3
    safety_days: float = 1.0
    holding_cost_weight: float = 1.0
    stockout_cost_weight: float = 2.0
    waste_cost_weight: float = 2.0
    max_purchase_budget: float | None = None

    @classmethod
    def preset(cls, name: str, budget: float | None = None) -> StrategyConfig:
        presets = {
            "保守": cls(
                name="保守",
                service_level=0.88,
                review_period_days=2,
                safety_days=0.5,
                holding_cost_weight=2.0,
                stockout_cost_weight=1.0,
                waste_cost_weight=2.5,
            ),
            "平衡": cls(),
            "增长": cls(
                name="增长",
                service_level=0.98,
                review_period_days=5,
                safety_days=2.0,
                holding_cost_weight=0.8,
                stockout_cost_weight=3.0,
                waste_cost_weight=1.5,
            ),
        }
        if name not in presets:
            raise ValueError(f"未知策略：{name}")
        return replace(presets[name], max_purchase_budget=budget)

    def validate(self) -> None:
        values = [
            self.service_level,
            self.safety_days,
            self.holding_cost_weight,
            self.stockout_cost_weight,
            self.waste_cost_weight,
        ]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("策略参数必须为非负有限数值")
        if self.max_purchase_budget is not None and not math.isfinite(self.max_purchase_budget):
            raise ValueError("采购预算必须为有限数值")
        if not 0.5 <= self.service_level < 1:
            raise ValueError("service_level 必须在 [0.5, 1) 范围内")
        if (
            not math.isfinite(self.review_period_days)
            or self.review_period_days < 1
            or self.review_period_days % 1
        ):
            raise ValueError("盘点周期必须为正整数")
        if self.max_purchase_budget is not None and self.max_purchase_budget < 0:
            raise ValueError("采购预算不能为负数")


@dataclass(frozen=True)
class ScenarioConfig:
    """Traceable what-if assumptions applied after the base forecast."""

    price_change_pct: float = 0.0
    promotion_lift_pct: float = 0.0
    traffic_change_pct: float = 0.0
    holiday_lift_pct: float = 0.0
    supplier_delay_days: int = 0
    price_elasticity: float = -1.2

    def validate(self) -> None:
        values = [
            self.price_change_pct,
            self.promotion_lift_pct,
            self.traffic_change_pct,
            self.holiday_lift_pct,
            self.price_elasticity,
            self.supplier_delay_days,
        ]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("情景参数必须为有限数值")
        if self.price_change_pct <= -100:
            raise ValueError("价格降幅必须小于 100%")
        if any(
            v < -100
            for v in [self.promotion_lift_pct, self.traffic_change_pct, self.holiday_lift_pct]
        ):
            raise ValueError("需求变化不能低于 -100%")
        if self.supplier_delay_days < 0 or self.supplier_delay_days % 1:
            raise ValueError("供应延迟必须为非负整数")
        if self.price_elasticity > 0:
            raise ValueError("本版价格弹性假设需为非正数")

    def demand_multiplier(self) -> float:
        self.validate()
        price_effect = 1 + self.price_elasticity * (self.price_change_pct / 100)
        multipliers = [
            max(0.0, price_effect),
            max(0.0, 1 + self.promotion_lift_pct / 100),
            max(0.0, 1 + self.traffic_change_pct / 100),
            max(0.0, 1 + self.holiday_lift_pct / 100),
        ]
        result = 1.0
        for value in multipliers:
            result *= value
        return result
