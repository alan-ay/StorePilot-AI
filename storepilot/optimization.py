from __future__ import annotations

import math
from decimal import Decimal
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import ScenarioConfig, StrategyConfig


def _round_order(quantity: float, case_pack: int, minimum: int) -> int:
    if quantity <= 0:
        return 0
    pack = max(1, int(case_pack))
    return int(math.ceil(max(quantity, minimum) / pack) * pack)


class InventoryOptimizer:
    """Cost-aware reorder and transfer decisions on top of demand forecasts."""

    def recommend(
        self,
        forecast: pd.DataFrame,
        products: pd.DataFrame,
        inventory: pd.DataFrame,
        strategy: StrategyConfig,
        scenario: ScenarioConfig | None = None,
    ) -> pd.DataFrame:
        strategy.validate()
        scenario = scenario or ScenarioConfig()
        scenario.validate()
        product_map = products.set_index("sku")
        rows: list[dict] = []

        for stock in inventory.itertuples(index=False):
            sku = str(stock.sku)
            if sku not in product_map.index:
                continue
            product = product_map.loc[sku]
            sku_forecast = forecast[
                (forecast["store_id"] == str(stock.store_id)) & (forecast["sku"] == sku)
            ].sort_values("date")
            if sku_forecast.empty:
                continue
            lead_time = max(1, int(product["lead_time_days"]) + scenario.supplier_delay_days)
            protection_days = int(lead_time + strategy.review_period_days)
            if len(sku_forecast) < protection_days:
                raise ValueError("预测天数不足以覆盖交货周期与补货周期")
            protected = sku_forecast.head(protection_days)
            expected_demand = float(protected["predicted_units"].sum())
            # A zero-demand scenario also has zero uncertainty; do not invent safety stock.
            minimum_std = 0.5 if protected["upper_units"].gt(0).any() else 0.0
            daily_std = float(
                np.maximum(
                    minimum_std, (protected["upper_units"] - protected["lower_units"]) / 2.56
                ).mean()
            )
            z_score = NormalDist().inv_cdf(strategy.service_level)
            safety_stock = z_score * daily_std * math.sqrt(lead_time)
            safety_stock += strategy.safety_days * float(protected["predicted_units"].mean())
            target_stock = expected_demand + safety_stock

            # Perishables should not be replenished beyond their useful demand window.
            shelf_life = max(1, int(product["shelf_life_days"]))
            demand_before_expiry = float(sku_forecast.head(shelf_life).predicted_units.sum())
            if shelf_life < protection_days + 7:
                target_stock = min(target_stock, demand_before_expiry * 0.90)

            on_hand = float(stock.on_hand)
            on_order = float(getattr(stock, "on_order", 0))
            net_need = target_stock - on_hand - on_order
            order_qty = _round_order(
                net_need,
                int(product["case_pack"]),
                int(product["min_order_qty"]),
            )
            daily_demand = float(protected.predicted_units.mean())
            days_of_cover = on_hand / daily_demand if daily_demand > 0 else math.inf
            expected_shortage = max(0.0, expected_demand - on_hand - on_order)
            excess_units = max(0.0, on_hand + on_order - target_stock)
            expiry_risk = (
                max(0.0, on_hand - demand_before_expiry) if shelf_life <= len(sku_forecast) else 0.0
            )
            margin = float(product["price"] - product["cost"])
            risk_score = (
                strategy.stockout_cost_weight * expected_shortage * max(margin, 0.1)
                + strategy.waste_cost_weight * expiry_risk * float(product["cost"])
                - strategy.holding_cost_weight * excess_units * float(product["cost"]) * 0.01
            )
            if days_of_cover < lead_time:
                status = "高缺货风险"
                action = "立即补货"
            elif order_qty > 0:
                status = "需要关注"
                action = "计划补货"
            elif expiry_risk > 0:
                status = "临期/报废风险"
                action = "促销或调拨"
            elif excess_units > daily_demand * 7:
                status = "库存偏高"
                action = "暂停补货"
            else:
                status = "正常"
                action = "无需操作"
            rows.append(
                {
                    "store_id": str(stock.store_id),
                    "sku": sku,
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "supplier_id": product["supplier_id"],
                    "on_hand": round(on_hand, 1),
                    "on_order": round(on_order, 1),
                    "daily_demand": round(daily_demand, 2),
                    "days_of_cover": round(days_of_cover, 1),
                    "lead_time_days": lead_time,
                    "target_stock": round(target_stock, 1),
                    "safety_stock": round(safety_stock, 1),
                    "suggested_order_qty": order_qty,
                    "case_pack": int(product["case_pack"]),
                    "min_order_qty": int(product["min_order_qty"]),
                    "unit_cost": float(product["cost"]),
                    "purchase_cost": round(order_qty * float(product["cost"]), 2),
                    "expected_shortage": round(expected_shortage, 1),
                    "excess_units": round(excess_units, 1),
                    "expiry_risk_units": round(expiry_risk, 1),
                    "priority_score": round(risk_score, 2),
                    "status": status,
                    "action": action,
                }
            )

        if not rows:
            raise ValueError("库存与销售数据中没有可匹配的门店商品")
        recommendations = pd.DataFrame(rows).sort_values(
            ["priority_score", "expected_shortage"], ascending=False
        )
        return self._apply_budget(recommendations, strategy.max_purchase_budget)

    @staticmethod
    def _apply_budget(recommendations: pd.DataFrame, budget: float | None) -> pd.DataFrame:
        result = recommendations.copy()
        result["budget_adjusted"] = False
        if budget is None:
            return result
        remaining = Decimal(str(budget))
        for index, row in result.sort_values("priority_score", ascending=False).iterrows():
            cost = Decimal(str(row["purchase_cost"]))
            if cost <= remaining:
                remaining -= cost
                continue
            affordable = int(remaining // Decimal(str(row["unit_cost"])))
            original = int(row["suggested_order_qty"])
            pack = max(1, int(row["case_pack"]))
            minimum = max(1, int(row["min_order_qty"]))
            affordable = (affordable // pack) * pack
            adjusted = min(original, affordable) if affordable >= minimum else 0
            result.at[index, "suggested_order_qty"] = adjusted
            result.at[index, "purchase_cost"] = round(
                result.at[index, "suggested_order_qty"] * row["unit_cost"], 2
            )
            result.at[index, "budget_adjusted"] = True
            remaining = max(Decimal(0), remaining - Decimal(str(result.at[index, "purchase_cost"])))
        return result

    def suggest_transfers(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        transfer_rows: list[dict] = []
        physical = recommendations.copy()
        physical["excess_units"] = (physical["on_hand"] - physical["target_stock"]).clip(lower=0)
        for sku, group in physical.groupby("sku"):
            donors = group[group["excess_units"] > group["daily_demand"] * 3].copy()
            receivers = group[group["expected_shortage"] > 0].copy()
            for receiver_index, receiver in receivers.iterrows():
                shortage = float(receiver["expected_shortage"])
                for donor_index, donor in donors.iterrows():
                    if donor["store_id"] == receiver["store_id"] or shortage <= 0:
                        continue
                    available = max(
                        0, float(donor["excess_units"]) - float(donor["daily_demand"]) * 2
                    )
                    quantity = int(min(shortage, available))
                    if quantity <= 0:
                        continue
                    transfer_rows.append(
                        {
                            "sku": sku,
                            "product_name": receiver["product_name"],
                            "from_store": donor["store_id"],
                            "to_store": receiver["store_id"],
                            "quantity": quantity,
                            "estimated_purchase_saving": round(quantity * receiver["unit_cost"], 2),
                            "reason": "有现货余量可供调拨，请核对运输时间与费用",
                        }
                    )
                    shortage -= quantity
                    donors.at[donor_index, "excess_units"] = max(
                        0, donors.at[donor_index, "excess_units"] - quantity
                    )
        columns = [
            "sku",
            "product_name",
            "from_store",
            "to_store",
            "quantity",
            "estimated_purchase_saving",
            "reason",
        ]
        return pd.DataFrame(transfer_rows, columns=columns)


def build_purchase_orders(recommendations: pd.DataFrame) -> pd.DataFrame:
    orders = recommendations[recommendations["suggested_order_qty"] > 0].copy()
    if orders.empty:
        return pd.DataFrame(
            columns=[
                "purchase_order",
                "supplier_id",
                "store_id",
                "sku",
                "product_name",
                "quantity",
                "unit_cost",
                "line_total",
                "approval_status",
            ]
        )
    today = pd.Timestamp.today().strftime("%Y%m%d")
    orders["purchase_order"] = orders.apply(
        lambda row: f"PO-{today}-{row['supplier_id']}-{row['store_id']}", axis=1
    )
    orders = orders.rename(
        columns={"suggested_order_qty": "quantity", "purchase_cost": "line_total"}
    )
    orders["approval_status"] = "待店主审批"
    return orders[
        [
            "purchase_order",
            "supplier_id",
            "store_id",
            "sku",
            "product_name",
            "quantity",
            "unit_cost",
            "line_total",
            "approval_status",
        ]
    ].sort_values(["supplier_id", "store_id", "sku"])
