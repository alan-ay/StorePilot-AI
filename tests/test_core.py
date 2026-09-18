from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from storepilot.config import ScenarioConfig, StrategyConfig
from storepilot.data import RetailData, generate_demo_data, validate_data
from storepilot.optimization import build_purchase_orders
from storepilot.pipeline import StorePilotPipeline
from storepilot.repository import FeedbackRepository
from storepilot.scenarios import apply_scenario


class ScenarioTests(unittest.TestCase):
    def test_scenario_does_not_overwrite_base(self) -> None:
        base = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-01")],
                "store_id": ["S1"],
                "sku": ["A"],
                "predicted_units": [10.0],
                "lower_units": [8.0],
                "upper_units": [12.0],
            }
        )
        adjusted = apply_scenario(base, ScenarioConfig(traffic_change_pct=20))
        self.assertEqual(base.loc[0, "predicted_units"], 10.0)
        self.assertAlmostEqual(adjusted.loc[0, "predicted_units"], 12.0)
        self.assertAlmostEqual(adjusted.loc[0, "base_predicted_units"], 10.0)
        self.assertLessEqual(adjusted.loc[0, "lower_units"], adjusted.loc[0, "predicted_units"])
        self.assertGreaterEqual(adjusted.loc[0, "upper_units"], adjusted.loc[0, "predicted_units"])

    def test_price_increase_reduces_demand(self) -> None:
        scenario = ScenarioConfig(price_change_pct=10, price_elasticity=-1.2)
        self.assertLess(scenario.demand_multiplier(), 1.0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = generate_demo_data(days=75, seed=7)
        cls.result = StorePilotPipeline().run(
            cls.data,
            StrategyConfig.preset("平衡", budget=10_000),
            ScenarioConfig(),
            horizon_days=30,
        )

    def test_forecast_has_all_store_sku_days(self) -> None:
        groups = len(self.data.inventory)
        self.assertEqual(len(self.result.base_forecast), groups * 30)
        self.assertTrue((self.result.base_forecast["predicted_units"] >= 0).all())
        self.assertTrue(
            (
                self.result.base_forecast["lower_units"]
                <= self.result.base_forecast["predicted_units"]
            ).all()
        )
        self.assertTrue(
            (
                self.result.base_forecast["upper_units"]
                >= self.result.base_forecast["predicted_units"]
            ).all()
        )

    def test_recommendations_and_orders_are_consistent(self) -> None:
        self.assertEqual(len(self.result.recommendations), len(self.data.inventory))
        orders = build_purchase_orders(self.result.recommendations)
        self.assertTrue((orders["quantity"] > 0).all())
        self.assertTrue((orders["approval_status"] == "待店主审批").all())
        for row in self.result.recommendations.itertuples(index=False):
            if row.suggested_order_qty:
                self.assertEqual(row.suggested_order_qty % row.case_pack, 0)

    def test_budget_is_respected(self) -> None:
        result = StorePilotPipeline().run(
            self.data,
            StrategyConfig.preset("平衡", budget=500),
            ScenarioConfig(),
            horizon_days=30,
        )
        self.assertLessEqual(result.recommendations["purchase_cost"].sum(), 500.01)

    def test_report_contains_required_sections(self) -> None:
        for section in ("今日重点", "销售趋势", "采购与调拨", "使用说明"):
            self.assertIn(section, self.result.report)


class RepositoryTests(unittest.TestCase):
    def test_feedback_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = FeedbackRepository(Path(directory) / "test.db")
            repo.add_feedback("D-1", "S1", "A", "修改", 12, 8, "资金限制", "test")
            feedback = repo.list_feedback()
            self.assertEqual(len(feedback), 1)
            self.assertEqual(feedback.loc[0, "final_qty"], 8)


class ValidationTests(unittest.TestCase):
    def test_negative_sales_are_rejected(self) -> None:
        data = generate_demo_data(days=45, seed=3)
        data.sales.loc[0, "units"] = -1
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            validate_data(data)

    def test_unknown_inventory_sku_is_rejected(self) -> None:
        data = generate_demo_data(days=45, seed=3)
        extra = pd.DataFrame(
            [{"store_id": "STORE-01", "sku": "UNKNOWN", "on_hand": 1, "on_order": 0}]
        )
        invalid = RetailData(
            sales=data.sales,
            products=data.products,
            inventory=pd.concat([data.inventory, extra], ignore_index=True),
        )
        with self.assertRaisesRegex(ValueError, "未出现在商品表"):
            validate_data(invalid)


if __name__ == "__main__":
    unittest.main()
