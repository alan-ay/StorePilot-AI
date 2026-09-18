from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from storepilot.config import ScenarioConfig, StrategyConfig
from storepilot.data import RetailData, generate_demo_data, validate_data
from storepilot.optimization import InventoryOptimizer, _round_order
from storepilot.repository import FeedbackRepository
from storepilot.workbench import dataset_key, decision_scope, download_csv, prepare_review


def row(store="S1", sku="A", supplier="V1"):
    return {
        "store_id": store,
        "sku": sku,
        "supplier_id": supplier,
        "product_name": "矿泉水",
        "suggested_order_qty": 24,
        "case_pack": 6,
        "min_order_qty": 6,
        "unit_cost": 2.5,
    }


class ImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_demo_data(days=75, end_date=date(2026, 9, 5))

    def minimal(self):
        return RetailData(
            self.data.sales[["store_id", "sku", "date", "units"]],
            self.data.products[["sku", "product_name", "category", "supplier_id", "cost", "price"]],
            self.data.inventory[["store_id", "sku", "on_hand"]],
        )

    def test_optional_fields_are_actually_optional(self):
        result = validate_data(self.minimal())
        self.assertTrue((result.sales.promotion == 0).all())
        self.assertTrue((result.sales.stockout == 0).all())
        self.assertTrue((result.inventory.on_order == 0).all())
        self.assertTrue((result.products.case_pack == 1).all())

    def test_missing_infinite_and_negative_values_rejected(self):
        for value in [np.nan, np.inf, -1]:
            with self.subTest(value=value):
                data = self.minimal()
                data.inventory = data.inventory.copy()
                data.inventory["on_hand"] = data.inventory.on_hand.astype(float)
                data.inventory.loc[0, "on_hand"] = value
                with self.assertRaises(ValueError):
                    validate_data(data)

    def test_duplicate_daily_record_rejected(self):
        data = self.minimal()
        data.sales = pd.concat([data.sales, data.sales.head(1)])
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_data(data)

    def test_invalid_pack_and_binary_flag_rejected(self):
        for frame_name, column, value in [
            ("products", "case_pack", 0),
            ("products", "min_order_qty", 1.2),
            ("sales", "stockout", 2),
        ]:
            with self.subTest(column=column):
                data = self.minimal()
                frame = getattr(data, frame_name).copy()
                frame[column] = value
                setattr(data, frame_name, frame)
                with self.assertRaises(ValueError):
                    validate_data(data)

    def test_stale_series_not_silently_used_as_current(self):
        data = self.minimal()
        data.sales = data.sales.drop(data.sales.index[-1])
        with self.assertRaisesRegex(ValueError, "同一截止日期"):
            validate_data(data)

    def test_no_mutation_of_inputs(self):
        data = self.minimal()
        original = data.sales.copy(deep=True)
        validate_data(data)
        pd.testing.assert_frame_equal(data.sales, original)

    def test_scopes_separate_data_and_preferences(self):
        key = dataset_key(self.data, "门店数据")
        self.assertNotEqual(key, dataset_key(self.data, "演示数据"))
        first = decision_scope(key, StrategyConfig())
        second = decision_scope(key, replace(StrategyConfig(), max_purchase_budget=100))
        self.assertNotEqual(first, second)
        self.assertEqual(first, decision_scope(key, StrategyConfig()))


class DecisionTests(unittest.TestCase):
    def test_minimum_is_rounded_to_case_multiple(self):
        self.assertEqual(_round_order(1, 6, 10), 12)

    def test_reject_always_sets_zero_and_accept_uses_original(self):
        self.assertEqual(prepare_review(row(), "拒绝", 100, "已有货", "")["final_qty"], 0)
        self.assertEqual(prepare_review(row(), "接受", 100, "", "")["final_qty"], 24)

    def test_manual_quantity_and_reason_are_validated(self):
        for qty, reason in [
            (7, "修正"),
            (-1, "修正"),
            (1.5, "修正"),
            (float("nan"), "修正"),
            (12, ""),
        ]:
            with self.subTest(qty=qty), self.assertRaises(ValueError):
                prepare_review(row(), "修改", qty, reason, "")

    def test_changed_quantity_recalculates_cost(self):
        self.assertEqual(prepare_review(row(), "修改", 12, "复核", "")["line_total"], 30)

    def test_transfers_never_use_goods_still_on_order(self):
        rec = pd.DataFrame(
            [
                {
                    "store_id": "S1",
                    "sku": "A",
                    "product_name": "Water",
                    "on_hand": 0,
                    "on_order": 500,
                    "target_stock": 100,
                    "excess_units": 400,
                    "daily_demand": 10,
                    "expected_shortage": 0,
                    "unit_cost": 2,
                },
                {
                    "store_id": "S2",
                    "sku": "A",
                    "product_name": "Water",
                    "on_hand": 0,
                    "on_order": 0,
                    "target_stock": 100,
                    "excess_units": 0,
                    "daily_demand": 10,
                    "expected_shortage": 100,
                    "unit_cost": 2,
                },
            ]
        )
        self.assertTrue(InventoryOptimizer().suggest_transfers(rec).empty)

    def test_total_transfer_does_not_exceed_donor_stock(self):
        rec = pd.DataFrame(
            [
                {
                    "store_id": "S1",
                    "sku": "A",
                    "product_name": "Water",
                    "on_hand": 170,
                    "target_stock": 100,
                    "excess_units": 70,
                    "daily_demand": 10,
                    "expected_shortage": 0,
                    "unit_cost": 2,
                },
                *[
                    {
                        "store_id": store,
                        "sku": "A",
                        "product_name": "Water",
                        "on_hand": 0,
                        "target_stock": 100,
                        "excess_units": 0,
                        "daily_demand": 10,
                        "expected_shortage": 100,
                        "unit_cost": 2,
                    }
                    for store in ["S2", "S3"]
                ],
            ]
        )
        transfers = InventoryOptimizer().suggest_transfers(rec)
        self.assertGreater(transfers.quantity.sum(), 0)
        self.assertLessEqual(transfers.quantity.sum(), 50)

    def test_zero_demand_scenario_is_zero(self):
        self.assertEqual(ScenarioConfig(traffic_change_pct=-100).demand_multiplier(), 0)
        with self.assertRaises(ValueError):
            ScenarioConfig(price_change_pct=-100).demand_multiplier()

    def test_spreadsheet_formula_escaped(self):
        result = download_csv(pd.DataFrame({"name": ["=1+1", "ordinary"], "qty": [1, 2]})).decode(
            "utf-8-sig"
        )
        self.assertIn("'=1+1", result)
        self.assertIn("ordinary,2", result)


class PurchasingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "store.db"
        self.repo = FeedbackRepository(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_review_roundtrip_updates_single_draft(self):
        self.repo.review("run", row(), "接受", 24)
        self.repo.review("run", row(), "修改", 12, "复核")
        self.assertEqual(len(self.repo.reviews("run")), 1)
        restarted = FeedbackRepository(self.path)
        self.assertEqual(restarted.reviews("run").iloc[0].line_total, 30)
        self.assertEqual(len(restarted.list_feedback()), 2)

    def test_approval_is_idempotent_and_immutable(self):
        self.repo.review("run", row(), "修改", 12, "复核")
        first = self.repo.approve("run", "S1", "V1")
        self.assertEqual(first, self.repo.approve("run", "S1", "V1"))
        self.assertEqual(len(self.repo.orders()), 1)
        self.assertEqual(self.repo.orders().iloc[0].final_qty, 12)
        with self.assertRaisesRegex(ValueError, "已审核"):
            self.repo.review("run", row(), "修改", 6, "复核")

    def test_budget_checked_again_after_manual_changes_across_stores(self):
        self.repo.review("run", row(), "接受", 24)
        self.repo.review("run", row(store="S2"), "接受", 24)
        with self.assertRaisesRegex(ValueError, "预算"):
            self.repo.approve("run", "S1", "V1", budget=100)
        self.assertTrue(self.repo.orders().empty)

    def test_rejected_line_not_added_to_order(self):
        self.repo.review("run", row(), "拒绝", 24, "已有货")
        with self.assertRaisesRegex(ValueError, "没有"):
            self.repo.approve("run", "S1", "V1")

    def test_scope_does_not_mix_data_imports(self):
        self.repo.review("old", row(), "接受", 24)
        self.repo.approve("old", "S1", "V1")
        self.assertTrue(self.repo.reviews("new").empty)
        self.assertTrue(self.repo.orders("new").empty)
        self.assertEqual(len(self.repo.orders()), 1)

    def test_reasons_are_stored_literally(self):
        text = "'; DROP TABLE reviewed_lines; --"
        self.repo.review("run", row(), "修改", 12, text)
        self.assertEqual(self.repo.reviews("run").iloc[0].reason, text)
