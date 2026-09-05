from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest, app_test

from storepilot.repository import FeedbackRepository

APP = Path(__file__).resolve().parents[1] / "app.py"


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "test.db"
        self.env = patch.dict(os.environ, {"STOREPILOT_DB": str(self.database)})
        self.env.start()
        self.app = AppTest.from_file(str(APP), default_timeout=90).run()
        self.assertNoFailure()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def assertNoFailure(self):
        self.assertFalse(list(self.app.exception), [e.message for e in self.app.exception])
        self.assertFalse(list(self.app.error), [e.value for e in self.app.error])

    def visit(self, page):
        self.app.radio(key="nav").set_value(page).run()
        self.assertNoFailure()

    def button(self, label):
        return next(b for b in self.app.button if b.label == label)

    def test_all_pages_render_without_retraining(self):
        with patch(
            "storepilot.pipeline.StorePilotPipeline.run",
            side_effect=AssertionError("unexpected retrain"),
        ):
            for page in [
                "补货清单",
                "采购与记录",
                "销售走势",
                "情景测算",
                "门店调拨",
                "数据与设置",
                "今日概览",
            ]:
                self.visit(page)

    def test_search_empty_state_and_scope(self):
        self.visit("补货清单")
        self.app.text_input(key="product_search").set_value("NONEXISTENT-PRODUCT").run()
        self.assertNoFailure()
        self.assertTrue(any("当前筛选下没有商品" in x.value for x in self.app.markdown))
        self.app.text_input(key="product_search").set_value("").run()
        self.app.selectbox(key="store_scope").set_value("STORE-01").run()
        self.assertNoFailure()
        frame = self.app.dataframe[0].value
        self.assertTrue((frame["门店"] == "STORE-01").all())

    def test_review_to_approval_and_repeat(self):
        self.visit("补货清单")
        self.app.selectbox(key="review_item").set_value(("STORE-01", "SKU001")).run()
        action = next(r for r in self.app.radio if r.key.startswith("action:"))
        action.set_value("修改").run()
        qty = next(n for n in self.app.number_input if n.key.startswith("qty:"))
        qty.set_value(24)
        reason = next(s for s in self.app.selectbox if s.key.startswith("reason:"))
        reason.set_value("库存复核")
        self.button("保存安排").click().run()
        self.assertNoFailure()
        self.visit("采购与记录")
        self.assertTrue(any("合计 **¥24.00**" in x.value for x in self.app.markdown))
        self.button("审核本单").click().run()
        self.assertTrue(any("勾选" in e.value for e in self.app.error))
        next(c for c in self.app.checkbox if "已核对" in c.label).check()
        self.button("审核本单").click().run()
        self.assertNoFailure()
        repo = FeedbackRepository(self.database.with_name("test-demo.db"))
        self.assertEqual(len(repo.orders()), 1)
        self.assertEqual(repo.orders().iloc[0].final_qty, 24)
        self.app.run()
        self.assertNoFailure()
        self.assertEqual(len(repo.orders()), 1)

    def test_scenario_does_not_modify_orders_or_policy(self):
        self.visit("情景测算")
        self.app.slider(key="scenario_traffic").set_value(20)
        self.button("比较这组方案").click().run()
        self.assertNoFailure()
        self.assertAlmostEqual(self.app.session_state["scenario_result"][1].traffic_change_pct, 20)
        self.assertEqual(self.app.session_state["policy"].name, "平衡")
        self.assertTrue(FeedbackRepository(self.database.with_name("test-demo.db")).orders().empty)

    def test_policy_and_reset_from_single_store(self):
        self.app.selectbox(key="store_scope").set_value("STORE-01").run()
        self.visit("数据与设置")
        self.app.selectbox(key="policy_name").set_value("保守")
        self.app.checkbox(key="policy_budget_enabled").check()
        self.app.number_input(key="policy_budget").set_value(1000.0)
        self.button("保存备货偏好").click().run()
        self.assertNoFailure()
        self.assertEqual(self.app.session_state["policy"].max_purchase_budget, 1000)
        self.app.button(key="reset_sample").click().run()
        self.assertNoFailure()
        self.assertEqual(self.app.selectbox(key="store_scope").value, "全部门店")

    def test_incomplete_import_keeps_current_data(self):
        self.visit("数据与设置")
        original = len(self.app.session_state["active_data"].sales)
        self.button("检查并使用这组数据").click().run()
        self.assertTrue(any("三份" in e.value for e in self.app.error))
        self.assertEqual(len(self.app.session_state["active_data"].sales), original)


def tearDownModule():
    # AppTest owns a module-level temporary directory; close it before warning-strict exit.
    directory = getattr(app_test, "TMP_DIR", None)
    if directory is not None:
        directory.cleanup()
