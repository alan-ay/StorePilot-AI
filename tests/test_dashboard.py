from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest, app_test

from storepilot.config import StrategyConfig
from storepilot.dashboard import analyse, selected_plan
from storepilot.data import RetailData
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

    def test_saved_review_is_restored_after_navigation(self):
        self.visit("补货清单")
        self.app.selectbox(key="review_item").set_value(("STORE-01", "SKU001")).run()
        next(r for r in self.app.radio if r.key.startswith("action:")).set_value("修改").run()
        next(n for n in self.app.number_input if n.key.startswith("qty:")).set_value(24)
        next(s for s in self.app.selectbox if s.key.startswith("reason:")).set_value("库存复核")
        next(t for t in self.app.text_input if t.key.startswith("note:")).set_value("已盘点")
        self.button("保存安排").click().run()
        self.visit("今日概览")
        self.visit("补货清单")
        self.app.selectbox(key="status_filter").set_value("已处理").run()
        self.assertNoFailure()
        self.assertEqual(
            next(r for r in self.app.radio if r.key.startswith("action:")).value, "修改"
        )
        self.assertEqual(
            next(n for n in self.app.number_input if n.key.startswith("qty:")).value, 24
        )
        self.assertEqual(
            next(s for s in self.app.selectbox if s.key.startswith("reason:")).value, "库存复核"
        )
        self.assertEqual(
            next(t for t in self.app.text_input if t.key.startswith("note:")).value, "已盘点"
        )

    def test_approval_confirmation_does_not_carry_to_another_order(self):
        from storepilot.workbench import dataset_key, decision_scope

        scope = decision_scope(
            dataset_key(self.app.session_state["active_data"], "演示数据"), StrategyConfig()
        )
        repo = FeedbackRepository(self.database.with_name("test-demo.db"))
        for supplier in ["V1", "V2"]:
            repo.review(
                scope,
                {
                    "store_id": "STORE-01",
                    "sku": supplier,
                    "supplier_id": supplier,
                    "product_name": "Water",
                    "suggested_order_qty": 24,
                    "case_pack": 6,
                    "min_order_qty": 6,
                    "unit_cost": 2.5,
                },
                "接受",
                24,
            )
        self.visit("采购与记录")
        next(c for c in self.app.checkbox if "已核对" in c.label).check().run()
        self.app.selectbox(key="order_group").set_value(("STORE-01", "V2")).run()
        self.assertNoFailure()
        self.assertFalse(next(c for c in self.app.checkbox if "已核对" in c.label).value)
        self.button("审核本单").click().run()
        self.assertTrue(repo.orders().empty)

    def test_store_without_sales_has_empty_states(self):
        import pandas as pd

        data = self.app.session_state["active_data"]
        inventory = pd.concat([data.inventory, data.inventory.head(1).assign(store_id="NEW")])
        self.app.session_state["active_data"] = RetailData(data.sales, data.products, inventory)
        self.app.run()
        self.app.selectbox(key="store_scope").set_value("NEW").run()
        self.assertNoFailure()
        for page in ["今日概览", "销售走势", "情景测算", "补货清单", "数据与设置"]:
            self.visit(page)

    def test_zero_budget_and_custom_preferences_survive_navigation(self):
        self.visit("数据与设置")
        self.app.checkbox(key="policy_budget_enabled").check()
        self.app.number_input(key="policy_budget").set_value(0.0)
        self.app.checkbox(key="policy_custom").check()
        self.app.number_input(key="policy_safety").set_value(3.0)
        self.button("保存备货偏好").click().run()
        self.visit("今日概览")
        self.visit("数据与设置")
        self.assertEqual(self.app.number_input(key="policy_budget").value, 0.0)
        self.assertTrue(self.app.checkbox(key="policy_custom").value)
        self.button("保存备货偏好").click().run()
        self.assertEqual(self.app.session_state["policy"].safety_days, 3.0)

    def test_failed_policy_change_preserves_working_policy(self):
        self.visit("数据与设置")
        data = self.app.session_state["active_data"]
        policy = self.app.session_state["policy"]
        rec = selected_plan(data, analyse(data), policy)
        self.app.selectbox(key="policy_name").set_value("增长")
        with patch(
            "storepilot.dashboard.selected_plan", side_effect=[rec, ValueError("预测天数不足")]
        ):
            self.button("保存备货偏好").click().run()
        self.assertFalse(list(self.app.exception))
        self.assertTrue(any("未保存" in e.value for e in self.app.error))
        self.assertEqual(self.app.session_state["policy"], policy)
        self.visit("今日概览")

    def test_failed_import_plan_preserves_working_data(self):
        self.visit("数据与设置")
        data = self.app.session_state["active_data"]
        base = analyse(data)
        rec = selected_plan(data, base, self.app.session_state["policy"])
        uploads = []
        for frame in [data.sales, data.products, data.inventory.assign(on_hand=1)]:
            upload = io.BytesIO(frame.to_csv(index=False).encode())
            upload.size = len(upload.getvalue())
            uploads.append(upload)
        with (
            patch("streamlit.file_uploader", side_effect=uploads),
            patch("storepilot.dashboard.analyse", return_value=base),
            patch(
                "storepilot.dashboard.selected_plan", side_effect=[rec, ValueError("预测天数不足")]
            ),
        ):
            self.button("检查并使用这组数据").click().run()
        self.assertFalse(list(self.app.exception))
        self.assertTrue(any("未导入" in e.value for e in self.app.error))
        self.assertEqual(self.app.session_state["source"], "演示数据")
        self.assertTrue(self.app.session_state["active_data"].inventory.equals(data.inventory))
        self.visit("今日概览")

    def test_demo_catalog_refresh_does_not_replace_imported_data(self):
        from storepilot.data import DEMO_CATALOG_VERSION

        data = self.app.session_state["active_data"]
        data.products.loc[0, "product_name"] = "旧演示食品"
        self.app.session_state["demo_catalog_version"] = "old"
        self.app.run()
        self.assertNoFailure()
        self.assertEqual(self.app.session_state["demo_catalog_version"], DEMO_CATALOG_VERSION)
        self.assertNotIn(
            "旧演示食品", self.app.session_state["active_data"].products.product_name.tolist()
        )

        data = self.app.session_state["active_data"]
        data.products.loc[0, "product_name"] = "My imported product"
        self.app.session_state["source"] = "门店数据"
        self.app.session_state["demo_catalog_version"] = "old"
        self.app.run()
        self.assertNoFailure()
        self.assertIn(
            "My imported product",
            self.app.session_state["active_data"].products.product_name.tolist(),
        )

    def test_english_pages_translate_visible_content(self):
        self.app.radio(key="language").set_value("en").run()
        self.assertEqual(self.app.radio(key="nav").options[0], "Overview")
        with patch(
            "storepilot.pipeline.StorePilotPipeline.run",
            side_effect=AssertionError("unexpected retrain"),
        ):
            for page in [
                "今日概览",
                "补货清单",
                "采购与记录",
                "销售走势",
                "情景测算",
                "门店调拨",
                "数据与设置",
            ]:
                self.visit(page)
                for kind in ["markdown", "caption", "subheader", "info", "warning"]:
                    for element in self.app.get(kind):
                        self.assertNotRegex(
                            element.value, r"[\u4e00-\u9fff]", f"{page}: {element.value}"
                        )
                for kind in [
                    "button",
                    "selectbox",
                    "radio",
                    "checkbox",
                    "slider",
                    "number_input",
                    "text_input",
                    "file_uploader",
                    "download_button",
                ]:
                    for element in self.app.get(kind):
                        if kind == "radio" and element.key == "language":
                            continue
                        for label in [element.label] + list(getattr(element, "options", [])):
                            self.assertNotRegex(label, r"[\u4e00-\u9fff]", f"{page}: {label}")
                for element in self.app.dataframe:
                    for label in list(element.value.columns) + list(element.value.to_numpy().flat):
                        if isinstance(label, str):
                            self.assertNotRegex(label, r"[\u4e00-\u9fff]", f"{page}: {label}")

    def test_language_switch_preserves_review_scope_and_store(self):
        from storepilot.workbench import dataset_key, decision_scope

        self.visit("补货清单")
        self.app.selectbox(key="store_scope").set_value("STORE-01").run()
        self.app.selectbox(key="review_item").set_value(("STORE-01", "SKU001")).run()
        next(r for r in self.app.radio if r.key.startswith("action:")).set_value("修改").run()
        next(n for n in self.app.number_input if n.key.startswith("qty:")).set_value(24)
        next(s for s in self.app.selectbox if s.key.startswith("reason:")).set_value("库存复核")
        self.button("保存安排").click().run()
        data = self.app.session_state["active_data"]
        scope = decision_scope(dataset_key(data, "演示数据"), self.app.session_state["policy"])
        repo = FeedbackRepository(self.database.with_name("test-demo.db"))
        self.assertEqual(len(repo.reviews(scope)), 1)
        self.app.selectbox(key="status_filter").set_value("已处理").run()
        self.app.radio(key="language").set_value("en").run()
        self.assertNoFailure()
        self.assertEqual(self.app.radio(key="nav").value, "补货清单")
        self.assertEqual(self.app.selectbox(key="store_scope").value, "STORE-01")
        self.assertEqual(self.app.selectbox(key="status_filter").value, "已处理")
        self.assertEqual(
            next(n for n in self.app.number_input if n.key.startswith("qty:")).value, 24
        )
        self.visit("采购与记录")
        next(c for c in self.app.checkbox if "checked quantities" in c.label).check().run()
        self.app.radio(key="language").set_value("zh").run()
        self.assertTrue(next(c for c in self.app.checkbox if "已核对" in c.label).value)
        self.button("审核本单").click().run()
        self.assertNoFailure()
        self.assertEqual(len(repo.orders(scope)), 1)
        self.assertEqual(repo.orders(scope).iloc[0].final_qty, 24)

    def test_english_import_errors_and_examples(self):
        self.app.radio(key="language").set_value("en").run()
        self.visit("数据与设置")
        with patch("streamlit.download_button") as download:
            self.app.run()
        import pandas as pd

        examples = {
            call.args[2]: pd.read_csv(io.BytesIO(call.args[1]))
            for call in download.call_args_list
            if len(call.args) >= 3
            and call.args[2] in {"sales.csv", "products.csv", "inventory.csv"}
        }
        self.assertEqual(set(examples), {"sales.csv", "products.csv", "inventory.csv"})
        self.assertIn(
            "Dishwashing sponges (2 pack)", examples["products.csv"].product_name.tolist()
        )
        self.assertEqual(
            set(examples["products.csv"].category),
            {"Household cleaning", "Laundry care", "Personal care", "Paper goods"},
        )
        original = self.app.session_state["active_data"]
        examples["inventory.csv"].loc[0, "on_hand"] = -1
        uploads = []
        for name in ["sales.csv", "products.csv", "inventory.csv"]:
            upload = io.BytesIO(examples[name].to_csv(index=False).encode())
            upload.size = len(upload.getvalue())
            uploads.append(upload)
        with patch("streamlit.file_uploader", side_effect=uploads):
            self.button("Validate and use these files").click().run()
        self.assertFalse(list(self.app.exception))
        self.assertTrue(
            any("on_hand must not be negative" in error.value for error in self.app.error)
        )
        self.assertTrue(self.app.session_state["active_data"].inventory.equals(original.inventory))

    def test_imported_names_are_preserved_when_switching_language(self):
        data = self.app.session_state["active_data"]
        self.app.session_state["source"] = "门店数据"
        self.app.radio(key="language").set_value("en").run()
        self.visit("补货清单")
        self.app.selectbox(key="status_filter").set_value("全部商品").run()
        self.assertNoFailure()
        names = self.app.dataframe[0].value["Product"].tolist()
        self.assertTrue(set(names).issubset(set(data.products.product_name)))
        self.app.radio(key="language").set_value("zh").run()
        self.assertEqual(self.app.session_state["source"], "门店数据")
        self.assertTrue(self.app.session_state["active_data"].products.equals(data.products))

    def test_english_scenario_export_keeps_numbers_and_translates_labels(self):
        import pandas as pd

        self.app.radio(key="language").set_value("en").run()
        self.visit("情景测算")
        self.app.slider(key="scenario_traffic").set_value(20)
        with patch("streamlit.download_button") as download:
            self.button("Compare scenario").click().run()
        self.assertNoFailure()
        payload = next(
            call.args[1] for call in download.call_args_list if call.args[2] == "scenario-plan.csv"
        )
        exported = pd.read_csv(io.BytesIO(payload))
        self.assertIn("Product", exported.columns)
        self.assertIn("Suggested order", exported.columns)
        for value in exported.select_dtypes(include=["object", "string"]).to_numpy().flat:
            if isinstance(value, str):
                self.assertNotRegex(value, r"[\u4e00-\u9fff]")
        saved = self.app.session_state["scenario_result"]
        self.assertEqual(exported["Suggested order"].sum(), saved[3].suggested_order_qty.sum())
        self.app.radio(key="language").set_value("zh").run()
        self.assertNoFailure()
        self.assertEqual(self.app.session_state["scenario_result"][0], saved[0])
        self.assertEqual(self.app.session_state["scenario_result"][1], saved[1])

    def test_language_switch_keeps_saved_budget_and_custom_preferences(self):
        self.visit("数据与设置")
        self.app.checkbox(key="policy_budget_enabled").check()
        self.app.number_input(key="policy_budget").set_value(0.0)
        self.app.checkbox(key="policy_custom").check()
        self.app.number_input(key="policy_safety").set_value(3.0)
        self.button("保存备货偏好").click().run()
        policy = self.app.session_state["policy"]
        self.app.radio(key="language").set_value("en").run()
        self.assertNoFailure()
        self.assertEqual(self.app.number_input(key="policy_budget").value, 0.0)
        self.assertTrue(self.app.checkbox(key="policy_custom").value)
        self.assertEqual(self.app.number_input(key="policy_safety").value, 3.0)
        self.assertEqual(self.app.session_state["policy"], policy)


def tearDownModule():
    # AppTest owns a module-level temporary directory; close it before warning-strict exit.
    directory = getattr(app_test, "TMP_DIR", None)
    if directory is not None:
        directory.cleanup()
