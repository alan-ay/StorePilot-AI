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
        self.app = AppTest.from_file(str(APP), default_timeout=90)
        self.app.session_state["nav"] = "今日概览"
        self.app.run()
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

    def assertBrowserSelection(self, widget, value, label):
        self.assertEqual(widget.value, value)
        self.assertIn(label, widget.options)
        # Options alone do not update the browser's selected text or radio indicator.
        self.assertTrue(widget.proto.set_value, widget.key)
        self.assertEqual(widget.proto.raw_value, label)

    def test_all_pages_render_without_retraining(self):
        with patch(
            "storepilot.pipeline.StorePilotPipeline.run",
            side_effect=AssertionError("unexpected retrain"),
        ):
            for page in [
                "补货清单",
                "采购与记录",
                "销售走势",
                "预测评估",
                "时尚侦察员",
                "情景测算",
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
        self.assertEqual(self.app.selectbox(key="store_scope").value, "STORE-01")
        self.assertNoFailure()
        frame = self.app.dataframe[0].value
        self.assertTrue((frame["门店"] == "STORE-01").all())

    def test_review_to_approval_and_repeat(self):
        self.visit("补货清单")
        self.app.selectbox(key="status_filter").set_value("全部商品").run()
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

    def test_evaluation_run_survives_language_switch_and_preserves_operating_state(self):
        self.visit("预测评估")
        original_data = self.app.session_state["active_data"]
        policy = self.app.session_state["policy"]
        self.app.number_input(key="evaluation_origins").set_value(1)
        self.app.button(key="run_evaluation").click().run()
        self.assertNoFailure()
        saved_key, result = self.app.session_state["evaluation_result"]
        self.assertEqual(len(result.scores), 9)
        self.assertEqual(set(result.predictions.store_id), {"STORE-01"})
        self.assertIn("每日 MAE", self.app.dataframe[0].value.columns)
        with patch(
            "storepilot.dashboard.backtest", side_effect=AssertionError("unexpected retrain")
        ):
            with patch("streamlit.download_button") as download:
                self.app.radio(key="language").set_value("en").run()
            self.assertNoFailure()
            self.assertBrowserSelection(self.app.selectbox(key="evaluation_horizon"), 7, "7 days")
            self.assertIn("Daily MAE", self.app.dataframe[0].value.columns)
            for kind in ["markdown", "caption", "info"]:
                for element in self.app.get(kind):
                    self.assertNotRegex(element.value, r"[\u4e00-\u9fff]")
            for frame in self.app.dataframe:
                for label in list(frame.value.columns) + list(frame.value.to_numpy().flat):
                    if isinstance(label, str):
                        self.assertNotRegex(label, r"[\u4e00-\u9fff]")
            report = next(
                call.args[1]
                for call in download.call_args_list
                if call.args[2] == "forecast-evaluation.md"
            )
            self.assertIn(result.manifest["dataset_id"], report)
            self.assertIn("synthetic demo data", report)
            self.assertNotRegex(report, r"[\u4e00-\u9fff]")
            self.app.radio(key="language").set_value("zh").run()
            self.assertBrowserSelection(self.app.selectbox(key="evaluation_horizon"), 7, "7 天")
            self.assertEqual(self.app.session_state["evaluation_result"][0], saved_key)
            self.visit("今日概览")
            self.visit("预测评估")
            self.assertEqual(self.app.number_input(key="evaluation_origins").value, 1)
            self.assertTrue(list(self.app.dataframe))
            # Settings that no longer match the result must hide its tables and downloads.
            self.app.number_input(key="evaluation_step").set_value(14).run()
            self.assertNoFailure()
            self.assertFalse(list(self.app.dataframe))
            self.assertFalse(list(self.app.get("download_button")))
        self.assertEqual(self.app.session_state["policy"], policy)
        self.assertTrue(self.app.session_state["active_data"].sales.equals(original_data.sales))
        repo = FeedbackRepository(self.database.with_name("test-demo.db"))
        self.assertTrue(repo.orders().empty)
        from storepilot.workbench import dataset_key, decision_scope

        self.assertTrue(
            repo.reviews(decision_scope(dataset_key(original_data, "演示数据"), policy)).empty
        )

    def test_evaluation_reports_insufficient_history_in_english(self):
        self.app.radio(key="language").set_value("en").run()
        self.visit("预测评估")
        self.app.number_input(key="evaluation_origins").set_value(5)
        self.app.number_input(key="evaluation_step").set_value(90)
        self.app.button(key="run_evaluation").click().run()
        self.assertFalse(list(self.app.exception))
        self.assertTrue(any("460 consecutive days" in error.value for error in self.app.error))
        self.assertNotIn("evaluation_result", self.app.session_state)

    def test_policy_and_reset_from_single_store(self):
        self.assertEqual(self.app.selectbox(key="store_scope").value, "STORE-01")
        self.visit("数据与设置")
        self.app.selectbox(key="policy_name").set_value("保守")
        self.app.checkbox(key="policy_budget_enabled").check()
        self.app.number_input(key="policy_budget").set_value(1000.0)
        self.button("保存备货偏好").click().run()
        self.assertNoFailure()
        self.assertEqual(self.app.session_state["policy"].max_purchase_budget, 1000)
        self.app.button(key="reset_sample").click().run()
        self.assertNoFailure()
        self.assertEqual(self.app.selectbox(key="store_scope").value, "STORE-01")

    def test_incomplete_import_keeps_current_data(self):
        self.visit("数据与设置")
        original = len(self.app.session_state["active_data"].sales)
        self.button("检查并使用这组数据").click().run()
        self.assertTrue(any("三份" in e.value for e in self.app.error))
        self.assertEqual(len(self.app.session_state["active_data"].sales), original)

    def test_saved_review_is_restored_after_navigation(self):
        self.visit("补货清单")
        self.app.selectbox(key="status_filter").set_value("全部商品").run()
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

    def test_demo_has_one_store_and_no_transfers_page(self):
        self.assertEqual(self.app.selectbox(key="store_scope").options, ["STORE-01"])
        self.assertTrue(self.app.selectbox(key="store_scope").disabled)
        self.assertNotIn("门店调拨", self.app.radio(key="nav").options)
        self.visit("销售走势")
        self.assertEqual(self.app.selectbox(key="trend_store").options, ["STORE-01"])

    def test_previous_two_store_demo_is_merged_and_stale_navigation_is_reset(self):
        import pandas as pd

        from storepilot.data import DEMO_CATALOG_VERSION

        data = self.app.session_state["active_data"]
        old_data = RetailData(
            pd.concat([data.sales, data.sales.assign(store_id="STORE-02")], ignore_index=True),
            data.products,
            pd.concat(
                [data.inventory, data.inventory.assign(store_id="STORE-02")], ignore_index=True
            ),
        )
        self.app = AppTest.from_file(str(APP), default_timeout=90)
        self.app.session_state["active_data"] = old_data
        self.app.session_state["source"] = "演示数据"
        self.app.session_state["demo_catalog_version"] = "household-v1"
        self.app.session_state["store_scope"] = "STORE-02"
        self.app.session_state["trend_store"] = "STORE-02"
        self.app.session_state["nav"] = "门店调拨"
        self.app.run()
        self.assertNoFailure()
        merged = self.app.session_state["active_data"]
        self.assertEqual(merged.sales.units.sum(), old_data.sales.units.sum())
        self.assertEqual(merged.inventory.on_hand.sum(), old_data.inventory.on_hand.sum())
        self.assertAlmostEqual(
            (merged.sales.units * merged.sales.price).sum(),
            (old_data.sales.units * old_data.sales.price).sum(),
        )
        self.assertEqual(self.app.session_state["demo_catalog_version"], DEMO_CATALOG_VERSION)
        self.assertEqual(self.app.selectbox(key="store_scope").value, "STORE-01")
        self.assertEqual(self.app.radio(key="nav").value, "今日概览")
        self.visit("销售走势")
        self.assertEqual(self.app.selectbox(key="trend_store").value, "STORE-01")

    def test_imported_stores_are_kept_separate(self):
        import pandas as pd

        data = self.app.session_state["active_data"]
        imported = RetailData(
            pd.concat([data.sales, data.sales.assign(store_id="STORE-02")], ignore_index=True),
            data.products,
            pd.concat(
                [data.inventory, data.inventory.assign(store_id="STORE-02")], ignore_index=True
            ),
        )
        self.app.session_state["active_data"] = imported
        self.app.session_state["source"] = "门店数据"
        self.app.session_state["demo_catalog_version"] = "household-v1"
        self.app.run()
        self.assertNoFailure()
        self.assertEqual(
            self.app.selectbox(key="store_scope").options, ["全部门店", "STORE-01", "STORE-02"]
        )
        pd.testing.assert_frame_equal(self.app.session_state["active_data"].sales, imported.sales)
        self.app.selectbox(key="store_scope").set_value("全部门店").run()
        for lang, label in [("en", "All stores"), ("zh", "全部门店")]:
            self.app.radio(key="language").set_value(lang).run()
            self.assertBrowserSelection(self.app.selectbox(key="store_scope"), "全部门店", label)
            self.visit("门店调拨")

    def test_english_pages_translate_visible_content(self):
        self.app.radio(key="language").set_value("en").run()
        self.assertEqual(self.app.radio(key="nav").options[0], "Overview")
        with patch(
            "storepilot.pipeline.StorePilotPipeline.run",
            side_effect=AssertionError("unexpected retrain"),
        ):
            for page in [
                "今日概览",
                "时尚侦察员",
                "补货清单",
                "采购与记录",
                "销售走势",
                "预测评估",
                "情景测算",
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
        self.app.selectbox(key="status_filter").set_value("全部商品").run()
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
        reason = next(s for s in self.app.selectbox if s.key.startswith("reason:"))
        self.assertEqual(reason.value, "库存复核")
        self.assertTrue(reason.proto.set_value)
        self.assertEqual(reason.proto.raw_value, "Stock check")
        self.assertBrowserSelection(
            next(r for r in self.app.radio if r.key.startswith("action:")),
            "修改",
            "Adjust quantity",
        )
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

    def test_adjustment_reason_refreshes_in_both_languages_before_saving(self):
        self.visit("补货清单")
        self.app.selectbox(key="status_filter").set_value("全部商品").run()
        self.app.selectbox(key="review_item").set_value(("STORE-01", "SKU001")).run()
        next(r for r in self.app.radio if r.key.startswith("action:")).set_value("修改").run()
        next(n for n in self.app.number_input if n.key.startswith("qty:")).set_value(24).run()
        next(n for n in self.app.text_input if n.key.startswith("note:")).set_value(
            "Checked stock / 已盘点"
        ).run()
        repo = FeedbackRepository(self.database.with_name("test-demo.db"))

        for reason_value, language, label in [
            ("库存复核", "en", "Stock check"),
            ("资金安排", "zh", "资金安排"),
            ("资金安排", "en", "Budget planning"),
        ]:
            with self.subTest(language=language, reason=reason_value):
                next(s for s in self.app.selectbox if s.key.startswith("reason:")).set_value(
                    reason_value
                ).run()
                self.app.radio(key="language").set_value(language).run()
                self.assertNoFailure()
                reason = next(s for s in self.app.selectbox if s.key.startswith("reason:"))
                self.assertEqual(reason.value, reason_value)
                self.assertIn(label, reason.options)
                # The browser must receive the new selected label, not just new options.
                self.assertTrue(reason.proto.set_value)
                self.assertEqual(reason.proto.raw_value, label)
                # Draft edits must reach the server without submitting a purchase decision.
                self.assertEqual(reason.proto.form_id, "")
                self.assertEqual(
                    next(n for n in self.app.number_input if n.key.startswith("qty:")).value,
                    24,
                )
                self.assertEqual(
                    next(n for n in self.app.text_input if n.key.startswith("note:")).value,
                    "Checked stock / 已盘点",
                )
                self.assertTrue(repo.list_feedback().empty)

        self.button("Save decision").click().run()
        self.assertNoFailure()
        feedback = repo.list_feedback()
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback.iloc[0].reason, "资金安排")
        self.assertEqual(feedback.iloc[0].final_qty, 24)
        self.assertEqual(feedback.iloc[0].note, "Checked stock / 已盘点")

    def test_replenishment_selections_and_reason_placeholder_follow_language(self):
        self.visit("补货清单")
        self.app.selectbox(key="status_filter").set_value("全部商品").run()
        self.app.selectbox(key="category_filter").set_value("家居清洁").run()
        self.app.selectbox(key="review_item").set_value(("STORE-01", "SKU007")).run()
        labels = {
            "en": [
                "Replenishment",
                "Household cleaning",
                "All products",
                "Bin bags (20 pack) · STORE-01",
                "Use suggested quantity",
                "Choose a reason (required for adjustments or skipped purchases)",
            ],
            "zh": [
                "补货清单",
                "家居清洁",
                "全部商品",
                "垃圾袋 20只装 · STORE-01",
                "按建议补货",
                "请选择（调整或不采购时必填）",
            ],
        }
        for language in ["en", "zh", "en"]:
            self.app.radio(key="language").set_value(language).run()
            self.assertNoFailure()
            selections = [
                (self.app.radio(key="nav"), "补货清单"),
                (self.app.selectbox(key="category_filter"), "家居清洁"),
                (self.app.selectbox(key="status_filter"), "全部商品"),
                (self.app.selectbox(key="review_item"), ("STORE-01", "SKU007")),
                (next(r for r in self.app.radio if r.key.startswith("action:")), "接受"),
                (next(s for s in self.app.selectbox if s.key.startswith("reason:")), ""),
            ]
            for (widget, value), label in zip(selections, labels[language], strict=True):
                with self.subTest(language=language, widget=widget.key):
                    self.assertBrowserSelection(widget, value, label)

        # An already-open session from before this fix also needs its labels refreshed.
        del self.app.session_state["choice_label_language"]
        self.app.run()
        self.assertNoFailure()
        self.assertBrowserSelection(
            self.app.selectbox(key="review_item"),
            ("STORE-01", "SKU007"),
            "Bin bags (20 pack) · STORE-01",
        )

    def test_trend_product_and_horizon_selections_follow_language(self):
        self.visit("销售走势")
        self.app.selectbox(key="trend_sku").set_value("SKU007").run()
        self.app.selectbox(key="trend_horizon").set_value(14).run()
        for language, product, horizon in [
            ("en", "Bin bags (20 pack)", "14 days"),
            ("zh", "垃圾袋 20只装", "14 天"),
        ]:
            self.app.radio(key="language").set_value(language).run()
            self.assertNoFailure()
            with self.subTest(language=language):
                self.assertBrowserSelection(self.app.selectbox(key="trend_sku"), "SKU007", product)
                self.assertBrowserSelection(self.app.selectbox(key="trend_horizon"), 14, horizon)

    def test_unsaved_policy_selection_follows_language_without_applying_policy(self):
        self.visit("数据与设置")
        original = self.app.session_state["policy"]
        self.app.selectbox(key="policy_name").set_value("保守").run()
        self.app.checkbox(key="policy_budget_enabled").check().run()
        self.app.number_input(key="policy_budget").set_value(1200.0).run()
        for language, label in [("en", "Lean stock"), ("zh", "控制库存")]:
            self.app.radio(key="language").set_value(language).run()
            self.assertNoFailure()
            with self.subTest(language=language):
                widget = self.app.selectbox(key="policy_name")
                self.assertBrowserSelection(widget, "保守", label)
                self.assertEqual(widget.proto.form_id, "")
                self.assertEqual(self.app.number_input(key="policy_budget").value, 1200.0)
                self.assertEqual(self.app.session_state["policy"], original)
        self.button("保存备货偏好").click().run()
        self.assertNoFailure()
        self.assertEqual(self.app.session_state["policy"].name, "保守")
        self.assertEqual(self.app.session_state["policy"].max_purchase_budget, 1200)

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
        for name in ["sales.csv", "inventory.csv"]:
            self.assertEqual(set(examples[name].store_id), {"STORE-01"})
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
        selected = self.app.selectbox(key="review_item")
        name = data.products.set_index("sku").loc[selected.value[1], "product_name"]
        self.assertBrowserSelection(selected, selected.value, f"{name} · {selected.value[0]}")
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
