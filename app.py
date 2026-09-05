from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from storepilot.config import ScenarioConfig, StrategyConfig
from storepilot.data import RetailData, generate_demo_data, validate_data
from storepilot.pipeline import StorePilotPipeline
from storepilot.repository import FeedbackRepository

st.set_page_config(page_title="StorePilot", page_icon="🧭", layout="wide")
st.title("StorePilot｜智能零售经营驾驶舱")
st.caption("预测需求、优化补货、模拟经营情景，并通过真实反馈持续学习。")


@st.cache_resource
def repository() -> FeedbackRepository:
    return FeedbackRepository("storepilot.db")


@st.cache_data
def demo_data() -> RetailData:
    return generate_demo_data()


def read_upload(file: object) -> pd.DataFrame:
    return pd.read_csv(file)


def as_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


with st.sidebar:
    st.header("数据与策略")
    use_demo = st.toggle("使用内置演示数据", value=True)
    if use_demo:
        data = demo_data()
    else:
        sales_file = st.file_uploader("销售数据 sales.csv", type="csv")
        products_file = st.file_uploader("商品数据 products.csv", type="csv")
        inventory_file = st.file_uploader("库存数据 inventory.csv", type="csv")
        if not all((sales_file, products_file, inventory_file)):
            st.info("请上传三份 CSV 文件后继续。")
            st.stop()
        data = RetailData(
            sales=read_upload(sales_file),
            products=read_upload(products_file),
            inventory=read_upload(inventory_file),
        )
    strategy_name = st.selectbox("经营策略", ["保守", "平衡", "增长"], index=1)
    enable_budget = st.toggle("限制本次采购预算", value=False)
    budget = st.number_input("采购预算（元）", 0.0, 1_000_000.0, 5000.0, 100.0)
    budget_value = budget if enable_budget else None

    st.subheader("情景模拟")
    price_change = st.slider("价格变化", -30, 30, 0, format="%d%%")
    promotion_lift = st.slider("促销带来的需求提升", 0, 100, 0, format="%d%%")
    traffic_change = st.slider("预计客流变化", -50, 100, 0, format="%d%%")
    holiday_lift = st.slider("节假日需求提升", 0, 100, 0, format="%d%%")
    supplier_delay = st.slider("供应商额外延迟（天）", 0, 14, 0)
    run_button = st.button("运行经营分析", type="primary", width="stretch")


if run_button or "result" not in st.session_state:
    try:
        checked = validate_data(data)
        strategy = StrategyConfig.preset(strategy_name, budget_value)
        scenario = ScenarioConfig(
            price_change_pct=price_change,
            promotion_lift_pct=promotion_lift,
            traffic_change_pct=traffic_change,
            holiday_lift_pct=holiday_lift,
            supplier_delay_days=supplier_delay,
        )
        with st.spinner("正在训练模型并生成经营建议……"):
            result = StorePilotPipeline().run(checked, strategy, scenario)
        st.session_state["data"] = checked
        st.session_state["result"] = result
        st.session_state["strategy"] = strategy
        st.session_state["scenario"] = scenario
        repository().record_model_run(checked.sales["date"].max().isoformat(), result.metrics)
    except (ValueError, RuntimeError) as exc:
        st.error(f"分析失败：{exc}")
        st.stop()

result = st.session_state["result"]
data = st.session_state["data"]
strategy = st.session_state["strategy"]
scenario = st.session_state["scenario"]

high_risk_count = int((result.recommendations["status"] == "高缺货风险").sum())
purchase_total = (
    float(result.purchase_orders["line_total"].sum()) if not result.purchase_orders.empty else 0
)
expiry_units = float(result.recommendations["expiry_risk_units"].sum())
col1, col2, col3, col4 = st.columns(4)
col1.metric("高缺货风险", high_risk_count)
col2.metric("建议采购金额", f"¥{purchase_total:,.0f}")
col3.metric("临期风险件数", f"{expiry_units:,.0f}")
col4.metric("模型 WAPE", f"{result.metrics.wape:.1%}")

tabs = st.tabs(
    ["经营总览", "补货决策", "情景对比", "门店调拨", "采购单", "店主反馈", "中文日报", "模型状态"]
)

with tabs[0]:
    st.subheader("未来30天需求趋势")
    product_options = data.products.set_index("sku")["product_name"].to_dict()
    selected_sku = st.selectbox(
        "选择商品", list(product_options), format_func=lambda sku: f"{product_options[sku]} ({sku})"
    )
    chart = result.adjusted_forecast[result.adjusted_forecast["sku"] == selected_sku].pivot(
        index="date", columns="store_id", values="predicted_units"
    )
    st.line_chart(chart)
    horizon_rows = []
    for (store_id, sku), group in result.adjusted_forecast.groupby(["store_id", "sku"]):
        group = group.sort_values("date")
        horizon_rows.append(
            {
                "store_id": store_id,
                "sku": sku,
                "product_name": product_options.get(sku, sku),
                "未来1天": round(group.head(1)["predicted_units"].sum(), 1),
                "未来7天": round(group.head(7)["predicted_units"].sum(), 1),
                "未来30天": round(group.head(30)["predicted_units"].sum(), 1),
            }
        )
    st.subheader("1 / 7 / 30 天需求预测")
    st.dataframe(pd.DataFrame(horizon_rows), width="stretch", hide_index=True)
    st.subheader("异常与高风险商品")
    risk = result.recommendations[result.recommendations["status"] != "正常"]
    st.dataframe(risk, width="stretch", hide_index=True)

with tabs[1]:
    st.caption(
        f"当前策略：{strategy.name}；服务水平 {strategy.service_level:.0%}。"
        "采购建议同时考虑利润、缺货、库存资金和报废风险。"
    )
    st.dataframe(result.recommendations, width="stretch", hide_index=True)
    st.download_button(
        "下载补货建议 CSV",
        as_csv(result.recommendations),
        "storepilot_recommendations.csv",
        "text/csv",
    )

with tabs[2]:
    st.write(
        f"情景需求乘数：**{scenario.demand_multiplier():.3f}**。"
        "基础预测不会被覆盖，调整值单独保存，方便审计和回测。"
    )
    comparison = result.adjusted_forecast.groupby("date", as_index=False)[
        ["base_predicted_units", "predicted_units"]
    ].sum()
    comparison = comparison.rename(
        columns={"base_predicted_units": "基础预测", "predicted_units": "情景预测"}
    ).set_index("date")
    st.line_chart(comparison)
    st.dataframe(result.adjusted_forecast.head(100), width="stretch", hide_index=True)

with tabs[3]:
    if result.transfers.empty:
        st.info("当前没有优先于重新采购的跨店调拨机会。")
    else:
        st.dataframe(result.transfers, width="stretch", hide_index=True)
        st.download_button(
            "下载调拨建议 CSV", as_csv(result.transfers), "storepilot_transfers.csv", "text/csv"
        )

with tabs[4]:
    st.warning("采购单为草稿，必须由店主审批后才能提交给供应商。")
    st.dataframe(result.purchase_orders, width="stretch", hide_index=True)
    st.download_button(
        "下载采购单 CSV",
        as_csv(result.purchase_orders),
        "storepilot_purchase_orders.csv",
        "text/csv",
    )

with tabs[5]:
    options = result.recommendations.copy()
    options["label"] = (
        options["store_id"]
        + "｜"
        + options["product_name"]
        + "｜建议 "
        + options["suggested_order_qty"].astype(str)
        + " 件"
    )
    selection = st.selectbox(
        "选择一条建议", options.index, format_func=lambda index: options.loc[index, "label"]
    )
    selected = options.loc[selection]
    with st.form("feedback_form", clear_on_submit=True):
        action = st.radio("处理方式", ["接受", "修改", "拒绝"], horizontal=True)
        final_qty = st.number_input(
            "最终补货数量", min_value=0.0, value=float(selected["suggested_order_qty"]), step=1.0
        )
        reason = st.selectbox(
            "原因",
            [
                "系统建议合理",
                "供应商缺货",
                "准备促销",
                "资金限制",
                "仓储限制",
                "本地活动",
                "店主经验",
                "其他",
            ],
        )
        note = st.text_input("补充说明（可选）")
        submitted = st.form_submit_button("保存反馈")
        if submitted:
            repository().add_feedback(
                decision_id=str(uuid.uuid4()),
                store_id=selected["store_id"],
                sku=selected["sku"],
                action=action,
                original_qty=selected["suggested_order_qty"],
                final_qty=final_qty,
                reason=reason,
                note=note,
            )
            st.success("反馈已保存，将用于后续策略评估和模型迭代。")
    st.dataframe(repository().list_feedback(), width="stretch", hide_index=True)

with tabs[6]:
    st.markdown(result.report)
    st.download_button(
        "下载经营简报 Markdown",
        result.report.encode("utf-8"),
        "storepilot_daily_report.md",
        "text/markdown",
    )

with tabs[7]:
    st.write(f"模型：{result.metrics.model_name}")
    metric_frame = pd.DataFrame(
        {
            "指标": ["MAE", "WAPE", "预测区间覆盖率", "验证样本数"],
            "数值": [
                f"{result.metrics.mae:.3f}",
                f"{result.metrics.wape:.2%}",
                f"{result.metrics.interval_coverage:.2%}",
                str(result.metrics.validation_rows),
            ],
        }
    )
    st.dataframe(metric_frame, width="stretch", hide_index=True)
    st.info(
        "自适应路径：导入最新真实销量后重新运行分析，模型会用完整最新数据重训，"
        "并记录每次验证误差。生产环境可把这一步配置为每日定时任务。"
    )
    st.dataframe(repository().list_model_runs(), width="stretch", hide_index=True)
