from __future__ import annotations

import io
import os
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from html import escape
from importlib.resources import files
from pathlib import Path

import pandas as pd
import streamlit as st

from .config import ScenarioConfig, StrategyConfig
from .data import RetailData, generate_demo_data, validate_data
from .optimization import InventoryOptimizer
from .pipeline import StorePilotPipeline
from .reporting import generate_chinese_report
from .repository import FeedbackRepository
from .workbench import dataset_key, decision_scope, download_csv, scenario_plan, scoped

PAGES = ["今日概览", "补货清单", "采购与记录", "销售走势", "情景测算", "门店调拨", "数据与设置"]
STATUS = {
    "高缺货风险": "优先补货",
    "需要关注": "计划补货",
    "临期/报废风险": "核对保质期",
    "库存偏高": "库存偏多",
    "正常": "暂不处理",
}
COLUMNS = {
    "store_id": "门店",
    "sku": "商品编号",
    "product_name": "商品",
    "category": "品类",
    "supplier_id": "供应商",
    "on_hand": "现货",
    "on_order": "在途",
    "days_of_cover": "现货可售天数",
    "daily_demand": "预计日销量",
    "suggested_order_qty": "建议补货",
    "purchase_cost": "参考金额",
    "status": "处理事项",
    "original_qty": "原建议",
    "final_qty": "确认数量",
    "unit_cost": "进价",
    "line_total": "金额",
    "reason": "原因",
    "note": "备注",
    "action": "处理方式",
    "order_id": "单号",
    "approved_at": "审核时间",
    "reviewed_at": "处理时间",
    "approval_status": "状态",
    "from_store": "调出门店",
    "to_store": "调入门店",
    "quantity": "数量",
    "estimated_purchase_saving": "对应采购货值",
    "trend_pct": "预计变化 %",
    "trend": "趋势",
}


def table(frame, columns=None, **kwargs):
    selected = frame[columns].copy() if columns else frame.copy()
    if "status" in selected:
        selected["status"] = selected["status"].map(STATUS).fillna(selected["status"])
    st.dataframe(selected.rename(columns=COLUMNS), hide_index=True, width="stretch", **kwargs)


def empty(message):
    st.markdown(f'<div class="sp-empty">{escape(message)}</div>', unsafe_allow_html=True)


def go(page):
    st.session_state["nav"] = page


def notice(message):
    st.session_state["notice"] = message


def heading(title, subtitle):
    st.markdown(
        f'<h1 class="sp-heading">{escape(title)}</h1><p class="sp-intro">{escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def metrics(items):
    for col, (label, value, note, tone) in zip(st.columns(len(items)), items, strict=True):
        with col:
            st.markdown(
                f'<div class="sp-metric"><div class="sp-metric-label">{escape(label)}</div>'
                f'<div class="sp-metric-value {tone}">{escape(str(value))}</div>'
                f'<div class="sp-metric-note">{escape(note)}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")


@st.cache_data(show_spinner=False, max_entries=4)
def analyse(data):
    horizon = min(90, max(30, int(data.products.lead_time_days.max()) + 28))
    return StorePilotPipeline().run(data, horizon_days=horizon)


@st.cache_data(show_spinner=False, ttl=3600)
def sample_data():
    return generate_demo_data()


def selected_plan(data, base, policy):
    return InventoryOptimizer().recommend(base.base_forecast, data.products, data.inventory, policy)


def review_status(frame, reviews, orders):
    result = frame.copy()
    review_keys = {(r.store_id, r.sku): r for r in reviews.itertuples()}
    locked = set(zip(orders.store_id, orders.supplier_id))
    result["确认状态"] = [
        "已审核"
        if (r.store_id, r.supplier_id) in locked
        else "已处理"
        if (r.store_id, r.sku) in review_keys
        else "待处理"
        for r in result.itertuples()
    ]
    return result


def overview(data, base, rec, store, repo, scope, policy):
    local = scoped(rec, store)
    sales = scoped(data.sales, store)
    latest_date = sales.date.max()
    latest = sales[sales.date == latest_date]
    revenue = float((latest.units * latest.price).sum())
    before = sales[sales.date == latest_date - pd.Timedelta(days=7)]
    previous = float((before.units * before.price).sum())
    change = (
        f"较上周同日 {(revenue / previous - 1) * 100:+.1f}%" if previous else "上周同日数据不足"
    )
    heading("今天先处理这些", "先看缺货与积压，再核对补货清单。")
    urgent = local[local.status == "高缺货风险"]
    orders = scoped(repo.orders(scope), store)
    drafts = scoped(repo.reviews(scope), store)
    locked = set(zip(orders.store_id, orders.supplier_id))
    pending = (
        drafts[[(r.store_id, r.supplier_id) not in locked for r in drafts.itertuples()]]
        if not drafts.empty
        else drafts
    )
    metrics(
        [
            ("最近营业额", f"¥{revenue:,.0f}", f"{latest_date:%m月%d日} · {change}", ""),
            (
                "优先补货",
                f"{len(urgent)} 项",
                "现货预计撑不到补货到达",
                "attention" if len(urgent) else "",
            ),
            ("待审核采购", f"¥{pending.line_total.sum():,.0f}", "已确认清单的采购金额", "primary"),
            (
                "库存需复核",
                f"{int((local.expiry_risk_units > 0).sum())} 项",
                "需核对批次保质期",
                "",
            ),
        ]
    )
    left, right = st.columns([1.65, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("待办清单")
        tasks = local[local.status != "正常"].head(4)
        if tasks.empty:
            empty("目前没有紧急事项，可以继续查看本周销售情况。")
        for row in tasks.itertuples():
            text = (
                f"{row.store_id} · 现货 {row.on_hand:g} 件，约可售 {row.days_of_cover:g} 天；"
                f"交货需 {row.lead_time_days} 天。"
            )
            st.markdown(
                f'<div class="sp-task"><div class="sp-task-title">{escape(row.product_name)}'
                f'<span class="sp-label">{STATUS[row.status]}</span></div>'
                f'<div class="sp-task-detail">{escape(text)}</div></div>',
                unsafe_allow_html=True,
            )
        st.button(
            "查看并处理补货清单",
            key="open_replenishment",
            on_click=go,
            args=("补货清单",),
            type="primary",
        )
    with right, st.container(border=True):
        st.subheader("销售概况")
        daily = (
            sales.assign(营业额=sales.units * sales.price).groupby("date")["营业额"].sum().tail(14)
        )
        st.line_chart(daily, color="#2458d3", height=230, x_label="日期", y_label="营业额（元）")
        st.caption(f"当前采用{policy.name}策略 · {len(local)} 项门店商品")
        st.button("查看商品走势", key="open_trends", on_click=go, args=("销售走势",))
    with st.expander("查看当日经营摘要"):
        transfers = InventoryOptimizer().suggest_transfers(rec)
        if store != "全部门店":
            transfers = transfers[(transfers.from_store == store) | (transfers.to_store == store)]
        report = generate_chinese_report(
            sales, scoped(base.trends, store), local, transfers, policy.name
        )
        st.markdown(report)
        st.download_button("下载经营摘要", report, "storepilot-report.md", "text/markdown")


def replenishment(data, base, rec, store, repo, scope):
    heading("补货清单", "逐项确认后，前往采购页审核。数量调整会同步到采购草稿。")
    reviews, orders = repo.reviews(scope), repo.orders(scope)
    local = review_status(scoped(rec, store), reviews, orders)
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("搜索商品", placeholder="商品名或编号", key="product_search")
    category = f2.selectbox(
        "品类", ["全部品类"] + sorted(local.category.unique()), key="category_filter"
    )
    filter_status = f3.selectbox(
        "处理范围", ["需要处理", "全部商品", "已处理"], key="status_filter"
    )
    if query:
        local = local[
            local.product_name.str.contains(query, case=False, regex=False)
            | local.sku.str.contains(query, case=False, regex=False)
        ]
    if category != "全部品类":
        local = local[local.category == category]
    if filter_status == "需要处理":
        local = local[(local.status != "正常") & (local["确认状态"] == "待处理")]
    elif filter_status == "已处理":
        local = local[local["确认状态"] != "待处理"]
    if local.empty:
        empty("当前筛选下没有商品。可以更换筛选条件，或前往采购页核对已确认清单。")
        st.button("前往采购与记录", on_click=go, args=("采购与记录",))
        return
    table(
        local,
        [
            "store_id",
            "product_name",
            "on_hand",
            "days_of_cover",
            "suggested_order_qty",
            "purchase_cost",
            "status",
            "确认状态",
        ],
    )
    names = {(r.store_id, r.sku): f"{r.product_name} · {r.store_id}" for r in local.itertuples()}
    chosen = st.selectbox("选择要处理的商品", list(names), format_func=names.get, key="review_item")
    store_id, sku = chosen
    row = local[(local.store_id == store_id) & (local.sku == sku)].iloc[0]
    details, form = st.columns([1, 1.2], gap="large")
    with details, st.container(border=True):
        st.subheader(str(row.product_name))
        st.write(
            f"建议补货 **{row.suggested_order_qty} 件**，参考金额 **¥{row.purchase_cost:,.2f}**。"
        )
        st.write(
            f"当前现货 {row.on_hand:g} 件，在途 {row.on_order:g} 件；预计每天售出 {row.daily_demand:g} 件。"
        )
        st.write(
            f"交货需要 {row.lead_time_days} 天，目标库存约 {row.target_stock:g} 件，含安全余量 {row.safety_stock:g} 件。"
        )
        st.caption(
            f"每箱 {row.case_pack} 件 · 起订 {row.min_order_qty} 件 · 供应商 {row.supplier_id}"
        )
        if row.budget_adjusted:
            st.warning("本项建议已受预算限制，仍可能存在缺货。")
        if row.expiry_risk_units > 0:
            st.info("库存可能超过保质期内的销量。请先核对各批次到期日，再决定促销或调拨。")
        st.caption("在途到货日期尚未录入；请向供应商核对交货安排。")
    with form, st.container(border=True):
        st.subheader("确认这项安排")
        if ((orders.store_id == store_id) & (orders.supplier_id == row.supplier_id)).any():
            st.info("该供应商的采购单已审核，可在采购与记录中下载。")
            return
        identity = f"{scope}:{store_id}:{sku}"
        prior = reviews[(reviews.store_id == store_id) & (reviews.sku == sku)]
        default = int(prior.iloc[0].final_qty) if not prior.empty else int(row.suggested_order_qty)
        action = st.radio(
            "处理方式",
            ["接受", "修改", "拒绝"],
            horizontal=True,
            key=f"action:{identity}",
            format_func=lambda a: {"接受": "按建议补货", "修改": "调整数量", "拒绝": "本次不采购"}[
                a
            ],
        )
        with st.form(f"review:{identity}"):
            qty = st.number_input(
                "本次补货数量",
                min_value=0,
                value=default
                if action == "修改"
                else int(row.suggested_order_qty)
                if action == "接受"
                else 0,
                step=int(row.case_pack),
                disabled=action != "修改",
                key=f"qty:{identity}:{action}",
            )
            reason = st.selectbox(
                "调整原因",
                [
                    "",
                    "资金安排",
                    "供应商缺货",
                    "计划促销",
                    "库存复核",
                    "附近活动",
                    "经验判断",
                    "其他",
                ],
                format_func=lambda x: x or "请选择（调整或不采购时必填）",
                key=f"reason:{identity}",
            )
            note = st.text_input(
                "补充说明", placeholder="例如：周末活动取消，先少进一箱", key=f"note:{identity}"
            )
            submitted = st.form_submit_button("保存安排", type="primary", width="stretch")
        if submitted:
            try:
                repo.review(scope, row.to_dict(), action, qty, reason, note)
                notice("已保存，采购草稿已同步更新。")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.button("前往采购与记录", key="go_orders", on_click=go, args=("采购与记录",))


def purchases(data, base, rec, store, repo, scope, policy):
    heading("采购与记录", "按门店和供应商核对草稿，审核后下载订货清单。")
    reviews, orders = repo.reviews(scope), repo.orders(scope)
    locked = set(zip(orders.store_id, orders.supplier_id))
    pending = (
        reviews[[(r.store_id, r.supplier_id) not in locked for r in reviews.itertuples()]]
        if not reviews.empty
        else reviews
    )
    drafts = scoped(pending[pending.final_qty > 0], store)
    all_cost = reviews.line_total.sum()
    metrics(
        [
            (
                "待审核金额",
                f"¥{drafts.line_total.sum():,.2f}",
                f"{len(drafts)} 条确认明细",
                "primary",
            ),
            (
                "本批次已审核",
                f"{scoped(orders, store).order_id.nunique()} 单",
                "保留完整审核记录",
                "",
            ),
            (
                "采购预算",
                f"¥{policy.max_purchase_budget:,.0f}"
                if policy.max_purchase_budget is not None
                else "未设置",
                "预算按本批次全部门店合计",
                "",
            ),
        ]
    )
    if policy.max_purchase_budget is not None and all_cost > policy.max_purchase_budget:
        st.warning(f"全部门店的确认金额 ¥{all_cost:,.2f} 已超出预算，请先回补货清单调整。")
    draft_tab, history_tab = st.tabs(["待审核草稿", "处理记录"])
    with draft_tab:
        if drafts.empty:
            empty("还没有待审核的采购草稿。先在补货清单中确认需要采购的商品。")
            st.button("去确认补货", key="review_from_orders", on_click=go, args=("补货清单",))
        else:
            groups = list(drafts.groupby(["store_id", "supplier_id"]).groups)
            group = st.selectbox(
                "选择采购单", groups, format_func=lambda x: f"{x[0]} · {x[1]}", key="order_group"
            )
            rows = drafts[(drafts.store_id == group[0]) & (drafts.supplier_id == group[1])]
            table(
                rows,
                [
                    "product_name",
                    "sku",
                    "original_qty",
                    "final_qty",
                    "unit_cost",
                    "line_total",
                    "reason",
                ],
            )
            st.write(f"本单合计 **¥{rows.line_total.sum():,.2f}**")
            st.caption("当前调拨建议尚未扣减采购量。若决定调拨，请先调整补货清单后再审核。")
            with st.form("approve_order"):
                confirmed = st.checkbox("已核对数量、价格和供应商交货安排")
                approve = st.form_submit_button("审核本单", type="primary")
            if approve:
                if not confirmed:
                    st.error("请先勾选核对项。")
                else:
                    try:
                        order_id = repo.approve(scope, *group, policy.max_purchase_budget)
                        notice(f"采购单 {order_id} 已审核，可在处理记录中下载。")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            st.download_button(
                "导出待审核草稿",
                download_csv(rows.rename(columns=COLUMNS)),
                "purchase-draft.csv",
                "text/csv",
            )
    with history_tab:
        st.subheader("本批次已审核采购")
        local_orders = scoped(orders, store)
        if local_orders.empty:
            empty("本批次尚无已审核采购单。")
        else:
            table(
                local_orders,
                [
                    "order_id",
                    "supplier_id",
                    "product_name",
                    "final_qty",
                    "line_total",
                    "approval_status",
                ],
            )
            st.download_button(
                "下载已审核采购单",
                download_csv(local_orders.drop(columns="scope").rename(columns=COLUMNS)),
                "purchase-approved.csv",
                "text/csv",
            )
        st.subheader("商品处理记录")
        local_reviews = scoped(reviews, store)
        if not local_reviews.empty:
            table(
                local_reviews,
                [
                    "store_id",
                    "product_name",
                    "action",
                    "original_qty",
                    "final_qty",
                    "reason",
                    "note",
                ],
            )
        else:
            st.caption("确认或调整商品后，记录会出现在这里。")
        with st.expander("查看历史批次采购"):
            historical = scoped(repo.orders(), store)
            historical = historical[historical.scope != scope]
            table(
                historical,
                ["order_id", "store_id", "product_name", "final_qty", "line_total", "approved_at"],
            )
            st.caption("导入新数据或修改策略会产生新的批次；审核前请核对历史采购，避免重复订货。")


def trends(data, base, rec, store):
    import altair as alt

    heading("销售走势", "对照实际销量和未来需求，判断哪些商品需要多留意。")
    f1, f2, f3 = st.columns([2, 1, 1])
    history = scoped(data.sales, store)
    names = data.products.set_index("sku").product_name.to_dict()
    sku = f1.selectbox("商品", sorted(history.sku.unique()), format_func=names.get, key="trend_sku")
    stores = sorted(history[history.sku == sku].store_id.unique())
    trend_store = f2.selectbox("对照门店", stores, key="trend_store")
    horizon = f3.selectbox(
        "查看未来", [7, 14, 30], format_func=lambda d: f"{d} 天", key="trend_horizon"
    )
    future = base.base_forecast[
        (base.base_forecast.sku == sku) & (base.base_forecast.store_id == trend_store)
    ].sort_values("date")
    metrics(
        [
            (
                f"未来 {d} 天",
                f"{future.head(d).predicted_units.sum():,.0f} 件",
                "预计需求量",
                "primary" if d == 7 else "",
            )
            for d in [1, 7, 30]
        ]
    )
    actual = (
        history[(history.sku == sku) & (history.store_id == trend_store)]
        .sort_values("date")
        .tail(28)
    )
    actual = actual[["date", "units"]].rename(columns={"units": "销量"}).assign(类型="实际销量")
    estimate = (
        future.head(horizon).rename(columns={"predicted_units": "销量"}).assign(类型="预计需求")
    )
    lines = pd.concat([actual, estimate[["date", "销量", "类型"]]], ignore_index=True)
    line = (
        alt.Chart(lines)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title="日期", axis=alt.Axis(format="%m/%d")),
            y=alt.Y("销量:Q", title="件数", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "类型:N",
                scale=alt.Scale(domain=["实际销量", "预计需求"], range=["#182d4d", "#2458d3"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash(
                "类型:N",
                scale=alt.Scale(domain=["实际销量", "预计需求"], range=[[1, 0], [5, 4]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("销量:Q", format=".1f"),
                "类型:N",
            ],
        )
    )
    band = (
        alt.Chart(future.head(horizon))
        .mark_area(opacity=0.12, color="#2458d3")
        .encode(x="date:T", y="lower_units:Q", y2="upper_units:Q")
    )
    st.altair_chart(
        (band + line).properties(height=310).configure_view(stroke=None), width="stretch"
    )
    st.caption("浅色带为历史误差估算的需求范围，远期参考性较弱；实际销量也可能受到缺货影响。")
    st.subheader("商品趋势对照")
    comparison = scoped(base.trends, store).merge(data.products[["sku", "product_name"]], on="sku")
    table(
        comparison.sort_values("trend_pct", ascending=False),
        ["store_id", "product_name", "trend", "trend_pct"],
    )
    st.caption("变化比例对比未来 7 天的预计日销量与过去 28 天的平均日需求。")


def scenarios(data, base, rec, store, policy, scope):
    heading("先测算，再决定", "输入一组经营假设，看看需求和备货安排可能怎样变化。")
    with st.form("scenario_form"):
        left, right = st.columns(2, gap="large")
        with left:
            price = st.slider("售价调整（%）", -30, 30, 0, key="scenario_price")
            promo = st.slider("额外促销需求（%）", 0, 100, 0, key="scenario_promo")
            traffic = st.slider("客流变化（%）", -50, 100, 0, key="scenario_traffic")
        with right:
            holiday = st.slider("节假日额外需求（%）", 0, 100, 0, key="scenario_holiday")
            delay = st.slider("交货额外延迟（天）", 0, 14, 0, key="scenario_delay")
            elasticity = st.number_input(
                "价格弹性假设",
                -5.0,
                0.0,
                -1.2,
                0.1,
                key="scenario_elasticity",
                help="例如 -1.2 表示涨价 1% 时，假设需求下降约 1.2%。此系数尚未从你的门店数据中估计。",
            )
        st.caption("同一活动带来的客流与促销影响请避免重复填写。本次测算作用于所选门店的全部商品。")
        submitted = st.form_submit_button("比较这组方案", type="primary")
    comparison_key = f"{scope}:{store}"
    if submitted:
        scenario = ScenarioConfig(price, promo, traffic, holiday, delay, elasticity)
        try:
            future, result = scenario_plan(
                scoped(base.base_forecast, store),
                RetailData(scoped(data.sales, store), data.products, scoped(data.inventory, store)),
                policy,
                scenario,
            )
            st.session_state["scenario_result"] = (comparison_key, scenario, future, result)
        except ValueError as exc:
            st.error(str(exc))
    saved = st.session_state.get("scenario_result")
    if not saved or saved[0] != comparison_key:
        empty("设置好条件后，点击“比较这组方案”。测算结果不会修改已确认的采购安排。")
        return
    _, scenario, future, result = saved
    cutoff = future.date.min() + pd.Timedelta(days=6)
    new = future[future.date <= cutoff]
    old = scoped(base.base_forecast, store)
    old = old[old.date <= cutoff]
    metrics(
        [
            (
                "未来 7 天需求",
                f"{new.predicted_units.sum():,.0f} 件",
                f"当前方案 {old.predicted_units.sum():,.0f} 件",
                "primary",
            ),
            (
                "测算补货金额",
                f"¥{result.purchase_cost.sum():,.0f}",
                f"当前方案 ¥{scoped(rec, store).purchase_cost.sum():,.0f}",
                "",
            ),
            (
                "优先补货项",
                f"{int((result.status == '高缺货风险').sum())} 项",
                "按测算后的交货时间判断",
                "",
            ),
        ]
    )
    chart = pd.DataFrame(
        {
            "当前需求": old.groupby("date").predicted_units.sum(),
            "测算需求": new.groupby("date").predicted_units.sum(),
        }
    )
    st.line_chart(chart, color=["#718298", "#2458d3"], height=240, y_label="预计件数")
    st.caption(
        f"本次假设：售价 {scenario.price_change_pct:+g}%，促销 +{scenario.promotion_lift_pct:g}%，"
        f"客流 {scenario.traffic_change_pct:+g}%，节假日 +{scenario.holiday_lift_pct:g}%，交货延迟 {scenario.supplier_delay_days} 天。"
    )
    table(result, ["store_id", "product_name", "suggested_order_qty", "purchase_cost", "status"])
    if store != "全部门店" and policy.max_purchase_budget is not None:
        st.caption("单店测算使用完整的本批次预算；若需在门店之间分配预算，请切换到全部门店比较。")
    st.download_button(
        "下载测算结果",
        download_csv(result.rename(columns=COLUMNS)),
        "scenario-plan.csv",
        "text/csv",
    )


def transfers(data, base, rec, store):
    heading("门店调拨", "一边有余量、一边需要货时，先核对能否在门店之间调配。")
    result = InventoryOptimizer().suggest_transfers(rec)
    if store != "全部门店":
        result = result[(result.from_store == store) | (result.to_store == store)]
    if result.empty:
        empty("当前没有可用现货足够的调拨建议。单门店经营可直接使用补货清单。")
        return
    table(
        result, ["product_name", "from_store", "to_store", "quantity", "estimated_purchase_saving"]
    )
    st.info(
        "这里列出的是可调拨现货。请确认保质期、运费与到达时间，完成调拨后更新库存，再核对采购清单。"
    )
    st.caption("对应采购货值不等于净节省金额；本版尚未计算运输和人工费用，也不会自动调减采购数量。")
    st.download_button(
        "下载调拨清单",
        download_csv(result.rename(columns=COLUMNS)),
        "store-transfers.csv",
        "text/csv",
    )


def settings(data, base, rec, store, repo, scope, policy):
    heading("数据与设置", "更新经营数据、调整备货偏好，或查看计算依据。")
    settings_tab, import_tab, method_tab = st.tabs(["备货偏好", "导入数据", "计算与数据说明"])
    with settings_tab:
        with st.form("policy_form"):
            name = st.selectbox(
                "经营偏好",
                ["保守", "平衡", "增长"],
                index=["保守", "平衡", "增长"].index(policy.name),
                format_func=lambda x: {"保守": "控制库存", "平衡": "平衡备货", "增长": "备货充足"}[
                    x
                ],
                key="policy_name",
            )
            st.caption("控制库存会减少安全余量；备货充足会预留更多库存，并承担更高的占款压力。")
            enabled = st.checkbox(
                "设置本批次采购预算",
                value=policy.max_purchase_budget is not None,
                key="policy_budget_enabled",
            )
            amount = st.number_input(
                "全部门店合计预算（元）",
                0.0,
                1_000_000.0,
                float(policy.max_purchase_budget or 5000),
                100.0,
                key="policy_budget",
            )
            custom = st.checkbox("自定义安全余量", key="policy_custom")
            safety = st.number_input(
                "额外安全库存天数", 0.0, 7.0, float(policy.safety_days), 0.5, key="policy_safety"
            )
            service = st.slider(
                "目标备货服务水平（%）",
                75,
                99,
                int(policy.service_level * 100),
                key="policy_service",
                help="用于估算安全库存的目标，不是实际服务水平的保证。",
            )
            st.caption("保存后会重新计算建议并建立新的审核批次，已有采购记录会保留。")
            save = st.form_submit_button("保存备货偏好", type="primary")
        if save:
            new_policy = StrategyConfig.preset(name, amount if enabled else None)
            if custom:
                new_policy = replace(new_policy, safety_days=safety, service_level=service / 100)
            st.session_state["policy"] = new_policy
            notice("备货偏好已保存，补货建议已更新。")
            st.rerun()
    with import_tab:
        st.write("每次导入一组完整的销售、商品和库存文件。建议准备至少 70 天的每日销量。")
        with st.form("import_form"):
            sales_file = st.file_uploader("每日销售 · sales.csv", type=["csv"], key="upload_sales")
            products_file = st.file_uploader(
                "商品资料 · products.csv", type=["csv"], key="upload_products"
            )
            inventory_file = st.file_uploader(
                "当前库存 · inventory.csv", type=["csv"], key="upload_inventory"
            )
            load = st.form_submit_button("检查并使用这组数据", type="primary")
        if load:
            if any(f is None for f in (sales_file, products_file, inventory_file)):
                st.error("请同时选择销售、商品和库存三份文件。")
            elif (
                sum(f.size for f in (sales_file, products_file, inventory_file)) > 25 * 1024 * 1024
            ):
                st.error("当前版本每批导入上限为 25 MB。请先按日汇总或减少历史范围。")
            else:
                try:
                    frames = [
                        pd.read_csv(
                            io.BytesIO(f.getvalue()),
                            dtype={"store_id": str, "sku": str, "supplier_id": str},
                        )
                        for f in (sales_file, products_file, inventory_file)
                    ]
                    checked = validate_data(RetailData(*frames))
                    with st.spinner("正在检查记录并计算需求…"):
                        analyse(checked)
                    st.session_state["active_data"] = checked
                    st.session_state["source"] = "门店数据"
                    st.session_state["pending_store_reset"] = True
                    notice("数据已更新，建议已根据这组记录重新计算。")
                    st.rerun()
                except (
                    ValueError,
                    UnicodeError,
                    pd.errors.ParserError,
                    pd.errors.EmptyDataError,
                ) as exc:
                    st.error(f"未导入：{exc}")
        if st.button("切换到演示数据", key="reset_sample"):
            st.session_state["active_data"] = sample_data()
            st.session_state["source"] = "演示数据"
            st.session_state["pending_store_reset"] = True
            notice("已切换到演示数据。")
            st.rerun()
        with st.expander("下载示例文件，查看字段格式"):
            demo = sample_data()
            for title, frame in [
                ("sales", demo.sales),
                ("products", demo.products),
                ("inventory", demo.inventory),
            ]:
                st.download_button(
                    f"下载 {title}.csv", download_csv(frame), f"{title}.csv", "text/csv"
                )
            st.caption("示例中的销量与库存为模拟数据。编号按文本读取，可保留前导零。")
    with method_tab:
        st.write(
            f"当前数据：{data.sales.date.min():%Y-%m-%d} 至 {data.sales.date.max():%Y-%m-%d}，"
            f"共 {len(data.sales):,} 条日销量记录、{data.inventory.store_id.nunique()} 家门店。"
        )
        unmatched = data.inventory.merge(
            data.sales[["store_id", "sku"]].drop_duplicates(),
            on=["store_id", "sku"],
            how="left",
            indicator=True,
        )
        unmatched = unmatched[unmatched._merge == "left_only"]
        if not unmatched.empty:
            st.warning(f"有 {len(unmatched)} 项门店商品没有销售历史，暂未给出补货建议。")
            table(unmatched, ["store_id", "sku", "on_hand"])
        st.write(
            "需求预测使用梯度提升回归，参考近期销量、每周规律与历史价格。新销量导入后重新拟合；浏览页面和调整情景不会重复训练。"
        )
        st.write(
            "店主反馈用于保留经营判断与采购安排，当前不会直接用反馈文字修改预测模型。补货采用目标库存与预算优先分配规则。"
        )
        st.write(
            "库存文件没有批次到期日和预计到货日，因此临期提示仅用于提醒核查；无法据此判断某一批货是否已经临期。"
        )
        with st.expander("查看预测验证指标"):
            m = base.metrics
            table(
                pd.DataFrame(
                    {
                        "指标": [
                            "日销量平均绝对误差",
                            "日销量加权绝对误差",
                            "误差范围覆盖比例",
                            "校验样本数",
                        ],
                        "数值": [
                            f"{m.mae:.2f} 件",
                            f"{m.wape:.1%}",
                            f"{m.interval_coverage:.1%}",
                            str(m.validation_rows),
                        ],
                    }
                )
            )
            st.caption(
                "按日期留出历史记录做逐日验证。覆盖比例来自同一误差校准样本，不代表未来 30 天的独立验证表现。"
            )
        st.caption("当前版本适合单店主本地使用。已审核表示本地留档，未与供应商系统连接。")


def main():
    st.set_page_config(
        page_title="StorePilot · 门店工作台", layout="wide", initial_sidebar_state="expanded"
    )
    style = files("storepilot").joinpath("assets/workbench.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
    if "active_data" not in st.session_state:
        st.session_state["active_data"] = sample_data()
        st.session_state["source"] = "演示数据"
    if "policy" not in st.session_state:
        st.session_state["policy"] = StrategyConfig.preset("平衡")
    data, source, policy = (
        st.session_state["active_data"],
        st.session_state["source"],
        st.session_state["policy"],
    )
    if st.session_state.pop("pending_store_reset", False):
        st.session_state["store_scope"] = "全部门店"
    with st.sidebar:
        st.markdown(
            '<div class="sp-brand">StorePilot</div><div class="sp-brand-sub">门店经营工作台</div>',
            unsafe_allow_html=True,
        )
        stores = ["全部门店"] + sorted(data.inventory.store_id.unique())
        if st.session_state.get("store_scope") not in stores:
            st.session_state["store_scope"] = "全部门店"
        store = st.selectbox("当前门店", stores, key="store_scope")
        page = st.radio("工作台导航", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption(f"{source} · 销售截至 {data.sales.date.max():%m月%d日}")
        st.caption("更改数据与备货偏好，请前往数据与设置。")
    st.markdown(
        f'<div class="sp-top"><span class="sp-badge">{escape(source)}</span>'
        f"<span>{escape(store)}</span><span>销售截至 {data.sales.date.max():%Y-%m-%d}</span></div>",
        unsafe_allow_html=True,
    )
    message = st.session_state.pop("notice", None)
    if message:
        st.success(message)
    age = (pd.Timestamp(datetime.now(UTC).date()) - data.sales.date.max()).days
    if age > 2:
        st.warning(f"销售数据已 {age} 天未更新。建议先导入最新记录，再确认采购。")
    try:
        with st.spinner("正在整理销售与库存…"):
            base = analyse(data)
            rec = selected_plan(data, base, policy)
        data_id = dataset_key(data, source)
        scope = decision_scope(data_id, policy)
        database = os.environ.get("STOREPILOT_DB", "storepilot.db")
        if source == "演示数据":
            dbpath = Path(database)
            database = str(dbpath.with_name(dbpath.stem + "-demo" + dbpath.suffix))
        repo = FeedbackRepository(database)
        if st.session_state.get("recorded_data") != data_id:
            repo.record_model_run(data.sales.date.max().isoformat(), base.metrics)
            st.session_state["recorded_data"] = data_id
        if page == "今日概览":
            overview(data, base, rec, store, repo, scope, policy)
        elif page == "补货清单":
            replenishment(data, base, rec, store, repo, scope)
        elif page == "采购与记录":
            purchases(data, base, rec, store, repo, scope, policy)
        elif page == "销售走势":
            trends(data, base, rec, store)
        elif page == "情景测算":
            scenarios(data, base, rec, store, policy, scope)
        elif page == "门店调拨":
            transfers(data, base, rec, store)
        else:
            settings(data, base, rec, store, repo, scope, policy)
    except (ValueError, sqlite3.Error, OSError) as exc:
        st.error(f"暂时无法完成操作：{exc}")
        st.caption("请检查数据和本地文件的读写权限后重试。")
