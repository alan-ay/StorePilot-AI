from __future__ import annotations

import io
import math
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
from .data import DEMO_CATALOG_VERSION, RetailData, generate_demo_data, merge_stores, validate_data
from .i18n import ValidationError, demo_label, translate
from .optimization import InventoryOptimizer
from .pipeline import StorePilotPipeline
from .reporting import generate_report
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


def language():
    return st.session_state.get("language", "zh")


def t(message, *values):
    values = [
        value.render(language()) if isinstance(value, ValidationError) else value
        for value in values
    ]
    return translate(message, language(), *values)


def product_label(value):
    return demo_label(value, language()) if st.session_state.get("source") == "演示数据" else value


def option_labels(options, formatter=t):
    """Bind labels to this render without changing the options' stored values."""
    return {value: formatter(value) for value in options}.get


def display_frame(frame, *, rename_columns=True):
    selected = frame.copy()
    if "status" in selected:
        selected["status"] = selected["status"].map(STATUS).fillna(selected["status"])
    if "days_of_cover" in selected:
        selected["days_of_cover"] = selected["days_of_cover"].replace(math.inf, None)
    for column in ["status", "确认状态", "action", "reason", "approval_status", "trend", "指标"]:
        if column in selected:
            selected[column] = selected[column].map(
                lambda value: t(value) if isinstance(value, str) else value
            )
    for column in ["product_name", "category"]:
        if column in selected:
            selected[column] = selected[column].map(product_label)
    if rename_columns:
        selected = selected.rename(columns=lambda name: t(COLUMNS.get(name, name)))
    return selected


def table(frame, columns=None, **kwargs):
    selected = frame[columns] if columns else frame
    st.dataframe(display_frame(selected), hide_index=True, width="stretch", **kwargs)


def empty(message):
    st.markdown(f'<div class="sp-empty">{escape(t(message))}</div>', unsafe_allow_html=True)


def cover_text(days):
    return t("约可售 {0:g} 天", days) if math.isfinite(days) else t("暂无销售需求")


def go(page):
    st.session_state["nav"] = page


def notice(message):
    st.session_state["notice"] = t(message)


def heading(title, subtitle):
    st.markdown(
        f'<h1 class="sp-heading">{escape(t(title))}</h1><p class="sp-intro">{escape(t(subtitle))}</p>',
        unsafe_allow_html=True,
    )


def metrics(items):
    for col, (label, value, note, tone) in zip(st.columns(len(items)), items, strict=True):
        with col:
            st.markdown(
                f'<div class="sp-metric"><div class="sp-metric-label">{escape(t(label))}</div>'
                f'<div class="sp-metric-value {tone}">{escape(t(str(value)))}</div>'
                f'<div class="sp-metric-note">{escape(t(note))}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")


@st.cache_data(show_spinner=False, max_entries=4)
def analyse(data):
    horizon = min(90, max(30, int(data.products.lead_time_days.max()) + 28))
    return StorePilotPipeline().run(data, horizon_days=horizon)


@st.cache_data(show_spinner=False, ttl=3600)
def sample_data(catalog_version=DEMO_CATALOG_VERSION):
    """Include the demo version in the cache key when products or stores change."""
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
    heading("今天先处理这些", "先看缺货与积压，再核对补货清单。")
    if sales.empty:
        empty("该门店暂无销售历史。请在数据与设置中导入销售记录后查看经营概况。")
        return
    latest_date = sales.date.max()
    latest = sales[sales.date == latest_date]
    revenue = float((latest.units * latest.price).sum())
    before = sales[sales.date == latest_date - pd.Timedelta(days=7)]
    previous = float((before.units * before.price).sum())
    change = (
        t("较上周同日 {0:+.1f}%", (revenue / previous - 1) * 100)
        if previous
        else t("上周同日数据不足")
    )
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
            ("最近营业额", f"¥{revenue:,.0f}", t("{0:%m月%d日} · {1}", latest_date, change), ""),
            (
                "优先补货",
                t("{0} 项", len(urgent)),
                "现货预计撑不到补货到达",
                "attention" if len(urgent) else "",
            ),
            ("待审核采购", f"¥{pending.line_total.sum():,.0f}", "已确认清单的采购金额", "primary"),
            (
                "库存需复核",
                t("{0} 项", int((local.expiry_risk_units > 0).sum())),
                "需核对批次保质期",
                "",
            ),
        ]
    )
    left, right = st.columns([1.65, 1], gap="large")
    with left, st.container(border=True):
        st.subheader(t("待办清单"))
        tasks = local[local.status != "正常"].head(4)
        if tasks.empty:
            empty("目前没有紧急事项，可以继续查看本周销售情况。")
        for row in tasks.itertuples():
            text = t(
                "{0} · 现货 {1:g} 件，{2}；交货需 {3} 天。",
                row.store_id,
                row.on_hand,
                cover_text(row.days_of_cover),
                row.lead_time_days,
            )
            st.markdown(
                f'<div class="sp-task"><div class="sp-task-title">{escape(product_label(row.product_name))}'
                f'<span class="sp-label">{escape(t(STATUS[row.status]))}</span></div>'
                f'<div class="sp-task-detail">{escape(text)}</div></div>',
                unsafe_allow_html=True,
            )
        st.button(
            t("查看并处理补货清单"),
            key="open_replenishment",
            on_click=go,
            args=("补货清单",),
            type="primary",
        )
    with right, st.container(border=True):
        st.subheader(t("销售概况"))
        daily = (
            sales.assign(营业额=sales.units * sales.price).groupby("date")["营业额"].sum().tail(14)
        )
        st.line_chart(
            daily.rename(t("营业额")),
            color="#2458d3",
            height=230,
            x_label=t("日期"),
            y_label=t("营业额（元）"),
        )
        st.caption(t("当前采用{0}策略 · {1} 项门店商品", t(policy.name), len(local)))
        st.button(t("查看商品走势"), key="open_trends", on_click=go, args=("销售走势",))
    with st.expander(t("查看当日经营摘要")):
        transfers = InventoryOptimizer().suggest_transfers(rec)
        if store != "全部门店":
            transfers = transfers[(transfers.from_store == store) | (transfers.to_store == store)]
        report = generate_report(
            sales,
            scoped(base.trends, store),
            local.assign(product_name=local.product_name.map(product_label)),
            transfers,
            policy.name,
            language=language(),
        )
        st.markdown(report)
        st.download_button(t("下载经营摘要"), report, "storepilot-report.md", "text/markdown")


def replenishment(data, base, rec, store, repo, scope):
    heading("补货清单", "逐项确认后，前往采购页审核。数量调整会同步到采购草稿。")
    reviews, orders = repo.reviews(scope), repo.orders(scope)
    local = review_status(scoped(rec, store), reviews, orders)
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input(t("搜索商品"), placeholder=t("商品名或编号"), key="product_search")
    categories = ["全部品类"] + sorted(local.category.unique())
    category = f2.selectbox(
        t("品类"),
        categories,
        key="category_filter",
        format_func=option_labels(
            categories, lambda value: t(value) if value == "全部品类" else product_label(value)
        ),
    )
    filter_status = f3.selectbox(
        t("处理范围"),
        ["需要处理", "全部商品", "已处理"],
        key="status_filter",
        format_func=option_labels(["需要处理", "全部商品", "已处理"]),
    )
    if query:
        local = local[
            local.product_name.map(product_label).str.contains(query, case=False, regex=False)
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
        st.button(t("前往采购与记录"), on_click=go, args=("采购与记录",))
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
    names = {
        (r.store_id, r.sku): f"{product_label(r.product_name)} · {r.store_id}"
        for r in local.itertuples()
    }
    chosen = st.selectbox(
        t("选择要处理的商品"), list(names), format_func=names.get, key="review_item"
    )
    store_id, sku = chosen
    row = local[(local.store_id == store_id) & (local.sku == sku)].iloc[0]
    details, form = st.columns([1, 1.2], gap="large")
    with details, st.container(border=True):
        st.subheader(product_label(str(row.product_name)))
        st.write(
            t(
                "建议补货 **{0} 件**，参考金额 **¥{1:,.2f}**。",
                row.suggested_order_qty,
                row.purchase_cost,
            )
        )
        st.write(
            t(
                "当前现货 {0:g} 件，在途 {1:g} 件；预计每天售出 {2:g} 件。",
                row.on_hand,
                row.on_order,
                row.daily_demand,
            )
        )
        st.write(
            t(
                "交货需要 {0} 天，目标库存约 {1:g} 件，含安全余量 {2:g} 件。",
                row.lead_time_days,
                row.target_stock,
                row.safety_stock,
            )
        )
        st.caption(
            t(
                "每箱 {0} 件 · 起订 {1} 件 · 供应商 {2}",
                row.case_pack,
                row.min_order_qty,
                row.supplier_id,
            )
        )
        if row.budget_adjusted:
            st.warning(t("本项建议已受预算限制，仍可能存在缺货。"))
        if row.expiry_risk_units > 0:
            st.info(t("库存可能超过保质期内的销量。请先核对各批次到期日，再决定促销或调拨。"))
        st.caption(t("在途到货日期尚未录入；请向供应商核对交货安排。"))
    with form, st.container(border=True):
        st.subheader(t("确认这项安排"))
        if ((orders.store_id == store_id) & (orders.supplier_id == row.supplier_id)).any():
            st.info(t("该供应商的采购单已审核，可在采购与记录中下载。"))
            return
        identity = f"{scope}:{store_id}:{sku}"
        prior = reviews[(reviews.store_id == store_id) & (reviews.sku == sku)]
        default = int(prior.iloc[0].final_qty) if not prior.empty else int(row.suggested_order_qty)
        action = st.radio(
            t("处理方式"),
            ["接受", "修改", "拒绝"],
            index=["接受", "修改", "拒绝"].index(prior.iloc[0].action) if not prior.empty else 0,
            horizontal=True,
            key=f"action:{identity}",
            format_func={
                key: t(label)
                for key, label in {
                    "接受": "按建议补货",
                    "修改": "调整数量",
                    "拒绝": "本次不采购",
                }.items()
            }.get,
        )
        # Keep draft edits in session state so a language switch can retain them.
        with st.container(key=f"review:{identity}", border=True):
            qty = st.number_input(
                t("本次补货数量"),
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
            reasons = [
                "",
                "资金安排",
                "供应商缺货",
                "计划促销",
                "库存复核",
                "附近活动",
                "经验判断",
                "其他",
            ]
            saved_reason = prior.iloc[0].reason if not prior.empty else ""
            reason_key = f"reason:{identity}"
            # Re-send the selected label; changing options alone leaves stale browser text.
            st.session_state[reason_key] = st.session_state.get(
                reason_key, saved_reason if saved_reason in reasons else ""
            )
            reason = st.selectbox(
                t("调整原因"),
                reasons,
                format_func=option_labels(
                    reasons, lambda x: t(x or "请选择（调整或不采购时必填）")
                ),
                key=reason_key,
            )
            note = st.text_input(
                t("补充说明"),
                value=prior.iloc[0].note if not prior.empty else "",
                placeholder=t("例如：周末活动取消，先少进一箱"),
                key=f"note:{identity}",
            )
            submitted = st.button(
                t("保存安排"), key=f"save:{identity}", type="primary", width="stretch"
            )
        if submitted:
            try:
                repo.review(scope, row.to_dict(), action, qty, reason, note)
                notice("已保存，采购草稿已同步更新。")
                st.rerun()
            except ValueError as exc:
                st.error(t("{0}", exc))
    st.button(t("前往采购与记录"), key="go_orders", on_click=go, args=("采购与记录",))


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
                t("{0} 条确认明细", len(drafts)),
                "primary",
            ),
            (
                "本批次已审核",
                t("{0} 单", scoped(orders, store).order_id.nunique()),
                "保留完整审核记录",
                "",
            ),
            (
                "采购预算",
                f"¥{policy.max_purchase_budget:,.0f}"
                if policy.max_purchase_budget is not None
                else "未设置",
                "预算按本批次合计",
                "",
            ),
        ]
    )
    if policy.max_purchase_budget is not None and all_cost > policy.max_purchase_budget:
        st.warning(t("本批次的确认金额 ¥{0:,.2f} 已超出预算，请先回补货清单调整。", all_cost))
    draft_tab, history_tab = st.tabs([t("待审核草稿"), t("处理记录")])
    with draft_tab:
        if drafts.empty:
            empty("还没有待审核的采购草稿。先在补货清单中确认需要采购的商品。")
            st.button(t("去确认补货"), key="review_from_orders", on_click=go, args=("补货清单",))
        else:
            groups = list(drafts.groupby(["store_id", "supplier_id"]).groups)
            group = st.selectbox(
                t("选择采购单"), groups, format_func=lambda x: f"{x[0]} · {x[1]}", key="order_group"
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
            st.write(t("本单合计 **¥{0:,.2f}**", rows.line_total.sum()))
            if data.inventory.store_id.nunique() > 1:
                st.caption(t("当前调拨建议尚未扣减采购量。若决定调拨，请先调整补货清单后再审核。"))
            # Confirmation belongs to the exact draft the owner checked.
            approval_key = f"{scope}:{group}:{','.join(rows.reviewed_at)}"
            with st.form(f"approve_order:{approval_key}"):
                confirmed = st.checkbox(
                    t("已核对数量、价格和供应商交货安排"), key=f"confirm:{approval_key}"
                )
                approve = st.form_submit_button(t("审核本单"), type="primary")
            if approve:
                if not confirmed:
                    st.error(t("请先勾选核对项。"))
                else:
                    try:
                        order_id = repo.approve(scope, *group, policy.max_purchase_budget)
                        notice(t("采购单 {0} 已审核，可在处理记录中下载。", order_id))
                        st.rerun()
                    except ValueError as exc:
                        st.error(t("{0}", exc))
            st.download_button(
                t("导出待审核草稿"),
                download_csv(display_frame(rows)),
                "purchase-draft.csv",
                "text/csv",
            )
    with history_tab:
        st.subheader(t("本批次已审核采购"))
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
                t("下载已审核采购单"),
                download_csv(display_frame(local_orders.drop(columns="scope"))),
                "purchase-approved.csv",
                "text/csv",
            )
        st.subheader(t("商品处理记录"))
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
            st.caption(t("确认或调整商品后，记录会出现在这里。"))
        with st.expander(t("查看历史批次采购")):
            historical = scoped(repo.orders(), store)
            historical = historical[historical.scope != scope]
            table(
                historical,
                ["order_id", "store_id", "product_name", "final_qty", "line_total", "approved_at"],
            )
            st.caption(
                t("导入新数据或修改策略会产生新的批次；审核前请核对历史采购，避免重复订货。")
            )


def trends(data, base, rec, store):
    import altair as alt

    heading("销售走势", "对照实际销量和未来需求，判断哪些商品需要多留意。")
    history = scoped(data.sales, store)
    if history.empty:
        empty("该门店暂无销售历史，导入销售记录后即可查看走势。")
        return
    f1, f2, f3 = st.columns([2, 1, 1])
    names = data.products.set_index("sku").product_name.map(product_label).to_dict()
    sku = f1.selectbox(
        t("商品"),
        sorted(history.sku.unique()),
        format_func=names.get,
        key="trend_sku",
    )
    stores = sorted(history[history.sku == sku].store_id.unique())
    if st.session_state.get("trend_store") not in stores:
        st.session_state["trend_store"] = stores[0]
    trend_store = f2.selectbox(t("对照门店"), stores, key="trend_store", disabled=len(stores) == 1)
    horizon = f3.selectbox(
        t("查看未来"),
        [7, 14, 30],
        format_func=option_labels([7, 14, 30], lambda d: t("{0} 天", d)),
        key="trend_horizon",
    )
    future = base.base_forecast[
        (base.base_forecast.sku == sku) & (base.base_forecast.store_id == trend_store)
    ].sort_values("date")
    metrics(
        [
            (
                t("未来 {0} 天", d),
                t("{0:,.0f} 件", future.head(d).predicted_units.sum()),
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
    actual = actual[["date", "units"]].assign(kind=t("实际销量"))
    estimate = (
        future.head(horizon).rename(columns={"predicted_units": "units"}).assign(kind=t("预计需求"))
    )
    lines = pd.concat([actual, estimate[["date", "units", "kind"]]], ignore_index=True)
    line = (
        alt.Chart(lines)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title=t("日期"), axis=alt.Axis(format="%m/%d")),
            y=alt.Y("units:Q", title=t("件数"), scale=alt.Scale(zero=True)),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=[t("实际销量"), t("预计需求")], range=["#182d4d", "#2458d3"]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash(
                "kind:N",
                scale=alt.Scale(domain=[t("实际销量"), t("预计需求")], range=[[1, 0], [5, 4]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("date:T", title=t("日期"), format="%Y-%m-%d"),
                alt.Tooltip("units:Q", title=t("件数"), format=".1f"),
                alt.Tooltip("kind:N", title=t("类型")),
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
    st.caption(t("浅色带为历史误差估算的需求范围，远期参考性较弱；实际销量也可能受到缺货影响。"))
    st.subheader(t("商品趋势对照"))
    comparison = scoped(base.trends, store).merge(data.products[["sku", "product_name"]], on="sku")
    table(
        comparison.sort_values("trend_pct", ascending=False),
        ["store_id", "product_name", "trend", "trend_pct"],
    )
    st.caption(t("变化比例对比未来 7 天的预计日销量与过去 28 天的平均日需求。"))


def scenarios(data, base, rec, store, policy, scope):
    heading("先测算，再决定", "输入一组经营假设，看看需求和备货安排可能怎样变化。")
    if scoped(rec, store).empty:
        empty("该门店暂无可测算的商品，请先导入对应的销售和库存记录。")
        return
    with st.form("scenario_form"):
        left, right = st.columns(2, gap="large")
        with left:
            price = st.slider(t("售价调整（%）"), -30, 30, 0, key="scenario_price")
            promo = st.slider(t("额外促销需求（%）"), 0, 100, 0, key="scenario_promo")
            traffic = st.slider(t("客流变化（%）"), -50, 100, 0, key="scenario_traffic")
        with right:
            holiday = st.slider(t("节假日额外需求（%）"), 0, 100, 0, key="scenario_holiday")
            delay = st.slider(t("交货额外延迟（天）"), 0, 14, 0, key="scenario_delay")
            elasticity = st.number_input(
                t("价格弹性假设"),
                -5.0,
                0.0,
                -1.2,
                0.1,
                key="scenario_elasticity",
                help=t(
                    "例如 -1.2 表示涨价 1% 时，假设需求下降约 1.2%。此系数尚未从你的门店数据中估计。"
                ),
            )
        st.caption(
            t("同一活动带来的客流与促销影响请避免重复填写。本次测算作用于所选门店的全部商品。")
        )
        submitted = st.form_submit_button(t("比较这组方案"), type="primary")
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
            st.error(t("{0}", exc))
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
                t("{0:,.0f} 件", new.predicted_units.sum()),
                t("当前方案 {0:,.0f} 件", old.predicted_units.sum()),
                "primary",
            ),
            (
                "测算补货金额",
                f"¥{result.purchase_cost.sum():,.0f}",
                t("当前方案 ¥{0:,.0f}", scoped(rec, store).purchase_cost.sum()),
                "",
            ),
            (
                "优先补货项",
                t("{0} 项", int((result.status == "高缺货风险").sum())),
                "按测算后的交货时间判断",
                "",
            ),
        ]
    )
    chart = pd.DataFrame(
        {
            t("当前需求"): old.groupby("date").predicted_units.sum(),
            t("测算需求"): new.groupby("date").predicted_units.sum(),
        }
    )
    st.line_chart(chart, color=["#718298", "#2458d3"], height=240, y_label=t("预计件数"))
    st.caption(
        t(
            "本次假设：售价 {0:+g}%，促销 +{1:g}%，客流 {2:+g}%，节假日 +{3:g}%，交货延迟 {4} 天。",
            scenario.price_change_pct,
            scenario.promotion_lift_pct,
            scenario.traffic_change_pct,
            scenario.holiday_lift_pct,
            scenario.supplier_delay_days,
        )
    )
    table(result, ["store_id", "product_name", "suggested_order_qty", "purchase_cost", "status"])
    if (
        data.inventory.store_id.nunique() > 1
        and store != "全部门店"
        and policy.max_purchase_budget is not None
    ):
        st.caption(
            t("单店测算使用完整的本批次预算；若需在门店之间分配预算，请切换到全部门店比较。")
        )
    st.download_button(
        t("下载测算结果"),
        download_csv(display_frame(result)),
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
        t(
            "这里列出的是可调拨现货。请确认保质期、运费与到达时间，完成调拨后更新库存，再核对采购清单。"
        )
    )
    st.caption(
        t("对应采购货值不等于净节省金额；本版尚未计算运输和人工费用，也不会自动调减采购数量。")
    )
    st.download_button(
        t("下载调拨清单"),
        download_csv(display_frame(result)),
        "store-transfers.csv",
        "text/csv",
    )


def settings(data, base, rec, store, repo, scope, policy):
    heading("数据与设置", "更新经营数据、调整备货偏好，或查看计算依据。")
    settings_tab, import_tab, method_tab = st.tabs(
        [t("备货偏好"), t("导入数据"), t("计算与数据说明")]
    )
    with settings_tab:
        with st.form("policy_form"):
            name = st.selectbox(
                t("经营偏好"),
                ["保守", "平衡", "增长"],
                index=["保守", "平衡", "增长"].index(policy.name),
                format_func={
                    key: t(label)
                    for key, label in {
                        "保守": "控制库存",
                        "平衡": "平衡备货",
                        "增长": "备货充足",
                    }.items()
                }.get,
                key="policy_name",
            )
            st.caption(t("控制库存会减少安全余量；备货充足会预留更多库存，并承担更高的占款压力。"))
            enabled = st.checkbox(
                t("设置本批次采购预算"),
                value=policy.max_purchase_budget is not None,
                key="policy_budget_enabled",
            )
            amount = st.number_input(
                t("本批次采购预算（元）"),
                0.0,
                1_000_000.0,
                float(
                    policy.max_purchase_budget if policy.max_purchase_budget is not None else 5000
                ),
                100.0,
                key="policy_budget",
            )
            preset = StrategyConfig.preset(policy.name)
            custom = st.checkbox(
                t("自定义安全余量"),
                value=(
                    policy.safety_days != preset.safety_days
                    or policy.service_level != preset.service_level
                ),
                key="policy_custom",
            )
            safety = st.number_input(
                t("额外安全库存天数"), 0.0, 7.0, float(policy.safety_days), 0.5, key="policy_safety"
            )
            service = st.slider(
                t("目标备货服务水平（%）"),
                75,
                99,
                int(policy.service_level * 100),
                key="policy_service",
                help=t("用于估算安全库存的目标，不是实际服务水平的保证。"),
            )
            st.caption(t("保存后会重新计算建议并建立新的审核批次，已有采购记录会保留。"))
            save = st.form_submit_button(t("保存备货偏好"), type="primary")
        if save:
            new_policy = StrategyConfig.preset(name, amount if enabled else None)
            if custom:
                new_policy = replace(new_policy, safety_days=safety, service_level=service / 100)
            try:
                selected_plan(data, base, new_policy)
            except ValueError as exc:
                st.error(t("未保存：{0}", exc))
            else:
                st.session_state["policy"] = new_policy
                notice("备货偏好已保存，补货建议已更新。")
                st.rerun()
    with import_tab:
        st.info(
            t(
                "演示与示例文件仅包含非食品日用品：家居清洁、衣物清洁、个人护理和纸品。上传文件保留你自己的商品与品类。"
            )
        )
        st.write(t("每次导入一组完整的销售、商品和库存文件。建议准备至少 70 天的每日销量。"))
        with st.form("import_form"):
            sales_file = st.file_uploader(
                t("每日销售 · sales.csv"), type=["csv"], key="upload_sales"
            )
            products_file = st.file_uploader(
                t("商品资料 · products.csv"), type=["csv"], key="upload_products"
            )
            inventory_file = st.file_uploader(
                t("当前库存 · inventory.csv"), type=["csv"], key="upload_inventory"
            )
            load = st.form_submit_button(t("检查并使用这组数据"), type="primary")
        if load:
            if any(f is None for f in (sales_file, products_file, inventory_file)):
                st.error(t("请同时选择销售、商品和库存三份文件。"))
            elif (
                sum(f.size for f in (sales_file, products_file, inventory_file)) > 25 * 1024 * 1024
            ):
                st.error(t("当前版本每批导入上限为 25 MB。请先按日汇总或减少历史范围。"))
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
                    with st.spinner(t("正在检查记录并计算需求…")):
                        checked_base = analyse(checked)
                        selected_plan(checked, checked_base, policy)
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
                    st.error(t("未导入：{0}", exc))
        if st.button(t("切换到演示数据"), key="reset_sample"):
            st.session_state["active_data"] = sample_data()
            st.session_state["source"] = "演示数据"
            st.session_state["pending_store_reset"] = True
            notice("已切换到演示数据。")
            st.rerun()
        with st.expander(t("下载示例文件，查看字段格式")):
            demo = sample_data()
            example_products = demo.products.copy()
            for column in ["product_name", "category"]:
                example_products[column] = example_products[column].map(
                    lambda value: demo_label(value, language())
                )
            for title, frame in [
                ("sales", demo.sales),
                ("products", example_products),
                ("inventory", demo.inventory),
            ]:
                st.download_button(
                    t("下载 {0}.csv", title), download_csv(frame), f"{title}.csv", "text/csv"
                )
            st.caption(t("示例中的销量与库存为模拟数据。编号按文本读取，可保留前导零。"))
    with method_tab:
        st.write(
            t(
                "当前数据：{0:%Y-%m-%d} 至 {1:%Y-%m-%d}，共 {2:,} 条日销量记录、{3} 家门店。",
                data.sales.date.min(),
                data.sales.date.max(),
                len(data.sales),
                data.inventory.store_id.nunique(),
            )
        )
        unmatched = data.inventory.merge(
            data.sales[["store_id", "sku"]].drop_duplicates(),
            on=["store_id", "sku"],
            how="left",
            indicator=True,
        )
        unmatched = unmatched[unmatched._merge == "left_only"]
        if not unmatched.empty:
            st.warning(t("有 {0} 项门店商品没有销售历史，暂未给出补货建议。", len(unmatched)))
            table(unmatched, ["store_id", "sku", "on_hand"])
        st.write(
            t(
                "需求预测使用梯度提升回归，参考近期销量、每周规律与历史价格。新销量导入后重新拟合；浏览页面和调整情景不会重复训练。"
            )
        )
        st.write(
            t(
                "店主反馈用于保留经营判断与采购安排，当前不会直接用反馈文字修改预测模型。补货采用目标库存与预算优先分配规则。"
            )
        )
        st.write(
            t(
                "库存文件没有批次到期日和预计到货日，因此临期提示仅用于提醒核查；无法据此判断某一批货是否已经临期。"
            )
        )
        with st.expander(t("查看预测验证指标")):
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
                            t("{0:.2f} 件", m.mae),
                            f"{m.wape:.1%}",
                            f"{m.interval_coverage:.1%}",
                            str(m.validation_rows),
                        ],
                    }
                )
            )
            st.caption(
                t(
                    "按日期留出历史记录做逐日验证。覆盖比例来自同一误差校准样本，不代表未来 30 天的独立验证表现。"
                )
            )
        st.caption(t("当前版本适合单店主本地使用。已审核表示本地留档，未与供应商系统连接。"))


def main():
    if "language" not in st.session_state:
        st.session_state["language"] = "zh"
    st.set_page_config(
        page_title=t("StorePilot · 门店工作台"), layout="wide", initial_sidebar_state="expanded"
    )
    style = files("storepilot").joinpath("assets/workbench.css").read_text(encoding="utf-8")
    st.html(f"<style>{style}</style>")
    if "active_data" not in st.session_state or (
        st.session_state.get("source") == "演示数据"
        and st.session_state.get("demo_catalog_version") != DEMO_CATALOG_VERSION
    ):
        if (
            "active_data" in st.session_state
            and st.session_state.get("source") == "演示数据"
            and st.session_state.get("demo_catalog_version") == "household-v1"
        ):
            st.session_state["active_data"] = merge_stores(
                st.session_state["active_data"], "STORE-01"
            )
        else:
            st.session_state["active_data"] = sample_data()
        st.session_state["source"] = "演示数据"
        st.session_state["demo_catalog_version"] = DEMO_CATALOG_VERSION
        st.session_state["pending_store_reset"] = True
    if "policy" not in st.session_state:
        st.session_state["policy"] = StrategyConfig.preset("平衡")
    data, source, policy = (
        st.session_state["active_data"],
        st.session_state["source"],
        st.session_state["policy"],
    )
    store_ids = sorted(data.inventory.store_id.unique())
    stores = ["全部门店", *store_ids] if len(store_ids) > 1 else store_ids
    pages = [page for page in PAGES if len(store_ids) > 1 or page != "门店调拨"]
    if st.session_state.pop("pending_store_reset", False):
        st.session_state["store_scope"] = stores[0]
    if st.session_state.get("store_scope") not in stores:
        st.session_state["store_scope"] = stores[0]
    if st.session_state.get("nav") not in pages:
        st.session_state["nav"] = pages[0]
    with st.sidebar:
        st.radio(
            "Language / 语言",
            ["zh", "en"],
            format_func={"zh": "中文", "en": "English"}.get,
            key="language",
            horizontal=True,
        )
        st.markdown(
            f'<div class="sp-brand">StorePilot</div><div class="sp-brand-sub">{escape(t("门店经营工作台"))}</div>',
            unsafe_allow_html=True,
        )
        store = st.selectbox(
            t("当前门店"),
            stores,
            key="store_scope",
            disabled=len(store_ids) == 1,
            format_func=option_labels(
                stores, lambda value: t(value) if value == "全部门店" else value
            ),
        )
        page = st.radio(
            t("工作台导航"),
            pages,
            key="nav",
            format_func=option_labels(pages),
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(t("{0} · 销售截至 {1:%m月%d日}", t(source), data.sales.date.max()))
        st.caption(t("更改数据与备货偏好，请前往数据与设置。"))
    st.markdown(
        f'<div class="sp-top"><span class="sp-badge">{escape(t(source))}</span>'
        f"<span>{escape(t(store) if store == '全部门店' else store)}</span>"
        f"<span>{escape(t('销售截至 {0:%Y-%m-%d}', data.sales.date.max()))}</span></div>",
        unsafe_allow_html=True,
    )
    message = st.session_state.pop("notice", None)
    if message:
        st.success(message)
    age = (pd.Timestamp(datetime.now(UTC).date()) - data.sales.date.max()).days
    if age > 2:
        st.warning(t("销售数据已 {0} 天未更新。建议先导入最新记录，再确认采购。", age))
    try:
        with st.spinner(t("正在整理销售与库存…")):
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
        st.error(t("暂时无法完成操作：{0}", exc))
        st.caption(t("请检查数据和本地文件的读写权限后重试。"))
