from __future__ import annotations

import pandas as pd

STATUS_PRIORITY = ["高缺货风险", "临期/报废风险", "需要关注", "库存偏高"]


def _product_list(frame: pd.DataFrame, limit: int = 3) -> str:
    if frame.empty:
        return "无"
    return "、".join(frame["product_name"].astype(str).drop_duplicates().head(limit).tolist())


def generate_chinese_report(
    sales: pd.DataFrame,
    trends: pd.DataFrame,
    recommendations: pd.DataFrame,
    transfers: pd.DataFrame,
    strategy_name: str,
) -> str:
    latest_date = pd.to_datetime(sales["date"]).max()
    latest = sales[pd.to_datetime(sales["date"]) == latest_date]
    sales_units = float(latest["units"].sum())
    revenue = float((latest["units"] * latest["price"]).sum())
    high_risk = recommendations[recommendations["status"] == "高缺货风险"].sort_values(
        "expected_shortage", ascending=False
    )
    expiry = recommendations[recommendations["expiry_risk_units"] > 0].sort_values(
        "expiry_risk_units", ascending=False
    )
    overstock = recommendations[recommendations["status"] == "库存偏高"].sort_values(
        "excess_units", ascending=False
    )
    order_lines = recommendations[recommendations["suggested_order_qty"] > 0]
    trend_names = recommendations[["store_id", "sku", "product_name"]].merge(
        trends, on=["store_id", "sku"], how="right"
    )
    growing_names = trend_names[trend_names["trend"] == "增长"].sort_values(
        "trend_pct", ascending=False
    )
    declining_names = trend_names[trend_names["trend"] == "下降"].sort_values("trend_pct")

    lines = [
        f"# StorePilot 每日经营简报｜{latest_date:%Y-%m-%d}",
        "",
        (
            f"当前采用“{strategy_name}”经营策略。昨日共销售 **{sales_units:.0f} 件**商品，"
            f"估算营业额为 **¥{revenue:,.2f}**。"
        ),
        "",
        "## 今日重点",
        "",
    ]
    if high_risk.empty:
        lines.append("- 暂未发现需要立即处理的高缺货风险。")
    else:
        top = high_risk.iloc[0]
        lines.append(
            f"- **缺货预警：** {_product_list(high_risk)}。其中 {top['product_name']} "
            f"预计短缺 {top['expected_shortage']:.0f} 件，建议补货 "
            f"{top['suggested_order_qty']:.0f} 件。"
        )
    if expiry.empty:
        lines.append("- 暂未发现明显临期或报废风险。")
    else:
        lines.append(f"- **临期处理：** {_product_list(expiry)}，建议促销、减少采购或跨店调拨。")
    if overstock.empty:
        lines.append("- 库存总体处于正常区间。")
    else:
        lines.append(f"- **库存偏高：** {_product_list(overstock)}，建议暂停补货并观察销售速度。")

    lines.extend(
        [
            "",
            "## 销售趋势",
            "",
            f"- 增长商品：{_product_list(growing_names)}。",
            f"- 下降商品：{_product_list(declining_names)}。",
            "",
            "## 采购与调拨",
            "",
            (
                f"- 今日建议生成 {len(order_lines)} 条采购明细，预计采购金额 "
                f"**¥{order_lines['purchase_cost'].sum():,.2f}**。"
            ),
        ]
    )
    if transfers.empty:
        lines.append("- 暂无明显优于重新采购的跨店调拨机会。")
    else:
        total_transfer = int(transfers["quantity"].sum())
        total_saving = float(transfers["estimated_purchase_saving"].sum())
        lines.append(
            f"- 建议执行 {len(transfers)} 笔门店调拨，共 {total_transfer} 件，"
            f"预计减少采购支出 ¥{total_saving:,.2f}。"
        )
    lines.extend(
        [
            "",
            "## 风险提示",
            "",
            (
                "预测结果存在不确定性。供应延误、天气突变、社区活动和竞争对手促销等"
                "异常事件需要店主确认；所有采购单在发送前必须人工审批。"
            ),
        ]
    )
    return "\n".join(lines)
