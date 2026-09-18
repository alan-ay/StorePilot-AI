from __future__ import annotations

import pandas as pd

from .i18n import translate


def _product_list(frame: pd.DataFrame, language: str, limit: int = 3) -> str:
    if frame.empty:
        return translate("无", language)
    return (", " if language == "en" else "、").join(
        frame["product_name"].astype(str).drop_duplicates().head(limit).tolist()
    )


def generate_report(
    sales: pd.DataFrame,
    trends: pd.DataFrame,
    recommendations: pd.DataFrame,
    transfers: pd.DataFrame,
    strategy_name: str,
    *,
    language: str = "zh",
) -> str:
    def t(message, *values):
        return translate(message, language, *values)

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
        t("# StorePilot 每日经营简报｜{0:%Y-%m-%d}", latest_date),
        "",
        (
            t(
                "当前采用“{0}”经营策略。最近营业日共销售 **{1:.0f} 件**商品，估算营业额为 **¥{2:,.2f}**。",
                t(strategy_name),
                sales_units,
                revenue,
            )
        ),
        "",
        t("## 今日重点"),
        "",
    ]
    if high_risk.empty:
        lines.append(t("- 暂未发现需要立即处理的高缺货风险。"))
    else:
        top = high_risk.iloc[0]
        lines.append(
            t(
                "- **缺货预警：** {0}。其中 {1} 预计短缺 {2:.0f} 件，建议补货 {3:.0f} 件。",
                _product_list(high_risk, language),
                top["product_name"],
                top["expected_shortage"],
                top["suggested_order_qty"],
            )
        )
    if expiry.empty:
        lines.append(t("- 当前没有保质期窗口内的库存积压提示；批次到期日仍需人工核对。"))
    else:
        lines.append(
            t(
                "- **库存复核：** {0} 库存偏多，请核对批次到期日，再决定促销或调拨。",
                _product_list(expiry, language),
            )
        )
    if overstock.empty:
        lines.append(t("- 暂无其他库存积压提示。"))
    else:
        lines.append(
            t(
                "- **库存偏高：** {0}，建议暂停补货并观察销售速度。",
                _product_list(overstock, language),
            )
        )

    lines.extend(
        [
            "",
            t("## 销售趋势"),
            "",
            t("- 增长商品：{0}。", _product_list(growing_names, language)),
            t("- 下降商品：{0}。", _product_list(declining_names, language)),
            "",
            t("## 采购与调拨"),
            "",
            (
                t(
                    "- 今日建议生成 {0} 条采购明细，预计采购金额 **¥{1:,.2f}**。",
                    len(order_lines),
                    order_lines["purchase_cost"].sum(),
                )
            ),
        ]
    )
    if transfers.empty:
        lines.append(t("- 暂无现货余量充足的跨店调拨建议。"))
    else:
        total_transfer = int(transfers["quantity"].sum())
        total_saving = float(transfers["estimated_purchase_saving"].sum())
        lines.append(
            t(
                "- 建议执行 {0} 笔门店调拨，共 {1} 件，对应采购货值 ¥{2:,.2f}，尚未扣除运输成本，也未调减采购清单。",
                len(transfers),
                total_transfer,
                total_saving,
            )
        )
    lines.extend(
        [
            "",
            t("## 使用说明"),
            "",
            (
                t(
                    "预测结果存在不确定性。供应延误、天气突变、社区活动和竞争对手促销等异常事件需要店主确认；所有采购单在发送前必须人工审批。"
                )
            ),
        ]
    )
    return "\n".join(lines)


def generate_chinese_report(
    sales: pd.DataFrame,
    trends: pd.DataFrame,
    recommendations: pd.DataFrame,
    transfers: pd.DataFrame,
    strategy_name: str,
) -> str:
    """Keep the existing CLI and pipeline report interface."""
    return generate_report(sales, trends, recommendations, transfers, strategy_name)
