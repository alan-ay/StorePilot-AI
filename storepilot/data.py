from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

SALES_REQUIRED = {"date", "store_id", "sku", "units"}
PRODUCT_REQUIRED = {
    "sku",
    "product_name",
    "category",
    "supplier_id",
    "cost",
    "price",
}
INVENTORY_REQUIRED = {"store_id", "sku", "on_hand"}
DEMO_CATALOG_VERSION = "household-v1"


@dataclass
class RetailData:
    sales: pd.DataFrame
    products: pd.DataFrame
    inventory: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} 缺少字段：{', '.join(missing)}")


def validate_data(data: RetailData) -> RetailData:
    sales, products, inventory = (data.sales.copy(), data.products.copy(), data.inventory.copy())
    for frame, required, label in (
        (sales, SALES_REQUIRED, "销售数据"),
        (products, PRODUCT_REQUIRED, "商品数据"),
        (inventory, INVENTORY_REQUIRED, "库存数据"),
    ):
        _require_columns(frame, required, label)
        if frame.empty:
            raise ValueError(f"{label}没有数据，请检查文件内容")
        for key in required - {"date", "units", "cost", "price", "on_hand"}:
            if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
                raise ValueError(f"{label}的 {key} 不能为空")
            frame[key] = frame[key].astype(str).str.strip()

    sales["date"] = pd.to_datetime(sales["date"], errors="raise").dt.normalize()
    if sales["date"].isna().any() or sales["date"].dt.tz is not None:
        raise ValueError("日期不能为空，请使用门店当地的 YYYY-MM-DD 日期")
    for frame, keys, label in (
        (sales, ["store_id", "sku", "date"], "每日销量"),
        (products, ["sku"], "商品"),
        (inventory, ["store_id", "sku"], "库存"),
    ):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label}存在重复记录，请按 {', '.join(keys)} 合并后导入")

    def numeric(frame, column, default=None, minimum=0, integer=False):
        if column not in frame:
            frame[column] = default
        if default is not None:
            frame[column] = frame[column].fillna(default)
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} 必须为完整的有限数值")
        if (frame[column] < minimum).any():
            raise ValueError(f"{column} 不能为负数或小于 {minimum}")
        if integer and (frame[column] % 1 != 0).any():
            raise ValueError(f"{column} 必须为整数")

    for frame, column in ((sales, "units"), (inventory, "on_hand"), (products, "cost")):
        numeric(frame, column)
    numeric(products, "price", minimum=0.01)
    numeric(inventory, "on_order", default=0)
    for column in ("promotion", "stockout"):
        numeric(sales, column, default=0, integer=True)
        if not sales[column].isin([0, 1]).all():
            raise ValueError(f"{column} 只能为 0 或 1")
    for column, default in {
        "shelf_life_days": 365,
        "lead_time_days": 2,
        "case_pack": 1,
        "min_order_qty": 1,
    }.items():
        numeric(products, column, default=default, minimum=1, integer=True)

    unknown = (set(sales["sku"]) | set(inventory["sku"])) - set(products["sku"])
    if unknown:
        raise ValueError(f"以下 SKU 未出现在商品表：{', '.join(sorted(unknown))}")
    if "price" not in sales:
        sales["price"] = sales["sku"].map(products.set_index("sku")["price"])
    else:
        sales["price"] = sales["price"].fillna(sales["sku"].map(products.set_index("sku")["price"]))
    numeric(sales, "price", minimum=0.01)
    ends = sales.groupby(["store_id", "sku"])["date"].max()
    if ends.nunique() > 1:
        raise ValueError("各门店商品的销售记录需更新至同一截止日期，零销量日也请保留记录")
    return RetailData(
        sales.sort_values(["store_id", "sku", "date"]).reset_index(drop=True),
        products.sort_values("sku").reset_index(drop=True),
        inventory.sort_values(["store_id", "sku"]).reset_index(drop=True),
    )


def generate_demo_data(days: int = 140, seed: int = 42, end_date: date | None = None) -> RetailData:
    """Create reproducible non-food household retail data for two stores."""

    rng = np.random.default_rng(seed)
    product_rows = [
        ("SKU001", "洗碗海绵 2片装", "家居清洁", "SUP-A", 1.0, 2.2, 1825, 2, 24, 24, 18, 0.30),
        ("SKU002", "洗洁精 500ml", "家居清洁", "SUP-B", 7.0, 11.5, 730, 1, 6, 6, 9, 0.05),
        ("SKU003", "洗衣皂 200g", "衣物清洁", "SUP-C", 2.6, 5.0, 1095, 3, 12, 12, 12, 0.08),
        ("SKU004", "洗手液 500ml", "个人护理", "SUP-B", 7.8, 12.8, 730, 1, 6, 6, 11, 0.04),
        ("SKU005", "牙刷", "个人护理", "SUP-C", 3.2, 7.0, 1825, 3, 12, 12, 7, -0.02),
        ("SKU006", "纸巾 3包装", "纸品", "SUP-D", 8.0, 14.9, 730, 4, 8, 8, 5, 0.02),
        ("SKU007", "垃圾袋 20只装", "家居清洁", "SUP-E", 3.8, 8.0, 1825, 2, 20, 20, 8, 0.18),
        ("SKU008", "厨房纸 2卷装", "纸品", "SUP-F", 6.5, 12.0, 1095, 1, 5, 5, 10, -0.03),
    ]
    columns = [
        "sku",
        "product_name",
        "category",
        "supplier_id",
        "cost",
        "price",
        "shelf_life_days",
        "lead_time_days",
        "case_pack",
        "min_order_qty",
        "base_demand",
        "trend",
    ]
    products = pd.DataFrame(product_rows, columns=columns)
    start = (end_date or datetime.now(UTC).date()) - timedelta(days=days)
    sales_rows: list[dict] = []
    stores = ["STORE-01", "STORE-02"]

    for store_index, store_id in enumerate(stores):
        store_factor = 1.0 if store_index == 0 else 0.78
        for product in products.itertuples(index=False):
            for day_index in range(days):
                current = start + timedelta(days=day_index)
                weekend = 1.18 if current.weekday() >= 5 else 1.0
                summer = 1 + 0.12 * np.sin(2 * np.pi * day_index / 45)
                promo = int(rng.random() < 0.07)
                promo_factor = 1.35 if promo else 1.0
                trend_factor = max(0.55, 1 + product.trend * day_index / days)
                demand = product.base_demand * store_factor * weekend * summer
                demand *= promo_factor * trend_factor
                units = max(0, round(rng.normal(demand, max(1.2, np.sqrt(demand)))))
                stockout = int(rng.random() < 0.025)
                if stockout:
                    units = int(units * rng.uniform(0.1, 0.5))
                sales_rows.append(
                    {
                        "date": current.isoformat(),
                        "store_id": store_id,
                        "sku": product.sku,
                        "units": units,
                        "price": product.price * (0.9 if promo else 1.0),
                        "promotion": promo,
                        "stockout": stockout,
                    }
                )

    inventory_rows: list[dict] = []
    for store_id in stores:
        for product in products.itertuples(index=False):
            stock_factor = rng.uniform(0.3, 2.2)
            inventory_rows.append(
                {
                    "store_id": store_id,
                    "sku": product.sku,
                    "on_hand": int(product.base_demand * product.lead_time_days * stock_factor),
                    "on_order": 0,
                }
            )
    for stock in inventory_rows:
        if stock["store_id"] == "STORE-02" and stock["sku"] == "SKU001":
            stock["on_hand"] = 300
        if stock["store_id"] == "STORE-01" and stock["sku"] == "SKU002":
            stock["on_hand"] = 160
    products = products.drop(columns=["base_demand", "trend"])
    return validate_data(
        RetailData(
            sales=pd.DataFrame(sales_rows),
            products=products,
            inventory=pd.DataFrame(inventory_rows),
        )
    )
