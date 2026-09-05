from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
    sales = data.sales.copy()
    products = data.products.copy()
    inventory = data.inventory.copy()
    _require_columns(sales, SALES_REQUIRED, "销售数据")
    _require_columns(products, PRODUCT_REQUIRED, "商品数据")
    _require_columns(inventory, INVENTORY_REQUIRED, "库存数据")

    sales["date"] = pd.to_datetime(sales["date"], errors="raise").dt.normalize()
    for frame in (sales, products, inventory):
        frame["sku"] = frame["sku"].astype(str)
    for frame in (sales, inventory):
        frame["store_id"] = frame["store_id"].astype(str)

    sales["units"] = pd.to_numeric(sales["units"], errors="raise")
    inventory["on_hand"] = pd.to_numeric(inventory["on_hand"], errors="raise")
    products["cost"] = pd.to_numeric(products["cost"], errors="raise")
    products["price"] = pd.to_numeric(products["price"], errors="raise")
    if (sales["units"] < 0).any() or (inventory["on_hand"] < 0).any():
        raise ValueError("销量和库存不能为负数")
    if (products["cost"] < 0).any() or (products["price"] <= 0).any():
        raise ValueError("商品成本不能为负数，售价必须大于0")

    sales["promotion"] = sales.get("promotion", 0).fillna(0).astype(int)
    sales["stockout"] = sales.get("stockout", 0).fillna(0).astype(int)
    inventory["on_order"] = inventory.get("on_order", 0).fillna(0).astype(float)

    defaults = {
        "shelf_life_days": 365,
        "lead_time_days": 2,
        "case_pack": 1,
        "min_order_qty": 1,
    }
    for column, default in defaults.items():
        products[column] = pd.to_numeric(
            products.get(column, pd.Series(default, index=products.index)).fillna(default),
            errors="raise",
        )

    if "price" not in sales:
        sales = sales.merge(products[["sku", "price"]], on="sku", how="left")
    else:
        price_map = products.set_index("sku")["price"]
        sales["price"] = sales["price"].fillna(sales["sku"].map(price_map))

    product_skus = set(products["sku"])
    unknown_sales = sorted(set(sales["sku"]) - product_skus)
    unknown_inventory = sorted(set(inventory["sku"]) - product_skus)
    if unknown_sales or unknown_inventory:
        unknown = sorted(set(unknown_sales + unknown_inventory))
        raise ValueError(f"以下 SKU 未出现在商品表：{', '.join(unknown)}")

    return RetailData(
        sales=sales.sort_values(["store_id", "sku", "date"]).reset_index(drop=True),
        products=products.drop_duplicates("sku").reset_index(drop=True),
        inventory=inventory.drop_duplicates(["store_id", "sku"]).reset_index(drop=True),
    )


def generate_demo_data(days: int = 140, seed: int = 42) -> RetailData:
    """Create reproducible two-store data with trends, seasonality and stockouts."""

    rng = np.random.default_rng(seed)
    product_rows = [
        ("SKU001", "矿泉水 550ml", "饮料", "SUP-A", 1.0, 2.2, 540, 2, 24, 24, 18, 0.30),
        ("SKU002", "鲜牛奶 1L", "乳制品", "SUP-B", 7.0, 11.5, 10, 1, 6, 6, 9, 0.05),
        ("SKU003", "方便面", "食品", "SUP-C", 2.6, 5.0, 240, 3, 12, 12, 12, 0.08),
        ("SKU004", "鸡蛋 10枚", "生鲜", "SUP-B", 7.8, 12.8, 20, 1, 6, 6, 11, 0.04),
        ("SKU005", "薯片", "零食", "SUP-C", 3.2, 7.0, 180, 3, 12, 12, 7, -0.02),
        ("SKU006", "纸巾 3包装", "日用品", "SUP-D", 8.0, 14.9, 730, 4, 8, 8, 5, 0.02),
        ("SKU007", "冰淇淋", "冷冻食品", "SUP-E", 3.8, 8.0, 120, 2, 20, 20, 8, 0.18),
        ("SKU008", "苹果 1kg", "生鲜", "SUP-F", 6.5, 12.0, 14, 1, 5, 5, 10, -0.03),
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
    start = datetime.now(UTC).date() - timedelta(days=days)
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
    products = products.drop(columns=["base_demand", "trend"])
    return validate_data(
        RetailData(
            sales=pd.DataFrame(sales_rows),
            products=products,
            inventory=pd.DataFrame(inventory_rows),
        )
    )
