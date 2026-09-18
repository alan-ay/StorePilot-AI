from datetime import date

import pandas as pd
import pytest

from storepilot.data import RetailData, generate_demo_data, merge_stores, validate_data


def test_demo_catalog_contains_only_non_food_household_products():
    data = generate_demo_data(days=3, end_date=date(2026, 9, 18))
    assert set(data.products.category) == {"家居清洁", "衣物清洁", "个人护理", "纸品"}
    assert set(data.products.product_name) == {
        "洗碗海绵 2片装",
        "洗洁精 500ml",
        "洗衣皂 200g",
        "洗手液 500ml",
        "牙刷",
        "纸巾 3包装",
        "垃圾袋 20只装",
        "厨房纸 2卷装",
    }
    assert data.products.shelf_life_days.ge(365).all()
    assert set(data.sales.sku) == set(data.products.sku) == set(data.inventory.sku)
    assert len(data.sales) == 3 * len(data.inventory)
    assert set(data.sales.store_id) == set(data.inventory.store_id) == {"STORE-01"}
    assert len(data.inventory) == 8
    assert not data.sales.duplicated(["date", "sku"]).any()


def test_merge_stores_preserves_daily_sales_revenue_stock_and_flags():
    products = pd.DataFrame(
        [
            {
                "sku": "A",
                "product_name": "Soap",
                "category": "Care",
                "supplier_id": "SUP",
                "cost": 5,
                "price": 20,
            },
            {
                "sku": "B",
                "product_name": "Tissues",
                "category": "Paper",
                "supplier_id": "SUP",
                "cost": 2,
                "price": 5,
            },
        ]
    )
    sales = pd.DataFrame(
        [
            ["2026-09-16", "STORE-01", "A", 2, 10, 1, 0],
            ["2026-09-16", "STORE-02", "A", 3, 20, 0, 1],
            ["2026-09-17", "STORE-01", "A", 0, 10, 0, 0],
            ["2026-09-17", "STORE-02", "A", 0, 20, 0, 0],
            ["2026-09-17", "STORE-02", "B", 4, 5, 0, 0],
        ],
        columns=["date", "store_id", "sku", "units", "price", "promotion", "stockout"],
    )
    inventory = pd.DataFrame(
        [["STORE-01", "A", 3, 1], ["STORE-02", "A", 4, 2], ["STORE-02", "B", 8, 0]],
        columns=["store_id", "sku", "on_hand", "on_order"],
    )
    original = validate_data(RetailData(sales, products, inventory))
    merged = merge_stores(original, "STORE-01")

    assert merged.sales.units.tolist() == [5, 0, 4]
    assert merged.sales.price.tolist() == [16, 15, 5]
    assert merged.sales.promotion.tolist() == [1, 0, 0]
    assert merged.sales.stockout.tolist() == [1, 0, 0]
    assert (merged.sales.units * merged.sales.price).sum() == pytest.approx(
        (original.sales.units * original.sales.price).sum()
    )
    assert merged.inventory.on_hand.tolist() == [7, 8]
    assert merged.inventory.on_order.tolist() == [3, 0]
    assert set(merged.sales.store_id) == set(merged.inventory.store_id) == {"STORE-01"}
    pd.testing.assert_frame_equal(merged.products, original.products)
    assert set(original.sales.store_id) == {"STORE-01", "STORE-02"}
    assert original.inventory.on_hand.tolist() == [3, 4, 8]


def test_uploaded_product_names_and_categories_are_not_restricted():
    data = generate_demo_data(days=3, end_date=date(2026, 9, 18))
    data.products.loc[0, ["product_name", "category"]] = ["Customer's rice", "Food"]
    original = data.products.copy(deep=True)
    checked = validate_data(data)
    pd.testing.assert_frame_equal(checked.products, original)
