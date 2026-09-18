from datetime import date

import pandas as pd

from storepilot.data import generate_demo_data, validate_data


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


def test_uploaded_product_names_and_categories_are_not_restricted():
    data = generate_demo_data(days=3, end_date=date(2026, 9, 18))
    data.products.loc[0, ["product_name", "category"]] = ["Customer's rice", "Food"]
    original = data.products.copy(deep=True)
    checked = validate_data(data)
    pd.testing.assert_frame_equal(checked.products, original)
