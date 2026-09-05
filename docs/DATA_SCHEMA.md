# Data schema

All identifiers are treated as strings. Dates must be parseable by pandas.
Rows should represent daily SKU sales per store. Missing calendar days are
filled with zero demand by the forecasting pipeline.

## Sales

| Column | Type | Required | Meaning |
|---|---:|:---:|---|
| date | date | yes | trading date |
| store_id | string | yes | store identifier |
| sku | string | yes | product identifier |
| units | number >= 0 | yes | units sold |
| price | number > 0 | no | actual selling price |
| promotion | 0/1 | no | promotion active |
| stockout | 0/1 | no | demand may have been censored by no stock |

When `stockout=1`, StorePilot replaces obviously censored sales with a recent
rolling demand estimate during model fitting. This avoids learning that a
stockout means low customer demand.

## Products

| Column | Type | Required | Default |
|---|---:|:---:|---:|
| sku | string | yes | - |
| product_name | string | yes | - |
| category | string | yes | - |
| supplier_id | string | yes | - |
| cost | number | yes | - |
| price | number | yes | - |
| shelf_life_days | integer | no | 365 |
| lead_time_days | integer | no | 2 |
| case_pack | integer | no | 1 |
| min_order_qty | integer | no | 1 |

## Inventory

| Column | Type | Required | Default |
|---|---:|:---:|---:|
| store_id | string | yes | - |
| sku | string | yes | - |
| on_hand | number >= 0 | yes | - |
| on_order | number >= 0 | no | 0 |
