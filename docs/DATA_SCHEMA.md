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

## Validation and assumptions

Sales must be unique by `(store_id, sku, date)`, products by `sku`, and inventory
by `(store_id, sku)`. Duplicate rows are rejected rather than dropped. Required
identifiers must be nonblank and are imported as text, preserving leading zeros.

Numeric fields must be finite. Units, stock, on-order stock and cost cannot be
negative. Selling price must be positive. Product shelf life, lead time, case
packs and minimum quantities must be positive integers. Binary flags allow only
0 or 1. Optional columns may be omitted entirely; defaults then apply.

Each sales series must end on the same local trading date. The app rejects
misaligned end dates instead of treating an old series as up to date. Dates must
be timezone-free local dates. Missing days *inside* a series are still zero-filled:
include explicit records for zero-sales days, and resolve missing uploads first.

No batch-level expiry date or incoming delivery date is available in this schema.
Expiry-related output is an inventory-pressure estimate. On-order stock reduces
net purchase need but does not count as physical cover or transfer stock.

An inventory item without matching store/SKU sales history is listed in the data
settings as unforecastable. If no inventory matches sales, the import is rejected.
An upload is limited to 25 MB across all three files and is activated only after
validation and forecast fitting succeed.

## Sample catalog

The built-in demo and downloadable CSV examples use only non-food household
essentials: dishwashing sponges and liquid, laundry soap, hand wash, toothbrushes,
tissues, bin bags, and kitchen towels. Sales and inventory reference the same
eight SKUs across two stores. Shelf lives are illustrative product storage periods,
not batch expiry dates.

This sample catalog does not restrict imports. Your uploaded product names and
categories are retained, including when the interface language changes. CSV
column names such as `product_name` and `category` stay the same in both languages.
