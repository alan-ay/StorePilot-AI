# Changelog

All notable changes to StorePilot are recorded in this file.

## Unreleased

- Refresh selected product names, filters, navigation, and decision controls when
  switching languages. Keep draft purchasing preferences until explicitly saved.
- Refresh the selected adjustment reason when switching languages, preserving
  unsaved replenishment quantities and notes until the decision is saved.
- Combine the two demo stores into `STORE-01`, preserving sales revenue and stock
  totals. Use the same merged data in CSV examples and hide transfer navigation
  when only one store is present.
- Replace food and drink demo products with non-food household essentials in the
  dashboard and downloadable import examples; keep uploaded catalogs unchanged.
- Add a Chinese/English sidebar selector for the interface, reports, and exports.
  Switching language preserves store selections, purchasing preferences, and reviews.

## 0.2.0 — 2026-09-05

- Replaced the tab-heavy dashboard with a store-scoped daily workbench.
- Added product search, clear stock explanations and inline review actions.
- Connected owner decisions to persistent supplier drafts and immutable approvals.
- Prevented duplicate approvals within a batch and rechecked budgets after manual edits.
- Kept scenario assumptions separate from active purchasing decisions.
- Isolated sample orders from store records and retained historical purchase batches.
- Fixed omitted optional CSV columns, invalid quantities, minimum/case rounding and in-transit transfers.
- Added workflow tests and updated the operating guide and calculation boundaries.

## 0.1.0 — 2026-09-04

- Added daily demand forecasts with uncertainty intervals.
- Added conservative, balanced and growth inventory strategies.
- Added scenario modelling for price, promotion, traffic, holidays and delivery delays.
- Added replenishment, purchase-order and inter-store transfer recommendations.
- Added SQLite-backed owner feedback and model-run records.
- Added a Streamlit dashboard, sample data, tests and container configuration.
