from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict

import pandas as pd

from .config import StrategyConfig
from .data import RetailData
from .optimization import InventoryOptimizer
from .scenarios import apply_scenario


def dataset_key(data: RetailData, source: str) -> str:
    digest = hashlib.sha256(source.encode())
    for frame in (data.sales, data.products, data.inventory):
        digest.update(frame.to_json(orient="split", date_format="iso").encode())
    return digest.hexdigest()[:24]


def decision_scope(data_key: str, strategy: StrategyConfig) -> str:
    payload = json.dumps(asdict(strategy), sort_keys=True)
    return hashlib.sha256(f"{data_key}:{payload}".encode()).hexdigest()[:24]


def scenario_plan(base_forecast, data, strategy, scenario):
    adjusted = apply_scenario(base_forecast, scenario)
    recommendations = InventoryOptimizer().recommend(
        adjusted, data.products, data.inventory, strategy, scenario
    )
    return adjusted, recommendations


def prepare_review(row: dict, action: str, quantity: float, reason: str, note: str) -> dict:
    original = int(row["suggested_order_qty"])
    if action not in {"接受", "修改", "拒绝"}:
        raise ValueError("请选择接受、修改或拒绝")
    quantity = original if action == "接受" else 0 if action == "拒绝" else quantity
    if not math.isfinite(quantity) or quantity < 0 or quantity % 1:
        raise ValueError("补货数量必须为非负整数")
    pack, minimum = int(row["case_pack"]), int(row["min_order_qty"])
    if quantity and (quantity % pack or quantity < minimum):
        raise ValueError(f"请按 {pack} 件整箱订购，起订量为 {minimum} 件；不订购请填 0")
    if action in {"修改", "拒绝"} and not reason.strip():
        raise ValueError("请填写调整原因")
    if not math.isfinite(float(row["unit_cost"])) or row["unit_cost"] < 0:
        raise ValueError("单价不能为负数或空值")
    return {
        "store_id": str(row["store_id"]),
        "sku": str(row["sku"]),
        "product_name": str(row["product_name"]),
        "supplier_id": str(row["supplier_id"]),
        "original_qty": original,
        "final_qty": int(quantity),
        "unit_cost": float(row["unit_cost"]),
        "line_total": round(quantity * row["unit_cost"], 2),
        "case_pack": pack,
        "min_order_qty": minimum,
        "action": action,
        "reason": reason.strip() or "按建议采购",
        "note": note.strip(),
    }


def scoped(frame: pd.DataFrame, store_id: str) -> pd.DataFrame:
    return frame.copy() if store_id == "全部门店" else frame[frame.store_id == store_id].copy()


def download_csv(frame: pd.DataFrame) -> bytes:
    result = frame.copy()
    # Spreadsheet programs interpret leading operators as formulas, even in quoted CSV cells.
    for col in result.select_dtypes(include=["object", "string"]).columns:
        result[col] = result[col].map(
            lambda v: (
                "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
            )
        )
    return result.to_csv(index=False).encode("utf-8-sig")
