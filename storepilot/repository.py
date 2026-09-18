from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .forecasting import ForecastMetrics
from .i18n import ValidationError
from .workbench import prepare_review


class FeedbackRepository:
    def __init__(self, database_path: str | Path = "storepilot.db") -> None:
        self.database_path = str(database_path)
        self._initialise()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS decision_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT NOT NULL,
                    store_id TEXT NOT NULL, sku TEXT NOT NULL, action TEXT NOT NULL,
                    original_qty REAL NOT NULL, final_qty REAL NOT NULL,
                    reason TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, trained_until TEXT NOT NULL,
                    mae REAL NOT NULL, wape REAL NOT NULL, interval_coverage REAL NOT NULL,
                    validation_rows INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviewed_lines (
                    scope TEXT NOT NULL, store_id TEXT NOT NULL, sku TEXT NOT NULL,
                    supplier_id TEXT NOT NULL, payload TEXT NOT NULL, reviewed_at TEXT NOT NULL,
                    PRIMARY KEY(scope, store_id, sku)
                );
                CREATE TABLE IF NOT EXISTS purchase_batches (
                    order_id TEXT PRIMARY KEY, scope TEXT NOT NULL, store_id TEXT NOT NULL,
                    supplier_id TEXT NOT NULL, payload TEXT NOT NULL, approved_at TEXT NOT NULL,
                    UNIQUE(scope, store_id, supplier_id)
                );
                """
            )

    def add_feedback(
        self, decision_id, store_id, sku, action, original_qty, final_qty, reason, note=""
    ):
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO decision_feedback
                   (decision_id, store_id, sku, action, original_qty, final_qty, reason, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision_id,
                    store_id,
                    sku,
                    action,
                    original_qty,
                    final_qty,
                    reason,
                    note,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def list_feedback(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query("SELECT * FROM decision_feedback ORDER BY id DESC", connection)

    def record_model_run(self, trained_until: str, metrics: ForecastMetrics) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO model_runs
                    (trained_until, mae, wape, interval_coverage, validation_rows, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    trained_until,
                    metrics.mae,
                    metrics.wape,
                    metrics.interval_coverage,
                    metrics.validation_rows,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def list_model_runs(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query("SELECT * FROM model_runs ORDER BY id DESC", connection)

    def review(self, scope, row, action, quantity, reason="", note="") -> dict:
        payload = prepare_review(row, action, quantity, reason, note)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            locked = connection.execute(
                "SELECT order_id FROM purchase_batches WHERE scope=? AND store_id=? AND supplier_id=?",
                (scope, payload["store_id"], payload["supplier_id"]),
            ).fetchone()
            if locked:
                raise ValidationError("该供应商的采购单已审核。请在新的数据批次中处理后续采购")
            connection.execute(
                """INSERT INTO reviewed_lines VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(scope, store_id, sku) DO UPDATE SET
                   payload=excluded.payload, reviewed_at=excluded.reviewed_at""",
                (
                    scope,
                    payload["store_id"],
                    payload["sku"],
                    payload["supplier_id"],
                    json.dumps(payload, ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                """INSERT INTO decision_feedback
                   (decision_id, store_id, sku, action, original_qty, final_qty, reason, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"{scope}:{payload['store_id']}:{payload['sku']}",
                    payload["store_id"],
                    payload["sku"],
                    action,
                    payload["original_qty"],
                    payload["final_qty"],
                    payload["reason"],
                    payload["note"],
                    now,
                ),
            )
        return payload

    def reviews(self, scope) -> pd.DataFrame:
        with self._connect() as connection:
            records = connection.execute(
                "SELECT payload, reviewed_at FROM reviewed_lines WHERE scope=? ORDER BY store_id, sku",
                (scope,),
            ).fetchall()
        columns = [
            "store_id",
            "sku",
            "product_name",
            "supplier_id",
            "original_qty",
            "final_qty",
            "unit_cost",
            "line_total",
            "case_pack",
            "min_order_qty",
            "action",
            "reason",
            "note",
            "reviewed_at",
        ]
        return pd.DataFrame(
            [dict(json.loads(r["payload"]), reviewed_at=r["reviewed_at"]) for r in records],
            columns=columns,
        )

    def approve(self, scope, store_id, supplier_id, budget=None) -> str:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT order_id FROM purchase_batches WHERE scope=? AND store_id=? AND supplier_id=?",
                (scope, store_id, supplier_id),
            ).fetchone()
            if existing:
                return existing["order_id"]
            records = connection.execute(
                "SELECT payload FROM reviewed_lines WHERE scope=?", (scope,)
            ).fetchall()
            all_lines = [json.loads(r["payload"]) for r in records]
            if budget is not None and sum(r["line_total"] for r in all_lines) > budget + 0.001:
                raise ValidationError("本批次的确认金额超过采购预算，请调整清单后再审核")
            lines = [
                r
                for r in all_lines
                if r["store_id"] == store_id
                and r["supplier_id"] == supplier_id
                and r["final_qty"] > 0
            ]
            if not lines:
                raise ValidationError("该供应商没有待审核的采购明细")
            order_id = "PO-" + uuid.uuid4().hex[:12].upper()
            connection.execute(
                "INSERT INTO purchase_batches VALUES (?, ?, ?, ?, ?, ?)",
                (
                    order_id,
                    scope,
                    store_id,
                    supplier_id,
                    json.dumps(lines, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                ),
            )
            return order_id

    def orders(self, scope: str | None = None) -> pd.DataFrame:
        with self._connect() as connection:
            query = "SELECT * FROM purchase_batches"
            records = connection.execute(
                query + (" WHERE scope=?" if scope else "") + " ORDER BY approved_at DESC",
                (scope,) if scope else (),
            ).fetchall()
        rows = []
        for record in records:
            for line in json.loads(record["payload"]):
                rows.append(
                    dict(
                        line,
                        order_id=record["order_id"],
                        scope=record["scope"],
                        approved_at=record["approved_at"],
                        approval_status="已审核 · 未发送",
                    )
                )
        return pd.DataFrame(
            rows,
            columns=[
                "order_id",
                "scope",
                "store_id",
                "supplier_id",
                "sku",
                "product_name",
                "original_qty",
                "final_qty",
                "unit_cost",
                "line_total",
                "reason",
                "note",
                "approved_at",
                "approval_status",
            ],
        )
