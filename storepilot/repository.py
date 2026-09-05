from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


class FeedbackRepository:
    def __init__(self, database_path: str | Path = "storepilot.db") -> None:
        self.database_path = str(database_path)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS decision_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    store_id TEXT NOT NULL,
                    sku TEXT NOT NULL,
                    action TEXT NOT NULL,
                    original_qty REAL NOT NULL,
                    final_qty REAL NOT NULL,
                    reason TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_until TEXT NOT NULL,
                    mae REAL NOT NULL,
                    wape REAL NOT NULL,
                    interval_coverage REAL NOT NULL,
                    validation_rows INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def add_feedback(
        self,
        decision_id: str,
        store_id: str,
        sku: str,
        action: str,
        original_qty: float,
        final_qty: float,
        reason: str,
        note: str = "",
    ) -> None:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO decision_feedback
                    (decision_id, store_id, sku, action, original_qty, final_qty,
                     reason, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    store_id,
                    sku,
                    action,
                    float(original_qty),
                    float(final_qty),
                    reason,
                    note,
                    created_at,
                ),
            )

    def list_feedback(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                "SELECT * FROM decision_feedback ORDER BY created_at DESC", connection
            )

    def record_model_run(self, trained_until: str, metrics: object) -> None:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_runs
                    (trained_until, mae, wape, interval_coverage, validation_rows, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trained_until,
                    float(metrics.mae),
                    float(metrics.wape),
                    float(metrics.interval_coverage),
                    int(metrics.validation_rows),
                    created_at,
                ),
            )

    def list_model_runs(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                "SELECT * FROM model_runs ORDER BY created_at DESC", connection
            )
