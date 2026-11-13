"""SQLite persistence layer for GreenGen datasets, models, optimizations, and action plans."""
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
DB_PATH = os.environ.get("GREENGEN_DB_PATH", os.path.join(tempfile.gettempdir(), "greengen.db"))


def get_connection() -> sqlite3.Connection:
    db_target = os.environ.get("GREENGEN_DB_PATH", DB_PATH)
    try:
        conn = sqlite3.connect(db_target)
        conn.execute("CREATE TABLE IF NOT EXISTS _healthcheck (id INT);")
    except sqlite3.OperationalError:
        conn = sqlite3.connect(os.path.join(tempfile.gettempdir(), "greengen_fallback.db"))
    conn.row_factory = sqlite3.Row
    return conn
def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            cleaned_file_path TEXT,
            row_count INTEGER NOT NULL,
            column_count INTEGER NOT NULL,
            is_synthetic INTEGER DEFAULT 0,
            profile_json TEXT,
            mapping_json TEXT,
            preprocessing_log_json TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trained_models (
            run_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            target_column TEXT NOT NULL,
            target_unit TEXT,
            feature_columns_json TEXT NOT NULL,
            split_strategy TEXT NOT NULL,
            test_size REAL NOT NULL,
            models_summary_json TEXT NOT NULL,
            best_model_name TEXT,
            selection_criterion TEXT,
            artifact_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS optimization_runs (
            opt_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            target_reduction_pct REAL NOT NULL,
            baseline_emissions REAL NOT NULL,
            optimized_emissions REAL NOT NULL,
            absolute_reduction REAL NOT NULL,
            percentage_reduction REAL NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS action_plans (
            plan_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            items_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
    """)
    conn.commit()
    conn.close()


def save_dataset_record(
    dataset_id: str,
    filename: str,
    file_path: str,
    row_count: int,
    column_count: int,
    profile: Dict[str, Any],
    mapping: Dict[str, Any],
    is_synthetic: bool = False,
) -> None:
    init_db()
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO datasets
        (dataset_id, filename, file_path, cleaned_file_path, row_count, column_count,
         is_synthetic, profile_json, mapping_json, preprocessing_log_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            filename,
            file_path,
            file_path,
            row_count,
            column_count,
            1 if is_synthetic else 0,
            json.dumps(profile),
            json.dumps(mapping),
            json.dumps([]),
            datetime.utcnow().isoformat() + "Z",
        ),
    )
    conn.commit()
    conn.close()


def update_dataset_cleaned(
    dataset_id: str,
    cleaned_file_path: str,
    row_count: int,
    column_count: int,
    profile: Dict[str, Any],
    mapping: Dict[str, Any],
    preprocessing_log: List[str],
) -> None:
    conn = get_connection()
    conn.execute(
        """
        UPDATE datasets
        SET cleaned_file_path = ?, row_count = ?, column_count = ?,
            profile_json = ?, mapping_json = ?, preprocessing_log_json = ?
        WHERE dataset_id = ?
        """,
        (
            cleaned_file_path,
            row_count,
            column_count,
            json.dumps(profile),
            json.dumps(mapping),
            json.dumps(preprocessing_log),
            dataset_id,
        ),
    )
    conn.commit()
    conn.close()


def get_dataset_record(dataset_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["is_synthetic"] = bool(d["is_synthetic"])
    d["profile"] = json.loads(d["profile_json"]) if d["profile_json"] else {}
    d["mapping"] = json.loads(d["mapping_json"]) if d["mapping_json"] else {}
    d["preprocessing_log"] = json.loads(d["preprocessing_log_json"]) if d["preprocessing_log_json"] else []
    return d


def list_dataset_records() -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["is_synthetic"] = bool(d["is_synthetic"])
        d["profile"] = json.loads(d["profile_json"]) if d["profile_json"] else {}
        d["mapping"] = json.loads(d["mapping_json"]) if d["mapping_json"] else {}
        result.append(d)
    return result


def save_model_run(
    run_id: str,
    dataset_id: str,
    target_column: str,
    target_unit: str,
    feature_columns: List[str],
    split_strategy: str,
    test_size: float,
    models_summary: Dict[str, Any],
    best_model_name: str,
    selection_criterion: str,
    artifact_path: str,
) -> None:
    init_db()
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO trained_models
        (run_id, dataset_id, target_column, target_unit, feature_columns_json,
         split_strategy, test_size, models_summary_json, best_model_name,
         selection_criterion, artifact_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            dataset_id,
            target_column,
            target_unit,
            json.dumps(feature_columns),
            split_strategy,
            test_size,
            json.dumps(models_summary),
            best_model_name,
            selection_criterion,
            artifact_path,
            datetime.utcnow().isoformat() + "Z",
        ),
    )
    conn.commit()
    conn.close()


def get_latest_model_run(dataset_id: str, target_column: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    if target_column:
        row = conn.execute(
            "SELECT * FROM trained_models WHERE dataset_id = ? AND target_column = ? ORDER BY created_at DESC LIMIT 1",
            (dataset_id, target_column),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM trained_models WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 1",
            (dataset_id,),
        ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["feature_columns"] = json.loads(d["feature_columns_json"])
    d["models_summary"] = json.loads(d["models_summary_json"])
    return d


def list_model_runs(dataset_id: str) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM trained_models WHERE dataset_id = ? ORDER BY created_at DESC",
        (dataset_id,),
    ).fetchall()
    conn.close()
    res = []
    for r in rows:
        d = dict(r)
        d["feature_columns"] = json.loads(d["feature_columns_json"])
        d["models_summary"] = json.loads(d["models_summary_json"])
        res.append(d)
    return res


def save_optimization_run(
    opt_id: str,
    dataset_id: str,
    run_id: str,
    model_name: str,
    target_reduction_pct: float,
    baseline_emissions: float,
    optimized_emissions: float,
    absolute_reduction: float,
    percentage_reduction: float,
    status: str,
    result_data: Dict[str, Any],
) -> None:
    init_db()
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO optimization_runs
        (opt_id, dataset_id, run_id, model_name, target_reduction_pct,
         baseline_emissions, optimized_emissions, absolute_reduction,
         percentage_reduction, status, result_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            opt_id,
            dataset_id,
            run_id,
            model_name,
            target_reduction_pct,
            baseline_emissions,
            optimized_emissions,
            absolute_reduction,
            percentage_reduction,
            status,
            json.dumps(result_data),
            datetime.utcnow().isoformat() + "Z",
        ),
    )
    conn.commit()
    conn.close()


def get_latest_optimization_run(dataset_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM optimization_runs WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 1",
        (dataset_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["result"] = json.loads(d["result_json"])
    return d


def save_action_plan(dataset_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    init_db()
    plan_id = f"plan_{dataset_id}"
    now = datetime.utcnow().isoformat() + "Z"
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO action_plans (plan_id, dataset_id, items_json, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (plan_id, dataset_id, json.dumps(items), now),
    )
    conn.commit()
    conn.close()
    return {"plan_id": plan_id, "dataset_id": dataset_id, "items": items, "updated_at": now}


def get_action_plan(dataset_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    row = conn.execute("SELECT * FROM action_plans WHERE dataset_id = ?", (dataset_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["items"] = json.loads(d["items_json"])
    return d
