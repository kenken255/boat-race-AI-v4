from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from supabase import create_client
except Exception:
    create_client = None


def _json_safe(v: Any):
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, tuple):
        return [_json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if pd.isna(v) if not isinstance(v, (str, bytes, dict, list, tuple)) else False:
        return None
    return v


def dataframe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [_json_safe(r) for r in df.to_dict(orient="records")]


def client_from_secrets(secrets):
    if create_client is None:
        return None, "supabase パッケージを読み込めません"
    try:
        sec = secrets["supabase"]
        url = str(sec["url"])
        key = str(sec["key"])
        if not url or not key:
            return None, "Supabase URL/Key が空です"
        return create_client(url, key), "接続設定あり"
    except Exception:
        return None, "Supabase未設定"


def insert_snapshot(client, snapshot: Dict[str, Any], tickets: List[Dict[str, Any]]) -> str:
    payload = _json_safe(snapshot)
    res = client.table("race_snapshots").insert(payload).execute()
    data = getattr(res, "data", None) or []
    if not data:
        raise RuntimeError("race_snapshots への保存結果を取得できませんでした")
    snapshot_id = str(data[0]["id"])
    if tickets:
        rows = []
        for t in tickets:
            x = dict(t)
            x["snapshot_id"] = snapshot_id
            rows.append(_json_safe(x))
        client.table("ticket_predictions").insert(rows).execute()
    return snapshot_id


def load_unsettled(client, limit: int = 30) -> List[Dict[str, Any]]:
    res = client.table("race_snapshots").select("*").eq("settled", False).order("race_date").limit(limit).execute()
    return getattr(res, "data", None) or []


def load_tickets(client, snapshot_id: str) -> List[Dict[str, Any]]:
    res = client.table("ticket_predictions").select("id,bet_type,combo").eq("snapshot_id", snapshot_id).execute()
    return getattr(res, "data", None) or []


def settle_snapshot(client, snapshot_id: str, result_json: Dict[str, Any], ranks: Dict[int, int], payoff_by_type: Dict[str, Dict[str, int]]):
    client.table("race_snapshots").update({"settled": True, "result_json": _json_safe(result_json)}).eq("id", snapshot_id).execute()
    tickets = load_tickets(client, snapshot_id)
    for row in tickets:
        bt = str(row.get("bet_type"))
        combo = str(row.get("combo"))
        payoff = int(payoff_by_type.get(bt, {}).get(combo, 0))
        client.table("ticket_predictions").update({"result_hit": payoff > 0, "payoff": payoff}).eq("id", row["id"]).execute()


def load_settled_snapshots(client, limit: int = 1000) -> List[Dict[str, Any]]:
    res = client.table("race_snapshots").select("id,race_id,race_date,snapshot_at,stadium_code,stadium_name,race_no,weather,boats,result_json").eq("settled", True).order("snapshot_at", desc=True).limit(limit).execute()
    return getattr(res, "data", None) or []


def snapshots_to_history(rows: List[Dict[str, Any]], feature_cols: List[str]) -> pd.DataFrame:
    # Keep latest snapshot for each race to avoid duplicated outcomes in training.
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("race_id", ""))
        if rid and rid not in latest:
            latest[rid] = row
    out = []
    for rid, row in latest.items():
        result = row.get("result_json") or {}
        rank_map = {}
        for item in result.get("result", []) or []:
            try:
                rank_map[int(item.get("boat"))] = int(item.get("rank"))
            except Exception:
                pass
        for b in row.get("boats") or []:
            try:
                boat_no = int(b.get("boat_no"))
            except Exception:
                continue
            rec = {"race_id": rid, "race_date": row.get("race_date"), "boat_no": boat_no, "result_rank": rank_map.get(boat_no)}
            for c in feature_cols:
                rec[c] = b.get(c)
            out.append(rec)
    return pd.DataFrame(out)


def insert_model_run(client, metrics: Dict[str, Any], accepted: bool, note: str):
    payload = {"accepted": bool(accepted), "note": note, "metrics": _json_safe(metrics)}
    client.table("model_runs").insert(payload).execute()

# ---- V5 backtest / backfill -------------------------------------------------

def create_backtest_job(client, config: Dict[str, Any]) -> str:
    payload = _json_safe(dict(config))
    payload.setdefault("status", "queued")
    payload.setdefault("processed", 0)
    payload.setdefault("succeeded", 0)
    payload.setdefault("failed", 0)
    res = client.table("backtest_jobs").insert(payload).execute()
    data = getattr(res, "data", None) or []
    if not data:
        raise RuntimeError("backtest_jobs 作成失敗")
    return str(data[0]["id"])


def insert_backtest_targets(client, job_id: str, targets: List[Dict[str, Any]]):
    if not targets:
        return
    rows = []
    for t in targets:
        x = dict(t)
        x["job_id"] = job_id
        x["status"] = "pending"
        rows.append(_json_safe(x))
    # Supabase/PostgREST payloads can get large; chunk them.
    for i in range(0, len(rows), 300):
        client.table("backtest_queue").insert(rows[i:i+300]).execute()
    client.table("backtest_jobs").update({"total_targets": len(rows), "status": "ready"}).eq("id", job_id).execute()


def list_backtest_jobs(client, limit: int = 20) -> List[Dict[str, Any]]:
    res = client.table("backtest_jobs").select("*").order("created_at", desc=True).limit(limit).execute()
    return getattr(res, "data", None) or []


def get_backtest_job(client, job_id: str) -> Dict[str, Any] | None:
    res = client.table("backtest_jobs").select("*").eq("id", job_id).limit(1).execute()
    rows = getattr(res, "data", None) or []
    return rows[0] if rows else None


def get_pending_targets(client, job_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    res = (client.table("backtest_queue").select("*").eq("job_id", job_id).eq("status", "pending")
           .order("race_date").order("stadium_code").order("race_no").limit(limit).execute())
    return getattr(res, "data", None) or []


def update_target_status(client, queue_id: int, status: str, snapshot_id: str | None = None, error: str | None = None):
    payload = {"status": status, "snapshot_id": snapshot_id, "error": error}
    client.table("backtest_queue").update(_json_safe(payload)).eq("id", queue_id).execute()


def refresh_backtest_job_counts(client, job_id: str, status: str | None = None, message: str | None = None) -> Dict[str, int]:
    counts = {}
    for s in ("pending", "done", "error", "skipped"):
        res = client.table("backtest_queue").select("id", count="exact").eq("job_id", job_id).eq("status", s).limit(1).execute()
        counts[s] = int(getattr(res, "count", 0) or 0)
    processed = counts["done"] + counts["error"] + counts["skipped"]
    payload = {"processed": processed, "succeeded": counts["done"], "failed": counts["error"] + counts["skipped"]}
    if status is not None:
        payload["status"] = status
    if message is not None:
        payload["message"] = message
    client.table("backtest_jobs").update(payload).eq("id", job_id).execute()
    return counts


def load_settled_before(client, before_date: str, limit: int = 5000) -> List[Dict[str, Any]]:
    res = (client.table("race_snapshots")
           .select("id,race_id,race_date,snapshot_at,stadium_code,stadium_name,race_no,weather,boats,result_json")
           .eq("settled", True).lt("race_date", before_date).order("race_date", desc=True).limit(limit).execute())
    return getattr(res, "data", None) or []


def insert_backtest_snapshot(client, snapshot: Dict[str, Any], tickets: List[Dict[str, Any]]) -> str:
    payload = _json_safe(snapshot)
    # Idempotent for the same job/race: return existing snapshot instead of duplicating.
    job_id = str(payload.get("backtest_job_id", ""))
    race_id = str(payload.get("race_id", ""))
    if job_id and race_id:
        q = client.table("race_snapshots").select("id").eq("backtest_job_id", job_id).eq("race_id", race_id).limit(1).execute()
        existing = getattr(q, "data", None) or []
        if existing:
            return str(existing[0]["id"])
    res = client.table("race_snapshots").insert(payload).execute()
    data = getattr(res, "data", None) or []
    if not data:
        raise RuntimeError("backtest snapshot 保存失敗")
    sid = str(data[0]["id"])
    if tickets:
        rows = []
        for t in tickets:
            x = dict(t); x["snapshot_id"] = sid; rows.append(_json_safe(x))
        for i in range(0, len(rows), 400):
            client.table("ticket_predictions").insert(rows[i:i+400]).execute()
    return sid


def reset_backtest_errors(client, job_id: str):
    client.table("backtest_queue").update({"status": "pending", "error": None}).eq("job_id", job_id).in_("status", ["error", "skipped"]).execute()
    refresh_backtest_job_counts(client, job_id, status="ready", message="失敗分を再試行待ちへ戻しました")
