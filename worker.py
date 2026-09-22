from __future__ import annotations

import os
import sys

from supabase import create_client

from backtest import run_backtest_batch
from core import MODEL_FEATURES, evaluate_and_train_history
from storage import (
    get_backtest_job,
    insert_model_run,
    list_backtest_jobs,
    load_settled_snapshots,
    snapshots_to_history,
)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def main() -> int:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        print("SUPABASE_URL / SUPABASE_KEY が未設定です。", file=sys.stderr)
        return 2

    max_races = max(1, env_int("V5_MAX_RACES_PER_RUN", 100))
    batch_size = min(25, max(1, env_int("V5_BATCH_SIZE", 10)))
    request_delay = max(0.3, env_float("V5_REQUEST_DELAY", 0.8))
    client = create_client(url, key)

    jobs = [j for j in list_backtest_jobs(client, limit=50) if str(j.get("status")) in {"ready", "running", "queued"}]
    if not jobs:
        print("処理対象の一括学習ジョブはありません。")
        return 0

    processed_this_run = 0
    for job in reversed(jobs):  # oldest first
        job_id = str(job["id"])
        print(f"job {job_id[:8]} start: {job.get('processed', 0)}/{job.get('total_targets', 0)}")
        while processed_this_run < max_races:
            fresh = get_backtest_job(client, job_id)
            if not fresh or str(fresh.get("status")) == "completed":
                break
            step = min(batch_size, max_races - processed_this_run)
            out = run_backtest_batch(client, fresh, limit=step, request_delay=request_delay)
            n = int(out.get("done", 0)) + int(out.get("failed", 0))
            processed_this_run += n
            print(
                f"  batch: success={out.get('done', 0)} failed={out.get('failed', 0)} "
                f"pending={out.get('counts', {}).get('pending', '?')}"
            )
            for msg in list(out.get("messages", []))[:10]:
                print("   WARN", msg)
            if out.get("completed") or n == 0:
                break
        if processed_this_run >= max_races:
            break

    # Record the latest validation result for traceability. The model object itself is
    # rebuilt from settled history by the Streamlit app, so no unsafe pickle is stored.
    try:
        rows = load_settled_snapshots(client, limit=5000)
        hist = snapshots_to_history(rows, MODEL_FEATURES)
        lb = evaluate_and_train_history(hist)
        insert_model_run(client, lb.metrics, lb.accepted, f"GitHub Actions worker: {lb.note}")
        print(f"model evaluation: accepted={lb.accepted} {lb.note}")
    except Exception as exc:
        print(f"model evaluation warning: {exc}")

    print(f"run complete: {processed_this_run} races attempted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
