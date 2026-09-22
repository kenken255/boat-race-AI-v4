from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from core import (
    BET_SPECS, MODEL_FEATURES, STADIUMS, LearningBundle,
    blend_probability_sets, build_all_bet_probabilities, calibrate_win_probs,
    data_quality, enrich_market, evaluate_and_train_history, fetch_race_bundle,
    fetch_race_result, fetch_stadiums, heuristic_strength, model_strength, combine_model_and_manual_strength,
    normalize_race_data, parse_odds_dict, payoff_map, probability_temperature,
    reliability_factor, simulate_races, softmax_strength,
)
from storage import (
    dataframe_records, insert_backtest_snapshot, load_settled_before,
    snapshots_to_history,
)


def iter_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def discover_targets(start: date, end: date, stadium_codes: List[int], max_races: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Discover actual meetings first so we do not hammer non-running stadiums."""
    targets: List[Dict[str, Any]] = []
    warnings: List[str] = []
    selected = set(int(x) for x in stadium_codes)
    for d in iter_dates(start, end):
        try:
            active = fetch_stadiums(d)
        except Exception as e:
            warnings.append(f"{d}: 開催場取得失敗 {e}")
            continue
        active_codes = []
        for name in (active or {}).keys():
            for code, nm in STADIUMS.items():
                if nm == name and code in selected:
                    active_codes.append(code)
                    break
        for code in sorted(set(active_codes)):
            for r in range(1, 13):
                targets.append({"race_date": d.isoformat(), "stadium_code": code, "race_no": r})
                if max_races and len(targets) >= max_races:
                    return targets, warnings
    return targets, warnings


def learning_before(client, target_date: date, min_races: int = 30) -> LearningBundle:
    rows = load_settled_before(client, target_date.isoformat(), limit=5000)
    hist = snapshots_to_history(rows, MODEL_FEATURES)
    return evaluate_and_train_history(hist, min_races=min_races)


def _model_for_race(df: pd.DataFrame, lb: LearningBundle):
    calibrator = None
    validation_gain = 0.0
    if lb.accepted and lb.trained is not None:
        trained = lb.trained
        calibrator = lb.calibrator
        bll = lb.metrics.get("baseline_log_loss", np.nan)
        cll = lb.metrics.get("candidate_log_loss", np.nan)
        if np.isfinite(bll) and bll > 0 and np.isfinite(cll):
            validation_gain = float(np.clip((bll-cll)/bll, 0, 0.12))
        model_kind = f"walk-forward learned ({int(lb.metrics.get('races',0))} races)"
    else:
        trained = None
        model_kind = "heuristic warmup"
    return trained, calibrator, validation_gain, model_kind


def process_historical_race(
    client,
    job_id: str,
    d: date,
    stadium: int,
    race_no: int,
    bet_types: Tuple[str, ...],
    include_before: bool,
    include_odds: bool,
    n_sims: int,
    sim_weight: float,
    uncertainty_scale: float,
    lb: LearningBundle,
    feature_weights: Dict[str, float] | None = None,
    manual_weight_mix: float = 0.30,
) -> Dict[str, Any]:
    # Fetch historical pre-race inputs first; result is fetched separately and never used in prediction.
    fetch_bets = bet_types if include_odds else tuple()
    race_info, before, odds_raw, odds_updates, errors = fetch_race_bundle(d, stadium, race_no, include_before, fetch_bets)
    if not race_info:
        raise RuntimeError("出走表なし")
    df, weather = normalize_race_data(race_info, before)
    if len(df) != 6:
        raise RuntimeError(f"艇データが6艇ではありません ({len(df)})")

    trained, calibrator, validation_gain, model_kind = _model_for_race(df, lb)
    strengths = combine_model_and_manual_strength(
        df, trained=trained, weights=feature_weights, manual_mix=manual_weight_mix
    )
    temp = probability_temperature(df)
    raw_win = softmax_strength(strengths, temp)
    cal_win = calibrate_win_probs(raw_win, calibrator)
    if calibrator is not None:
        analytic = build_all_bet_probabilities(df["boat_no"].astype(int).tolist(), np.log(np.clip(cal_win,1e-8,1.0)), 1.0)
    else:
        analytic = build_all_bet_probabilities(df["boat_no"].astype(int).tolist(), strengths, temp)

    race_id = f"{d:%Y%m%d}-{stadium:02d}-{race_no:02d}"
    seed = int(hashlib.sha256(race_id.encode()).hexdigest()[:8], 16)
    simulated = simulate_races(df, cal_win, n_sims=n_sims, seed=seed, uncertainty_scale=uncertainty_scale)
    probs = blend_probability_sets(analytic, simulated, sim_weight)
    win = probs["win"].copy()
    win["boat_no"] = win["combo"].astype(int)
    df = df.merge(win[["boat_no","analytic_prob","sim_prob","prob"]].rename(columns={"prob":"win_prob","analytic_prob":"analytic_win_prob","sim_prob":"sim_win_prob"}), on="boat_no", how="left")

    reliability = reliability_factor(df, trained is not None, validation_gain)
    ticket_rows: List[Dict[str, Any]] = []
    for bt in bet_types:
        odds_df = parse_odds_dict(odds_raw.get(bt, {}) or {}, bt) if include_odds else pd.DataFrame()
        market = enrich_market(probs[bt], odds_df, reliability, BET_SPECS[bt]["exclusive"])
        for _, r in market.dropna(subset=["odds"]).iterrows():
            ticket_rows.append({
                "bet_type": bt, "combo": str(r["combo"]), "odds": float(r["odds"]),
                "model_prob": float(r["prob"]), "analytic_prob": float(r.get("analytic_prob", np.nan)) if np.isfinite(r.get("analytic_prob", np.nan)) else None,
                "sim_prob": float(r.get("sim_prob", np.nan)) if np.isfinite(r.get("sim_prob", np.nan)) else None,
                "market_prob_proxy": float(r.get("market_prob_proxy", np.nan)) if np.isfinite(r.get("market_prob_proxy", np.nan)) else None,
                "conservative_prob": float(r.get("conservative_prob", np.nan)) if np.isfinite(r.get("conservative_prob", np.nan)) else None,
                "conservative_ev": float(r.get("conservative_EV", np.nan)) if np.isfinite(r.get("conservative_EV", np.nan)) else None,
                "consensus_class": str(r.get("consensus_class", "")),
            })

    result = fetch_race_result(d, stadium, race_no)
    if not result or not result.get("result"):
        raise RuntimeError("結果未取得")
    ranks = {int(x["boat"]): int(x["rank"]) for x in result.get("result", []) if str(x.get("rank", "")).isdigit()}
    if len(ranks) < 3:
        raise RuntimeError("着順データ不足")
    payoffs = payoff_map(result)
    for t in ticket_rows:
        p = int(payoffs.get(t["bet_type"], {}).get(t["combo"], 0))
        t["result_hit"] = p > 0
        t["payoff"] = p

    snapshot = {
        "race_id": race_id, "race_date": d.isoformat(), "stadium_code": stadium,
        "stadium_name": STADIUMS.get(stadium, str(stadium)), "race_no": race_no,
        "model_kind": model_kind, "data_quality": data_quality(df),
        "simulation_count": n_sims, "simulation_weight": sim_weight,
        "reliability": reliability, "weather": weather, "boats": dataframe_records(df),
        "feature_weights": feature_weights or {}, "manual_weight_mix": float(manual_weight_mix),
        "result_json": result, "settled": True, "source": "backfill", "backtest_job_id": job_id,
    }
    sid = insert_backtest_snapshot(client, snapshot, ticket_rows)
    return {"snapshot_id": sid, "race_id": race_id, "model_kind": model_kind, "quality": data_quality(df), "odds_errors": errors}


def run_backtest_batch(client, job: Dict[str, Any], limit: int = 10, request_delay: float = 0.5) -> Dict[str, Any]:
    """Run a resumable chronological batch. Training only sees races strictly before target date."""
    import time
    from storage import get_pending_targets, update_target_status, refresh_backtest_job_counts

    targets = get_pending_targets(client, str(job["id"]), limit=limit)
    if not targets:
        counts = refresh_backtest_job_counts(client, str(job["id"]), status="completed", message="キュー処理完了")
        return {"done": 0, "failed": 0, "counts": counts, "completed": True, "messages": []}

    bet_types = tuple(job.get("bet_types") or ["trifecta", "trio", "exacta", "quinella"])
    include_before = bool(job.get("include_before", True))
    include_odds = bool(job.get("include_odds", True))
    n_sims = int(job.get("simulation_count", 2000) or 2000)
    sim_weight = float(job.get("simulation_weight", 0.35) or 0.35)
    uncertainty = float(job.get("uncertainty_scale", 1.0) or 1.0)
    retrain_every = max(1, int(job.get("retrain_every", 50) or 50))
    feature_weights = job.get("feature_weights") or {}
    manual_weight_mix = float(job.get("manual_weight_mix", 0.30) or 0.30)

    lb = None
    lb_date = None
    since_train = retrain_every
    done = failed = 0
    messages: List[str] = []
    for t in targets:
        d = date.fromisoformat(str(t["race_date"]))
        # Daily walk-forward refresh, and at user interval. The query itself uses < target_date.
        if lb is None or d != lb_date or since_train >= retrain_every:
            lb = learning_before(client, d)
            lb_date = d
            since_train = 0
        try:
            result = process_historical_race(
                client=client, job_id=str(job["id"]), d=d,
                stadium=int(t["stadium_code"]), race_no=int(t["race_no"]),
                bet_types=bet_types, include_before=include_before, include_odds=include_odds,
                n_sims=n_sims, sim_weight=sim_weight, uncertainty_scale=uncertainty,
                lb=lb, feature_weights=feature_weights, manual_weight_mix=manual_weight_mix,
            )
            update_target_status(client, int(t["id"]), "done", snapshot_id=result["snapshot_id"], error=None)
            done += 1; since_train += 1
        except Exception as e:
            update_target_status(client, int(t["id"]), "error", error=str(e)[:800])
            failed += 1
            messages.append(f"{d} {STADIUMS.get(int(t['stadium_code']), t['stadium_code'])} {t['race_no']}R: {e}")
        if request_delay > 0:
            time.sleep(request_delay)

    counts = refresh_backtest_job_counts(client, str(job["id"]), status="running")
    completed = counts.get("pending", 0) == 0
    if completed:
        counts = refresh_backtest_job_counts(client, str(job["id"]), status="completed", message="キュー処理完了")
    return {"done": done, "failed": failed, "counts": counts, "completed": completed, "messages": messages}
