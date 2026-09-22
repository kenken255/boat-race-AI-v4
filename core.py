from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from pyjpboatrace import PyJPBoatrace
except Exception as exc:  # pragma: no cover
    PyJPBoatrace = None
    PYJP_IMPORT_ERROR = exc
else:
    PYJP_IMPORT_ERROR = None

STADIUMS = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}
NAME_TO_STADIUM = {v: k for k, v in STADIUMS.items()}
CLASS_SCORE = {"A1": 1.0, "A2": 0.55, "B1": 0.10, "B2": -0.35}
COURSE_PRIOR = {1: 1.35, 2: 0.36, 3: 0.17, 4: 0.05, 5: -0.20, 6: -0.43}

# V5 manual factor weighting. Values are multipliers around the standard model (1.00 = 100%).
DEFAULT_FACTOR_WEIGHTS = {
    "course": 1.00,
    "class": 1.00,
    "global": 1.00,
    "local": 1.00,
    "motor": 1.00,
    "boat": 1.00,
    "ave_st": 1.00,
    "display_time": 1.00,
    "display_st": 1.00,
    "penalty": 1.00,
}
FACTOR_LABELS = {
    "course": "コース/進入",
    "class": "級別",
    "global": "全国成績",
    "local": "当地成績",
    "motor": "モーター",
    "boat": "ボート",
    "ave_st": "平均ST",
    "display_time": "展示タイム",
    "display_st": "展示ST",
    "penalty": "F/Lペナルティ",
}
WEIGHT_PRESETS = {
    "標準": dict(DEFAULT_FACTOR_WEIGHTS),
    "ST重視": {**DEFAULT_FACTOR_WEIGHTS, "ave_st": 1.55, "display_st": 1.65, "course": 1.10},
    "機力重視": {**DEFAULT_FACTOR_WEIGHTS, "motor": 1.65, "boat": 1.35, "display_time": 1.35},
    "選手実力重視": {**DEFAULT_FACTOR_WEIGHTS, "class": 1.35, "global": 1.55, "local": 1.25},
    "当地・水面重視": {**DEFAULT_FACTOR_WEIGHTS, "local": 1.65, "display_time": 1.35, "course": 1.20},
    "展示重視": {**DEFAULT_FACTOR_WEIGHTS, "display_time": 1.65, "display_st": 1.65, "motor": 1.20},
}

BET_SPECS = {
    "win": {"label": "単勝", "exclusive": True, "range_odds": False},
    "place": {"label": "複勝", "exclusive": False, "range_odds": True},
    "exacta": {"label": "2連単", "exclusive": True, "range_odds": False},
    "quinella": {"label": "2連複", "exclusive": True, "range_odds": False},
    "quinella_place": {"label": "拡連複", "exclusive": False, "range_odds": True},
    "trifecta": {"label": "3連単", "exclusive": True, "range_odds": False},
    "trio": {"label": "3連複", "exclusive": True, "range_odds": False},
}
LABEL_TO_BET = {v["label"]: k for k, v in BET_SPECS.items()}
DEFAULT_BET_LABELS = ["3連単", "3連複", "2連単", "2連複"]

MODEL_FEATURES = [
    "course", "class_score", "aveST", "global_win_pt", "global_in2nd", "global_in3rd",
    "local_win_pt", "local_in2nd", "local_in3rd", "motor_in2nd", "motor_in3rd",
    "boat_in2nd", "boat_in3rd", "F", "L", "display_time", "start_display_st",
    "wind_speed", "wave_height", "temperature", "water_temperature",
]
DISPLAY_COLS = [
    "boat_no", "course", "name", "class", "aveST", "global_win_pt", "global_in2nd",
    "local_win_pt", "local_in2nd", "motor_in2nd", "boat_in2nd", "display_time",
    "start_display_st", "tilt", "F", "L",
]
EDITABLE_COLS = ["course", "display_time", "start_display_st", "tilt"]
QUALITY_COLS = [
    "aveST", "global_win_pt", "global_in2nd", "local_win_pt", "local_in2nd",
    "motor_in2nd", "display_time", "start_display_st",
]


def jst_today() -> date:
    return datetime.now(ZoneInfo("Asia/Tokyo")).date()


def now_jst_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")


def safe_float(v: Any, default=np.nan) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def zscore(s: pd.Series, reverse: bool = False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() <= 1 or float(s.std(ddof=0) or 0) < 1e-9:
        out = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    else:
        out = (s - s.mean()) / s.std(ddof=0)
    return -out if reverse else out


def fetch_stadiums(d: date) -> Dict[str, Any]:
    if PyJPBoatrace is None:
        raise RuntimeError(f"pyjpboatrace を読み込めません: {PYJP_IMPORT_ERROR}")
    cli = PyJPBoatrace()
    try:
        return cli.get_stadiums(d)
    finally:
        try:
            cli.close()
        except Exception:
            pass


def fetch_race_bundle(d: date, stadium: int, race: int, include_before: bool, bet_types: Tuple[str, ...]):
    if PyJPBoatrace is None:
        raise RuntimeError(f"pyjpboatrace を読み込めません: {PYJP_IMPORT_ERROR}")
    cli = PyJPBoatrace()
    try:
        race_info = cli.get_race_info(d=d, stadium=stadium, race=race)
        before: Dict[str, Any] = {}
        if include_before:
            try:
                before = cli.get_just_before_info(d=d, stadium=stadium, race=race)
            except Exception:
                before = {}

        odds_by_type: Dict[str, Any] = {}
        odds_updates: Dict[str, str] = {}
        errors: Dict[str, str] = {}

        if any(x in bet_types for x in ("win", "place")):
            try:
                wp = cli.get_odds_win_placeshow(d=d, stadium=stadium, race=race)
                if "win" in bet_types:
                    odds_by_type["win"] = wp.get("win", {})
                    odds_updates["win"] = str(wp.get("update", ""))
                if "place" in bet_types:
                    odds_by_type["place"] = wp.get("place_show", {})
                    odds_updates["place"] = str(wp.get("update", ""))
            except Exception as e:
                for x in ("win", "place"):
                    if x in bet_types:
                        errors[x] = str(e)

        if any(x in bet_types for x in ("exacta", "quinella")):
            try:
                eq = cli.get_odds_exacta_quinella(d=d, stadium=stadium, race=race)
                if "exacta" in bet_types:
                    odds_by_type["exacta"] = eq.get("exacta", {})
                    odds_updates["exacta"] = str(eq.get("update", ""))
                if "quinella" in bet_types:
                    odds_by_type["quinella"] = eq.get("quinella", {})
                    odds_updates["quinella"] = str(eq.get("update", ""))
            except Exception as e:
                for x in ("exacta", "quinella"):
                    if x in bet_types:
                        errors[x] = str(e)

        if "quinella_place" in bet_types:
            try:
                qp = cli.get_odds_quinellaplace(d=d, stadium=stadium, race=race)
                odds_by_type["quinella_place"] = qp
                odds_updates["quinella_place"] = str(qp.get("update", ""))
            except Exception as e:
                errors["quinella_place"] = str(e)

        if "trifecta" in bet_types:
            try:
                tri = cli.get_odds_trifecta(d=d, stadium=stadium, race=race)
                odds_by_type["trifecta"] = tri
                odds_updates["trifecta"] = str(tri.get("update", ""))
            except Exception as e:
                errors["trifecta"] = str(e)

        if "trio" in bet_types:
            try:
                trio = cli.get_odds_trio(d=d, stadium=stadium, race=race)
                odds_by_type["trio"] = trio
                odds_updates["trio"] = str(trio.get("update", ""))
            except Exception as e:
                errors["trio"] = str(e)

        return race_info, before, odds_by_type, odds_updates, errors
    finally:
        try:
            cli.close()
        except Exception:
            pass


def fetch_race_result(d: date, stadium: int, race: int) -> Dict[str, Any]:
    if PyJPBoatrace is None:
        raise RuntimeError(f"pyjpboatrace を読み込めません: {PYJP_IMPORT_ERROR}")
    cli = PyJPBoatrace()
    try:
        return cli.get_race_result(d=d, stadium=stadium, race=race)
    finally:
        try:
            cli.close()
        except Exception:
            pass


def normalize_race_data(race_info: Dict[str, Any], before: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    start_map: Dict[int, Dict[str, Any]] = {}
    for course_key, item in (before.get("start_display", {}) or {}).items():
        try:
            course = int(str(course_key).replace("course", ""))
        except Exception:
            continue
        boat = safe_int(item.get("boat"), 0)
        if boat:
            start_map[boat] = {"course": course, "ST": safe_float(item.get("ST"))}

    weather = before.get("weather_information", {}) or {}
    for boat_no in range(1, 7):
        k = f"boat{boat_no}"
        r = race_info.get(k, {}) or {}
        b = before.get(k, {}) or {}
        sd = start_map.get(boat_no, {})
        course = safe_int(sd.get("course"), boat_no)
        rows.append({
            "boat_no": boat_no,
            "course": course,
            "racerid": r.get("racerid"),
            "name": r.get("name", f"{boat_no}号艇"),
            "class": r.get("class", ""),
            "class_score": CLASS_SCORE.get(str(r.get("class", "")), 0.0),
            "F": safe_float(r.get("F"), 0.0),
            "L": safe_float(r.get("L"), 0.0),
            "age": safe_float(r.get("age")),
            "weight": safe_float(b.get("weight", r.get("weight"))),
            "aveST": safe_float(r.get("aveST")),
            "global_win_pt": safe_float(r.get("global_win_pt")),
            "global_in2nd": safe_float(r.get("global_in2nd")),
            "global_in3rd": safe_float(r.get("global_in3rd")),
            "local_win_pt": safe_float(r.get("local_win_pt")),
            "local_in2nd": safe_float(r.get("local_in2nd")),
            "local_in3rd": safe_float(r.get("local_in3rd")),
            "motor_no": r.get("motor"),
            "motor_in2nd": safe_float(r.get("motor_in2nd")),
            "motor_in3rd": safe_float(r.get("motor_in3rd")),
            "boat_no_machine": r.get("boat"),
            "boat_in2nd": safe_float(r.get("boat_in2nd")),
            "boat_in3rd": safe_float(r.get("boat_in3rd")),
            "display_time": safe_float(b.get("display_time")),
            "tilt": safe_float(b.get("tilt")),
            "weight_adjustment": safe_float(b.get("weight_adjustment"), 0.0),
            "start_display_st": safe_float(sd.get("ST")),
            "wind_speed": safe_float(weather.get("wind_speed")),
            "wave_height": safe_float(weather.get("wave_height")),
            "temperature": safe_float(weather.get("temperature")),
            "water_temperature": safe_float(weather.get("water_temperature")),
        })
    return pd.DataFrame(rows), weather


def apply_conditions(df: pd.DataFrame, weather: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c in ["wind_speed", "wave_height", "temperature", "water_temperature"]:
        out[c] = safe_float(weather.get(c))
    return out


def normalize_factor_weights(weights: Dict[str, float] | None = None) -> Dict[str, float]:
    out = dict(DEFAULT_FACTOR_WEIGHTS)
    if weights:
        for k in out:
            try:
                out[k] = float(np.clip(float(weights.get(k, out[k])), 0.0, 2.5))
            except Exception:
                pass
    return out


def heuristic_strength(df: pd.DataFrame, weights: Dict[str, float] | None = None) -> np.ndarray:
    """Human-readable heuristic score with V5 adjustable factor multipliers.

    Each multiplier is relative to the standard V3/V4 weight: 1.0 = standard,
    0.0 = ignore the factor, 2.0 = double its influence.
    """
    w = normalize_factor_weights(weights)
    score = df["course"].map(COURSE_PRIOR).fillna(0).astype(float) * w["course"]
    wind = safe_float(pd.to_numeric(df.get("wind_speed"), errors="coerce").median(), 0.0)
    wave = safe_float(pd.to_numeric(df.get("wave_height"), errors="coerce").median(), 0.0)
    roughness = float(np.clip(max(wind, 0) / 10 + max(wave, 0) / 20, 0, 1.0))
    score = score * (1.0 - 0.12 * roughness)
    components = [
        ("class_score", 0.42 * w["class"], False),
        ("global_win_pt", 0.52 * w["global"], False),
        ("global_in2nd", 0.23 * w["global"], False),
        ("local_win_pt", 0.18 * w["local"], False),
        ("local_in2nd", 0.12 * w["local"], False),
        ("motor_in2nd", 0.25 * w["motor"], False),
        ("boat_in2nd", 0.08 * w["boat"], False),
        ("aveST", 0.32 * w["ave_st"], True),
        ("display_time", (0.34 + 0.10 * roughness) * w["display_time"], True),
        ("start_display_st", (0.30 + 0.10 * roughness) * w["display_st"], True),
    ]
    for col, weight, reverse in components:
        if col in df:
            score = score + weight * zscore(df[col], reverse=reverse).fillna(0)
    score = score - (0.18 * w["penalty"]) * pd.to_numeric(df.get("F", 0), errors="coerce").fillna(0)
    score = score - (0.05 * w["penalty"]) * pd.to_numeric(df.get("L", 0), errors="coerce").fillna(0)
    return score.to_numpy(float)


def combine_model_and_manual_strength(
    df: pd.DataFrame,
    trained=None,
    weights: Dict[str, float] | None = None,
    manual_mix: float = 0.30,
) -> np.ndarray:
    """Blend validated learned strength with the user-adjustable transparent model.

    With no learned model this simply returns the weighted heuristic. With a learned
    model, both score vectors are standardized race-by-race before blending so that
    the manual slider layer has a stable, interpretable influence.
    """
    manual = heuristic_strength(df, weights)
    if trained is None:
        return manual
    learned = model_strength(df, trained)
    m = float(np.clip(manual_mix, 0.0, 1.0))

    def _std(v):
        a = np.asarray(v, dtype=float)
        sd = float(np.nanstd(a))
        if not np.isfinite(sd) or sd < 1e-9:
            return np.zeros_like(a)
        return (a - float(np.nanmean(a))) / sd

    return (1.0 - m) * _std(learned) + m * _std(manual)


def train_from_history(hist: pd.DataFrame):
    if "result_rank" not in hist.columns:
        raise ValueError("履歴に result_rank 列が必要です。")
    usable = [c for c in MODEL_FEATURES if c in hist.columns]
    if len(usable) < 4:
        raise ValueError("学習に使える特徴量が不足しています。")
    x = hist[usable].copy()
    y = (pd.to_numeric(hist["result_rank"], errors="coerce") == 1).astype(int)
    if len(x) < 120 or y.nunique() < 2:
        raise ValueError("最低120行（目安20レース）以上の履歴が必要です。")
    pre = ColumnTransformer([("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
    ]), usable)])
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1800, class_weight="balanced", C=0.7)),
    ])
    model.fit(x, y)
    return model, usable


def model_strength(df: pd.DataFrame, trained) -> np.ndarray:
    model, usable = trained
    x = df.reindex(columns=usable)
    clf = model.named_steps["clf"]
    xt = model.named_steps["pre"].transform(x)
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(xt), dtype=float)
    p = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def probability_temperature(df: pd.DataFrame) -> float:
    wind = safe_float(pd.to_numeric(df.get("wind_speed"), errors="coerce").median(), 0.0)
    wave = safe_float(pd.to_numeric(df.get("wave_height"), errors="coerce").median(), 0.0)
    return float(1.0 + np.clip(max(wind, 0) / 25 + max(wave, 0) / 60, 0, 0.40))


def softmax_strength(strength_scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    s = np.asarray(strength_scores, dtype=float) / max(temperature, 0.75)
    s = s - np.nanmax(s)
    w = np.exp(np.clip(s, -12, 12))
    w = np.where(np.isfinite(w), w, 1e-6)
    return w / w.sum()


def ordered_probabilities(boats: List[int], strength_scores: np.ndarray, temperature: float = 1.0):
    s = np.asarray(strength_scores, dtype=float) / max(temperature, 0.75)
    s = s - np.nanmax(s)
    w = np.exp(np.clip(s, -12, 12))
    w = np.where(np.isfinite(w), w, 1e-6)
    total = w.sum()
    idx = {b: i for i, b in enumerate(boats)}

    win_df = pd.DataFrame({"combo": [str(b) for b in boats], "prob": w / total})
    pair_rows = []
    for a, b in itertools.permutations(boats, 2):
        ia, ib = idx[a], idx[b]
        p = (w[ia] / total) * (w[ib] / (total - w[ia]))
        pair_rows.append({"combo": f"{a}-{b}", "prob": float(p)})
    exacta = pd.DataFrame(pair_rows).sort_values("prob", ascending=False, ignore_index=True)

    tri_rows = []
    for a, b, c in itertools.permutations(boats, 3):
        ia, ib, ic = idx[a], idx[b], idx[c]
        p1 = w[ia] / total
        denom2 = total - w[ia]
        p2 = w[ib] / denom2
        denom3 = denom2 - w[ib]
        p3 = w[ic] / denom3
        tri_rows.append({"combo": f"{a}-{b}-{c}", "prob": float(p1 * p2 * p3)})
    trifecta = pd.DataFrame(tri_rows).sort_values("prob", ascending=False, ignore_index=True)
    return win_df, exacta, trifecta


def build_all_bet_probabilities(boats: List[int], strength_scores: np.ndarray, temperature: float = 1.0) -> Dict[str, pd.DataFrame]:
    win, exacta, trifecta = ordered_probabilities(boats, strength_scores, temperature)
    return derive_bet_probabilities_from_ordered(boats, win, exacta, trifecta)


def derive_bet_probabilities_from_ordered(boats, win, exacta, trifecta):
    place_rows = []
    for boat in boats:
        mask = exacta["combo"].str.split("-").apply(lambda xs: str(boat) in xs)
        place_rows.append({"combo": str(boat), "prob": float(exacta.loc[mask, "prob"].sum())})
    place = pd.DataFrame(place_rows).sort_values("prob", ascending=False, ignore_index=True)

    q_rows: Dict[str, float] = {}
    for _, r in exacta.iterrows():
        a, b = map(int, str(r["combo"]).split("-"))
        key = "=".join(map(str, sorted((a, b))))
        q_rows[key] = q_rows.get(key, 0.0) + float(r["prob"])
    quinella = pd.DataFrame([{"combo": k, "prob": v} for k, v in q_rows.items()]).sort_values("prob", ascending=False, ignore_index=True)

    trio_rows: Dict[str, float] = {}
    wide_rows: Dict[str, float] = {"=".join(map(str, p)): 0.0 for p in itertools.combinations(boats, 2)}
    for _, r in trifecta.iterrows():
        arr = tuple(map(int, str(r["combo"]).split("-")))
        trio_key = "=".join(map(str, sorted(arr)))
        trio_rows[trio_key] = trio_rows.get(trio_key, 0.0) + float(r["prob"])
        aset = set(arr)
        for a, b in itertools.combinations(boats, 2):
            if a in aset and b in aset:
                wide_rows[f"{a}={b}"] += float(r["prob"])
    trio = pd.DataFrame([{"combo": k, "prob": v} for k, v in trio_rows.items()]).sort_values("prob", ascending=False, ignore_index=True)
    quinella_place = pd.DataFrame([{"combo": k, "prob": v} for k, v in wide_rows.items()]).sort_values("prob", ascending=False, ignore_index=True)

    return {
        "win": win.sort_values("prob", ascending=False, ignore_index=True),
        "place": place,
        "exacta": exacta,
        "quinella": quinella,
        "quinella_place": quinella_place,
        "trifecta": trifecta,
        "trio": trio,
    }


def data_quality(df: pd.DataFrame) -> float:
    available = [c for c in QUALITY_COLS if c in df.columns]
    if not available:
        return 0.0
    return float(df[available].notna().mean().mean())


def simulate_races(df: pd.DataFrame, base_win_probs: np.ndarray, n_sims: int = 10000, seed: int = 42, uncertainty_scale: float = 1.0) -> Dict[str, pd.DataFrame]:
    """Latent performance + start variability Monte Carlo.

    This is an uncertainty model, not a physical CFD race simulator. It perturbs each boat's
    latent strength and start timing, with more noise when data are incomplete or conditions rough.
    """
    rng = np.random.default_rng(seed)
    boats = df["boat_no"].astype(int).to_numpy()
    n = len(boats)
    base = np.log(np.clip(np.asarray(base_win_probs, dtype=float), 1e-8, 1.0))
    q = data_quality(df)
    temp = probability_temperature(df)
    rough = np.clip((temp - 1.0) / 0.40, 0.0, 1.0)
    latent_sigma = (0.38 + 0.30 * (1 - q) + 0.16 * rough) * float(uncertainty_scale)

    ave = pd.to_numeric(df.get("aveST"), errors="coerce").to_numpy(float)
    dsp = pd.to_numeric(df.get("start_display_st"), errors="coerce").to_numpy(float)
    mu = np.where(np.isfinite(dsp), 0.62 * dsp + 0.38 * np.where(np.isfinite(ave), ave, dsp), ave)
    fallback = np.nanmedian(mu) if np.isfinite(mu).any() else 0.17
    mu = np.where(np.isfinite(mu), mu, fallback)
    st_sd = (0.024 + 0.012 * rough) * float(uncertainty_scale)

    chunk = 5000
    win_counts = {str(b): 0 for b in boats}
    exacta_counts: Dict[str, int] = {}
    trifecta_counts: Dict[str, int] = {}

    remaining = int(n_sims)
    while remaining > 0:
        m = min(chunk, remaining)
        st = rng.normal(mu, st_sd, size=(m, n))
        st_center = st - st.mean(axis=1, keepdims=True)
        latent_noise = rng.normal(0.0, latent_sigma, size=(m, n))
        turn_noise = rng.gumbel(0.0, 0.18 + 0.08 * rough, size=(m, n))
        latent = base[None, :] + latent_noise + turn_noise - 2.6 * st_center
        order = np.argsort(-latent, axis=1)
        top3 = boats[order[:, :3]]
        for row in top3:
            a, b, c = map(int, row)
            win_counts[str(a)] += 1
            exacta_counts[f"{a}-{b}"] = exacta_counts.get(f"{a}-{b}", 0) + 1
            trifecta_counts[f"{a}-{b}-{c}"] = trifecta_counts.get(f"{a}-{b}-{c}", 0) + 1
        remaining -= m

    win = pd.DataFrame([{"combo": k, "prob": v / n_sims} for k, v in win_counts.items()])
    exacta = pd.DataFrame([{"combo": k, "prob": v / n_sims} for k, v in exacta_counts.items()])
    trifecta = pd.DataFrame([{"combo": k, "prob": v / n_sims} for k, v in trifecta_counts.items()])

    # Ensure zero-count combinations are present so merging is stable.
    all_exacta = [f"{a}-{b}" for a, b in itertools.permutations(boats, 2)]
    all_tri = [f"{a}-{b}-{c}" for a, b, c in itertools.permutations(boats, 3)]
    exacta = pd.DataFrame({"combo": all_exacta}).merge(exacta, on="combo", how="left").fillna({"prob": 0.0})
    trifecta = pd.DataFrame({"combo": all_tri}).merge(trifecta, on="combo", how="left").fillna({"prob": 0.0})
    return derive_bet_probabilities_from_ordered(list(map(int, boats)), win, exacta, trifecta)


def blend_probability_sets(analytical: Dict[str, pd.DataFrame], simulated: Dict[str, pd.DataFrame], sim_weight: float) -> Dict[str, pd.DataFrame]:
    w = float(np.clip(sim_weight, 0.0, 1.0))
    out: Dict[str, pd.DataFrame] = {}
    for bt in BET_SPECS:
        a = analytical[bt].rename(columns={"prob": "analytic_prob"})
        s = simulated[bt].rename(columns={"prob": "sim_prob"})
        m = a.merge(s, on="combo", how="outer").fillna(0.0)
        m["prob"] = (1.0 - w) * m["analytic_prob"] + w * m["sim_prob"]
        out[bt] = m.sort_values("prob", ascending=False, ignore_index=True)
    return out


def calibrate_win_probs(win_probs: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    p = np.asarray(win_probs, dtype=float)
    if calibrator is None:
        return p / p.sum()
    try:
        q = np.asarray(calibrator.predict(np.clip(p, 1e-6, 1 - 1e-6)), dtype=float)
        q = np.clip(q, 1e-5, 1.0)
        if q.sum() > 0:
            return q / q.sum()
    except Exception:
        pass
    return p / p.sum()


def parse_odds_dict(raw: Dict[str, Any], bet_type: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    meta = {"date", "stadium", "race", "update", "win", "place_show", "exacta", "quinella"}
    for k, v in (raw or {}).items():
        if str(k) in meta:
            continue
        key = str(k)
        if bet_type in ("win", "place") and not re.fullmatch(r"[1-6]", key):
            continue
        if bet_type == "exacta" and not re.fullmatch(r"[1-6]-[1-6]", key):
            continue
        if bet_type in ("quinella", "quinella_place") and not re.fullmatch(r"[1-6]=[1-6]", key):
            continue
        if bet_type == "trifecta" and not re.fullmatch(r"[1-6]-[1-6]-[1-6]", key):
            continue
        if bet_type == "trio" and not re.fullmatch(r"[1-6]=[1-6]=[1-6]", key):
            continue

        if isinstance(v, (list, tuple)) and len(v) >= 2:
            lo, hi = safe_float(v[0]), safe_float(v[1])
            if np.isfinite(lo) and lo > 0:
                rows.append({"combo": key, "odds": lo, "odds_low": lo, "odds_high": hi})
        else:
            val = safe_float(v)
            if np.isfinite(val) and val > 0:
                rows.append({"combo": key, "odds": val, "odds_low": val, "odds_high": val})
    return pd.DataFrame(rows)


def reliability_factor(df: pd.DataFrame, trained: bool, validation_gain: float = 0.0) -> float:
    q = data_quality(df)
    base = 0.62 if trained else 0.43
    base += float(np.clip(validation_gain, 0.0, 0.12))
    return float(np.clip(base * (0.72 + 0.28 * q), 0.24, 0.80))


def _percentile_score(s: pd.Series) -> pd.Series:
    if s.notna().sum() <= 1:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return s.rank(pct=True, method="average")


def classify_consensus(out: pd.DataFrame) -> pd.Series:
    ai = out["ai_percentile"]
    mk = out["market_percentile"]
    edge = out["edge_ratio"]
    cev = out["conservative_EV"]
    labels = np.full(len(out), "中立・評価混在", dtype=object)
    both = (ai >= 0.72) & (mk >= 0.72) & (cev >= 0.90)
    underv = (ai >= 0.65) & ((mk <= 0.58) | (edge >= 1.12)) & (edge > 1.02)
    market_only = (mk >= 0.72) & ((ai <= 0.58) | (edge <= 0.90))
    low = (ai <= 0.42) & (mk <= 0.42)
    labels[low.fillna(False)] = "双方低評価・見送り"
    labels[market_only.fillna(False)] = "市場優位・AI慎重"
    labels[underv.fillna(False)] = "AI優位・過小評価"
    labels[both.fillna(False)] = "両者一致・手堅さ"
    return pd.Series(labels, index=out.index)


def enrich_market(pred: pd.DataFrame, odds_df: pd.DataFrame, reliability: float, exclusive: bool) -> pd.DataFrame:
    out = pred.copy()
    if odds_df.empty:
        for c in ["odds", "odds_low", "odds_high", "market_prob_proxy", "break_even_prob", "EV", "edge_ratio", "conservative_prob", "conservative_EV", "conservative_edge", "kelly", "ai_percentile", "market_percentile", "consensus_class"]:
            out[c] = np.nan
        return out
    out = out.merge(odds_df, on="combo", how="left")
    out["break_even_prob"] = 1.0 / out["odds"]
    inv = 1.0 / out["odds"]
    if exclusive:
        denom = inv.sum(skipna=True)
        out["market_prob_proxy"] = inv / denom if denom > 0 else np.nan
    else:
        out["market_prob_proxy"] = out["break_even_prob"].clip(upper=1.0)
    out["EV"] = out["prob"] * out["odds"]
    out["edge_ratio"] = out["prob"] / out["market_prob_proxy"]
    out["conservative_prob"] = reliability * out["prob"] + (1.0 - reliability) * out["market_prob_proxy"]
    out["conservative_prob"] = out["conservative_prob"].clip(lower=0.0, upper=1.0)
    out["conservative_EV"] = out["conservative_prob"] * out["odds"]
    out["conservative_edge"] = out["conservative_prob"] / out["market_prob_proxy"]
    b = out["odds"] - 1.0
    out["kelly"] = np.where(b > 0, (out["conservative_prob"] * out["odds"] - 1.0) / b, 0.0)
    out["kelly"] = pd.to_numeric(out["kelly"], errors="coerce").clip(lower=0.0, upper=0.25)
    out["ai_percentile"] = _percentile_score(out["prob"])
    out["market_percentile"] = _percentile_score(out["market_prob_proxy"])
    out["consensus_class"] = classify_consensus(out)
    return out


def round_to_unit(x: float, unit: int) -> int:
    if not np.isfinite(x) or x < unit:
        return 0
    return int(math.floor(x / unit) * unit)


def allocate_stakes(cands: pd.DataFrame, bankroll: int, unit: int, race_cap_pct: float, kelly_scale: float, max_each_pct: float) -> pd.DataFrame:
    if cands.empty:
        return cands.copy()
    x = cands.copy()
    raw = []
    for _, r in x.iterrows():
        k = safe_float(r.get("kelly"), 0.0)
        amt = min(bankroll * k * kelly_scale, bankroll * max_each_pct)
        raw.append(max(0.0, amt))
    x["raw_stake"] = raw
    total_raw = float(x["raw_stake"].sum())
    cap = bankroll * race_cap_pct
    if total_raw > cap and total_raw > 0:
        x["raw_stake"] *= cap / total_raw
    x["stake_yen"] = x["raw_stake"].map(lambda v: round_to_unit(v, unit))
    y = x[x["stake_yen"] >= unit].copy()
    if y.empty and not x.empty and cap >= unit:
        y = x.head(1).copy()
        y["stake_yen"] = unit
    return y.drop(columns=["raw_stake"], errors="ignore")


def build_bet_strategies(market: pd.DataFrame, bankroll: int, unit: int) -> Dict[str, pd.DataFrame]:
    keys = ["balanced", "safe_consensus", "undervalued_ai", "market_caution"]
    if market.empty or market.get("odds", pd.Series(dtype=float)).notna().sum() == 0:
        return {k: pd.DataFrame() for k in keys}
    t = market.dropna(subset=["odds", "conservative_EV", "conservative_prob", "conservative_edge", "market_prob_proxy"]).copy()
    if t.empty:
        return {k: pd.DataFrame() for k in keys}

    p_med = float(t["conservative_prob"].median())
    balanced = t[(t["conservative_EV"] >= 1.04) & (t["conservative_edge"] >= 1.05) & (t["conservative_prob"] >= max(0.004, p_med * 0.45))].copy()
    balanced["strategy_score"] = np.log(np.clip(balanced["conservative_EV"], 1.0001, None)) * np.sqrt(balanced["conservative_prob"])
    balanced = allocate_stakes(balanced.sort_values(["strategy_score", "conservative_prob"], ascending=False).head(5), bankroll, unit, 0.03, 0.125, 0.012)

    safe = t[(t["consensus_class"] == "両者一致・手堅さ") & (t["conservative_EV"] >= 0.90)].copy()
    safe["strategy_score"] = 0.55 * safe["ai_percentile"] + 0.45 * safe["market_percentile"] + 0.10 * np.clip(safe["conservative_EV"] - 1.0, -0.2, 0.4)
    safe = allocate_stakes(safe.sort_values(["strategy_score", "conservative_prob"], ascending=False).head(4), bankroll, unit, 0.015, 0.075, 0.0075)

    underv = t[(t["consensus_class"] == "AI優位・過小評価") & (t["conservative_EV"] >= 1.03) & (t["conservative_edge"] >= 1.05)].copy()
    underv["strategy_score"] = (underv["edge_ratio"] - 1.0) * np.sqrt(np.clip(underv["prob"], 1e-6, None))
    underv = allocate_stakes(underv.sort_values(["strategy_score", "conservative_EV"], ascending=False).head(5), bankroll, unit, 0.02, 0.06, 0.0065)

    caution = t[t["consensus_class"] == "市場優位・AI慎重"].copy()
    caution = caution.sort_values(["market_percentile", "edge_ratio"], ascending=[False, True]).head(5)
    return {"balanced": balanced, "safe_consensus": safe, "undervalued_ai": underv, "market_caution": caution}


def portfolio_metrics(bets: pd.DataFrame, exclusive: bool) -> Dict[str, float]:
    if bets.empty or "stake_yen" not in bets:
        return {"stake": 0, "hit_prob": 0, "expected_payout": 0, "expected_roi": np.nan}
    stake = float(bets["stake_yen"].sum())
    if stake <= 0:
        return {"stake": 0, "hit_prob": 0, "expected_payout": 0, "expected_roi": np.nan}
    hit_prob = float(bets["conservative_prob"].sum())
    if not exclusive:
        hit_prob = min(hit_prob, 1.0)
    expected_payout = float((bets["stake_yen"] * bets["conservative_prob"] * bets["odds"]).sum())
    return {"stake": stake, "hit_prob": hit_prob, "expected_payout": expected_payout, "expected_roi": expected_payout / stake - 1.0}


def strategy_view(df: pd.DataFrame, label: str, range_odds: bool) -> pd.DataFrame:
    if df.empty:
        return df
    cols = ["combo", "consensus_class", "prob", "market_prob_proxy", "conservative_prob", "odds", "conservative_EV", "conservative_edge"]
    if "stake_yen" in df.columns:
        cols.append("stake_yen")
    if range_odds and "odds_high" in df.columns:
        cols.insert(6, "odds_high")
    v = df[[c for c in cols if c in df.columns]].copy()
    for c in ["prob", "market_prob_proxy", "conservative_prob"]:
        if c in v:
            v[c] = (v[c] * 100).round(2)
    for c in ["odds", "odds_high"]:
        if c in v:
            v[c] = v[c].round(1)
    if "conservative_EV" in v:
        v["conservative_EV"] = v["conservative_EV"].round(3)
    if "conservative_edge" in v:
        v["conservative_edge"] = v["conservative_edge"].round(2)
    return v.rename(columns={
        "combo": label, "consensus_class": "市場×AI評価", "prob": "AI確率(%)",
        "market_prob_proxy": "市場代理確率(%)", "conservative_prob": "保守確率(%)",
        "odds": "保守オッズ", "odds_high": "オッズ上限", "conservative_EV": "保守EV",
        "conservative_edge": "市場比", "stake_yen": "参考配分(円)",
    })


def ticket_type_guide(markets: Dict[str, pd.DataFrame], strategies: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for bt, market in markets.items():
        spec = BET_SPECS[bt]
        valid = market.dropna(subset=["odds", "conservative_EV", "conservative_prob", "conservative_edge"]) if not market.empty else pd.DataFrame()
        if valid.empty:
            rows.append({"券種": spec["label"], "取得": "オッズなし", "最大保守EV": np.nan, "最高保守確率(%)": np.nan, "両者一致": 0, "AI過小評価": 0, "採算候補": 0, "向き": "判定不可"})
            continue
        bal = strategies[bt]["balanced"]
        safe = strategies[bt]["safe_consensus"]
        und = strategies[bt]["undervalued_ai"]
        best_ev = float(valid["conservative_EV"].max())
        best_p = float(valid["conservative_prob"].max())
        if not safe.empty:
            direction = "両者一致・手堅さ候補"
        elif not bal.empty:
            direction = "採算×リスク候補"
        elif not und.empty:
            direction = "AI優位・妙味候補"
        else:
            direction = "見送り寄り"
        rows.append({
            "券種": spec["label"], "取得": "OK", "最大保守EV": round(best_ev, 3),
            "最高保守確率(%)": round(best_p * 100, 1), "両者一致": len(safe),
            "AI過小評価": len(und), "採算候補": len(bal), "向き": direction,
        })
    return pd.DataFrame(rows)


def market_view(df: pd.DataFrame, bet_type: str, top_n: int) -> pd.DataFrame:
    spec = BET_SPECS[bet_type]
    v = df.head(top_n).copy()
    for c in ["prob", "analytic_prob", "sim_prob", "market_prob_proxy", "conservative_prob"]:
        if c in v:
            v[c] = (v[c] * 100).round(2)
    for c in ["EV", "conservative_EV", "conservative_edge"]:
        if c in v:
            v[c] = v[c].round(3 if "EV" in c else 2)
    for c in ["odds", "odds_high"]:
        if c in v:
            v[c] = v[c].round(1)
    cols = ["combo", "consensus_class", "prob", "analytic_prob", "sim_prob", "odds"]
    if spec["range_odds"] and "odds_high" in v.columns:
        cols.append("odds_high")
    cols += ["EV", "market_prob_proxy", "conservative_prob", "conservative_EV", "conservative_edge"]
    v = v[[c for c in cols if c in v.columns]]
    return v.rename(columns={
        "combo": spec["label"], "consensus_class": "市場×AI評価", "prob": "統合AI確率(%)",
        "analytic_prob": "解析確率(%)", "sim_prob": "仮想レース確率(%)", "odds": "保守オッズ",
        "odds_high": "オッズ上限", "EV": "生EV", "market_prob_proxy": "市場代理確率(%)",
        "conservative_prob": "保守確率(%)", "conservative_EV": "保守EV", "conservative_edge": "保守市場比",
    })


def history_template() -> pd.DataFrame:
    return pd.DataFrame(columns=["race_id", "race_date", "boat_no"] + MODEL_FEATURES + ["result_rank"])


def _racewise_probs_from_model(df: pd.DataFrame, trained) -> pd.Series:
    vals = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("race_id", sort=False):
        s = model_strength(g, trained)
        p = softmax_strength(s, 1.0)
        vals.loc[g.index] = p
    return vals


def _racewise_probs_heuristic(df: pd.DataFrame) -> pd.Series:
    vals = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("race_id", sort=False):
        s = heuristic_strength(g)
        p = softmax_strength(s, probability_temperature(g))
        vals.loc[g.index] = p
    return vals


def probability_metrics(df: pd.DataFrame, probs: pd.Series) -> Dict[str, float]:
    tmp = df[["race_id", "result_rank"]].copy()
    tmp["p"] = pd.to_numeric(probs, errors="coerce").clip(1e-8, 1 - 1e-8)
    tmp["y"] = (pd.to_numeric(tmp["result_rank"], errors="coerce") == 1).astype(int)
    race_losses, race_briers = [], []
    for _, g in tmp.groupby("race_id"):
        if g["y"].sum() != 1:
            continue
        p = g["p"].to_numpy(float)
        p = p / p.sum()
        y = g["y"].to_numpy(int)
        winner_p = float(p[np.argmax(y)])
        race_losses.append(-math.log(max(winner_p, 1e-8)))
        race_briers.append(float(np.mean((p - y) ** 2)))
    return {
        "log_loss": float(np.mean(race_losses)) if race_losses else np.nan,
        "brier": float(np.mean(race_briers)) if race_briers else np.nan,
        "races": len(race_losses),
    }


@dataclass
class LearningBundle:
    accepted: bool
    trained: Any | None
    calibrator: Any | None
    metrics: Dict[str, float]
    note: str


def evaluate_and_train_history(hist: pd.DataFrame, min_races: int = 30) -> LearningBundle:
    if hist.empty or "race_id" not in hist or "result_rank" not in hist:
        return LearningBundle(False, None, None, {}, "履歴不足")
    hist = hist.copy()
    hist = hist[pd.to_numeric(hist["result_rank"], errors="coerce").notna()]
    race_order = hist[["race_id"] + (["race_date"] if "race_date" in hist.columns else [])].drop_duplicates()
    if "race_date" in race_order:
        race_order = race_order.sort_values(["race_date", "race_id"])
    else:
        race_order = race_order.sort_values("race_id")
    races = race_order["race_id"].tolist()
    if len(races) < min_races:
        return LearningBundle(False, None, None, {"races": len(races)}, f"自己学習には最低{min_races}レースを推奨（現在{len(races)}）")

    cut = max(20, int(len(races) * 0.80))
    cut = min(cut, len(races) - max(8, int(len(races) * 0.15)))
    train_races, valid_races = set(races[:cut]), set(races[cut:])
    tr = hist[hist["race_id"].isin(train_races)].copy()
    va = hist[hist["race_id"].isin(valid_races)].copy()
    try:
        candidate = train_from_history(tr)
    except Exception as e:
        return LearningBundle(False, None, None, {"races": len(races)}, f"候補モデル学習失敗: {e}")

    cand_p = _racewise_probs_from_model(va, candidate)
    base_p = _racewise_probs_heuristic(va)
    cm = probability_metrics(va, cand_p)
    bm = probability_metrics(va, base_p)

    # Candidate must improve one metric materially and not degrade the other badly.
    improve_ll = np.isfinite(cm["log_loss"]) and np.isfinite(bm["log_loss"]) and cm["log_loss"] <= bm["log_loss"] * 0.995
    improve_br = np.isfinite(cm["brier"]) and np.isfinite(bm["brier"]) and cm["brier"] <= bm["brier"] * 0.99
    not_bad_ll = not np.isfinite(bm["log_loss"]) or cm["log_loss"] <= bm["log_loss"] * 1.03
    not_bad_br = not np.isfinite(bm["brier"]) or cm["brier"] <= bm["brier"] * 1.03
    accepted = bool((improve_ll or improve_br) and not_bad_ll and not_bad_br)

    calibrator = None
    if len(va) >= 90 and va["result_rank"].notna().sum() >= 90:
        y = (pd.to_numeric(va["result_rank"], errors="coerce") == 1).astype(int).to_numpy()
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(cand_p), y)
        except Exception:
            calibrator = None

    trained_all = None
    if accepted:
        try:
            trained_all = train_from_history(hist)
        except Exception:
            accepted = False

    metrics = {
        "races": len(races),
        "validation_races": len(valid_races),
        "baseline_log_loss": bm.get("log_loss", np.nan),
        "candidate_log_loss": cm.get("log_loss", np.nan),
        "baseline_brier": bm.get("brier", np.nan),
        "candidate_brier": cm.get("brier", np.nan),
    }
    if accepted:
        note = "候補モデルが旧V3基準モデルを検証データで上回ったため採用"
    else:
        note = "候補モデルが旧V3基準モデルを十分上回らなかったため不採用"
    return LearningBundle(accepted, trained_all, calibrator if accepted else None, metrics, note)


def payoff_map(result: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    payoff = result.get("payoff", {}) or {}
    out: Dict[str, Dict[str, int]] = {k: {} for k in BET_SPECS}
    source_keys = {
        "win": "win_all", "place": "place_show_all", "exacta": "exacta_all",
        "quinella": "quinella_all", "quinella_place": "quinella_place_all",
        "trifecta": "trifecta_all", "trio": "trio_all",
    }
    for bt, key in source_keys.items():
        vals = payoff.get(key)
        if vals is None:
            vals = payoff.get(key.replace("_all", ""))
        if isinstance(vals, dict):
            vals = [vals]
        for item in vals or []:
            combo = str(item.get("result", ""))
            if combo:
                out[bt][combo] = safe_int(item.get("payoff"), 0)
    return out
