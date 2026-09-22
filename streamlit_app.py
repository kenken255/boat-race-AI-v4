from __future__ import annotations

import hashlib
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    BET_SPECS, DEFAULT_BET_LABELS, DISPLAY_COLS, EDITABLE_COLS, LABEL_TO_BET,
    MODEL_FEATURES, NAME_TO_STADIUM, STADIUMS, LearningBundle,
    DEFAULT_FACTOR_WEIGHTS, FACTOR_LABELS, WEIGHT_PRESETS, combine_model_and_manual_strength,
    apply_conditions, blend_probability_sets, build_all_bet_probabilities,
    build_bet_strategies, calibrate_win_probs, data_quality, enrich_market,
    evaluate_and_train_history, fetch_race_bundle, fetch_race_result, fetch_stadiums,
    heuristic_strength, history_template, jst_today, market_view, model_strength,
    normalize_race_data, now_jst_iso, parse_odds_dict, payoff_map,
    portfolio_metrics, probability_temperature, reliability_factor, safe_float,
    simulate_races, softmax_strength, strategy_view, ticket_type_guide, train_from_history,
)
from storage import (
    client_from_secrets, dataframe_records, insert_model_run, insert_snapshot,
    load_settled_snapshots, load_unsettled, settle_snapshot, snapshots_to_history,
    create_backtest_job, insert_backtest_targets, list_backtest_jobs, get_backtest_job,
    reset_backtest_errors, refresh_backtest_job_counts,
)
from backtest import discover_targets, run_backtest_batch

st.set_page_config(page_title="BOAT RACE AI v5", page_icon="🚤", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
.block-container {max-width: 760px; padding-top: 0.7rem; padding-bottom: 4rem;}
[data-testid="stSidebar"] {display:none;}
[data-testid="stMetric"] {background:#f7f9fc; border:1px solid #e7ebf0; padding:0.45rem; border-radius:0.8rem;}
.boat-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin:8px 0 14px;}
.boat-card {border:1px solid #e4e8ee; border-radius:14px; padding:12px; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04);}
.boat-title {font-weight:700; font-size:1.02rem; margin-bottom:4px;}
.boat-prob {font-size:1.45rem; font-weight:800; margin:2px 0;}
.guide-card {border:1px solid #e4e8ee; border-radius:14px; padding:12px; margin:8px 0; background:#fff;}
.tag {display:inline-block; padding:2px 8px; border-radius:999px; background:#eef4ff; font-size:.78rem; margin-right:4px;}
.small-muted {font-size:.82rem; color:#667085;}
div[data-testid="stDataFrame"] {font-size:.82rem;}
button[kind="primary"] {min-height:3rem; font-weight:700;}
@media (max-width: 640px) {
  .block-container {padding-left:.75rem; padding-right:.75rem;}
  h1 {font-size:1.55rem !important;}
  h2 {font-size:1.25rem !important;}
  h3 {font-size:1.05rem !important;}
}
</style>
""",
    unsafe_allow_html=True,
)


def get_secret_pin() -> str:
    try:
        return str(st.secrets["app"].get("pin", ""))
    except Exception:
        return ""


def auth_gate() -> bool:
    pin = get_secret_pin()
    if not pin:
        return True
    if st.session_state.get("pin_ok"):
        return True
    st.title("🚤 BOAT RACE AI v5")
    st.caption("このアプリはPINで保護されています。")
    entered = st.text_input("PIN", type="password")
    if st.button("開く", type="primary", use_container_width=True):
        if entered == pin:
            st.session_state["pin_ok"] = True
            st.rerun()
        else:
            st.error("PINが違います。")
    return False


if not auth_gate():
    st.stop()


@st.cache_data(ttl=90, show_spinner=False)
def cached_stadiums(d: date):
    return fetch_stadiums(d)


@st.cache_data(ttl=45, show_spinner=False)
def cached_race_bundle(d: date, stadium: int, race: int, include_before: bool, bet_types: tuple[str, ...]):
    return fetch_race_bundle(d, stadium, race, include_before, bet_types)


def db_client():
    return client_from_secrets(st.secrets)


def load_learning(client) -> tuple[LearningBundle, pd.DataFrame, List[Dict[str, Any]]]:
    if client is None:
        return LearningBundle(False, None, None, {}, "Supabase未設定"), pd.DataFrame(), []
    try:
        rows = load_settled_snapshots(client, limit=1000)
        hist = snapshots_to_history(rows, MODEL_FEATURES)
        bundle = evaluate_and_train_history(hist)
        return bundle, hist, rows
    except Exception as e:
        return LearningBundle(False, None, None, {}, f"DB学習読込失敗: {e}"), pd.DataFrame(), []


def ensure_learning_state(client):
    if "learning_bundle" not in st.session_state:
        lb, hist, rows = load_learning(client)
        st.session_state["learning_bundle"] = lb
        st.session_state["learning_hist"] = hist
        st.session_state["settled_rows"] = rows


def json_num(v):
    try:
        x = float(v)
        return None if not np.isfinite(x) else x
    except Exception:
        return None


def compact_pct(v):
    return "–" if v is None or not np.isfinite(v) else f"{v*100:.1f}%"


def compact_num(v, nd=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def learning_report(rows: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    race_recs, course_recs = [], []
    for row in rows:
        result = row.get("result_json") or {}
        rank_map = {}
        for x in result.get("result", []) or []:
            try:
                rank_map[int(x.get("boat"))] = int(x.get("rank"))
            except Exception:
                pass
        boats = row.get("boats") or []
        probs, ys = [], []
        for b in boats:
            p = b.get("win_prob")
            try:
                p = float(p)
            except Exception:
                p = np.nan
            if np.isfinite(p):
                y = 1.0 if rank_map.get(int(b.get("boat_no", 0))) == 1 else 0.0
                probs.append(p); ys.append(y)
                course_recs.append({
                    "course": b.get("course"), "brier": (p-y)**2,
                    "stadium": row.get("stadium_name"),
                })
        if probs and sum(ys) == 1:
            p = np.asarray(probs); y = np.asarray(ys)
            p = p / p.sum()
            brier = float(np.mean((p-y)**2))
            winner_p = float(p[np.argmax(y)])
            weather = row.get("weather") or {}
            rough = safe_float(weather.get("wind_speed"),0)/8 + safe_float(weather.get("wave_height"),0)/15
            race_recs.append({
                "race_id": row.get("race_id"), "stadium": row.get("stadium_name"),
                "brier": brier, "log_loss": -math.log(max(winner_p,1e-8)),
                "condition": "荒れ気味" if rough >= 0.7 else "通常",
            })
    race_df = pd.DataFrame(race_recs)
    course_df = pd.DataFrame(course_recs)
    if race_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    by_stadium = race_df.groupby("stadium", as_index=False).agg(レース数=("race_id","count"), Brier=("brier","mean"), LogLoss=("log_loss","mean")).sort_values("Brier", ascending=False)
    by_cond = race_df.groupby("condition", as_index=False).agg(レース数=("race_id","count"), Brier=("brier","mean"), LogLoss=("log_loss","mean")).sort_values("Brier", ascending=False)
    by_course = course_df.groupby("course", as_index=False).agg(サンプル=("brier","count"), Brier=("brier","mean")).sort_values("Brier", ascending=False) if not course_df.empty else pd.DataFrame()
    return by_stadium, by_cond, by_course


def ticket_roi_report(client) -> pd.DataFrame:
    if client is None:
        return pd.DataFrame()
    try:
        res = client.table("ticket_predictions").select("bet_type,odds,result_hit,payoff,consensus_class").limit(10000).execute()
        rows = getattr(res, "data", None) or []
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["result_hit"].notna()].copy()
    if df.empty:
        return df
    df["stake"] = 100
    df["return"] = pd.to_numeric(df["payoff"], errors="coerce").fillna(0)
    label = {k:v["label"] for k,v in BET_SPECS.items()}
    g = df.groupby(["bet_type","consensus_class"], dropna=False).agg(件数=("stake","count"), 的中数=("result_hit","sum"), 投入=("stake","sum"), 払戻=("return","sum")).reset_index()
    g["回収率(%)"] = (g["払戻"] / g["投入"] * 100).round(1)
    g["的中率(%)"] = (g["的中数"] / g["件数"] * 100).round(1)
    g["券種"] = g["bet_type"].map(label).fillna(g["bet_type"])
    return g[["券種","consensus_class","件数","的中率(%)","回収率(%)"]].sort_values(["券種","回収率(%)"], ascending=[True,False])


client, db_status = db_client()
ensure_learning_state(client)
learning_bundle: LearningBundle = st.session_state["learning_bundle"]
learning_hist: pd.DataFrame = st.session_state.get("learning_hist", pd.DataFrame())
settled_rows: List[Dict[str, Any]] = st.session_state.get("settled_rows", [])

st.title("🚤 BOAT RACE AI v5")
st.caption("Web自動取得 + 市場×AI評価 + 仮想レース + 過去一括バックテスト/自己改善。自動投票は行いません。")

main_tab, learn_tab, backtest_tab, info_tab = st.tabs(["🎯 予想", "🧠 学習", "⏪ 一括学習", "⚙️ 情報"])

with main_tab:
    with st.expander("🏁 レース設定", expanded=True):
        target_date = st.date_input("日付", value=jst_today(), key="target_date")
        if st.button("開催場を自動検出", use_container_width=True):
            try:
                with st.spinner("開催場を確認中…"):
                    sd = cached_stadiums(target_date)
                names = [k for k in sd.keys() if k in NAME_TO_STADIUM]
                st.session_state["active_names"] = names
                st.success(f"{len(names)}場を検出" if names else "開催場を検出できませんでした")
            except Exception as e:
                st.error(f"開催場取得に失敗: {e}")
        choices = st.session_state.get("active_names") or list(NAME_TO_STADIUM.keys())
        stadium_name = st.selectbox("場", choices, key="stadium_name")
        race_no = st.selectbox("レース", list(range(1,13)), format_func=lambda x:f"{x}R", key="race_no")
        include_before = st.toggle("展示・直前情報も取得", value=True)
        selected_labels = st.multiselect(
            "分析する券種", [BET_SPECS[k]["label"] for k in BET_SPECS],
            default=DEFAULT_BET_LABELS,
        )
        selected_types = tuple(LABEL_TO_BET[x] for x in selected_labels)
        fetch_clicked = st.button("出走表・オッズを取得", type="primary", use_container_width=True, disabled=not selected_types)

    with st.expander("💰 資金・シミュレーション設定", expanded=False):
        bankroll = st.number_input("分析用資金（円）", min_value=1000, max_value=10_000_000, value=10_000, step=1000)
        unit = st.selectbox("購入単位", [100,200,500,1000], index=0)
        n_sims = st.select_slider("仮想レース回数", options=[2000,5000,10000,20000,50000], value=10000)
        sim_weight = st.slider("統合確率に占める仮想レース比率", 0.0, 0.8, 0.40, 0.05)
        uncertainty_scale = st.slider("仮想レースの不確実性", 0.6, 1.6, 1.0, 0.1)
        st.caption("既定では解析モデル60% + 仮想レース40%。仮想レースはSTぶれ・当日不確実性を反復して着順を生成します。")

    with st.expander("🎚️ 要素の重みづけ", expanded=False):
        st.caption("100%が標準。0%でその要素を無視、200%で標準の2倍の影響にします。過去一括学習にも同じ設定を固定保存できます。")
        preset_name = st.selectbox("プリセット", list(WEIGHT_PRESETS.keys()), key="weight-preset-name")
        if st.button("このプリセットをバーに反映", use_container_width=True, key="apply-weight-preset"):
            for k, v in WEIGHT_PRESETS[preset_name].items():
                st.session_state[f"factor-weight-{k}"] = int(round(float(v) * 100))
            st.rerun()
        feature_weights = {}
        for k, label in FACTOR_LABELS.items():
            default_pct = int(round(DEFAULT_FACTOR_WEIGHTS[k] * 100))
            current = int(st.session_state.get(f"factor-weight-{k}", default_pct))
            pct = st.slider(label, 0, 200, current, 5, key=f"factor-weight-{k}")
            feature_weights[k] = pct / 100.0
        manual_weight_mix = st.slider(
            "自己学習モデルがある場合の手動重み反映度", 0, 100,
            int(st.session_state.get("manual-weight-mix-pct", 30)), 5, key="manual-weight-mix-pct"
        ) / 100.0
        st.caption("自己学習モデル未採用時は、この重みモデルを100%使用します。採用済みの場合は学習モデルと手動重みモデルを上の比率で混合します。")

    with st.expander("📄 任意：外部CSV学習", expanded=False):
        hist_file = st.file_uploader("過去CSV", type=["csv"])
        st.download_button("CSVテンプレート", history_template().to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "boatrace_history_template.csv", "text/csv", use_container_width=True)

    csv_trained = None
    csv_note = ""
    if hist_file is not None:
        try:
            hist = pd.read_csv(hist_file)
            csv_trained = train_from_history(hist)
            csv_note = f"CSV学習 {len(hist):,}行"
        except Exception as e:
            st.warning(f"CSV学習は使いません: {e}")

    if fetch_clicked:
        stadium = NAME_TO_STADIUM[stadium_name]
        try:
            with st.spinner("出走表・直前情報・オッズを取得中…"):
                race_info, before, odds_raw, odds_updates, odds_errors = cached_race_bundle(target_date, stadium, race_no, include_before, selected_types)
            base_df, weather = normalize_race_data(race_info, before)
            if len(base_df) != 6:
                raise RuntimeError("6艇のデータを取得できませんでした")
            st.session_state["race_bundle"] = {
                "base_df":base_df, "weather":weather,
                "odds_frames":{bt:parse_odds_dict(odds_raw.get(bt,{}),bt) for bt in selected_types},
                "odds_updates":odds_updates, "odds_errors":odds_errors,
                "selected_types":selected_types, "date":str(target_date), "stadium":stadium_name,
                "stadium_code":stadium, "race_no":race_no,
            }
            st.session_state["race_key"] = f"{target_date:%Y%m%d}-{stadium:02d}-{race_no:02d}"
        except Exception as e:
            st.error(f"Web取得に失敗しました: {e}")
            st.info("直前情報がまだ公開されていない場合は『展示・直前情報も取得』をOFFにして再取得できます。")

    bundle = st.session_state.get("race_bundle")
    if bundle:
        st.markdown(f"## {bundle['stadium']} {bundle['race_no']}R")
        st.caption(f"{bundle['date']} / オッズ更新: " + "・".join(f"{BET_SPECS[k]['label']} {v}" for k,v in bundle.get('odds_updates',{}).items() if v))
        if bundle.get("odds_errors"):
            st.warning("一部オッズ取得失敗: " + " / ".join(BET_SPECS[k]["label"] for k in bundle["odds_errors"]))

        base_df = bundle["base_df"].copy()
        weather = bundle.get("weather") or {}
        race_key = st.session_state.get("race_key", "race")

        with st.expander("🌬️ 当日コンディションを確認・修正", expanded=False):
            c1,c2 = st.columns(2)
            wind_speed = c1.number_input("風速 m/s", 0.0, 30.0, float(safe_float(weather.get("wind_speed"),0.0)), 0.1, key=f"wind-{race_key}")
            wave_height = c2.number_input("波高 cm", 0.0, 100.0, float(safe_float(weather.get("wave_height"),0.0)), 1.0, key=f"wave-{race_key}")
            c3,c4 = st.columns(2)
            temperature = c3.number_input("気温 ℃", -10.0, 50.0, float(safe_float(weather.get("temperature"),20.0)), 0.1, key=f"temp-{race_key}")
            water_temperature = c4.number_input("水温 ℃", 0.0, 40.0, float(safe_float(weather.get("water_temperature"),20.0)), 0.1, key=f"water-{race_key}")
            st.markdown("**艇ごとの直前情報**")
            for i,row in base_df.iterrows():
                bno = int(row["boat_no"]); name = str(row.get("name", ""))
                with st.expander(f"{bno}号艇 {name}"):
                    x1,x2 = st.columns(2)
                    base_df.at[i,"course"] = x1.number_input("進入",1,6,int(safe_float(row.get("course"),bno)),1,key=f"course-{race_key}-{bno}")
                    base_df.at[i,"display_time"] = x2.number_input("展示タイム",5.0,9.0,float(safe_float(row.get("display_time"),6.8)),0.01,key=f"disp-{race_key}-{bno}")
                    x3,x4 = st.columns(2)
                    base_df.at[i,"start_display_st"] = x3.number_input("展示ST",-0.5,1.0,float(safe_float(row.get("start_display_st"),0.17)),0.01,key=f"dst-{race_key}-{bno}")
                    base_df.at[i,"tilt"] = x4.number_input("チルト",-0.5,3.0,float(safe_float(row.get("tilt"),-0.5)),0.5,key=f"tilt-{race_key}-{bno}")

        manual_weather = {"wind_speed":wind_speed,"wave_height":wave_height,"temperature":temperature,"water_temperature":water_temperature}
        df = apply_conditions(base_df, manual_weather)

        # Model selection: validated DB model > uploaded CSV > V3 heuristic.
        calibrator = None
        validation_gain = 0.0
        if learning_bundle.accepted and learning_bundle.trained is not None:
            trained = learning_bundle.trained
            calibrator = learning_bundle.calibrator
            model_kind = f"自己学習モデル ({int(learning_bundle.metrics.get('races',0))}レース)"
            bll = learning_bundle.metrics.get("baseline_log_loss", np.nan)
            cll = learning_bundle.metrics.get("candidate_log_loss", np.nan)
            if np.isfinite(bll) and bll > 0 and np.isfinite(cll):
                validation_gain = float(np.clip((bll-cll)/bll,0,0.12))
        elif csv_trained is not None:
            trained = csv_trained
            model_kind = csv_note
        else:
            trained = None
            model_kind = "旧V3基準ヒューリスティック"

        strengths = combine_model_and_manual_strength(
            df, trained=trained, weights=feature_weights, manual_mix=manual_weight_mix
        )
        temp_factor = probability_temperature(df)
        raw_win = softmax_strength(strengths, temp_factor)
        calibrated_win = calibrate_win_probs(raw_win, calibrator)
        if calibrator is not None:
            analytical = build_all_bet_probabilities(df["boat_no"].astype(int).tolist(), np.log(np.clip(calibrated_win,1e-8,1.0)), 1.0)
        else:
            analytical = build_all_bet_probabilities(df["boat_no"].astype(int).tolist(), strengths, temp_factor)

        seed = int(hashlib.sha256(race_key.encode()).hexdigest()[:8],16)
        simulated = simulate_races(df, calibrated_win, int(n_sims), seed=seed, uncertainty_scale=float(uncertainty_scale))
        all_probs = blend_probability_sets(analytical, simulated, float(sim_weight))

        win = all_probs["win"].copy()
        win["boat_no"] = win["combo"].astype(int)
        df = df.merge(win[["boat_no","analytic_prob","sim_prob","prob"]].rename(columns={"prob":"win_prob","analytic_prob":"analytic_win_prob","sim_prob":"sim_win_prob"}), on="boat_no", how="left")

        reliability = reliability_factor(df, trained is not None, validation_gain)
        selected_types_now = tuple(bundle.get("selected_types",("trifecta",)))
        markets, strategies = {}, {}
        for bt in selected_types_now:
            odds_df = bundle.get("odds_frames",{}).get(bt,pd.DataFrame())
            markets[bt] = enrich_market(all_probs[bt], odds_df, reliability, BET_SPECS[bt]["exclusive"])
            strategies[bt] = build_bet_strategies(markets[bt], int(bankroll), int(unit))

        q = data_quality(df)
        a,b = st.columns(2)
        a.metric("データ充足度", f"{q*100:.0f}%")
        b.metric("保守補正", f"{reliability:.2f}")
        c,d = st.columns(2)
        c.metric("仮想レース", f"{int(n_sims):,}回")
        d.metric("荒天平坦化", f"×{temp_factor:.2f}")
        st.caption(
            f"モデル: {model_kind} / 統合: 解析{(1-sim_weight)*100:.0f}% + 仮想レース{sim_weight*100:.0f}% / "
            f"手動重み反映 {manual_weight_mix*100:.0f}%"
        )

        st.markdown("### 🚤 各艇評価")
        cards = ['<div class="boat-grid">']
        for _,r in df.sort_values("boat_no").iterrows():
            cards.append(
                f'<div class="boat-card"><div class="boat-title">{int(r.boat_no)}号艇 {r.get("name","")}</div>'
                f'<div class="boat-prob">{r.win_prob*100:.1f}%</div>'
                f'<div class="small-muted">統合1着率</div>'
                f'<div><span class="tag">解析 {r.analytic_win_prob*100:.1f}%</span><span class="tag">仮想 {r.sim_win_prob*100:.1f}%</span></div>'
                f'<div class="small-muted">進入 {int(r.course)} / 全国勝率 {compact_num(r.get("global_win_pt"))} / 平均ST {compact_num(r.get("aveST"))}</div>'
                f'<div class="small-muted">モーター2連 {compact_num(r.get("motor_in2nd"),1)} / 展示 {compact_num(r.get("display_time"))}</div></div>'
            )
        cards.append('</div>')
        st.markdown("".join(cards), unsafe_allow_html=True)
        with st.expander("各艇の詳細表"):
            cols = [c for c in DISPLAY_COLS if c in df.columns] + ["analytic_win_prob","sim_win_prob","win_prob"]
            view = df[cols].copy()
            for c in ["analytic_win_prob","sim_win_prob","win_prob"]:
                view[c] = (view[c]*100).round(1)
            st.dataframe(view.rename(columns={"analytic_win_prob":"解析1着%","sim_win_prob":"仮想1着%","win_prob":"統合1着%"}), use_container_width=True, hide_index=True)

        st.markdown("### 🧭 券種ナビ")
        guide = ticket_type_guide(markets, strategies)
        for _,g in guide.iterrows():
            st.markdown(
                f'<div class="guide-card"><b>{g["券種"]}</b>　<span class="tag">{g["向き"]}</span><br>'
                f'<span class="small-muted">最大保守EV {g["最大保守EV"] if pd.notna(g["最大保守EV"]) else "–"} / '
                f'両者一致 {int(g["両者一致"])}点 / AI過小評価 {int(g["AI過小評価"])}点 / 採算候補 {int(g["採算候補"])}点</span></div>',
                unsafe_allow_html=True,
            )
        with st.expander("券種ナビの詳細表"):
            st.dataframe(guide, use_container_width=True, hide_index=True)

        st.markdown("### 🧾 買い方アシスト")
        active_label = st.selectbox("券種", [BET_SPECS[x]["label"] for x in selected_types_now], key=f"assist-bt-{race_key}")
        active_bt = LABEL_TO_BET[active_label]
        strategy_label = st.selectbox(
            "評価軸",
            ["🤝 両者一致・手堅さ", "🔎 AI優位・過小評価", "⚖️ 採算×リスク", "⚠️ 市場優位・AI慎重", "⛔ 見送り判断"],
            key=f"assist-strat-{race_key}",
        )
        spec = BET_SPECS[active_bt]
        ss = strategies[active_bt]
        if strategy_label.startswith("🤝"):
            bets = ss["safe_consensus"]
            st.info("市場もAIも上位評価している買い目。人気側であることを前提に、極端に割高な候補は除外します。")
        elif strategy_label.startswith("🔎"):
            bets = ss["undervalued_ai"]
            st.info("AIは高く評価する一方、市場評価が相対的に低い買い目。妙味候補ですがモデル誤差にも注意します。")
        elif strategy_label.startswith("⚖️"):
            bets = ss["balanced"]
            st.info("保守EVと的中確率を両方見て、1点集中を抑えた候補です。")
        elif strategy_label.startswith("⚠️"):
            bets = ss["market_caution"]
            st.warning("市場では人気ですがAIは相対的に低評価。『人気だから買う』を避けるための確認欄です。")
        else:
            bets = pd.DataFrame()
            reasons=[]
            if q < 0.72: reasons.append("データ欠損が多い")
            if trained is None: reasons.append("検証済み自己学習モデルがまだない")
            if ss["balanced"].empty and ss["safe_consensus"].empty and ss["undervalued_ai"].empty: reasons.append("有力候補がない")
            if temp_factor >= 1.22: reasons.append("風・波による不確実性が高い")
            if bundle.get("odds_frames",{}).get(active_bt,pd.DataFrame()).empty: reasons.append("オッズ未取得")
            if reasons:
                st.warning("見送り寄り: " + " / ".join(reasons))
            else:
                st.success("明確な見送り条件は少なめ。ただし購入を推奨・保証するものではありません。")
        if not strategy_label.startswith("⛔"):
            if bets.empty:
                st.info("この条件を満たす買い目はありません。")
            else:
                if "stake_yen" in bets.columns:
                    m = portfolio_metrics(bets, spec["exclusive"])
                    x,y = st.columns(2); x.metric("参考投入", f"{m['stake']:,.0f}円"); y.metric("保守的中率", f"{m['hit_prob']*100:.1f}%")
                    st.metric("保守期待ROI", f"{m['expected_roi']*100:+.1f}%")
                st.dataframe(strategy_view(bets, active_label, spec["range_odds"]), use_container_width=True, hide_index=True)

        with st.expander("📊 券種別・全候補（詳細）"):
            detail_label = st.selectbox("一覧券種", [BET_SPECS[x]["label"] for x in selected_types_now], key=f"detail-{race_key}")
            detail_bt = LABEL_TO_BET[detail_label]
            detail = markets[detail_bt]
            n = st.slider("表示件数",1,max(1,len(detail)),min(30,max(1,len(detail))), key=f"topn-{race_key}")
            st.dataframe(market_view(detail, detail_bt, n), use_container_width=True, hide_index=True)

        if client is not None:
            if st.button("💾 この予想を学習履歴に保存", use_container_width=True, key=f"save-{race_key}"):
                try:
                    boats_to_save = df.copy()
                    snapshot = {
                        "race_id": race_key,
                        "snapshot_at": now_jst_iso(),
                        "race_date": bundle["date"],
                        "stadium_code": int(bundle["stadium_code"]),
                        "stadium_name": bundle["stadium"],
                        "race_no": int(bundle["race_no"]),
                        "model_kind": model_kind,
                        "data_quality": float(q),
                        "simulation_count": int(n_sims),
                        "simulation_weight": float(sim_weight),
                        "reliability": float(reliability),
                        "feature_weights": feature_weights,
                        "manual_weight_mix": float(manual_weight_mix),
                        "weather": manual_weather,
                        "boats": dataframe_records(boats_to_save),
                    }
                    ticket_rows=[]
                    for bt,m in markets.items():
                        for _,r in m.dropna(subset=["odds"]).iterrows():
                            ticket_rows.append({
                                "bet_type":bt, "combo":str(r["combo"]), "odds":json_num(r.get("odds")),
                                "model_prob":json_num(r.get("prob")), "analytic_prob":json_num(r.get("analytic_prob")),
                                "sim_prob":json_num(r.get("sim_prob")), "market_prob_proxy":json_num(r.get("market_prob_proxy")),
                                "conservative_prob":json_num(r.get("conservative_prob")), "conservative_ev":json_num(r.get("conservative_EV")),
                                "consensus_class":str(r.get("consensus_class","")),
                            })
                    sid = insert_snapshot(client, snapshot, ticket_rows)
                    st.success(f"保存しました。ID: {sid[:8]}… レース終了後に『学習』タブで結果を自動照合できます。")
                except Exception as e:
                    st.error(f"保存失敗: {e}")
        else:
            st.caption("自己改善を使うにはSupabaseを設定してください。予想機能だけなら未設定でも使えます。")
    else:
        st.info("上の『レース設定』から日付・場・Rを選び、出走表・オッズを取得してください。")

with learn_tab:
    st.markdown("## 🧠 予想→結果→自己改善")
    if client is None:
        st.warning("Supabaseが未設定です。『⚙️ 情報』タブと同梱 SETUP.md の手順で設定すると、履歴保存・結果照合・自己学習が有効になります。")
    else:
        st.success("Supabase接続設定あり")
        c1,c2 = st.columns(2)
        c1.metric("学習済みレース", int(learning_bundle.metrics.get("races",0) or 0))
        c2.metric("現在のモデル", "採用" if learning_bundle.accepted else "基準モデル")
        st.caption(learning_bundle.note)
        if learning_bundle.metrics:
            mm = learning_bundle.metrics
            comp = pd.DataFrame([
                {"指標":"Log Loss（小さいほど良い）","旧V3基準":mm.get("baseline_log_loss"),"新候補":mm.get("candidate_log_loss")},
                {"指標":"Brier（小さいほど良い）","旧V3基準":mm.get("baseline_brier"),"新候補":mm.get("candidate_brier")},
            ])
            st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
            st.caption("新候補は検証用の直近レースで旧V3基準を改善し、もう一方の指標も大きく悪化しない場合だけ採用します。")

        if st.button("🔄 終了レースの結果を自動照合", type="primary", use_container_width=True):
            pending = load_unsettled(client, limit=50)
            done=0; skipped=0; result_cache={}
            prog = st.progress(0.0)
            for i,row in enumerate(pending):
                try:
                    d = date.fromisoformat(str(row["race_date"]))
                    if d > jst_today():
                        skipped += 1; continue
                    key=(str(d),int(row["stadium_code"]),int(row["race_no"]))
                    if key not in result_cache:
                        result_cache[key]=fetch_race_result(d,key[1],key[2])
                    result=result_cache[key]
                    if not result.get("result"):
                        skipped += 1; continue
                    ranks={int(x["boat"]):int(x["rank"]) for x in result.get("result",[]) if str(x.get("rank","")).isdigit()}
                    settle_snapshot(client, row["id"], result, ranks, payoff_map(result))
                    done += 1
                except Exception:
                    skipped += 1
                prog.progress((i+1)/max(1,len(pending)))
            st.success(f"照合完了: {done}件 / 未確定・取得不可 {skipped}件")
            lb,hist,rows = load_learning(client)
            st.session_state["learning_bundle"]=lb; st.session_state["learning_hist"]=hist; st.session_state["settled_rows"]=rows
            try: insert_model_run(client, lb.metrics, lb.accepted, lb.note)
            except Exception: pass

        if st.button("🧪 学習モデルを再評価", use_container_width=True):
            lb,hist,rows = load_learning(client)
            st.session_state["learning_bundle"]=lb; st.session_state["learning_hist"]=hist; st.session_state["settled_rows"]=rows
            try: insert_model_run(client, lb.metrics, lb.accepted, lb.note)
            except Exception: pass
            st.success("再評価しました。次回の予測から採用判定を反映します。")

        st.markdown("### 誤差の傾向")
        by_stadium, by_cond, by_course = learning_report(settled_rows)
        if by_stadium.empty:
            st.info("結果照合済みの予想が蓄積されると、苦手な場・コース・水面条件を表示します。")
        else:
            with st.expander("場別（Brierが大きい順）", expanded=True): st.dataframe(by_stadium.round(4),use_container_width=True,hide_index=True)
            with st.expander("コンディション別"): st.dataframe(by_cond.round(4),use_container_width=True,hide_index=True)
            with st.expander("進入コース別"): st.dataframe(by_course.round(4),use_container_width=True,hide_index=True)

        st.markdown("### 券種・評価タイプ別の実績")
        roi = ticket_roi_report(client)
        if roi.empty:
            st.info("払戻まで照合されると、『両者一致』『AI過小評価』などの分類別に100円均等購入した場合の参考回収率を表示します。")
        else:
            st.dataframe(roi, use_container_width=True, hide_index=True)
            st.caption("回収率は各候補を100円ずつ買った仮想集計であり、実際の購入履歴ではありません。")

with backtest_tab:
    st.markdown("## ⏪ 過去レース一括学習")
    st.caption("過去のレースを時系列順に再生し、その時点より前の結果だけで学習するウォークフォワード方式です。各レース終了ごとにDBへ保存するため、中断しても続きから再開できます。")
    if client is None:
        st.warning("一括学習にはSupabase設定が必要です。SETUP.mdのV5手順でテーブルを作成してください。")
    else:
        with st.expander("➕ 新しい一括学習ジョブ", expanded=True):
            c1, c2 = st.columns(2)
            bt_start = c1.date_input("開始日", value=jst_today()-timedelta(days=30), max_value=jst_today()-timedelta(days=1), key="bt-start")
            bt_end = c2.date_input("終了日", value=jst_today()-timedelta(days=1), max_value=jst_today()-timedelta(days=1), key="bt-end")
            selected_stadium_names = st.multiselect("対象場", list(NAME_TO_STADIUM.keys()), default=list(NAME_TO_STADIUM.keys()), key="bt-stadiums")
            bt_labels = st.multiselect(
                "検証する券種", [BET_SPECS[k]["label"] for k in BET_SPECS],
                default=DEFAULT_BET_LABELS, key="bt-bets"
            )
            c3, c4 = st.columns(2)
            bt_max = c3.selectbox("最大レース数", [100,300,500,1000,3000,5000], index=2, key="bt-max")
            bt_sims = c4.selectbox("1レースの仮想レース回数", [500,1000,2000,5000,10000], index=2, key="bt-sims")
            c5, c6 = st.columns(2)
            bt_sim_weight = c5.slider("仮想レース比率", 0.0, 0.8, float(sim_weight), 0.05, key="bt-sim-weight")
            bt_uncertainty = c6.slider("不確実性", 0.6, 1.6, float(uncertainty_scale), 0.1, key="bt-uncertainty")
            bt_retrain = st.selectbox("再学習チェック間隔（目安）", [25,50,100,200], index=1, key="bt-retrain")
            bt_before = st.toggle("過去の展示・直前情報を使用", value=True, key="bt-before")
            bt_odds = st.toggle("過去オッズも取得して買い方を検証", value=True, key="bt-odds")
            st.markdown("**このジョブに固定する要素重み**")
            st.caption("現在の『予想』タブのバー設定をコピーします。ジョブ作成後にバーを変えても、このジョブの条件は変わりません。")
            w_summary = " / ".join(f"{FACTOR_LABELS[k]} {feature_weights[k]*100:.0f}%" for k in feature_weights)
            st.caption(w_summary)
            if st.button("一括学習ジョブを作成", type="primary", use_container_width=True, key="create-bt-job"):
                if bt_start > bt_end:
                    st.error("開始日は終了日以前にしてください。")
                elif not selected_stadium_names or not bt_labels:
                    st.error("対象場と券種を1つ以上選んでください。")
                else:
                    try:
                        stadium_codes = [NAME_TO_STADIUM[x] for x in selected_stadium_names]
                        with st.spinner("開催日を確認して対象レースを作成中…"):
                            targets, warnings = discover_targets(bt_start, bt_end, stadium_codes, int(bt_max))
                        if not targets:
                            st.error("対象レースを検出できませんでした。期間・場を確認してください。")
                        else:
                            config = {
                                "start_date": bt_start.isoformat(), "end_date": bt_end.isoformat(),
                                "stadium_codes": stadium_codes,
                                "bet_types": [LABEL_TO_BET[x] for x in bt_labels],
                                "include_before": bool(bt_before), "include_odds": bool(bt_odds),
                                "simulation_count": int(bt_sims), "simulation_weight": float(bt_sim_weight),
                                "uncertainty_scale": float(bt_uncertainty), "retrain_every": int(bt_retrain),
                                "max_races": int(bt_max), "feature_weights": feature_weights,
                                "manual_weight_mix": float(manual_weight_mix),
                            }
                            job_id = create_backtest_job(client, config)
                            insert_backtest_targets(client, job_id, targets)
                            st.session_state["active_backtest_job"] = job_id
                            st.success(f"{len(targets):,}レースのジョブを作成しました。ID: {job_id[:8]}…")
                            if warnings:
                                st.warning(f"開催確認で{len(warnings)}件の警告がありました。ジョブは作成済みです。")
                    except Exception as e:
                        st.error(f"ジョブ作成失敗: {e}")

        jobs = list_backtest_jobs(client, limit=20)
        if not jobs:
            st.info("まだ一括学習ジョブがありません。上で作成してください。")
        else:
            job_map = {str(j["id"]): j for j in jobs}
            default_job = st.session_state.get("active_backtest_job")
            ids = list(job_map.keys())
            index = ids.index(default_job) if default_job in ids else 0
            selected_job_id = st.selectbox(
                "ジョブ", ids, index=index,
                format_func=lambda jid: f"{str(job_map[jid].get('start_date'))}〜{str(job_map[jid].get('end_date'))} / {job_map[jid].get('processed',0)}/{job_map[jid].get('total_targets',0)}R / {str(job_map[jid].get('status',''))}",
                key="bt-job-select",
            )
            job = get_backtest_job(client, selected_job_id) or job_map[selected_job_id]
            total = int(job.get("total_targets",0) or 0); processed = int(job.get("processed",0) or 0)
            succeeded = int(job.get("succeeded",0) or 0); failed = int(job.get("failed",0) or 0)
            st.progress(min(1.0, processed/max(1,total)))
            m1,m2,m3 = st.columns(3)
            m1.metric("処理", f"{processed}/{total}")
            m2.metric("成功", succeeded)
            m3.metric("失敗", failed)
            st.caption(f"状態: {job.get('status')} / {job.get('message') or ''}")
            jw = job.get("feature_weights") or {}
            if jw:
                with st.expander("このジョブの固定重み"):
                    st.write({FACTOR_LABELS.get(k,k): f"{float(v)*100:.0f}%" for k,v in jw.items()})
                    st.caption(f"自己学習モデルへの手動重み反映度: {float(job.get('manual_weight_mix',0.30))*100:.0f}%")

            st.markdown("### ▶️ 処理を進める")
            st.caption("各レースごとに保存するため、途中で画面を閉じても処理済み分は残ります。大量処理は下記GitHub Actions自動ワーカーを推奨します。")
            b1,b2,b3 = st.columns(3)
            run5 = b1.button("5R", use_container_width=True, key="bt-run5")
            run20 = b2.button("20R", use_container_width=True, key="bt-run20")
            run50 = b3.button("50R", use_container_width=True, key="bt-run50")
            requested = 5 if run5 else 20 if run20 else 50 if run50 else 0
            if requested:
                try:
                    remaining = requested; total_done=total_fail=0; msgs=[]
                    prog = st.progress(0.0)
                    while remaining > 0:
                        fresh = get_backtest_job(client, selected_job_id)
                        if not fresh or str(fresh.get("status")) == "completed":
                            break
                        step = min(10, remaining)
                        out = run_backtest_batch(client, fresh, limit=step, request_delay=0.7)
                        total_done += int(out.get("done",0)); total_fail += int(out.get("failed",0)); msgs += list(out.get("messages",[]))
                        remaining -= step
                        prog.progress(min(1.0, (requested-remaining)/requested))
                        if out.get("completed") or (out.get("done",0)+out.get("failed",0) == 0):
                            break
                    st.success(f"今回: 成功 {total_done}R / 失敗 {total_fail}R。保存済みなので続きから再開できます。")
                    if msgs:
                        with st.expander("エラー詳細"):
                            for msg in msgs[:30]: st.write(msg)
                    lb,hist,rows = load_learning(client)
                    st.session_state["learning_bundle"]=lb; st.session_state["learning_hist"]=hist; st.session_state["settled_rows"]=rows
                    st.rerun()
                except Exception as e:
                    st.error(f"一括処理失敗: {e}")

            if failed > 0 and st.button("失敗分を再試行待ちに戻す", use_container_width=True, key="bt-reset-errors"):
                try:
                    reset_backtest_errors(client, selected_job_id)
                    st.success("失敗分をpendingへ戻しました。")
                    st.rerun()
                except Exception as e:
                    st.error(f"再試行設定失敗: {e}")

            st.markdown("### 🤖 完全自動で何百Rも進める")
            st.info("ZIPには `.github/workflows/v5-backfill.yml` と `worker.py` を同梱しています。GitHub SecretsへSupabase URL/Keyを登録すると、GitHub Actionsが定期的に未処理ジョブを続きから処理できます。Streamlitを開きっぱなしにする必要はありません。")

with info_tab:
    st.markdown("## ⚙️ V5の仕組み")
    st.markdown(
        """
**市場×AI評価**
- **両者一致・手堅さ**: AI確率も市場代理確率も券種内で上位。人気でも極端に割高な候補は除外。
- **AI優位・過小評価**: AIは上位評価だが、市場評価は相対的に低い。市場比・保守EVも確認。
- **市場優位・AI慎重**: 人気だがAIは低め。人気追随を避けるための警戒欄。
- **双方低評価・見送り**: 両者とも低評価。

**仮想レース**
- 基礎の勝率を出した後、STの当日ぶれ・能力の不確実性・風/波による乱れを試行ごとに変化させます。
- 解析モデルと仮想レースを混合して最終確率にします。
- これは物理シミュレーターではなく、予測不確実性を確率へ織り込むモンテカルロモデルです。

**要素重みスライダー**
- コース、級別、全国/当地成績、モーター、ボート、平均ST、展示タイム、展示ST、F/Lを0〜200%で調整できます。
- 標準/ST重視/機力重視/選手実力重視/当地・水面重視/展示重視のプリセットがあります。
- 一括学習ジョブには作成時の重みを固定保存し、別設定との比較ができます。

**自己改善**
1. 予想時点の特徴量・オッズ・確率をSupabaseへ保存
2. 終了後に公式結果・払戻を自動取得
3. Log Loss / Brier / 回収率 / 場・コース・水面別誤差を集計
4. 履歴から候補モデルを再学習
5. 直近の検証データで旧V3基準モデルより改善した場合だけ採用
6. 十分な履歴があればIsotonic calibrationで確率校正
7. 一括学習では過去レースを時系列順に再生し、未来の結果を混ぜずにウォークフォワード検証
8. GitHub Actionsワーカーを使えば未処理ジョブを定期的に自動継続
"""
    )
    st.markdown("### 接続状態")
    st.write(f"Supabase: **{db_status}**")
    st.write(f"PIN保護: **{'ON' if get_secret_pin() else 'OFF'}**")
    st.info("詳細なセットアップ手順はZIP内の SETUP.md にあります。Supabase Secret KeyはGitHubに置かず、StreamlitのSecretsにだけ保存してください。")
