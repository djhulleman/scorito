"""
MODEL AUDIT - completed before app code

Python files read:
- world_cup_score_forecaster.py: Elo plus Poisson score model, with a
  tournament-calibrated decision-tree outcome layer.
- world_cup_exact_score_forecaster.py: scoreline-specific model on top of the
  Elo/Poisson baseline; selected DecisionTreeClassifier(max_depth=3,
  min_samples_leaf=3).
- total_goals_odds_forecaster.py: recent-form total-goals-first model using
  weighted recent internationals, Elo ratings, Poisson score modes, and implied
  win odds.
- scorito_knockout_topscorer_picker.py: Scorito knockout top-scorer model using
  current group scorers, historical WC/EK carryover priors, inferred positions,
  bracket path xG, and Monte Carlo top-five probabilities.
- scorito_poule_optimizer.py: combined Scorito advice model for match picks,
  country/bracket picks, and phase top-scorer picks.
- scorito_historical_backtest.py: walk-forward historical WC/EK backtest for
  top-scorer and knockout match-pick advice.
- bookmaker_odds_backtest.py: benchmark harness comparing the knockout
  match-pick model with common-score, random-scoreline, and bookmaker-market
  baselines when odds are available.

Structured output files read:
- All CSV outputs under outputs/, outputs_exact_score_model/,
  outputs_total_goals_model/, outputs_scorito_topscorers/,
  outputs_scorito_poule/, outputs_scorito_backtest/,
  outputs_bookmaker_backtest/.
- Hidden source/cache CSVs under .cache_world_cup/:
  international_results.csv and topscorer_player_positions.csv.

Model findings:
- Outcome/bracket model candidates:
  * Historical-only Elo/Poisson baseline:
    outcome accuracy 62.5%, exact score 7.5% on 40 completed 2026 matches.
  * Tournament-calibrated Elo/Poisson outcome layer:
    outcome accuracy 77.5%, exact score 22.5% on the same in-sample 2026 set.
  * Exact-score tree model:
    outcome accuracy 72.5%, exact score 40.0%, within-one-goal 65.0%.
  * Total-goals-first model:
    rolling outcome accuracy 37.5%, exact score 8.3%, total-goals MAE 1.54
    on 24 completed matches in its output.
  * Historical Scorito knockout match heuristic:
    5,580 points, 7.8% exact, 41.7% toto across 115 historical knockout games.
  * Common 1-0 home benchmark in bookmaker backtest:
    7,125 points, 9.6% exact, 50.4% toto across the same 115 games.
    This is a benchmark, not a bracket model, so it is displayed as risk context
    but not used to simulate a dynamic bracket.

- Top-scorer model candidates:
  * The Scorito knockout top-scorer picker is the only model with player-level
    expected Scorito top-scorer points. Historical walk-forward output:
    768 selected points, 22.3% candidate-oracle efficiency, 7.7% all-scorer
    oracle efficiency, 11.0% scoring-pick rate.
  * Current top-five expected Scorito points total: 656.6. Current top pick:
    Vinicius Junior, 184.7 expected points, 5.51 expected knockout goals,
    77.0% simulated top-five Scorito probability.
  * Phase top-scorer model in scorito_poule_optimizer gives the current
    Round-of-32 phase 203.0 expected points across four picks.

- Match expected Scorito output:
  * scorito_poule_optimizer outputs 64 match picks with total expected match
    points 2,189.6, mean 34.2, max 71.6. This is the best current expected
    Scorito match-pick output, but its historical knockout backtest remains
    weaker than the simple 1-0 benchmark.

Best models used by this app:
- Topscorers: scorito_knockout_topscorer_picker.py recommendations and
  scorito_poule_optimizer.py phase expected-points output.
- Match winners / bracket: the tournament-calibrated Elo/Poisson bracket had
  the highest pure winner accuracy, but this app now uses the exact-score tree
  for bracket picks too because Scorito rewards exact scores heavily and the
  exact-score gain is larger than the small winner-accuracy gap.
- Exact scores and bracket picks: world_cup_exact_score_forecaster.py selected
  scoreline tree because it has the highest exact-score accuracy and
  within-one-goal rate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import streamlit as st

from world_cup_score_forecaster import (
    CURRENT_WORLD_CUP_URL,
    load_current_world_cup,
    load_historical_results,
)


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache_world_cup"
PREDICTIONS_PATH = ROOT / "predictions.json"
RESULTS_CACHE_PATH = ROOT / "results_cache.json"
MODEL_REFRESH_STATE_PATH = ROOT / "model_refresh_state.json"
AMSTERDAM = ZoneInfo("Europe/Amsterdam")

BEST_TOPSCORER_PATH = ROOT / "outputs_scorito_topscorers" / "scorito_topscorer_recommendations.csv"
BEST_TOPSCORER_PHASE_PATH = ROOT / "outputs_scorito_poule" / "scorito_topscorers_by_phase.csv"
BEST_EXACT_FORECAST_PATH = ROOT / "outputs_exact_score_model" / "exact_score_model_forecast.csv"
BEST_EXACT_EVAL_PATH = ROOT / "outputs_exact_score_model" / "exact_score_model_evaluation.csv"
BEST_EXACT_KNOCKOUT_PATH = ROOT / "outputs_exact_score_model" / "exact_score_knockout_forecast.csv"
BEST_BRACKET_PATH = BEST_EXACT_KNOCKOUT_PATH
MATCH_ADVICE_PATH = ROOT / "outputs_scorito_poule" / "scorito_match_picks.csv"
COUNTRY_ADVICE_PATH = ROOT / "outputs_scorito_poule" / "scorito_country_picks.csv"
BACKTEST_OVERALL_PATH = ROOT / "outputs_scorito_backtest" / "backtest_overall_summary.csv"
BACKTEST_TOURNAMENT_PATH = ROOT / "outputs_scorito_backtest" / "backtest_tournament_summary.csv"
BOOKMAKER_SUMMARY_PATH = ROOT / "outputs_bookmaker_backtest" / "bookmaker_backtest_summary.csv"
BASELINE_EVAL_PATH = ROOT / "outputs" / "world_cup_evaluation_baseline.csv"
CALIBRATED_EVAL_PATH = ROOT / "outputs" / "world_cup_evaluation_calibrated.csv"
TOTAL_GOALS_EVAL_PATH = ROOT / "outputs_total_goals_model" / "total_goals_model_evaluation.csv"
CURRENT_SCORERS_PATH = ROOT / "outputs_scorito_topscorers" / "current_group_scorers.csv"
MODEL_AUTO_REFRESH_INTERVAL = timedelta(days=2)
MODEL_REFRESH_RETRY_BACKOFF = timedelta(hours=6)
MODEL_OUTPUT_PATHS = [
    BEST_EXACT_FORECAST_PATH,
    BEST_EXACT_EVAL_PATH,
    BEST_EXACT_KNOCKOUT_PATH,
    BEST_TOPSCORER_PATH,
    BEST_TOPSCORER_PHASE_PATH,
    MATCH_ADVICE_PATH,
    COUNTRY_ADVICE_PATH,
]


st.set_page_config(
    page_title="WK Scorito",
    layout="centered",
)

CHART_TEXT = "#E8EDF3"
CHART_MUTED = "#9AA7B4"
CHART_GRID = "#273241"
CHART_BLUE = "#5BB8FF"
CHART_GREEN = "#68D391"
CHART_AMBER = "#F6C85F"


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #080d14;
            --panel: #101722;
            --border: rgba(154, 167, 180, 0.18);
            --text: #f2f5f8;
            --muted: #9aa7b4;
            --blue: #5bb8ff;
            --green: #68d391;
            --amber: #f6c85f;
        }

        .stApp {
            background: radial-gradient(circle at top left, rgba(91, 184, 255, 0.09), transparent 34rem), var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 760px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 1.65rem !important;
            line-height: 1.15 !important;
            font-weight: 760 !important;
            letter-spacing: 0 !important;
            margin: 0 0 0.9rem 0 !important;
        }

        h2, h3 {
            letter-spacing: 0 !important;
        }

        .page-kicker {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0 !important;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
        }

        [data-testid="stSidebar"] {
            background: #0a1019;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            letter-spacing: 0 !important;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.015));
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        [data-testid="stMetricValue"] {
            font-size: 1.28rem !important;
            font-weight: 780 !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.2rem;
            padding: 0 0.75rem;
            border-radius: 6px 6px 0 0;
            color: var(--muted);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(91, 184, 255, 0.12);
            color: var(--text);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(16, 23, 34, 0.55);
        }

        .match-row {
            padding: 0.72rem 0;
            border-bottom: 1px solid rgba(154, 167, 180, 0.13);
        }

        .match-row:last-child {
            border-bottom: 0;
        }

        .match-meta {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.15rem;
        }

        .match-title {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 680;
            line-height: 1.28;
            margin-top: 0.25rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            min-height: 1.25rem;
            padding: 0 0.48rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 780;
            line-height: 1;
            margin-right: 0.42rem;
            border: 1px solid transparent;
        }

        .status-final {
            color: #bdf6d2;
            background: rgba(104, 211, 145, 0.14);
            border-color: rgba(104, 211, 145, 0.28);
        }

        .status-next {
            color: #ffe6a3;
            background: rgba(246, 200, 95, 0.14);
            border-color: rgba(246, 200, 95, 0.28);
        }

        .status-model {
            color: #c8e9ff;
            background: rgba(91, 184, 255, 0.13);
            border-color: rgba(91, 184, 255, 0.26);
        }

        .status-pick {
            color: #f8dcff;
            background: rgba(207, 139, 255, 0.13);
            border-color: rgba(207, 139, 255, 0.25);
        }

        .status-neutral {
            color: var(--muted);
            background: rgba(154, 167, 180, 0.11);
            border-color: rgba(154, 167, 180, 0.18);
        }

        .bracket-scroll {
            overflow-x: auto;
            padding: 0.15rem 0 0.8rem 0;
            margin-bottom: 1rem;
        }

        .bracket-board {
            display: grid;
            grid-template-columns: repeat(6, minmax(170px, 1fr));
            gap: 0.85rem;
            min-width: 1120px;
            align-items: stretch;
        }

        .bracket-column {
            min-width: 0;
        }

        .bracket-heading {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 780;
            text-transform: uppercase;
            letter-spacing: 0 !important;
            margin: 0 0 0.55rem 0.15rem;
        }

        .bracket-stack {
            display: flex;
            flex-direction: column;
            gap: 0.56rem;
            height: 100%;
            justify-content: space-around;
        }

        .bracket-card {
            position: relative;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.015));
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.64rem 0.68rem;
            min-height: 5.7rem;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.16);
        }

        .bracket-card::after {
            content: "";
            position: absolute;
            top: 50%;
            right: -0.85rem;
            width: 0.85rem;
            height: 1px;
            background: rgba(154, 167, 180, 0.22);
        }

        .bracket-column:last-child .bracket-card::after {
            display: none;
        }

        .bracket-card-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.44rem;
        }

        .bracket-match {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 720;
            white-space: nowrap;
        }

        .bracket-team {
            color: var(--text);
            font-size: 0.82rem;
            line-height: 1.22;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .bracket-team.pick {
            color: #d4ecff;
            font-weight: 760;
        }

        .bracket-score {
            color: var(--muted);
            font-size: 0.72rem;
            margin-top: 0.42rem;
        }

        .bracket-help {
            color: var(--muted);
            font-size: 0.78rem;
            margin: -0.3rem 0 0.8rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


def page_heading(title: str) -> None:
    st.markdown(
        f"<div class='page-kicker'>WK 2026 Scorito</div><h1>{escape(title)}</h1>",
        unsafe_allow_html=True,
    )


def status_pill(label: str, kind: str = "neutral") -> str:
    return f"<span class='status-pill status-{kind}'>{escape(label)}</span>"


def style_axis(ax: plt.Axes, title: str | None = None, grid_axis: str | None = "both") -> None:
    ax.set_facecolor("#0f1722")
    ax.figure.patch.set_facecolor("#0f1722")
    ax.tick_params(colors=CHART_MUTED, labelsize=8)
    ax.xaxis.label.set_color(CHART_MUTED)
    ax.yaxis.label.set_color(CHART_MUTED)
    ax.title.set_color(CHART_TEXT)
    for spine in ax.spines.values():
        spine.set_color(CHART_GRID)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=CHART_GRID, alpha=0.55, linewidth=0.7)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)


def default_predictions() -> dict[str, object]:
    return {"topscorer_pick": "", "bracket_overrides": {}}


def read_json(path: Path, default: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        path.write_text(json.dumps(default, indent=2, ensure_ascii=False), encoding="utf-8")
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prediction_state() -> dict[str, object]:
    if "prediction_state" not in st.session_state:
        st.session_state.prediction_state = read_json(PREDICTIONS_PATH, default_predictions())
    state = st.session_state.prediction_state
    state.setdefault("topscorer_pick", "")
    state.setdefault("bracket_overrides", {})
    return state


def save_prediction_state() -> None:
    write_json(PREDICTIONS_PATH, st.session_state.prediction_state)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(file_path)
    for column in ["date", "odds_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def csv(path: Path) -> pd.DataFrame:
    return load_csv(str(path))


def format_percent(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def format_number(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def simple_points(row: pd.Series) -> int:
    if bool(row.get("exact_score_correct", False)):
        return 3
    if bool(row.get("outcome_correct", False)):
        return 1
    return 0


def score_text(row: pd.Series, prefix: str = "predicted") -> str:
    home = row.get(f"{prefix}_home_score")
    away = row.get(f"{prefix}_away_score")
    if pd.isna(home) or pd.isna(away):
        return "-"
    return f"{int(float(home))}-{int(float(away))}"


def actual_score_text(row: pd.Series) -> str:
    home = row.get("actual_home_score", row.get("home_score"))
    away = row.get("actual_away_score", row.get("away_score"))
    if pd.isna(home) or pd.isna(away):
        return "-"
    return f"{int(float(home))}-{int(float(away))}"


def local_date_text(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "Datum onbekend"
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(AMSTERDAM).strftime("%d %b %Y %H:%M")


def round_label(value: object) -> str:
    text = str(value)
    if text.startswith("Group"):
        return "Group Stage"
    aliases = {
        "Quarterfinals": "QF",
        "Quarter-finals": "QF",
        "Semifinals": "SF",
        "Semi-finals": "SF",
        "Final": "Final",
        "Match for third place": "Third Place",
    }
    return aliases.get(text, text)


def fallback_schedule() -> pd.DataFrame:
    completed = csv(BEST_EXACT_EVAL_PATH)
    future = csv(BEST_EXACT_FORECAST_PATH)
    knockout = csv(BEST_EXACT_KNOCKOUT_PATH)
    frames: list[pd.DataFrame] = []
    if not completed.empty:
        frames.append(
            completed.rename(
                columns={
                    "actual_home_score": "home_score",
                    "actual_away_score": "away_score",
                }
            )[
                ["date", "group", "match_no", "home_team", "away_team", "home_score", "away_score"]
            ]
        )
    if not future.empty:
        future_group = future.copy()
        future_group["home_score"] = np.nan
        future_group["away_score"] = np.nan
        frames.append(
            future_group[
                ["date", "group", "match_no", "home_team", "away_team", "home_score", "away_score"]
            ]
        )
    if not knockout.empty:
        knockout_frame = knockout.rename(columns={"round": "group"}).copy()
        if "actual_home_score" in knockout_frame.columns:
            knockout_frame["home_score"] = knockout_frame["actual_home_score"]
            knockout_frame["away_score"] = knockout_frame["actual_away_score"]
        else:
            knockout_frame["home_score"] = np.nan
            knockout_frame["away_score"] = np.nan
        frames.append(
            knockout_frame[
                ["date", "group", "match_no", "home_team", "away_team", "home_score", "away_score"]
            ]
        )
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    return result.sort_values(["date", "match_no"], na_position="last").reset_index(drop=True)


def read_results_cache() -> tuple[pd.DataFrame, datetime | None]:
    payload = read_json(RESULTS_CACHE_PATH, {"generated_at": None, "source_url": CURRENT_WORLD_CUP_URL, "matches": []})
    generated_at = None
    if payload.get("generated_at"):
        generated_at = datetime.fromisoformat(str(payload["generated_at"]))
    matches = pd.DataFrame(payload.get("matches", []))
    if not matches.empty and "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    return matches, generated_at


def write_results_cache(matches: pd.DataFrame) -> None:
    payload_frame = matches.copy()
    if "date" in payload_frame.columns:
        payload_frame["date"] = pd.to_datetime(payload_frame["date"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": CURRENT_WORLD_CUP_URL,
        "matches": payload_frame.replace({np.nan: None}).to_dict(orient="records"),
    }
    write_json(RESULTS_CACHE_PATH, payload)


@st.cache_data(ttl=300, show_spinner=False)
def load_schedule_with_cache() -> pd.DataFrame:
    cached, generated_at = read_results_cache()
    now = datetime.now(timezone.utc)
    if generated_at is not None and (now - generated_at) < timedelta(hours=1) and not cached.empty:
        return cached

    try:
        historical = load_historical_results(CACHE_DIR, refresh=False)
        current = load_current_world_cup(CACHE_DIR, refresh=True, historical=historical)
        schedule = current.rename(
            columns={"home_score": "home_score", "away_score": "away_score"}
        )[
            ["date", "group", "match_no", "home_team", "away_team", "home_score", "away_score"]
        ]
        write_results_cache(schedule)
        return schedule
    except Exception:
        if not cached.empty:
            return cached
        fallback = fallback_schedule()
        if not fallback.empty:
            write_results_cache(fallback)
        return fallback


def parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def completed_match_count(schedule: pd.DataFrame) -> int | None:
    if schedule.empty or "home_score" not in schedule or "away_score" not in schedule:
        return None
    return int((schedule["home_score"].notna() & schedule["away_score"].notna()).sum())


def model_refresh_commands() -> list[list[str]]:
    exact_dir = ROOT / "outputs_exact_score_model"
    top_dir = ROOT / "outputs_scorito_topscorers"
    poule_dir = ROOT / "outputs_scorito_poule"
    return [
        [
            sys.executable,
            "-B",
            "world_cup_exact_score_forecaster.py",
            "--cache-dir",
            str(CACHE_DIR),
            "--output-dir",
            str(exact_dir),
        ],
        [
            sys.executable,
            "-B",
            "scorito_knockout_topscorer_picker.py",
            "--cache-dir",
            str(CACHE_DIR),
            "--exact-score-dir",
            str(exact_dir),
            "--output-dir",
            str(top_dir),
            "--simulations",
            "20000",
        ],
        [
            sys.executable,
            "-B",
            "scorito_poule_optimizer.py",
            "--cache-dir",
            str(CACHE_DIR),
            "--exact-score-dir",
            str(exact_dir),
            "--topscorer-dir",
            str(top_dir),
            "--output-dir",
            str(poule_dir),
            "--simulations",
            "20000",
        ],
    ]


def run_model_refresh(reason: str, completed_count: int | None) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    logs: list[str] = []
    for command in model_refresh_commands():
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        logs.append(result.stdout[-1200:])

    load_csv.clear()
    finished = datetime.now(timezone.utc)
    state = {
        "generated_at": finished.isoformat(),
        "completed_match_count": completed_count,
        "refresh_interval_hours": int(MODEL_AUTO_REFRESH_INTERVAL.total_seconds() // 3600),
        "last_reason": reason,
        "status": "ok",
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "latest_log_tail": "\n".join(logs)[-3000:],
    }
    write_json(MODEL_REFRESH_STATE_PATH, state)
    return state


def ensure_model_outputs_current() -> dict[str, object]:
    schedule = load_schedule_with_cache()
    current_completed = completed_match_count(schedule)
    outputs_missing = [str(path.relative_to(ROOT)) for path in MODEL_OUTPUT_PATHS if not path.exists()]

    if not MODEL_REFRESH_STATE_PATH.exists() and not outputs_missing:
        state = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "completed_match_count": current_completed,
            "refresh_interval_hours": int(MODEL_AUTO_REFRESH_INTERVAL.total_seconds() // 3600),
            "last_reason": "initial tracked outputs",
            "status": "ok",
        }
        write_json(MODEL_REFRESH_STATE_PATH, state)
        return state

    state = read_json(
        MODEL_REFRESH_STATE_PATH,
        {
            "generated_at": None,
            "completed_match_count": current_completed,
            "refresh_interval_hours": int(MODEL_AUTO_REFRESH_INTERVAL.total_seconds() // 3600),
            "last_reason": "not refreshed yet",
            "status": "pending",
        },
    )
    now = datetime.now(timezone.utc)
    last_generated = parse_timestamp(state.get("generated_at"))
    last_attempted = parse_timestamp(state.get("last_attempted_at"))
    last_completed = state.get("completed_match_count")
    stale = last_generated is None or (now - last_generated) >= MODEL_AUTO_REFRESH_INTERVAL
    finals_changed = (
        current_completed is not None
        and last_completed is not None
        and int(last_completed) != current_completed
    )

    reason = ""
    if outputs_missing:
        reason = "missing output files: " + ", ".join(outputs_missing)
    elif finals_changed:
        reason = f"completed match count changed from {last_completed} to {current_completed}"
    elif stale:
        reason = "48 hour scheduled refresh"

    if not reason:
        return state

    if (
        state.get("status") == "failed"
        and last_attempted is not None
        and (now - last_attempted) < MODEL_REFRESH_RETRY_BACKOFF
    ):
        return state

    try:
        with st.spinner("Updating predictions from latest results..."):
            return run_model_refresh(reason, current_completed)
    except Exception as exc:
        failure_state = dict(state)
        failure_state.update(
            {
                "status": "failed",
                "last_attempted_at": now.isoformat(),
                "last_reason": reason,
                "error": str(exc)[-1200:],
            }
        )
        write_json(MODEL_REFRESH_STATE_PATH, failure_state)
        return failure_state


def accuracy_bar_chart(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_data = metrics.dropna(subset=["Outcome accuracy"]).copy()
    plot_data = plot_data.sort_values("Outcome accuracy", ascending=True)
    y_pos = np.arange(len(plot_data))
    bar_height = 0.36
    ax.barh(
        y_pos + bar_height / 2,
        plot_data["Outcome accuracy"],
        height=bar_height,
        color=CHART_BLUE,
        alpha=0.9,
        label="Winner",
    )
    ax.barh(
        y_pos - bar_height / 2,
        plot_data["Exact accuracy"],
        height=bar_height,
        color=CHART_GREEN,
        alpha=0.9,
        label="Exact",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["Model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=CHART_MUTED)
    style_axis(ax, "Model accuracy", grid_axis="x")
    fig.tight_layout()
    return fig


def calibration_chart(predictions: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    if predictions.empty or "outcome_probability" not in predictions or "outcome_correct" not in predictions:
        style_axis(ax, title)
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center", color=CHART_MUTED)
        ax.axis("off")
        fig.tight_layout()
        return fig
    work = predictions.dropna(subset=["outcome_probability", "outcome_correct"]).copy()
    if work.empty:
        style_axis(ax, title)
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center", color=CHART_MUTED)
        ax.axis("off")
        fig.tight_layout()
        return fig
    work["bin"] = pd.cut(work["outcome_probability"].astype(float), bins=np.linspace(0, 1, 6), include_lowest=True)
    grouped = work.groupby("bin", observed=True).agg(
        confidence=("outcome_probability", "mean"),
        accuracy=("outcome_correct", "mean"),
        count=("outcome_correct", "size"),
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=CHART_MUTED, linewidth=1)
    ax.scatter(
        grouped["confidence"],
        grouped["accuracy"],
        s=grouped["count"] * 18,
        color=CHART_AMBER,
        edgecolors="#0f1722",
        linewidths=1.2,
        alpha=0.95,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Actual accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_axis(ax, title)
    fig.tight_layout()
    return fig


def points_chart(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3))
    if summary.empty or "scorito_points" not in summary:
        style_axis(ax)
        ax.axis("off")
        fig.tight_layout()
        return fig
    plot_data = summary[summary["matches_evaluated"].astype(float).gt(0)].copy()
    plot_data = plot_data.sort_values("points_per_match", ascending=True)
    labels = plot_data["strategy"].str.replace("_", " ", regex=False)
    ax.barh(labels, plot_data["points_per_match"], color=CHART_GREEN, alpha=0.9)
    ax.set_xlabel("Points per match")
    style_axis(ax, "Backtest points per match", grid_axis="x")
    fig.tight_layout()
    return fig


def bookmaker_benchmark_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    labels = {
        "model_all": "Our historical model",
        "model_market_covered": "Our model on odds-covered matches",
        "market_favorite_common_score": "Bookmaker favorite proxy",
        "common_home_1_0_all": "Always 1-0 home",
        "common_home_1_0_market_covered": "Always 1-0 home on odds-covered matches",
        "random_historical_scoreline": "Random historical scoreline",
    }
    display = summary.copy()
    display["Benchmark"] = display["strategy"].map(labels).fillna(display["strategy"])
    cols = [
        "Benchmark",
        "matches_evaluated",
        "exact_accuracy",
        "toto_accuracy",
        "scorito_points",
        "points_per_match",
        "market_coverage_rate",
    ]
    display = display[[column for column in cols if column in display.columns]].copy()
    display = display.rename(
        columns={
            "matches_evaluated": "Matches",
            "exact_accuracy": "Exact",
            "toto_accuracy": "Winner",
            "scorito_points": "Points",
            "points_per_match": "Points per match",
            "market_coverage_rate": "Market coverage",
        }
    )
    for column in ["Exact", "Winner", "Market coverage"]:
        if column in display:
            display[column] = display[column].map(format_percent)
    for column in ["Points", "Points per match"]:
        if column in display:
            display[column] = display[column].map(lambda value: format_number(value, 1))
    return display


def model_metrics_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline = csv(BASELINE_EVAL_PATH)
    calibrated = csv(CALIBRATED_EVAL_PATH)
    exact = csv(BEST_EXACT_EVAL_PATH)
    total = csv(TOTAL_GOALS_EVAL_PATH)
    if not baseline.empty:
        rows.append(
            {
                "Model": "Historical Elo/Poisson",
                "Category": "Outcome",
                "Outcome accuracy": baseline["outcome_correct"].mean(),
                "Exact accuracy": baseline["exact_score_correct"].mean(),
                "Scorito points": np.nan,
            }
        )
    if not calibrated.empty:
        rows.append(
            {
                "Model": "Tournament-calibrated Elo/Poisson",
                "Category": "Winner / bracket",
                "Outcome accuracy": calibrated["outcome_correct"].mean(),
                "Exact accuracy": calibrated["exact_score_correct"].mean(),
                "Scorito points": np.nan,
            }
        )
    if not exact.empty:
        rows.append(
            {
                "Model": "Exact-score tree",
                "Category": "Exact score",
                "Outcome accuracy": exact["outcome_correct"].mean(),
                "Exact accuracy": exact["exact_score_correct"].mean(),
                "Within one": exact["within_one_goal_correct"].mean(),
                "Scorito points": np.nan,
            }
        )
    if not total.empty:
        rows.append(
            {
                "Model": "Total-goals first",
                "Category": "Goals",
                "Outcome accuracy": total["outcome_correct"].mean(),
                "Exact accuracy": total["exact_score_correct"].mean(),
                "Total-goals MAE": total["total_goals_absolute_error"].mean(),
                "Scorito points": np.nan,
            }
        )
    backtest = csv(BACKTEST_OVERALL_PATH)
    if not backtest.empty:
        row = backtest.iloc[0]
        rows.append(
            {
                "Model": "Historical Scorito knockout heuristic",
                "Category": "Topscorer / match backtest",
                "Outcome accuracy": row["match_toto_accuracy"],
                "Exact accuracy": row["match_exact_accuracy"],
                "Scorito points": row["match_total_points"],
                "Topscorer points": row["topscorer_total_points"],
            }
        )
    book = csv(BOOKMAKER_SUMMARY_PATH)
    if not book.empty:
        common = book[book["strategy"].eq("common_home_1_0_all")]
        if not common.empty:
            row = common.iloc[0]
            rows.append(
                {
                    "Model": "Common 1-0 benchmark",
                    "Category": "Benchmark",
                    "Outcome accuracy": row["toto_accuracy"],
                    "Exact accuracy": row["exact_accuracy"],
                    "Scorito points": row["scorito_points"],
                }
            )
    return pd.DataFrame(rows)


def page_backtest_stats() -> None:
    page_heading("Backtest Stats")
    metrics = model_metrics_table()
    exact_summary = csv(ROOT / "outputs_exact_score_model" / "exact_score_model_summary.csv")
    backtest = csv(BACKTEST_OVERALL_PATH)
    book = csv(BOOKMAKER_SUMMARY_PATH)
    top = csv(BEST_TOPSCORER_PATH)
    match_advice = csv(MATCH_ADVICE_PATH)

    col1, col2 = st.columns(2)
    if not exact_summary.empty:
        row = exact_summary.iloc[0]
        col1.metric("Exact score", format_percent(row["exact_score_accuracy"]))
        col2.metric("Within one goal", format_percent(row["within_one_accuracy"]))
    if not match_advice.empty:
        st.metric("Current match EV", f"{match_advice['expected_match_points'].sum():.0f} pts")
    if not top.empty:
        st.metric("Top 5 scorer EV", f"{top.head(5)['expected_scorito_points'].sum():.0f} pts")

    tabs = st.tabs(["Accuracy", "Calibration", "Benchmarks"])
    with tabs[0]:
        st.pyplot(accuracy_bar_chart(metrics), clear_figure=True)
        st.dataframe(
            metrics.assign(
                **{
                    "Outcome accuracy": metrics["Outcome accuracy"].map(format_percent),
                    "Exact accuracy": metrics["Exact accuracy"].map(format_percent),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with tabs[1]:
        st.pyplot(calibration_chart(csv(CALIBRATED_EVAL_PATH), "Calibrated winner model"), clear_figure=True)
        st.pyplot(calibration_chart(csv(BEST_EXACT_EVAL_PATH), "Exact-score tree confidence"), clear_figure=True)
    with tabs[2]:
        st.markdown(
            "Historical Scorito-style match scoring. This compares the model with simple baselines "
            "and an OddsPortal bookmaker-favorite proxy; it is not real Scorito user data."
        )
        if not backtest.empty:
            st.dataframe(backtest, use_container_width=True, hide_index=True)
        if not book.empty:
            st.pyplot(points_chart(book), clear_figure=True)
            st.dataframe(bookmaker_benchmark_table(book), use_container_width=True, hide_index=True)
            odds_path = ROOT / "outputs_bookmaker_backtest" / "bookmaker_1x2_odds_used.csv"
            if odds_path.exists():
                odds = csv(odds_path)
                if not odds.empty:
                    st.caption(
                        f"Bookmaker source: {len(odds)} OddsPortal 1X2 rows saved locally; "
                        "the app reads this CSV and does not fetch odds during page navigation."
                    )


def page_schedule() -> None:
    page_heading("Schedule & Live Results")
    schedule = load_schedule_with_cache()
    if schedule.empty:
        st.warning("No schedule data available.")
        return

    schedule = schedule.copy()
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    schedule["completed"] = schedule["home_score"].notna() & schedule["away_score"].notna()
    now_local = datetime.now(AMSTERDAM)
    next_24 = now_local + timedelta(hours=24)

    def is_next_24(value: object) -> bool:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            return False
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        local = timestamp.tz_convert(AMSTERDAM)
        return now_local <= local <= next_24

    schedule["next_24h"] = schedule["date"].map(is_next_24) & ~schedule["completed"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", len(schedule))
    c2.metric("Final", int(schedule["completed"].sum()))
    c3.metric("Next 24h", int(schedule["next_24h"].sum()))

    for round_name, round_df in schedule.groupby(schedule["group"].map(round_label), sort=False):
        with st.expander(str(round_name), expanded=bool(round_df["next_24h"].any())):
            for row in round_df.sort_values(["date", "match_no"]).itertuples(index=False):
                if bool(row.completed):
                    label, kind = "Final", "final"
                elif bool(row.next_24h):
                    label, kind = "Next", "next"
                else:
                    label, kind = "Scheduled", "neutral"
                score = (
                    f"{int(float(row.home_score))}-{int(float(row.away_score))}"
                    if bool(row.completed)
                    else "vs"
                )
                st.markdown(
                    "<div class='match-row'>"
                    f"{status_pill(label, kind)}"
                    f"<span class='match-meta'>{escape(local_date_text(row.date))}</span>"
                    f"<div class='match-title'>{escape(str(row.home_team))} "
                    f"<span>{escape(score)}</span> {escape(str(row.away_team))}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )


def page_topscorers() -> None:
    page_heading("Topscorer Predictions")
    recommendations = csv(BEST_TOPSCORER_PATH)
    phase = csv(BEST_TOPSCORER_PHASE_PATH)
    scorers = csv(CURRENT_SCORERS_PATH)
    if recommendations.empty:
        st.warning("No top-scorer recommendations found.")
        return

    scorer_lookup = (
        scorers.groupby(["player", "team"], as_index=False)["group_goals"].sum()
        if not scorers.empty and "group_goals" in scorers
        else pd.DataFrame(columns=["player", "team", "group_goals"])
    )
    display = recommendations.merge(scorer_lookup, how="left", on=["player", "team"], suffixes=("", "_actual"))
    display["actual_goals_so_far"] = display["group_goals_actual"].fillna(display.get("group_goals", 0)).fillna(0)

    state = prediction_state()
    names = display["player"].astype(str).tolist()
    current_pick = str(state.get("topscorer_pick", ""))
    index = names.index(current_pick) if current_pick in names else 0
    selected = st.selectbox("My Pick", names, index=index)
    if selected != current_pick:
        state["topscorer_pick"] = selected
        save_prediction_state()

    picked = display[display["player"].eq(selected)].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Goals now", int(picked["actual_goals_so_far"]))
    cols[1].metric("Projected goals", format_number(picked["expected_knockout_goals"], 2))
    cols[2].metric("Scorito EV", format_number(picked["expected_scorito_points"], 1))
    cols[3].metric("Top 5 prob", format_percent(picked["top5_scorito_probability"]))

    show_cols = [
        "recommended_rank",
        "player",
        "team",
        "scorito_position",
        "actual_goals_so_far",
        "expected_knockout_goals",
        "top5_scorito_probability",
        "expected_scorito_points",
    ]
    st.dataframe(
        display[show_cols]
        .head(25)
        .rename(
            columns={
                "recommended_rank": "Rank",
                "player": "Player",
                "team": "Team",
                "scorito_position": "Position",
                "actual_goals_so_far": "Goals now",
                "expected_knockout_goals": "Projected goals",
                "top5_scorito_probability": "Top 5 probability",
                "expected_scorito_points": "Scorito EV",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Phase picks", expanded=False):
        phase_cols = [
            "phase",
            "phase_pick_rank",
            "player",
            "team",
            "scorito_position",
            "expected_round_goals",
            "expected_round_points",
        ]
        st.dataframe(
            phase[phase_cols].rename(
                columns={
                    "phase": "Phase",
                    "phase_pick_rank": "Rank",
                    "player": "Player",
                    "team": "Team",
                    "scorito_position": "Position",
                    "expected_round_goals": "Round goals",
                    "expected_round_points": "Round EV",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def merged_match_predictions() -> pd.DataFrame:
    forecast = csv(BEST_EXACT_FORECAST_PATH)
    knockout = csv(BEST_EXACT_KNOCKOUT_PATH)
    advice = csv(MATCH_ADVICE_PATH)
    frames: list[pd.DataFrame] = []
    if not forecast.empty:
        group = forecast.copy()
        group["round"] = group["group"]
        frames.append(group)
    if not knockout.empty:
        frames.append(knockout.copy())
    if not frames:
        return pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True, sort=False)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    if not advice.empty:
        key = ["date", "home_team", "away_team"]
        advice_keyed = advice.copy()
        advice_keyed["date"] = pd.to_datetime(advice_keyed["date"], errors="coerce")
        predictions = predictions.merge(
            advice_keyed[
                key
                + [
                    "expected_match_points",
                    "estimated_exact_probability",
                    "estimated_toto_probability",
                    "toto_points",
                    "exact_points",
                ]
            ],
            how="left",
            on=key,
        )
    predictions["predicted_total_goals"] = (
        pd.to_numeric(predictions["home_xg"], errors="coerce")
        + pd.to_numeric(predictions["away_xg"], errors="coerce")
    )
    return predictions.sort_values(["date", "match_no"], na_position="last").reset_index(drop=True)


def page_matches() -> None:
    page_heading("Match Predictions & Goals")
    completed = csv(BEST_EXACT_EVAL_PATH)
    future = merged_match_predictions()
    if not completed.empty:
        completed = completed.copy()
        completed["simple_points"] = completed.apply(simple_points, axis=1)
        st.metric("Actual points so far", int(completed["simple_points"].sum()))
        with st.expander("Completed matches", expanded=False):
            for group, group_df in completed.groupby("group", sort=False):
                st.markdown(f"**{group}**")
                rows = []
                for _, row in group_df.sort_values(["date", "match_no"]).iterrows():
                    rows.append(
                        {
                            "Match": f"{row['home_team']} - {row['away_team']}",
                            "Prediction": score_text(row),
                            "Actual": actual_score_text(row),
                            "Points": int(row["simple_points"]),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if future.empty:
        st.warning("No upcoming match predictions found.")
        return

    for round_name, round_df in future.groupby(future["round"].map(round_label), sort=False):
        with st.expander(str(round_name), expanded=str(round_name) == "Group Stage"):
            for row in round_df.itertuples(index=False):
                prob = getattr(row, "outcome_probability", np.nan)
                if pd.isna(prob):
                    prob = getattr(row, "estimated_toto_probability", np.nan)
                total_goals = getattr(row, "predicted_total_goals", np.nan)
                ev = getattr(row, "expected_match_points", np.nan)
                home_xg = getattr(row, "home_xg", np.nan)
                away_xg = getattr(row, "away_xg", np.nan)
                elo_diff = getattr(row, "elo_diff", np.nan)
                xg_edge = (
                    float(home_xg) - float(away_xg)
                    if not pd.isna(home_xg) and not pd.isna(away_xg)
                    else np.nan
                )
                st.markdown(
                    "<div class='match-row'>"
                    f"{status_pill('Model', 'model')}"
                    f"<span class='match-meta'>p {escape(format_percent(prob))} | "
                    f"goals {escape(format_number(total_goals, 2))} | EV {escape(format_number(ev, 1))}</span>"
                    f"<div class='match-title'>{escape(str(row.home_team))} - {escape(str(row.away_team))}</div>"
                    f"<div class='match-meta'>"
                    f"{int(float(row.predicted_home_score))}-{int(float(row.predicted_away_score))} | "
                    f"{escape(str(getattr(row, 'predicted_outcome', '')))} | "
                    f"xG {escape(format_number(home_xg, 2))}-{escape(format_number(away_xg, 2))} "
                    f"(edge {escape(format_number(xg_edge, 2))}) | "
                    f"Elo edge {escape(format_number(elo_diff, 0))}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )


def bracket_tag(row: pd.Series, selected: str, model_pick: str) -> str:
    if pd.notna(row.get("actual_home_score")) and pd.notna(row.get("actual_away_score")):
        return status_pill("Confirmed", "final")
    if selected and selected != model_pick:
        return status_pill("Your pick", "pick")
    return status_pill("Model", "model")


BRACKET_ROUND_ORDER = {
    "Round of 32": 0,
    "Round of 16": 1,
    "Quarterfinals": 2,
    "Quarter-finals": 2,
    "Semifinals": 3,
    "Semi-finals": 3,
    "Final": 4,
    "Match for third place": 5,
}


def bracket_round_order(round_name: object) -> int:
    return BRACKET_ROUND_ORDER.get(str(round_name), 99)


def bracket_card(row: pd.Series, overrides: dict[str, str]) -> str:
    match_no = int(row["match_no"])
    home = str(row.get("home_team", "") or row.get("home_slot", "") or "")
    away = str(row.get("away_team", "") or row.get("away_slot", "") or "")
    model_pick = str(row.get("advancing_team", "") or home)
    selected = str(overrides.get(str(match_no), model_pick) or model_pick)
    tag = bracket_tag(row, selected, model_pick)
    score = score_text(row)
    home_class = "bracket-team pick" if selected == home else "bracket-team"
    away_class = "bracket-team pick" if selected == away else "bracket-team"
    return (
        "<div class='bracket-card'>"
        "<div class='bracket-card-meta'>"
        f"<span class='bracket-match'>M{match_no}</span>{tag}"
        "</div>"
        f"<div class='{home_class}'>{escape(home)}</div>"
        f"<div class='{away_class}'>{escape(away)}</div>"
        f"<div class='bracket-score'>Score {escape(score)} | Pick {escape(selected)}</div>"
        "</div>"
    )


def horizontal_bracket_html(bracket: pd.DataFrame, overrides: dict[str, str]) -> str:
    board = bracket.copy()
    board["round_order"] = board["round"].map(bracket_round_order)
    board = board.sort_values(["round_order", "date", "match_no"], na_position="last")
    columns: list[str] = []
    for round_name, round_df in board.groupby("round", sort=False):
        cards = "".join(bracket_card(row, overrides) for _, row in round_df.iterrows())
        columns.append(
            "<div class='bracket-column'>"
            f"<div class='bracket-heading'>{escape(str(round_name))}</div>"
            f"<div class='bracket-stack'>{cards}</div>"
            "</div>"
        )
    return (
        "<div class='bracket-help'>Scroll sideways to read the full knockout path.</div>"
        "<div class='bracket-scroll'>"
        f"<div class='bracket-board'>{''.join(columns)}</div>"
        "</div>"
    )


def page_bracket() -> None:
    page_heading("Bracket Prediction")
    bracket = csv(BEST_BRACKET_PATH)
    country = csv(COUNTRY_ADVICE_PATH)
    match_advice = csv(MATCH_ADVICE_PATH)
    if bracket.empty:
        st.warning("No bracket forecast found.")
        return
    state = prediction_state()
    overrides = state.setdefault("bracket_overrides", {})
    bracket = bracket.copy()
    bracket["date"] = pd.to_datetime(bracket["date"], errors="coerce")
    today = pd.Timestamp(datetime.now(AMSTERDAM).date())
    changed = False

    st.markdown(horizontal_bracket_html(bracket, overrides), unsafe_allow_html=True)

    if st.toggle("Show override controls", value=False):
        st.markdown("**Override Picks**")
        for round_name, round_df in bracket.groupby("round", sort=False):
            st.markdown(f"**{round_name}**")
            for _, row in round_df.sort_values(["date", "match_no"]).iterrows():
                match_no = int(row["match_no"])
                home = str(row.get("home_team", "") or "")
                away = str(row.get("away_team", "") or "")
                if not home or not away:
                    st.markdown(
                        "<div class='match-row'>"
                        f"{status_pill('Model', 'model')}"
                        f"<span class='match-meta'>M{match_no}</span>"
                        f"<div class='match-title'>{escape(str(row.get('home_slot', '')))} - "
                        f"{escape(str(row.get('away_slot', '')))}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    continue
                model_pick = str(row.get("advancing_team", "") or home)
                options = [home, away]
                current = str(overrides.get(str(match_no), model_pick))
                if current not in options:
                    current = model_pick if model_pick in options else home
                editable = bool(pd.isna(row.get("actual_home_score"))) and (
                    pd.isna(row["date"]) or pd.Timestamp(row["date"]).normalize() >= today
                )
                tag = bracket_tag(row, current, model_pick)
                st.markdown(
                    "<div class='match-row'>"
                    f"{tag}<span class='match-meta'>M{match_no} | Model: {escape(model_pick)} | "
                    f"Score: {escape(score_text(row))}</span>"
                    f"<div class='match-title'>{escape(home)} - {escape(away)}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                selected = st.selectbox(
                    f"Pick M{match_no}",
                    options,
                    index=options.index(current),
                    key=f"bracket_{match_no}",
                    disabled=not editable,
                    label_visibility="collapsed",
                )
                if selected != overrides.get(str(match_no), model_pick):
                    overrides[str(match_no)] = selected
                    changed = True

    if changed:
        save_prediction_state()

    knockout_ev = 0.0
    if not match_advice.empty:
        knockout_ev = float(
            match_advice[
                ~match_advice["round"].astype(str).str.startswith("Group")
            ]["expected_match_points"].sum()
        )
    milestone_points = 0
    if not country.empty:
        milestone_points = int(
            country[
                country["prediction_type"].isin(
                    ["Door naar kwartfinale", "Door naar halve finale", "Door naar finale", "Wereldkampioen"]
                )
            ]["points"].sum()
        )
    c1, c2 = st.columns(2)
    c1.metric("Bracket match EV", format_number(knockout_ev, 1))
    c2.metric("Country points at stake", milestone_points)


pages = [
    st.Page(page_backtest_stats, title="Backtest Stats"),
    st.Page(page_schedule, title="Schedule & Live Results"),
    st.Page(page_topscorers, title="Topscorer Predictions"),
    st.Page(page_matches, title="Match Predictions & Goals"),
    st.Page(page_bracket, title="Bracket Prediction"),
]

refresh_status = ensure_model_outputs_current()
st.sidebar.caption("WK 2026 Scorito")
if refresh_status.get("status") == "failed":
    st.sidebar.warning("Prediction refresh failed; showing last saved outputs.")
else:
    refreshed_at = parse_timestamp(refresh_status.get("generated_at"))
    if refreshed_at is not None:
        refreshed_local = refreshed_at.astimezone(AMSTERDAM).strftime("%d %b %H:%M")
        st.sidebar.caption(f"Predictions updated: {refreshed_local}")
    count = refresh_status.get("completed_match_count")
    if count is not None:
        st.sidebar.caption(f"Final scores used: {count}")
selected_page = st.navigation(pages, position="sidebar")
selected_page.run()
