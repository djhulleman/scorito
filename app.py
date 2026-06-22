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
- Match winners / bracket: tournament-calibrated Elo/Poisson bracket output
  from outputs/knockout_forecast.csv because it has the highest 2026 outcome
  accuracy among dynamic bracket-capable models.
- Exact scores: world_cup_exact_score_forecaster.py selected scoreline tree
  because it has the highest exact-score accuracy and within-one-goal rate.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
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
AMSTERDAM = ZoneInfo("Europe/Amsterdam")

BEST_TOPSCORER_PATH = ROOT / "outputs_scorito_topscorers" / "scorito_topscorer_recommendations.csv"
BEST_TOPSCORER_PHASE_PATH = ROOT / "outputs_scorito_poule" / "scorito_topscorers_by_phase.csv"
BEST_EXACT_FORECAST_PATH = ROOT / "outputs_exact_score_model" / "exact_score_model_forecast.csv"
BEST_EXACT_EVAL_PATH = ROOT / "outputs_exact_score_model" / "exact_score_model_evaluation.csv"
BEST_EXACT_KNOCKOUT_PATH = ROOT / "outputs_exact_score_model" / "exact_score_knockout_forecast.csv"
BEST_BRACKET_PATH = ROOT / "outputs" / "knockout_forecast.csv"
MATCH_ADVICE_PATH = ROOT / "outputs_scorito_poule" / "scorito_match_picks.csv"
COUNTRY_ADVICE_PATH = ROOT / "outputs_scorito_poule" / "scorito_country_picks.csv"
BACKTEST_OVERALL_PATH = ROOT / "outputs_scorito_backtest" / "backtest_overall_summary.csv"
BACKTEST_TOURNAMENT_PATH = ROOT / "outputs_scorito_backtest" / "backtest_tournament_summary.csv"
BOOKMAKER_SUMMARY_PATH = ROOT / "outputs_bookmaker_backtest" / "bookmaker_backtest_summary.csv"
BASELINE_EVAL_PATH = ROOT / "outputs" / "world_cup_evaluation_baseline.csv"
CALIBRATED_EVAL_PATH = ROOT / "outputs" / "world_cup_evaluation_calibrated.csv"
TOTAL_GOALS_EVAL_PATH = ROOT / "outputs_total_goals_model" / "total_goals_model_evaluation.csv"
CURRENT_SCORERS_PATH = ROOT / "outputs_scorito_topscorers" / "current_group_scorers.csv"


st.set_page_config(
    page_title="WK Scorito",
    page_icon="🏆",
    layout="centered",
)


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


def accuracy_bar_chart(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_data = metrics.dropna(subset=["Outcome accuracy"]).copy()
    ax.barh(plot_data["Model"], plot_data["Outcome accuracy"], color="#4e79a7", label="Outcome")
    ax.barh(plot_data["Model"], plot_data["Exact accuracy"], color="#59a14f", alpha=0.72, label="Exact")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def calibration_chart(predictions: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5, 3))
    if predictions.empty or "outcome_probability" not in predictions or "outcome_correct" not in predictions:
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    work = predictions.dropna(subset=["outcome_probability", "outcome_correct"]).copy()
    if work.empty:
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    work["bin"] = pd.cut(work["outcome_probability"].astype(float), bins=np.linspace(0, 1, 6), include_lowest=True)
    grouped = work.groupby("bin", observed=True).agg(
        confidence=("outcome_probability", "mean"),
        accuracy=("outcome_correct", "mean"),
        count=("outcome_correct", "size"),
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.scatter(grouped["confidence"], grouped["accuracy"], s=grouped["count"] * 16, color="#f28e2b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def points_chart(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3))
    if summary.empty or "scorito_points" not in summary:
        ax.axis("off")
        fig.tight_layout()
        return fig
    plot_data = summary[summary["matches_evaluated"].astype(float).gt(0)].copy()
    ax.barh(plot_data["strategy"].str.replace("_", " ", regex=False), plot_data["points_per_match"], color="#59a14f")
    ax.set_xlabel("Points per match")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


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
    st.title("📊 Backtest Stats")
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

    tabs = st.tabs(["Accuracy", "Calibration", "Scorito"])
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
        if not backtest.empty:
            st.dataframe(backtest, use_container_width=True, hide_index=True)
        if not book.empty:
            st.pyplot(points_chart(book), clear_figure=True)
            st.dataframe(book, use_container_width=True, hide_index=True)


def page_schedule() -> None:
    st.title("🗓️ Schedule & Live Results")
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
                badge = "✅" if bool(row.completed) else ("🔥" if bool(row.next_24h) else "•")
                score = (
                    f"{int(float(row.home_score))}-{int(float(row.away_score))}"
                    if bool(row.completed)
                    else "vs"
                )
                st.markdown(
                    f"{badge} **{local_date_text(row.date)}**  \n"
                    f"{row.home_team} **{score}** {row.away_team}"
                )


def page_topscorers() -> None:
    st.title("🏅 Topscorer Predictions")
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
    st.dataframe(display[show_cols].head(25), use_container_width=True, hide_index=True)

    with st.expander("Phase picks", expanded=False):
        st.dataframe(
            phase[
                [
                    "phase",
                    "phase_pick_rank",
                    "player",
                    "team",
                    "scorito_position",
                    "expected_round_goals",
                    "expected_round_points",
                ]
            ],
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
    st.title("⚽ Match Predictions & Goals")
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
                st.markdown(
                    f"**{row.home_team} - {row.away_team}**  \n"
                    f"{int(float(row.predicted_home_score))}-{int(float(row.predicted_away_score))} · "
                    f"{getattr(row, 'predicted_outcome', '')} · "
                    f"p {format_percent(prob)} · goals {format_number(total_goals, 2)} · EV {format_number(ev, 1)}"
                )


def bracket_tag(row: pd.Series, selected: str, model_pick: str) -> str:
    if pd.notna(row.get("actual_home_score")) and pd.notna(row.get("actual_away_score")):
        return "✅"
    if selected and selected != model_pick:
        return "📌"
    return "🤖"


def page_bracket() -> None:
    st.title("🏆 Bracket Prediction")
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

    for round_name, round_df in bracket.groupby("round", sort=False):
        with st.expander(str(round_name), expanded=str(round_name) == "Round of 32"):
            for _, row in round_df.sort_values(["date", "match_no"]).iterrows():
                match_no = int(row["match_no"])
                home = str(row.get("home_team", "") or "")
                away = str(row.get("away_team", "") or "")
                if not home or not away:
                    st.markdown(f"🤖 M{match_no}: {row.get('home_slot', '')} - {row.get('away_slot', '')}")
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
                    f"{tag} **M{match_no}: {home} - {away}**  \n"
                    f"Model: {model_pick} · Score: {score_text(row)}"
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
    st.Page(page_backtest_stats, title="Backtest Stats", icon="📊"),
    st.Page(page_schedule, title="Schedule & Live Results", icon="🗓️"),
    st.Page(page_topscorers, title="Topscorer Predictions", icon="🏅"),
    st.Page(page_matches, title="Match Predictions & Goals", icon="⚽"),
    st.Page(page_bracket, title="Bracket Prediction", icon="🏆"),
]

st.sidebar.caption("WK 2026 Scorito")
selected_page = st.navigation(pages, position="sidebar")
selected_page.run()
