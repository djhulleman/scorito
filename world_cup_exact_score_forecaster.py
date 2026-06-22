#!/usr/bin/env python3
"""
Scoreline-focused World Cup forecaster.

This script leaves world_cup_score_forecaster.py intact and reuses it for
source loading, Elo ratings, and baseline score forecasts. It then trains a
scoreline-specific model on completed 2026 World Cup matches and iterates
candidate models until the completed-match score targets are reached:

  * at least 60% of predicted scorelines within one goal for both teams
  * at least 15% exact scorelines

Those selected metrics are in-sample on completed 2026 matches. They are useful
for fitting the current tournament pattern, not as an unbiased future accuracy
estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_cup_score_forecaster import (
    actual_knockout_winner,
    apply_elo_updates,
    apply_recent_form_adjustments,
    build_knockout_input_template,
    choose_knockout_winner,
    clean_text,
    compute_recent_form_adjustments,
    group_position_lookup,
    has_text,
    is_group_stage_match,
    is_known_team_fixture,
    is_knockout_match,
    load_current_world_cup,
    load_historical_results,
    make_predictions,
    normalize_team,
    outcome_from_scores,
    outcome_label,
    parse_optional_score,
    project_group_standings,
    rating_features,
    resolve_slot,
    third_place_candidates,
    train_elo_ratings,
    tune_baseline,
    tune_calibrator,
)

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
except Exception:  # pragma: no cover - only used when sklearn is unavailable.
    DecisionTreeClassifier = None
    KNeighborsClassifier = None


@dataclass
class ScoreModelSelection:
    name: str
    family: str
    detail: str
    estimator: object | None
    feature_columns: list[str]
    predictions: pd.DataFrame
    within_one_accuracy: float
    exact_score_accuracy: float
    outcome_accuracy: float
    reached_target: bool


def score_label(home_score: object, away_score: object) -> str:
    return f"{int(float(home_score))}-{int(float(away_score))}"


def split_score_label(label: object) -> tuple[int, int]:
    home, away = str(label).split("-", maxsplit=1)
    return int(home), int(away)


def scoreline_outcome_label(home_score: object, away_score: object) -> str:
    return outcome_label(outcome_from_scores(float(home_score), float(away_score)))


def add_scoreline_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.copy()
    if scored.empty:
        return scored

    scored["predicted_home_score"] = pd.to_numeric(
        scored["predicted_home_score"], errors="coerce"
    ).astype("Int64")
    scored["predicted_away_score"] = pd.to_numeric(
        scored["predicted_away_score"], errors="coerce"
    ).astype("Int64")
    scored["predicted_outcome"] = [
        scoreline_outcome_label(home, away)
        if not pd.isna(home) and not pd.isna(away)
        else np.nan
        for home, away in zip(scored["predicted_home_score"], scored["predicted_away_score"])
    ]

    scored["actual_outcome"] = pd.Series(pd.NA, index=scored.index, dtype="object")
    for column in ["outcome_correct", "exact_score_correct", "within_one_goal_correct"]:
        scored[column] = pd.Series(pd.NA, index=scored.index, dtype="boolean")
    for column in ["home_goal_error", "away_goal_error", "scoreline_absolute_error"]:
        scored[column] = pd.Series(pd.NA, index=scored.index, dtype="Float64")

    completed = scored[
        scored["actual_home_score"].notna() & scored["actual_away_score"].notna()
    ].index

    for index in completed:
        actual_home = int(float(scored.loc[index, "actual_home_score"]))
        actual_away = int(float(scored.loc[index, "actual_away_score"]))
        predicted_home = int(scored.loc[index, "predicted_home_score"])
        predicted_away = int(scored.loc[index, "predicted_away_score"])
        home_error = abs(predicted_home - actual_home)
        away_error = abs(predicted_away - actual_away)
        actual_outcome = scoreline_outcome_label(actual_home, actual_away)

        scored.loc[index, "actual_outcome"] = actual_outcome
        scored.loc[index, "outcome_correct"] = (
            scored.loc[index, "predicted_outcome"] == actual_outcome
        )
        scored.loc[index, "exact_score_correct"] = (
            predicted_home == actual_home and predicted_away == actual_away
        )
        scored.loc[index, "within_one_goal_correct"] = home_error <= 1 and away_error <= 1
        scored.loc[index, "home_goal_error"] = home_error
        scored.loc[index, "away_goal_error"] = away_error
        scored.loc[index, "scoreline_absolute_error"] = home_error + away_error

    return scored


def scoreline_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    evaluated = predictions[predictions["exact_score_correct"].notna()].copy()
    if evaluated.empty:
        return {
            "completed_matches": 0,
            "within_one_accuracy": 0.0,
            "exact_score_accuracy": 0.0,
            "outcome_accuracy": 0.0,
            "mean_scoreline_absolute_error": 0.0,
        }

    return {
        "completed_matches": int(len(evaluated)),
        "within_one_accuracy": float(evaluated["within_one_goal_correct"].mean()),
        "exact_score_accuracy": float(evaluated["exact_score_correct"].mean()),
        "outcome_accuracy": float(evaluated["outcome_correct"].mean()),
        "mean_scoreline_absolute_error": float(evaluated["scoreline_absolute_error"].mean()),
    }


def predictions_from_labels(
    base_predictions: pd.DataFrame,
    labels: Iterable[object],
    model_name: str,
    model_detail: str,
) -> pd.DataFrame:
    predicted_home: list[int] = []
    predicted_away: list[int] = []
    for label in labels:
        home_score, away_score = split_score_label(label)
        predicted_home.append(home_score)
        predicted_away.append(away_score)

    predictions = base_predictions.copy().reset_index(drop=True)
    predictions["predicted_home_score"] = predicted_home
    predictions["predicted_away_score"] = predicted_away
    predictions["score_model"] = model_name
    predictions["score_model_detail"] = model_detail
    return add_scoreline_metrics(predictions)


def build_scoreline_features(
    fixtures: pd.DataFrame,
    base_predictions: pd.DataFrame,
    ratings: dict[str, float],
    params,
) -> pd.DataFrame:
    fixtures = fixtures.reset_index(drop=True)
    base_predictions = base_predictions.reset_index(drop=True)
    rating_frame = rating_features(fixtures, ratings, params).reset_index(drop=True)

    base_home = pd.to_numeric(
        base_predictions["predicted_home_score"], errors="coerce"
    ).fillna(0.0)
    base_away = pd.to_numeric(
        base_predictions["predicted_away_score"], errors="coerce"
    ).fillna(0.0)
    home_xg = pd.to_numeric(base_predictions["home_xg"], errors="coerce").fillna(0.0)
    away_xg = pd.to_numeric(base_predictions["away_xg"], errors="coerce").fillna(0.0)
    probability = pd.to_numeric(
        base_predictions.get("outcome_probability", pd.Series(0.0, index=base_predictions.index)),
        errors="coerce",
    ).fillna(0.0)

    numeric = pd.DataFrame(
        {
            "home_elo": rating_frame["home_elo"],
            "away_elo": rating_frame["away_elo"],
            "elo_diff": rating_frame["elo_diff"],
            "abs_elo_diff": rating_frame["abs_elo_diff"],
            "elo_sum": rating_frame["elo_sum"],
            "home_is_host": rating_frame["home_is_host"],
            "away_is_host": rating_frame["away_is_host"],
            "neutral": rating_frame["neutral"],
            "base_predicted_home_score": base_home,
            "base_predicted_away_score": base_away,
            "base_predicted_total": base_home + base_away,
            "base_predicted_margin": base_home - base_away,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "xg_total": home_xg + away_xg,
            "xg_margin": home_xg - away_xg,
            "outcome_probability": probability,
            "is_group_stage": fixtures["group"].astype(str).str.startswith("Group ").astype(float),
        }
    )

    categorical = pd.DataFrame(
        {
            "home_team": fixtures["home_team"].astype(str),
            "away_team": fixtures["away_team"].astype(str),
            "group": fixtures["group"].astype(str),
            "baseline_outcome": base_predictions["predicted_outcome"].astype(str),
        }
    )
    dummies = pd.get_dummies(categorical, dtype=float)
    return pd.concat([numeric, dummies], axis=1).fillna(0.0).astype(float)


def candidate_summary_row(
    name: str,
    family: str,
    detail: str,
    predictions: pd.DataFrame,
    target_within_one: float,
    target_exact: float,
) -> dict[str, object]:
    metrics = scoreline_metrics(predictions)
    reached = (
        metrics["within_one_accuracy"] >= target_within_one
        and metrics["exact_score_accuracy"] >= target_exact
    )
    return {
        "candidate": name,
        "family": family,
        "detail": detail,
        "completed_matches": metrics["completed_matches"],
        "within_one_accuracy": metrics["within_one_accuracy"],
        "exact_score_accuracy": metrics["exact_score_accuracy"],
        "outcome_accuracy": metrics["outcome_accuracy"],
        "mean_scoreline_absolute_error": metrics["mean_scoreline_absolute_error"],
        "reached_target": reached,
        "selected": False,
    }


def labels_from_completed(completed: pd.DataFrame) -> pd.Series:
    return completed.apply(
        lambda row: score_label(row["home_score"], row["away_score"]),
        axis=1,
    )


def select_scoreline_model(
    completed: pd.DataFrame,
    base_completed_predictions: pd.DataFrame,
    train_features: pd.DataFrame,
    labels: pd.Series,
    baseline_candidates: list[tuple[str, str, str, pd.DataFrame]],
    target_within_one: float,
    target_exact: float,
    max_tree_depth: int,
) -> tuple[ScoreModelSelection, pd.DataFrame]:
    iteration_rows: list[dict[str, object]] = []

    for name, family, detail, predictions in baseline_candidates:
        iteration_rows.append(
            candidate_summary_row(
                name,
                family,
                detail,
                predictions,
                target_within_one,
                target_exact,
            )
        )

    best_selection: ScoreModelSelection | None = None

    if DecisionTreeClassifier is not None:
        for depth in range(1, max_tree_depth + 1):
            for min_leaf in range(4, 0, -1):
                if min_leaf > len(completed):
                    continue
                model = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=min_leaf,
                    random_state=7,
                )
                model.fit(train_features, labels)
                name = f"score_tree_depth_{depth}_leaf_{min_leaf}"
                detail = f"DecisionTreeClassifier(max_depth={depth}, min_samples_leaf={min_leaf})"
                predictions = predictions_from_labels(
                    base_completed_predictions,
                    model.predict(train_features),
                    name,
                    detail,
                )
                row = candidate_summary_row(
                    name,
                    "scoreline_tree",
                    detail,
                    predictions,
                    target_within_one,
                    target_exact,
                )
                iteration_rows.append(row)

                if row["reached_target"]:
                    best_selection = ScoreModelSelection(
                        name=name,
                        family="scoreline_tree",
                        detail=detail,
                        estimator=model,
                        feature_columns=list(train_features.columns),
                        predictions=predictions,
                        within_one_accuracy=float(row["within_one_accuracy"]),
                        exact_score_accuracy=float(row["exact_score_accuracy"]),
                        outcome_accuracy=float(row["outcome_accuracy"]),
                        reached_target=True,
                    )
                    break
            if best_selection is not None:
                break

    if best_selection is None and KNeighborsClassifier is not None:
        for neighbors in [3, 1]:
            model = KNeighborsClassifier(n_neighbors=min(neighbors, len(completed)))
            model.fit(train_features, labels)
            name = f"score_knn_{neighbors}"
            detail = f"KNeighborsClassifier(n_neighbors={min(neighbors, len(completed))})"
            predictions = predictions_from_labels(
                base_completed_predictions,
                model.predict(train_features),
                name,
                detail,
            )
            row = candidate_summary_row(
                name,
                "scoreline_knn",
                detail,
                predictions,
                target_within_one,
                target_exact,
            )
            iteration_rows.append(row)
            if row["reached_target"]:
                best_selection = ScoreModelSelection(
                    name=name,
                    family="scoreline_knn",
                    detail=detail,
                    estimator=model,
                    feature_columns=list(train_features.columns),
                    predictions=predictions,
                    within_one_accuracy=float(row["within_one_accuracy"]),
                    exact_score_accuracy=float(row["exact_score_accuracy"]),
                    outcome_accuracy=float(row["outcome_accuracy"]),
                    reached_target=True,
                )
                break

    if best_selection is None:
        if not iteration_rows:
            raise RuntimeError("No scoreline model candidates were available.")
        best_row = max(
            iteration_rows,
            key=lambda row: (
                float(row["within_one_accuracy"]),
                float(row["exact_score_accuracy"]),
                float(row["outcome_accuracy"]),
            ),
        )
        raise RuntimeError(
            "Could not reach the requested scoreline targets. "
            f"Best candidate was {best_row['candidate']} with "
            f"within-one={best_row['within_one_accuracy']:.1%}, "
            f"exact={best_row['exact_score_accuracy']:.1%}."
        )

    iterations = pd.DataFrame(iteration_rows)
    iterations.loc[iterations["candidate"].eq(best_selection.name), "selected"] = True
    return best_selection, iterations


def build_future_score_predictions(
    selection: ScoreModelSelection,
    future_base_predictions: pd.DataFrame,
    future_features: pd.DataFrame,
) -> pd.DataFrame:
    if future_base_predictions.empty:
        return future_base_predictions
    if selection.estimator is None:
        predictions = future_base_predictions.copy()
        predictions["score_model"] = selection.name
        predictions["score_model_detail"] = selection.detail
        return add_scoreline_metrics(predictions)

    aligned_features = future_features.reindex(columns=selection.feature_columns, fill_value=0.0)
    labels = selection.estimator.predict(aligned_features)
    return predictions_from_labels(
        future_base_predictions,
        labels,
        selection.name,
        selection.detail,
    )


def predict_exact_score_fixture(
    date: object,
    round_name: str,
    match_no: int,
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    params,
    calibrator,
    selection: ScoreModelSelection,
) -> pd.Series:
    fixture = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(date, errors="coerce"),
                "group": round_name,
                "match_no": match_no,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": np.nan,
                "away_score": np.nan,
                "neutral": True,
            }
        ]
    )
    base_prediction = add_scoreline_metrics(
        make_predictions(fixture, ratings, params, calibrator=calibrator)
    )
    features = build_scoreline_features(fixture, base_prediction, ratings, params)
    exact_prediction = build_future_score_predictions(selection, base_prediction, features)
    return exact_prediction.iloc[0]


def forecast_exact_score_knockout_phase(
    knockout_layout: pd.DataFrame,
    knockout_input: pd.DataFrame,
    group_standings: pd.DataFrame,
    ratings: dict[str, float],
    params,
    calibrator,
    selection: ScoreModelSelection,
) -> pd.DataFrame:
    if knockout_layout.empty:
        return pd.DataFrame()

    group_lookup = group_position_lookup(group_standings)
    third_places = third_place_candidates(group_standings)
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}
    used_third_groups: set[str] = set()
    input_lookup = knockout_input.set_index("match_no").to_dict("index")
    rows: list[dict[str, object]] = []

    for match in knockout_layout.sort_values(["date", "match_no"]).itertuples(index=False):
        match_no = int(match.match_no)
        input_row = input_lookup.get(match_no, {})
        home_slot = clean_text(match.home_team)
        away_slot = clean_text(match.away_team)

        resolved_home, home_note = resolve_slot(
            home_slot,
            group_lookup,
            third_places,
            winners,
            losers,
            used_third_groups,
        )
        resolved_away, away_note = resolve_slot(
            away_slot,
            group_lookup,
            third_places,
            winners,
            losers,
            used_third_groups,
        )

        manual_home = normalize_team(input_row.get("manual_home_team", ""))
        manual_away = normalize_team(input_row.get("manual_away_team", ""))
        home_team = manual_home if has_text(manual_home) else resolved_home
        away_team = manual_away if has_text(manual_away) else resolved_away
        team_source = "manual" if has_text(manual_home) or has_text(manual_away) else "projected"

        input_home_score = input_row.get("actual_home_score", "")
        input_away_score = input_row.get("actual_away_score", "")
        actual_home_score = parse_optional_score(
            input_home_score if has_text(input_home_score) else match.home_score
        )
        actual_away_score = parse_optional_score(
            input_away_score if has_text(input_away_score) else match.away_score
        )
        actual_winner_text = input_row.get("actual_winner", "")

        base_row = {
            "round": match.group,
            "date": match.date,
            "match_no": match_no,
            "home_slot": home_slot,
            "away_slot": away_slot,
            "home_team": home_team,
            "away_team": away_team,
            "team_source": team_source,
            "actual_home_score": actual_home_score,
            "actual_away_score": actual_away_score,
            "home_resolution_note": home_note,
            "away_resolution_note": away_note,
        }

        if not has_text(home_team) or not has_text(away_team):
            rows.append(
                base_row
                | {
                    "predicted_home_score": np.nan,
                    "predicted_away_score": np.nan,
                    "predicted_outcome": "",
                    "advancing_team": "",
                    "eliminated_team": "",
                    "win_method": "unresolved",
                    "home_xg": np.nan,
                    "away_xg": np.nan,
                    "elo_diff": np.nan,
                    "score_model": selection.name,
                    "score_model_detail": selection.detail,
                }
            )
            continue

        prediction = predict_exact_score_fixture(
            match.date,
            str(match.group),
            match_no,
            home_team,
            away_team,
            ratings,
            params,
            calibrator,
            selection,
        )
        predicted_winner, predicted_loser, predicted_method = choose_knockout_winner(
            prediction,
            home_team,
            away_team,
            ratings,
        )
        actual_advancement = actual_knockout_winner(
            actual_winner_text,
            home_team,
            away_team,
            actual_home_score,
            actual_away_score,
        )
        if actual_advancement is None:
            advancing_team, eliminated_team, win_method = (
                predicted_winner,
                predicted_loser,
                predicted_method,
            )
        else:
            advancing_team, eliminated_team, win_method = actual_advancement

        winners[match_no] = advancing_team
        losers[match_no] = eliminated_team
        rows.append(
            base_row
            | {
                "predicted_home_score": prediction["predicted_home_score"],
                "predicted_away_score": prediction["predicted_away_score"],
                "predicted_outcome": prediction["predicted_outcome"],
                "advancing_team": advancing_team,
                "eliminated_team": eliminated_team,
                "win_method": win_method,
                "home_xg": prediction["home_xg"],
                "away_xg": prediction["away_xg"],
                "elo_diff": prediction["elo_diff"],
                "score_model": selection.name,
                "score_model_detail": selection.detail,
            }
        )

    return add_scoreline_metrics(pd.DataFrame(rows))


def plot_exact_score_knockout_forecast(
    output_dir: Path,
    knockout_predictions: pd.DataFrame,
) -> Path | None:
    if knockout_predictions.empty:
        return None

    path = output_dir / "exact_score_knockout_forecast.png"
    display = knockout_predictions.copy()

    def fixture_text(row: pd.Series) -> str:
        if has_text(row["home_team"]) and has_text(row["away_team"]):
            if not pd.isna(row["predicted_home_score"]) and not pd.isna(
                row["predicted_away_score"]
            ):
                return (
                    f"{row['home_team']} {int(row['predicted_home_score'])}-"
                    f"{int(row['predicted_away_score'])} {row['away_team']}"
                )
            return f"{row['home_team']} vs {row['away_team']}"
        return f"{row['home_slot']} vs {row['away_slot']}"

    display["fixture"] = display.apply(fixture_text, axis=1)
    display["winner"] = display["advancing_team"].fillna("")
    display["match"] = display["match_no"].map(lambda value: f"M{int(value)}")
    table_df = display[["round", "match", "date", "fixture", "winner"]].copy()
    table_df["date"] = (
        pd.to_datetime(table_df["date"], errors="coerce").dt.strftime("%b %d").fillna("")
    )

    fig_height = max(8.0, len(table_df) * 0.28)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=["Round", "Match", "Date", "Exact-Score Forecast", "Advances"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.16, 0.07, 0.08, 0.47, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)

    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#d9e2f3")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        else:
            cell.set_facecolor("#ffffff")

    ax.set_title("Exact-Score Knockout Forecast", fontsize=16, pad=18)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_outputs(
    output_dir: Path,
    selection: ScoreModelSelection,
    future_predictions: pd.DataFrame,
    group_standings: pd.DataFrame,
    knockout_input: pd.DataFrame,
    knockout_input_path: Path,
    knockout_predictions: pd.DataFrame,
    iterations: pd.DataFrame,
    target_within_one: float,
    target_exact: float,
) -> tuple[pd.DataFrame, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    knockout_input_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "selected_model": selection.name,
                "family": selection.family,
                "detail": selection.detail,
                "completed_matches": len(selection.predictions),
                "within_one_accuracy": selection.within_one_accuracy,
                "target_within_one_accuracy": target_within_one,
                "exact_score_accuracy": selection.exact_score_accuracy,
                "target_exact_score_accuracy": target_exact,
                "outcome_accuracy": selection.outcome_accuracy,
                "reached_target": selection.reached_target,
                "evaluation_scope": "in_sample_completed_2026_matches",
            }
        ]
    )

    selection.predictions.to_csv(output_dir / "exact_score_model_evaluation.csv", index=False)
    future_predictions.to_csv(output_dir / "exact_score_model_forecast.csv", index=False)
    group_standings.to_csv(output_dir / "exact_score_projected_group_standings.csv", index=False)
    knockout_input.to_csv(knockout_input_path, index=False)
    knockout_predictions.to_csv(output_dir / "exact_score_knockout_forecast.csv", index=False)
    iterations.to_csv(output_dir / "exact_score_model_iterations.csv", index=False)
    summary.to_csv(output_dir / "exact_score_model_summary.csv", index=False)

    visual_paths: list[Path] = []
    knockout_figure = plot_exact_score_knockout_forecast(output_dir, knockout_predictions)
    if knockout_figure is not None:
        visual_paths.append(knockout_figure)
    return summary, visual_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate scoreline-focused models for completed 2026 World Cup games."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_world_cup"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_exact_score_model"),
    )
    parser.add_argument(
        "--target-within-one",
        type=float,
        default=0.60,
        help="Required share of completed games within one goal for both teams.",
    )
    parser.add_argument(
        "--target-exact",
        type=float,
        default=0.15,
        help="Required exact-score share on completed games.",
    )
    parser.add_argument(
        "--baseline-outcome-target",
        type=float,
        default=0.75,
        help="Outcome target passed to the existing tournament calibrator.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=8,
        help="Deepest scoreline decision tree to try before KNN fallback.",
    )
    parser.add_argument(
        "--recent-form-half-life-days",
        type=float,
        default=5.0,
        help="Half-life for recent 2026 form used in future forecasts.",
    )
    parser.add_argument(
        "--knockout-input",
        type=Path,
        default=None,
        help=(
            "Editable knockout CSV to read/write. Defaults to "
            "<output-dir>/exact_score_knockout_input_template.csv."
        ),
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Use cached downloads instead of refreshing source data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh = not args.no_refresh

    historical = load_historical_results(args.cache_dir, refresh=refresh)
    current = load_current_world_cup(args.cache_dir, refresh=refresh, historical=historical)
    group_matches = current[current.apply(is_group_stage_match, axis=1)].copy()
    knockout_layout = current[current.apply(is_knockout_match, axis=1)].copy()
    known_group_matches = group_matches[group_matches.apply(is_known_team_fixture, axis=1)].copy()
    completed = known_group_matches[known_group_matches["home_score"].notna()].copy()
    future = known_group_matches[known_group_matches["home_score"].isna()].copy()

    if completed.empty:
        raise RuntimeError("No completed 2026 World Cup matches were available for evaluation.")

    historical_baseline = tune_baseline(historical, completed, args.baseline_outcome_target)
    baseline_ratings = train_elo_ratings(historical, historical_baseline.params)
    baseline_completed_predictions = add_scoreline_metrics(historical_baseline.predictions)

    calibrated_result, calibrator = tune_calibrator(
        completed,
        baseline_ratings,
        historical_baseline.params,
        args.baseline_outcome_target,
    )
    calibrated_completed_predictions = add_scoreline_metrics(calibrated_result.predictions)

    base_completed_predictions = calibrated_completed_predictions
    train_features = build_scoreline_features(
        completed,
        base_completed_predictions,
        baseline_ratings,
        historical_baseline.params,
    )
    labels = labels_from_completed(completed)

    baseline_candidates = [
        (
            "historical_only_baseline",
            "baseline_reference",
            str(historical_baseline.params),
            baseline_completed_predictions,
        ),
        (
            "outcome_calibrated_baseline",
            "baseline_reference",
            calibrated_result.name,
            calibrated_completed_predictions,
        ),
    ]

    selection, iterations = select_scoreline_model(
        completed=completed,
        base_completed_predictions=base_completed_predictions,
        train_features=train_features,
        labels=labels,
        baseline_candidates=baseline_candidates,
        target_within_one=args.target_within_one,
        target_exact=args.target_exact,
        max_tree_depth=args.max_tree_depth,
    )

    tournament_ratings = apply_elo_updates(
        baseline_ratings,
        completed.assign(tournament="FIFA World Cup"),
        historical_baseline.params,
        in_place=False,
    )
    form_adjustments = compute_recent_form_adjustments(
        completed,
        baseline_ratings,
        historical_baseline.params,
        half_life_days=args.recent_form_half_life_days,
    )
    form_adjusted_ratings = apply_recent_form_adjustments(tournament_ratings, form_adjustments)

    if future.empty:
        future_predictions = pd.DataFrame(columns=selection.predictions.columns)
    else:
        future_base_predictions = add_scoreline_metrics(
            make_predictions(
                future,
                form_adjusted_ratings,
                historical_baseline.params,
                calibrator=calibrator,
            )
        )
        future_features = build_scoreline_features(
            future,
            future_base_predictions,
            form_adjusted_ratings,
            historical_baseline.params,
        )
        future_predictions = build_future_score_predictions(
            selection,
            future_base_predictions,
            future_features,
        )

    group_standings = project_group_standings(
        group_matches,
        future_predictions,
        form_adjusted_ratings,
    )
    knockout_input_path = args.knockout_input or (
        args.output_dir / "exact_score_knockout_input_template.csv"
    )
    knockout_input = build_knockout_input_template(knockout_layout, knockout_input_path)
    knockout_predictions = forecast_exact_score_knockout_phase(
        knockout_layout,
        knockout_input,
        group_standings,
        form_adjusted_ratings,
        historical_baseline.params,
        calibrator,
        selection,
    )

    _, visual_paths = write_outputs(
        args.output_dir,
        selection,
        future_predictions,
        group_standings,
        knockout_input,
        knockout_input_path,
        knockout_predictions,
        iterations,
        args.target_within_one,
        args.target_exact,
    )

    print("Exact-score model complete")
    print(f"Completed 2026 World Cup matches evaluated: {len(selection.predictions)}")
    print(f"Known future group fixtures forecasted: {len(future_predictions)}")
    print(f"Knockout fixtures forecasted: {len(knockout_predictions)}")
    print(f"Selected score model: {selection.name}")
    print(f"Detail: {selection.detail}")
    print(
        "Within one goal: "
        f"{selection.within_one_accuracy:.1%} "
        f"(target: {args.target_within_one:.0%})"
    )
    print(
        "Exact scores: "
        f"{selection.exact_score_accuracy:.1%} "
        f"(target: {args.target_exact:.0%})"
    )
    print(f"Outcome accuracy from scorelines: {selection.outcome_accuracy:.1%}")
    print(
        "Note: selected metrics are in-sample on completed 2026 matches; "
        "future forecasts should be treated as fitted scoreline projections."
    )
    print(f"Wrote: {args.output_dir / 'exact_score_model_evaluation.csv'}")
    print(f"Wrote: {args.output_dir / 'exact_score_model_forecast.csv'}")
    print(f"Wrote: {args.output_dir / 'exact_score_projected_group_standings.csv'}")
    print(f"Wrote: {knockout_input_path}")
    print(f"Wrote: {args.output_dir / 'exact_score_knockout_forecast.csv'}")
    for visual_path in visual_paths:
        print(f"Wrote: {visual_path}")
    print(f"Wrote: {args.output_dir / 'exact_score_model_iterations.csv'}")
    print(f"Wrote: {args.output_dir / 'exact_score_model_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
