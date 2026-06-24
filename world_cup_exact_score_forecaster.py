#!/usr/bin/env python3
"""
Scoreline-focused World Cup forecaster.

Every displayed schedule prediction is generated from a dated model snapshot:

  * group-stage matches use only results strictly before the match date;
  * knockout matches share one snapshot per round, using only results available
    before that round starts.

The Big strategy update's scoreline guardrails are retained: tree predictions
must agree with the model's outcome signal and remain plausible relative to xG.
Completed-match evaluation is walk-forward and never exposes the actual result
to either model or those guardrails.
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
    ModelParams,
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


OUTCOME_LABELS = {"home_win", "draw", "away_win"}
XG_OUTCOME_MARGIN = 1.00
MAX_SCORELINE_GOALS = 7
FUTURE_SCORELINE_XG_DISTANCE = 2.50


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


CAUSAL_MODEL_PARAMS = ModelParams(
    start_year=1980,
    k_factor=8.0,
    home_advantage=30.0,
    draw_margin=10.0,
)
EXACT_TREE_MAX_DEPTH = 5
EXACT_TREE_MIN_SAMPLES_LEAF = 1


@dataclass
class PredictionSnapshot:
    cutoff_date: pd.Timestamp
    training_data_through: pd.Timestamp | None
    ratings: dict[str, float]
    calibrator: object | None
    exact_estimator: object | None
    exact_feature_columns: list[str]
    elo_model_detail: str
    exact_model_detail: str


def score_label(home_score: object, away_score: object) -> str:
    return f"{int(float(home_score))}-{int(float(away_score))}"


def split_score_label(label: object) -> tuple[int, int]:
    home, away = str(label).split("-", maxsplit=1)
    return int(home), int(away)


def scoreline_outcome_label(home_score: object, away_score: object) -> str:
    return outcome_label(outcome_from_scores(float(home_score), float(away_score)))


def scoreline_label_outcome(label: object) -> str:
    home_score, away_score = split_score_label(label)
    return scoreline_outcome_label(home_score, away_score)


def expected_outcome_from_base(row: pd.Series) -> str:
    home_xg = pd.to_numeric(row.get("home_xg"), errors="coerce")
    away_xg = pd.to_numeric(row.get("away_xg"), errors="coerce")
    if not pd.isna(home_xg) and not pd.isna(away_xg):
        margin = float(home_xg) - float(away_xg)
        if margin >= XG_OUTCOME_MARGIN:
            return "home_win"
        if margin <= -XG_OUTCOME_MARGIN:
            return "away_win"

    base_outcome = clean_text(row.get("predicted_outcome", ""))
    if base_outcome in OUTCOME_LABELS:
        return base_outcome
    return "draw"


def bounded_score(value: float) -> int:
    if pd.isna(value):
        return 0
    return int(max(0, min(MAX_SCORELINE_GOALS, round(float(value)))))


def fallback_label_for_outcome(row: pd.Series, required_outcome: str) -> str:
    base_home = pd.to_numeric(row.get("predicted_home_score"), errors="coerce")
    base_away = pd.to_numeric(row.get("predicted_away_score"), errors="coerce")
    if not pd.isna(base_home) and not pd.isna(base_away):
        base_label = score_label(base_home, base_away)
        if scoreline_label_outcome(base_label) == required_outcome:
            return base_label

    home = bounded_score(pd.to_numeric(row.get("home_xg"), errors="coerce"))
    away = bounded_score(pd.to_numeric(row.get("away_xg"), errors="coerce"))
    if required_outcome == "home_win" and home <= away:
        home = min(MAX_SCORELINE_GOALS, away + 1)
    elif required_outcome == "away_win" and away <= home:
        away = min(MAX_SCORELINE_GOALS, home + 1)
    elif required_outcome == "draw":
        level = bounded_score(
            (
                pd.to_numeric(row.get("home_xg"), errors="coerce")
                + pd.to_numeric(row.get("away_xg"), errors="coerce")
            )
            / 2
        )
        home = level
        away = level
    return score_label(home, away)


def has_actual_score(row: pd.Series) -> bool:
    return (
        not pd.isna(pd.to_numeric(row.get("actual_home_score"), errors="coerce"))
        and not pd.isna(pd.to_numeric(row.get("actual_away_score"), errors="coerce"))
    )


def scoreline_xg_distance(label: object, row: pd.Series) -> float:
    home_score, away_score = split_score_label(label)
    home_xg = pd.to_numeric(row.get("home_xg"), errors="coerce")
    away_xg = pd.to_numeric(row.get("away_xg"), errors="coerce")
    if pd.isna(home_xg) or pd.isna(away_xg):
        return 0.0
    return abs(home_score - float(home_xg)) + abs(away_score - float(away_xg))


def plausible_future_label(label: object, row: pd.Series) -> bool:
    if has_actual_score(row):
        return True
    return scoreline_xg_distance(label, row) <= FUTURE_SCORELINE_XG_DISTANCE


def constrained_score_label(
    raw_label: object,
    base_row: pd.Series,
    class_labels: Iterable[object] | None = None,
    probabilities: Iterable[float] | None = None,
) -> str:
    required_outcome = expected_outcome_from_base(base_row)
    if (
        scoreline_label_outcome(raw_label) == required_outcome
        and plausible_future_label(raw_label, base_row)
    ):
        return str(raw_label)

    if class_labels is not None and probabilities is not None:
        best_label = None
        best_probability = 0.0
        for candidate_label, probability in zip(class_labels, probabilities):
            if scoreline_label_outcome(candidate_label) != required_outcome:
                continue
            if not plausible_future_label(candidate_label, base_row):
                continue
            probability = float(probability)
            if best_label is None or probability > best_probability:
                best_label = str(candidate_label)
                best_probability = probability
        if best_label is not None and best_probability > 0.0:
            return best_label

    return fallback_label_for_outcome(base_row, required_outcome)


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
    class_labels: Iterable[object] | None = None,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    predicted_home: list[int] = []
    predicted_away: list[int] = []
    base_predictions = base_predictions.copy().reset_index(drop=True)
    class_label_list = list(class_labels) if class_labels is not None else None

    for row_index, label in enumerate(labels):
        probability_row = probabilities[row_index] if probabilities is not None else None
        constrained_label = constrained_score_label(
            label,
            base_predictions.iloc[row_index],
            class_label_list,
            probability_row,
        )
        home_score, away_score = split_score_label(constrained_label)
        predicted_home.append(home_score)
        predicted_away.append(away_score)

    predictions = base_predictions
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
                probabilities = (
                    model.predict_proba(train_features)
                    if hasattr(model, "predict_proba")
                    else None
                )
                predictions = predictions_from_labels(
                    base_completed_predictions,
                    model.predict(train_features),
                    name,
                    detail,
                    class_labels=getattr(model, "classes_", None),
                    probabilities=probabilities,
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
            probabilities = (
                model.predict_proba(train_features)
                if hasattr(model, "predict_proba")
                else None
            )
            predictions = predictions_from_labels(
                base_completed_predictions,
                model.predict(train_features),
                name,
                detail,
                class_labels=getattr(model, "classes_", None),
                probabilities=probabilities,
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
    probabilities = (
        selection.estimator.predict_proba(aligned_features)
        if hasattr(selection.estimator, "predict_proba")
        else None
    )
    return predictions_from_labels(
        future_base_predictions,
        labels,
        selection.name,
        selection.detail,
        class_labels=getattr(selection.estimator, "classes_", None),
        probabilities=probabilities,
    )


def completed_before_cutoff(
    completed_matches: pd.DataFrame,
    cutoff_date: object,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(cutoff_date, errors="coerce")
    if pd.isna(cutoff) or completed_matches.empty:
        return completed_matches.iloc[0:0].copy()

    prior = completed_matches.copy()
    prior["date"] = pd.to_datetime(prior["date"], errors="coerce")
    return prior[
        prior["home_score"].notna()
        & prior["away_score"].notna()
        & (prior["date"] < cutoff)
    ].sort_values(["date", "match_no"], na_position="last")


def build_prediction_snapshot(
    historical: pd.DataFrame,
    completed_matches: pd.DataFrame,
    cutoff_date: object,
    params: ModelParams,
    baseline_outcome_target: float,
    recent_form_half_life_days: float,
) -> PredictionSnapshot:
    cutoff = pd.to_datetime(cutoff_date, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("A valid cutoff date is required for date-safe predictions.")

    prior = completed_before_cutoff(completed_matches, cutoff)
    baseline_ratings = train_elo_ratings(
        historical,
        params,
        cutoff_date=cutoff,
    )
    tournament_ratings = apply_elo_updates(
        baseline_ratings,
        prior,
        params,
        in_place=False,
    )
    form_adjustments = compute_recent_form_adjustments(
        prior,
        baseline_ratings,
        params,
        half_life_days=recent_form_half_life_days,
    )
    ratings = apply_recent_form_adjustments(tournament_ratings, form_adjustments)

    calibrator = None
    elo_model_detail = "Historical Elo/Poisson fallback; no prior tournament results"
    if not prior.empty:
        calibrated_result, calibrator = tune_calibrator(
            prior,
            ratings,
            params,
            baseline_outcome_target,
        )
        if calibrator is not None:
            elo_model_detail = (
                "Tournament-calibrated Elo/Poisson "
                f"(depth={calibrated_result.calibrator_depth}, "
                f"leaf={calibrated_result.calibrator_min_leaf})"
            )

    exact_estimator = None
    exact_feature_columns: list[str] = []
    exact_model_detail = "Elo/Poisson fallback; no prior scorelines"
    if not prior.empty and DecisionTreeClassifier is not None:
        prior_base_predictions = add_scoreline_metrics(
            make_predictions(
                prior,
                ratings,
                params,
                calibrator=calibrator,
            )
        )
        prior_features = build_scoreline_features(
            prior,
            prior_base_predictions,
            ratings,
            params,
        )
        exact_estimator = DecisionTreeClassifier(
            max_depth=EXACT_TREE_MAX_DEPTH,
            min_samples_leaf=EXACT_TREE_MIN_SAMPLES_LEAF,
            random_state=7,
        )
        exact_estimator.fit(prior_features, labels_from_completed(prior))
        exact_feature_columns = list(prior_features.columns)
        exact_model_detail = (
            "DecisionTreeClassifier("
            f"max_depth={EXACT_TREE_MAX_DEPTH}, "
            f"min_samples_leaf={EXACT_TREE_MIN_SAMPLES_LEAF})"
        )

    training_data_through = None if prior.empty else pd.Timestamp(prior["date"].max())
    return PredictionSnapshot(
        cutoff_date=pd.Timestamp(cutoff),
        training_data_through=training_data_through,
        ratings=ratings,
        calibrator=calibrator,
        exact_estimator=exact_estimator,
        exact_feature_columns=exact_feature_columns,
        elo_model_detail=elo_model_detail,
        exact_model_detail=exact_model_detail,
    )


def predict_fixtures_from_snapshot(
    fixtures: pd.DataFrame,
    snapshot: PredictionSnapshot,
    params: ModelParams,
    training_scope: str,
) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()

    prediction_fixtures = fixtures.copy().reset_index(drop=True)
    actual_home_scores = pd.to_numeric(
        prediction_fixtures["home_score"],
        errors="coerce",
    )
    actual_away_scores = pd.to_numeric(
        prediction_fixtures["away_score"],
        errors="coerce",
    )
    prediction_fixtures["home_score"] = np.nan
    prediction_fixtures["away_score"] = np.nan

    base_predictions = make_predictions(
        prediction_fixtures,
        snapshot.ratings,
        params,
        calibrator=snapshot.calibrator,
    )
    if snapshot.exact_estimator is None:
        exact_predictions = base_predictions.copy()
        exact_predictions["score_model"] = "exact_score_tree_fallback"
        exact_predictions["score_model_detail"] = snapshot.exact_model_detail
    else:
        exact_features = build_scoreline_features(
            prediction_fixtures,
            base_predictions,
            snapshot.ratings,
            params,
        ).reindex(columns=snapshot.exact_feature_columns, fill_value=0.0)
        labels = snapshot.exact_estimator.predict(exact_features)
        probabilities = (
            snapshot.exact_estimator.predict_proba(exact_features)
            if hasattr(snapshot.exact_estimator, "predict_proba")
            else None
        )
        exact_predictions = predictions_from_labels(
            base_predictions,
            labels,
            "exact_score_tree_walk_forward",
            snapshot.exact_model_detail,
            class_labels=getattr(snapshot.exact_estimator, "classes_", None),
            probabilities=probabilities,
        )

    exact_predictions["actual_home_score"] = actual_home_scores.to_numpy()
    exact_predictions["actual_away_score"] = actual_away_scores.to_numpy()
    exact_predictions = add_scoreline_metrics(exact_predictions)
    exact_predictions["elo_poisson_predicted_home_score"] = base_predictions[
        "predicted_home_score"
    ].to_numpy()
    exact_predictions["elo_poisson_predicted_away_score"] = base_predictions[
        "predicted_away_score"
    ].to_numpy()
    exact_predictions["elo_poisson_predicted_outcome"] = base_predictions[
        "predicted_outcome"
    ].to_numpy()
    exact_predictions["elo_poisson_model_detail"] = snapshot.elo_model_detail
    exact_predictions["training_cutoff"] = snapshot.cutoff_date
    exact_predictions["training_data_through"] = snapshot.training_data_through
    exact_predictions["training_scope"] = training_scope
    return exact_predictions


def build_date_safe_group_predictions(
    group_matches: pd.DataFrame,
    historical: pd.DataFrame,
    completed_matches: pd.DataFrame,
    params: ModelParams,
    baseline_outcome_target: float,
    recent_form_half_life_days: float,
) -> pd.DataFrame:
    if group_matches.empty:
        return pd.DataFrame()

    fixtures = group_matches.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    frames: list[pd.DataFrame] = []
    for match_date, date_fixtures in fixtures.groupby("date", sort=True, dropna=False):
        if pd.isna(match_date):
            continue
        snapshot = build_prediction_snapshot(
            historical,
            completed_matches,
            match_date,
            params,
            baseline_outcome_target,
            recent_form_half_life_days,
        )
        frames.append(
            predict_fixtures_from_snapshot(
                date_fixtures.sort_values("match_no"),
                snapshot,
                params,
                training_scope="before_match_date",
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["date", "match_no"],
        na_position="last",
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
    historical: pd.DataFrame,
    completed_matches: pd.DataFrame,
    params: ModelParams,
    baseline_outcome_target: float,
    recent_form_half_life_days: float,
) -> pd.DataFrame:
    if knockout_layout.empty:
        return pd.DataFrame()

    layout = knockout_layout.copy()
    layout["date"] = pd.to_datetime(layout["date"], errors="coerce")
    round_starts = layout.groupby("group", dropna=False)["date"].min().to_dict()
    group_lookup = group_position_lookup(group_standings)
    third_places = third_place_candidates(group_standings)
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}
    used_third_groups: set[str] = set()
    input_lookup = knockout_input.set_index("match_no").to_dict("index")
    rows: list[dict[str, object]] = []
    completed_context = completed_matches.copy()
    snapshots: dict[str, PredictionSnapshot] = {}

    for match in layout.sort_values(["date", "match_no"]).itertuples(index=False):
        match_no = int(match.match_no)
        round_name = str(match.group)
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
                    "score_model": "exact_score_tree_walk_forward",
                    "score_model_detail": "Unresolved fixture",
                    "elo_poisson_predicted_home_score": np.nan,
                    "elo_poisson_predicted_away_score": np.nan,
                    "elo_poisson_predicted_outcome": "",
                    "elo_poisson_model_detail": "Unresolved fixture",
                    "training_cutoff": round_starts.get(match.group),
                    "training_data_through": pd.NaT,
                    "training_scope": "before_knockout_round",
                }
            )
            continue

        if round_name not in snapshots:
            snapshots[round_name] = build_prediction_snapshot(
                historical,
                completed_context,
                round_starts.get(match.group, match.date),
                params,
                baseline_outcome_target,
                recent_form_half_life_days,
            )
        snapshot = snapshots[round_name]
        fixture = pd.DataFrame(
            [
                {
                    "date": match.date,
                    "group": match.group,
                    "match_no": match_no,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": actual_home_score,
                    "away_score": actual_away_score,
                    "neutral": True,
                }
            ]
        )
        prediction = predict_fixtures_from_snapshot(
            fixture,
            snapshot,
            params,
            training_scope="before_knockout_round",
        ).iloc[0]
        predicted_winner, predicted_loser, predicted_method = choose_knockout_winner(
            prediction,
            home_team,
            away_team,
            snapshot.ratings,
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
                "score_model": prediction["score_model"],
                "score_model_detail": prediction["score_model_detail"],
                "elo_poisson_predicted_home_score": prediction[
                    "elo_poisson_predicted_home_score"
                ],
                "elo_poisson_predicted_away_score": prediction[
                    "elo_poisson_predicted_away_score"
                ],
                "elo_poisson_predicted_outcome": prediction[
                    "elo_poisson_predicted_outcome"
                ],
                "elo_poisson_model_detail": prediction["elo_poisson_model_detail"],
                "training_cutoff": prediction["training_cutoff"],
                "training_data_through": prediction["training_data_through"],
                "training_scope": prediction["training_scope"],
            }
        )

        if not pd.isna(actual_home_score) and not pd.isna(actual_away_score):
            completed_context = pd.concat(
                [
                    completed_context,
                    pd.DataFrame(
                        [
                            {
                                "date": match.date,
                                "group": match.group,
                                "match_no": match_no,
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_score": actual_home_score,
                                "away_score": actual_away_score,
                                "neutral": True,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
                sort=False,
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


def build_schedule_prediction_export(
    group_predictions: pd.DataFrame,
    knockout_predictions: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not group_predictions.empty:
        frames.append(group_predictions.copy())
    if not knockout_predictions.empty:
        knockout = knockout_predictions.rename(columns={"round": "group"}).copy()
        frames.append(knockout)
    if not frames:
        return pd.DataFrame()

    schedule = pd.concat(frames, ignore_index=True, sort=False)
    schedule = schedule.rename(
        columns={
            "predicted_home_score": "exact_tree_predicted_home_score",
            "predicted_away_score": "exact_tree_predicted_away_score",
            "predicted_outcome": "exact_tree_predicted_outcome",
        }
    )
    columns = [
        "date",
        "group",
        "match_no",
        "home_team",
        "away_team",
        "actual_home_score",
        "actual_away_score",
        "exact_tree_predicted_home_score",
        "exact_tree_predicted_away_score",
        "exact_tree_predicted_outcome",
        "elo_poisson_predicted_home_score",
        "elo_poisson_predicted_away_score",
        "elo_poisson_predicted_outcome",
        "score_model",
        "score_model_detail",
        "elo_poisson_model_detail",
        "training_cutoff",
        "training_data_through",
        "training_scope",
    ]
    for column in columns:
        if column not in schedule:
            schedule[column] = np.nan
    return schedule[columns].sort_values(["date", "match_no"], na_position="last")


def write_outputs(
    output_dir: Path,
    selection: ScoreModelSelection,
    future_predictions: pd.DataFrame,
    group_standings: pd.DataFrame,
    knockout_input: pd.DataFrame,
    knockout_input_path: Path,
    knockout_predictions: pd.DataFrame,
    schedule_predictions: pd.DataFrame,
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
                "evaluation_scope": "date_safe_walk_forward_and_round_start_snapshots",
            }
        ]
    )

    selection.predictions.to_csv(output_dir / "exact_score_model_evaluation.csv", index=False)
    future_predictions.to_csv(output_dir / "exact_score_model_forecast.csv", index=False)
    group_standings.to_csv(output_dir / "exact_score_projected_group_standings.csv", index=False)
    knockout_input.to_csv(knockout_input_path, index=False)
    knockout_predictions.to_csv(output_dir / "exact_score_knockout_forecast.csv", index=False)
    schedule_predictions.to_csv(output_dir / "schedule_model_predictions.csv", index=False)
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
        default=0.50,
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

    group_predictions = build_date_safe_group_predictions(
        known_group_matches,
        historical,
        completed,
        CAUSAL_MODEL_PARAMS,
        args.baseline_outcome_target,
        args.recent_form_half_life_days,
    )
    completed_predictions = group_predictions[
        group_predictions["actual_home_score"].notna()
        & group_predictions["actual_away_score"].notna()
    ].copy()
    future_predictions = group_predictions[
        group_predictions["actual_home_score"].isna()
        | group_predictions["actual_away_score"].isna()
    ].copy()

    metrics = scoreline_metrics(completed_predictions)
    reached_target = (
        metrics["within_one_accuracy"] >= args.target_within_one
        and metrics["exact_score_accuracy"] >= args.target_exact
    )
    selection = ScoreModelSelection(
        name="exact_score_tree_walk_forward",
        family="scoreline_tree",
        detail=(
            "DecisionTreeClassifier("
            f"max_depth={EXACT_TREE_MAX_DEPTH}, "
            f"min_samples_leaf={EXACT_TREE_MIN_SAMPLES_LEAF}); "
            "retrained before each match date / knockout round"
        ),
        estimator=None,
        feature_columns=[],
        predictions=completed_predictions,
        within_one_accuracy=float(metrics["within_one_accuracy"]),
        exact_score_accuracy=float(metrics["exact_score_accuracy"]),
        outcome_accuracy=float(metrics["outcome_accuracy"]),
        reached_target=reached_target,
    )
    iterations = pd.DataFrame(
        [
            {
                "candidate": selection.name,
                "family": selection.family,
                "detail": selection.detail,
                "completed_matches": metrics["completed_matches"],
                "within_one_accuracy": metrics["within_one_accuracy"],
                "exact_score_accuracy": metrics["exact_score_accuracy"],
                "outcome_accuracy": metrics["outcome_accuracy"],
                "mean_scoreline_absolute_error": metrics[
                    "mean_scoreline_absolute_error"
                ],
                "reached_target": reached_target,
                "selected": True,
            }
        ]
    )

    if not future.empty:
        standings_cutoff = pd.to_datetime(future["date"], errors="coerce").min()
    else:
        standings_cutoff = pd.to_datetime(completed["date"], errors="coerce").max() + pd.Timedelta(
            days=1
        )
    standings_snapshot = build_prediction_snapshot(
        historical,
        completed,
        standings_cutoff,
        CAUSAL_MODEL_PARAMS,
        args.baseline_outcome_target,
        args.recent_form_half_life_days,
    )

    group_standings = project_group_standings(
        group_matches,
        future_predictions,
        standings_snapshot.ratings,
    )
    knockout_input_path = args.knockout_input or (
        args.output_dir / "exact_score_knockout_input_template.csv"
    )
    knockout_input = build_knockout_input_template(knockout_layout, knockout_input_path)
    knockout_predictions = forecast_exact_score_knockout_phase(
        knockout_layout,
        knockout_input,
        group_standings,
        historical,
        completed,
        CAUSAL_MODEL_PARAMS,
        args.baseline_outcome_target,
        args.recent_form_half_life_days,
    )
    schedule_predictions = build_schedule_prediction_export(
        group_predictions,
        knockout_predictions,
    )

    _, visual_paths = write_outputs(
        args.output_dir,
        selection,
        future_predictions,
        group_standings,
        knockout_input,
        knockout_input_path,
        knockout_predictions,
        schedule_predictions,
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
        "Note: completed-match metrics are date-safe walk-forward results; "
        "knockout models are retrained once at each round start."
    )
    print(f"Wrote: {args.output_dir / 'exact_score_model_evaluation.csv'}")
    print(f"Wrote: {args.output_dir / 'exact_score_model_forecast.csv'}")
    print(f"Wrote: {args.output_dir / 'exact_score_projected_group_standings.csv'}")
    print(f"Wrote: {knockout_input_path}")
    print(f"Wrote: {args.output_dir / 'exact_score_knockout_forecast.csv'}")
    print(f"Wrote: {args.output_dir / 'schedule_model_predictions.csv'}")
    for visual_path in visual_paths:
        print(f"Wrote: {visual_path}")
    print(f"Wrote: {args.output_dir / 'exact_score_model_iterations.csv'}")
    print(f"Wrote: {args.output_dir / 'exact_score_model_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
