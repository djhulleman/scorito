#!/usr/bin/env python3
"""
Alternative World Cup model: forecast total goals first, estimate win odds,
then split expected goals across the two teams.

This script intentionally avoids leaning on very old country results. By
default it only uses the last four years of internationals, applies exponential
recency decay inside that window, and gives completed 2026 World Cup matches
extra weight. The model is built for entering upcoming group and knockout
predictions where current form matters more than an eight-year-old tournament.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from world_cup_score_forecaster import (
    DEFAULT_RATING,
    TOURNAMENT_START,
    actual_knockout_winner,
    build_knockout_input_template,
    clean_text,
    clipped,
    has_text,
    is_group_stage_match,
    is_known_team_fixture,
    is_knockout_match,
    load_current_world_cup,
    load_historical_results,
    normalize_team,
    parse_optional_score,
    project_group_standings,
    resolve_slot,
    third_place_candidates,
    group_position_lookup,
)


@dataclass(frozen=True)
class TotalGoalsConfig:
    lookback_years: float = 4.0
    half_life_days: float = 365.0
    current_world_cup_weight: float = 2.75
    elo_k: float = 32.0
    home_advantage: float = 55.0
    min_group_total_goals: float = 2.05
    min_knockout_total_goals: float = 2.35
    max_total_goals: float = 5.25
    score_max_goals: int = 8


@dataclass
class TotalGoalsModel:
    config: TotalGoalsConfig
    as_of_date: pd.Timestamp
    ratings: dict[str, float]
    team_goal_rates: pd.DataFrame
    global_total_goals: float
    current_tournament_total_goals: float


def outcome_from_scores(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return "home_win"
    if away_score > home_score:
        return "away_win"
    return "draw"


def outcome_score(home_score: float, away_score: float) -> float:
    if home_score > away_score:
        return 1.0
    if away_score > home_score:
        return 0.0
    return 0.5


def poisson_prob(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def decimal_odds(probability: float) -> float:
    if probability <= 0:
        return np.nan
    return round(1.0 / probability, 2)


def prepare_recent_training_matches(
    historical: pd.DataFrame,
    completed_current: pd.DataFrame,
    as_of_date: pd.Timestamp,
    config: TotalGoalsConfig,
) -> pd.DataFrame:
    cutoff = as_of_date - pd.Timedelta(days=int(config.lookback_years * 365.25))
    columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
        "is_current_world_cup",
    ]
    base = historical[
        historical["home_score"].notna()
        & historical["away_score"].notna()
        & (historical["date"] < as_of_date)
        & (historical["date"] >= cutoff)
        & ~(
            historical["tournament"].eq("FIFA World Cup")
            & (historical["date"] >= TOURNAMENT_START)
        )
    ].copy()
    base["is_current_world_cup"] = False

    current = completed_current[
        completed_current["home_score"].notna()
        & completed_current["away_score"].notna()
        & (completed_current["date"] < as_of_date)
        & (completed_current["date"] >= cutoff)
    ].copy()
    if not current.empty:
        current["tournament"] = "FIFA World Cup"
        current["is_current_world_cup"] = True
        for column in ["city", "country"]:
            if column not in current.columns:
                current[column] = ""
        if "neutral" not in current.columns:
            current["neutral"] = True
    else:
        current = pd.DataFrame(columns=columns)

    frames = [base[columns]]
    if not current.empty:
        frames.append(current[columns])
    training = pd.concat(frames, ignore_index=True)
    if training.empty:
        return training

    training["home_team"] = training["home_team"].map(normalize_team)
    training["away_team"] = training["away_team"].map(normalize_team)
    training["neutral"] = training["neutral"].where(training["neutral"].notna(), True).astype(bool)
    days_old = (as_of_date - training["date"]).dt.days.clip(lower=0)
    training["time_weight"] = 0.5 ** (days_old / config.half_life_days)
    training["weight"] = training["time_weight"] * np.where(
        training["is_current_world_cup"],
        config.current_world_cup_weight,
        1.0,
    )
    return training.sort_values("date").reset_index(drop=True)


def fit_recent_ratings(training: pd.DataFrame, config: TotalGoalsConfig) -> dict[str, float]:
    ratings: dict[str, float] = {}
    if training.empty:
        return ratings

    for row in training.sort_values("date").itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_rating = ratings.get(home, DEFAULT_RATING)
        away_rating = ratings.get(away, DEFAULT_RATING)
        home_edge = 0.0 if bool(row.neutral) else config.home_advantage
        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_rating - home_rating - home_edge) / 400.0)
        )
        actual_home = outcome_score(float(row.home_score), float(row.away_score))
        goal_diff = abs(float(row.home_score) - float(row.away_score))
        margin_multiplier = 1.0 + min(goal_diff, 4.0) * 0.18
        delta = config.elo_k * float(row.weight) * margin_multiplier * (
            actual_home - expected_home
        )
        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta
    return ratings


def fit_team_goal_rates(training: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    if training.empty:
        empty = pd.DataFrame(
            columns=[
                "team",
                "weighted_matches",
                "goals_for_rate",
                "goals_against_rate",
                "match_total_rate",
                "goal_difference_rate",
            ]
        )
        return empty, 2.65, 2.65

    global_total = float(
        np.average(
            training["home_score"].astype(float) + training["away_score"].astype(float),
            weights=training["weight"].astype(float),
        )
    )
    current = training[training["is_current_world_cup"]]
    if current.empty:
        current_total = global_total
    else:
        current_total = float(
            np.average(
                current["home_score"].astype(float) + current["away_score"].astype(float),
                weights=current["weight"].astype(float),
            )
        )

    team_rows: list[dict[str, float | str]] = []
    teams = sorted(set(training["home_team"]) | set(training["away_team"]))
    for team in teams:
        home_rows = training[training["home_team"].eq(team)]
        away_rows = training[training["away_team"].eq(team)]
        weighted_matches = float(home_rows["weight"].sum() + away_rows["weight"].sum())
        if weighted_matches <= 0:
            continue

        goals_for = (
            (home_rows["home_score"].astype(float) * home_rows["weight"]).sum()
            + (away_rows["away_score"].astype(float) * away_rows["weight"]).sum()
        )
        goals_against = (
            (home_rows["away_score"].astype(float) * home_rows["weight"]).sum()
            + (away_rows["home_score"].astype(float) * away_rows["weight"]).sum()
        )
        total_goals = (
            (
                (home_rows["home_score"].astype(float) + home_rows["away_score"].astype(float))
                * home_rows["weight"]
            ).sum()
            + (
                (away_rows["home_score"].astype(float) + away_rows["away_score"].astype(float))
                * away_rows["weight"]
            ).sum()
        )
        team_rows.append(
            {
                "team": team,
                "weighted_matches": round(weighted_matches, 3),
                "goals_for_rate": goals_for / weighted_matches,
                "goals_against_rate": goals_against / weighted_matches,
                "match_total_rate": total_goals / weighted_matches,
                "goal_difference_rate": (goals_for - goals_against) / weighted_matches,
            }
        )

    rates = pd.DataFrame(team_rows)
    return rates, global_total, current_total


def fit_total_goals_model(
    historical: pd.DataFrame,
    completed_current: pd.DataFrame,
    as_of_date: pd.Timestamp,
    config: TotalGoalsConfig,
) -> TotalGoalsModel:
    training = prepare_recent_training_matches(historical, completed_current, as_of_date, config)
    ratings = fit_recent_ratings(training, config)
    team_goal_rates, global_total, current_total = fit_team_goal_rates(training)
    return TotalGoalsModel(
        config=config,
        as_of_date=as_of_date,
        ratings=ratings,
        team_goal_rates=team_goal_rates,
        global_total_goals=global_total,
        current_tournament_total_goals=current_total,
    )


def team_goal_profile(model: TotalGoalsModel, team: str) -> dict[str, float]:
    rates = model.team_goal_rates
    if rates.empty or team not in set(rates["team"]):
        return {
            "weighted_matches": 0.0,
            "goals_for_rate": model.global_total_goals / 2.0,
            "goals_against_rate": model.global_total_goals / 2.0,
            "match_total_rate": model.global_total_goals,
            "goal_difference_rate": 0.0,
        }
    row = rates[rates["team"].eq(team)].iloc[0]
    return {
        "weighted_matches": float(row["weighted_matches"]),
        "goals_for_rate": float(row["goals_for_rate"]),
        "goals_against_rate": float(row["goals_against_rate"]),
        "match_total_rate": float(row["match_total_rate"]),
        "goal_difference_rate": float(row["goal_difference_rate"]),
    }


def fixture_round_floor(round_name: str, config: TotalGoalsConfig) -> float:
    if str(round_name).startswith("Group "):
        return config.min_group_total_goals
    return config.min_knockout_total_goals


def estimate_total_goals(
    model: TotalGoalsModel,
    home_team: str,
    away_team: str,
    round_name: str,
    rating_diff: float,
) -> float:
    home_profile = team_goal_profile(model, home_team)
    away_profile = team_goal_profile(model, away_team)
    pair_total_tendency = (
        home_profile["match_total_rate"] + away_profile["match_total_rate"]
    ) / 2.0
    pair_attack_defense = (
        home_profile["goals_for_rate"]
        + away_profile["goals_for_rate"]
        + home_profile["goals_against_rate"]
        + away_profile["goals_against_rate"]
    ) / 2.0
    mismatch_boost = min(abs(rating_diff) / 500.0, 0.55)

    total_goals = (
        0.34 * model.global_total_goals
        + 0.28 * model.current_tournament_total_goals
        + 0.23 * pair_total_tendency
        + 0.15 * pair_attack_defense
        + mismatch_boost
    )
    return round(
        clipped(
            total_goals,
            fixture_round_floor(round_name, model.config),
            model.config.max_total_goals,
        ),
        3,
    )


def estimate_win_probabilities(
    model: TotalGoalsModel,
    total_goals: float,
    rating_diff: float,
) -> tuple[float, float, float]:
    home_strength_share = 1.0 / (1.0 + math.exp(-rating_diff / 235.0))
    draw_probability = clipped(
        0.295 - 0.032 * (total_goals - 2.65) - 0.00023 * abs(rating_diff),
        0.145,
        0.335,
    )
    non_draw = 1.0 - draw_probability
    home_win = non_draw * home_strength_share
    away_win = non_draw * (1.0 - home_strength_share)
    return round(home_win, 4), round(draw_probability, 4), round(away_win, 4)


def split_total_goals(
    total_goals: float,
    home_win_probability: float,
    away_win_probability: float,
    rating_diff: float,
) -> tuple[float, float]:
    odds_dominance = home_win_probability - away_win_probability
    rating_dominance = math.tanh(rating_diff / 520.0)
    expected_goal_difference = total_goals * (
        0.42 * odds_dominance + 0.18 * rating_dominance
    )
    expected_goal_difference = clipped(
        expected_goal_difference,
        -(total_goals - 0.45),
        total_goals - 0.45,
    )
    home_xg = max(0.22, (total_goals + expected_goal_difference) / 2.0)
    away_xg = max(0.22, total_goals - home_xg)
    scale = total_goals / (home_xg + away_xg)
    return round(home_xg * scale, 3), round(away_xg * scale, 3)


def score_distribution_pick(
    home_xg: float,
    away_xg: float,
    max_goals: int,
    forced_outcome: str,
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for home_goals in range(max_goals + 1):
        home_prob = poisson_prob(home_xg, home_goals)
        for away_goals in range(max_goals + 1):
            outcome = outcome_from_scores(home_goals, away_goals)
            if forced_outcome != outcome:
                continue
            probability = home_prob * poisson_prob(away_xg, away_goals)
            if best is None or probability > best[0]:
                best = (probability, home_goals, away_goals)
    if best is None:
        return score_distribution_pick(home_xg, away_xg, max_goals, "draw")
    return best[1], best[2]


def predict_fixture(
    model: TotalGoalsModel,
    fixture: pd.Series,
) -> dict[str, object]:
    home_team = normalize_team(fixture["home_team"])
    away_team = normalize_team(fixture["away_team"])
    round_name = str(fixture["group"])
    neutral = bool(fixture.get("neutral", True))
    home_edge = 0.0 if neutral else model.config.home_advantage
    home_rating = model.ratings.get(home_team, DEFAULT_RATING)
    away_rating = model.ratings.get(away_team, DEFAULT_RATING)
    rating_diff = home_rating + home_edge - away_rating

    expected_total_goals = estimate_total_goals(
        model,
        home_team,
        away_team,
        round_name,
        rating_diff,
    )
    home_win, draw, away_win = estimate_win_probabilities(
        model,
        expected_total_goals,
        rating_diff,
    )
    home_xg, away_xg = split_total_goals(
        expected_total_goals,
        home_win,
        away_win,
        rating_diff,
    )
    if home_win >= draw and home_win >= away_win:
        predicted_outcome = "home_win"
    elif away_win >= draw and away_win >= home_win:
        predicted_outcome = "away_win"
    else:
        predicted_outcome = "draw"
    predicted_home, predicted_away = score_distribution_pick(
        home_xg,
        away_xg,
        model.config.score_max_goals,
        predicted_outcome,
    )

    return {
        "date": fixture["date"],
        "group": round_name,
        "match_no": int(fixture["match_no"]) if not pd.isna(fixture["match_no"]) else np.nan,
        "home_team": home_team,
        "away_team": away_team,
        "actual_home_score": fixture.get("home_score", np.nan),
        "actual_away_score": fixture.get("away_score", np.nan),
        "predicted_home_score": predicted_home,
        "predicted_away_score": predicted_away,
        "predicted_outcome": predicted_outcome,
        "expected_total_goals": expected_total_goals,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_win_probability": home_win,
        "draw_probability": draw,
        "away_win_probability": away_win,
        "home_decimal_odds": decimal_odds(home_win),
        "draw_decimal_odds": decimal_odds(draw),
        "away_decimal_odds": decimal_odds(away_win),
        "rating_diff": round(rating_diff, 2),
        "home_recent_rating": round(home_rating, 2),
        "away_recent_rating": round(away_rating, 2),
    }


def make_predictions(
    model: TotalGoalsModel,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    rows = [predict_fixture(model, row) for _, row in fixtures.iterrows()]
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return predictions

    predictions["actual_outcome"] = np.nan
    predictions["outcome_correct"] = np.nan
    predictions["exact_score_correct"] = np.nan
    predictions["total_goals_absolute_error"] = np.nan

    completed = predictions[
        predictions["actual_home_score"].notna() & predictions["actual_away_score"].notna()
    ].index
    for index in completed:
        actual_home = float(predictions.loc[index, "actual_home_score"])
        actual_away = float(predictions.loc[index, "actual_away_score"])
        actual_outcome = outcome_from_scores(actual_home, actual_away)
        predicted_total = float(predictions.loc[index, "expected_total_goals"])
        actual_total = actual_home + actual_away
        predictions.loc[index, "actual_outcome"] = actual_outcome
        predictions.loc[index, "outcome_correct"] = (
            predictions.loc[index, "predicted_outcome"] == actual_outcome
        )
        predictions.loc[index, "exact_score_correct"] = (
            int(actual_home) == int(predictions.loc[index, "predicted_home_score"])
            and int(actual_away) == int(predictions.loc[index, "predicted_away_score"])
        )
        predictions.loc[index, "total_goals_absolute_error"] = abs(predicted_total - actual_total)
    return predictions


def rolling_evaluate_completed(
    historical: pd.DataFrame,
    completed_current: pd.DataFrame,
    config: TotalGoalsConfig,
) -> pd.DataFrame:
    rows = []
    completed_sorted = completed_current.sort_values(["date", "match_no"]).reset_index(drop=True)
    for _, fixture in completed_sorted.iterrows():
        as_of = pd.to_datetime(fixture["date"])
        model = fit_total_goals_model(historical, completed_current, as_of, config)
        prediction = predict_fixture(model, fixture)
        actual_outcome = outcome_from_scores(
            float(fixture["home_score"]),
            float(fixture["away_score"]),
        )
        prediction["actual_outcome"] = actual_outcome
        prediction["outcome_correct"] = prediction["predicted_outcome"] == actual_outcome
        prediction["exact_score_correct"] = (
            int(fixture["home_score"]) == prediction["predicted_home_score"]
            and int(fixture["away_score"]) == prediction["predicted_away_score"]
        )
        prediction["total_goals_absolute_error"] = abs(
            prediction["expected_total_goals"]
            - float(fixture["home_score"])
            - float(fixture["away_score"])
        )
        rows.append(prediction)
    return pd.DataFrame(rows)


def choose_advancing_team(
    prediction: pd.Series,
    home_team: str,
    away_team: str,
    model: TotalGoalsModel,
) -> tuple[str, str, str]:
    if prediction["predicted_outcome"] == "home_win":
        return home_team, away_team, "regulation"
    if prediction["predicted_outcome"] == "away_win":
        return away_team, home_team, "regulation"

    home_win = float(prediction["home_win_probability"])
    away_win = float(prediction["away_win_probability"])
    if home_win > away_win:
        return home_team, away_team, "after_extra_time_or_penalties"
    if away_win > home_win:
        return away_team, home_team, "after_extra_time_or_penalties"
    if model.ratings.get(home_team, DEFAULT_RATING) >= model.ratings.get(away_team, DEFAULT_RATING):
        return home_team, away_team, "after_extra_time_or_penalties"
    return away_team, home_team, "after_extra_time_or_penalties"


def forecast_knockout_phase(
    model: TotalGoalsModel,
    knockout_layout: pd.DataFrame,
    knockout_input: pd.DataFrame,
    group_standings: pd.DataFrame,
) -> pd.DataFrame:
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
        actual_home_score = parse_optional_score(input_row.get("actual_home_score", match.home_score))
        actual_away_score = parse_optional_score(input_row.get("actual_away_score", match.away_score))
        actual_winner_text = input_row.get("actual_winner", "")

        base = {
            "round": match.group,
            "date": match.date,
            "match_no": match_no,
            "home_slot": home_slot,
            "away_slot": away_slot,
            "home_team": home_team,
            "away_team": away_team,
            "home_resolution_note": home_note,
            "away_resolution_note": away_note,
            "actual_home_score": actual_home_score,
            "actual_away_score": actual_away_score,
        }
        if not has_text(home_team) or not has_text(away_team):
            rows.append(base | {"advancing_team": "", "win_method": "unresolved"})
            continue

        fixture = pd.Series(
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
        )
        prediction = pd.Series(predict_fixture(model, fixture))
        predicted_winner, predicted_loser, predicted_method = choose_advancing_team(
            prediction,
            home_team,
            away_team,
            model,
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
            base
            | {
                "predicted_home_score": prediction["predicted_home_score"],
                "predicted_away_score": prediction["predicted_away_score"],
                "predicted_outcome": prediction["predicted_outcome"],
                "expected_total_goals": prediction["expected_total_goals"],
                "home_xg": prediction["home_xg"],
                "away_xg": prediction["away_xg"],
                "home_win_probability": prediction["home_win_probability"],
                "draw_probability": prediction["draw_probability"],
                "away_win_probability": prediction["away_win_probability"],
                "home_decimal_odds": prediction["home_decimal_odds"],
                "draw_decimal_odds": prediction["draw_decimal_odds"],
                "away_decimal_odds": prediction["away_decimal_odds"],
                "rating_diff": prediction["rating_diff"],
                "advancing_team": advancing_team,
                "eliminated_team": eliminated_team,
                "win_method": win_method,
            }
        )

    return pd.DataFrame(rows)


def ratings_report(model: TotalGoalsModel) -> pd.DataFrame:
    rate_lookup = model.team_goal_rates.set_index("team").to_dict("index")
    rows = []
    for team, rating in sorted(model.ratings.items(), key=lambda item: item[1], reverse=True):
        profile = rate_lookup.get(team, {})
        rows.append(
            {
                "team": team,
                "recent_rating": round(rating, 2),
                "weighted_matches": profile.get("weighted_matches", 0.0),
                "goals_for_rate": round(profile.get("goals_for_rate", model.global_total_goals / 2), 3),
                "goals_against_rate": round(
                    profile.get("goals_against_rate", model.global_total_goals / 2),
                    3,
                ),
                "match_total_rate": round(profile.get("match_total_rate", model.global_total_goals), 3),
            }
        )
    return pd.DataFrame(rows)


def plot_total_goal_forecasts(output_dir: Path, predictions: pd.DataFrame) -> Path | None:
    if predictions.empty:
        return None
    path = output_dir / "total_goals_next_fixtures.png"
    sample = predictions.sort_values(["date", "match_no"]).head(18).copy()
    labels = [
        f"{pd.to_datetime(row.date).strftime('%b %d')}  {row.home_team} vs {row.away_team}"
        for row in sample.itertuples(index=False)
    ]
    y = np.arange(len(sample))
    fig, ax = plt.subplots(figsize=(11, max(6, len(sample) * 0.38)), constrained_layout=True)
    ax.barh(y, sample["expected_total_goals"], color="#4e79a7")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Expected total goals")
    ax.set_title("Total-Goals-First Forecasts")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for idx, value in enumerate(sample["expected_total_goals"]):
        ax.text(value + 0.03, idx, f"{value:.2f}", va="center", fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_win_odds(output_dir: Path, predictions: pd.DataFrame) -> Path | None:
    if predictions.empty:
        return None
    path = output_dir / "win_odds_next_fixtures.png"
    sample = predictions.sort_values(["date", "match_no"]).head(14).copy()
    labels = [
        f"{row.home_team} vs {row.away_team}"
        for row in sample.itertuples(index=False)
    ]
    y = np.arange(len(sample))
    fig, ax = plt.subplots(figsize=(11, max(6, len(sample) * 0.42)), constrained_layout=True)
    left = np.zeros(len(sample))
    for column, color, label in [
        ("home_win_probability", "#4e79a7", "Home win"),
        ("draw_probability", "#f28e2b", "Draw"),
        ("away_win_probability", "#59a14f", "Away win"),
    ]:
        values = sample[column].astype(float).to_numpy()
        ax.barh(y, values, left=left, color=color, label=label)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Win/Draw/Away Odds")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_knockout_totals(output_dir: Path, knockout: pd.DataFrame) -> Path | None:
    if knockout.empty:
        return None
    path = output_dir / "total_goals_knockout_layout.png"
    display = knockout.copy()
    display["forecast"] = display.apply(
        lambda row: (
            f"{row['home_team']} {row['predicted_home_score']}-{row['predicted_away_score']} {row['away_team']}"
            if has_text(row.get("home_team", "")) and has_text(row.get("away_team", ""))
            else f"{row['home_slot']} vs {row['away_slot']}"
        ),
        axis=1,
    )
    table_df = display[
        ["round", "match_no", "date", "forecast", "expected_total_goals", "advancing_team"]
    ].copy()
    table_df["date"] = pd.to_datetime(table_df["date"]).dt.strftime("%b %d")
    table_df["match_no"] = table_df["match_no"].map(lambda value: f"M{int(value)}")
    table_df["expected_total_goals"] = table_df["expected_total_goals"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
    )

    fig, ax = plt.subplots(figsize=(15, max(8, len(table_df) * 0.28)))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=["Round", "Match", "Date", "Forecast", "Exp goals", "Advances"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.15, 0.06, 0.07, 0.42, 0.10, 0.20],
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
    ax.set_title("Total-Goals Model Knockout Layout", fontsize=16, pad=18)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_evaluation(output_dir: Path, evaluation: pd.DataFrame) -> Path | None:
    if evaluation.empty:
        return None
    path = output_dir / "total_goals_model_evaluation.png"
    values = [
        float(evaluation["outcome_correct"].mean()),
        float(evaluation["exact_score_correct"].mean()),
        1.0 / (1.0 + float(evaluation["total_goals_absolute_error"].mean())),
    ]
    labels = ["Outcome", "Exact score", "Total-goals score"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#4e79a7", "#59a14f", "#f28e2b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Rolling Evaluation on Completed 2026 Matches")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.0%}", ha="center")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_outputs(
    output_dir: Path,
    future_predictions: pd.DataFrame,
    evaluation: pd.DataFrame,
    group_standings: pd.DataFrame,
    knockout_input: pd.DataFrame,
    knockout_input_path: Path,
    knockout_predictions: pd.DataFrame,
    rating_report: pd.DataFrame,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    knockout_input_path.parent.mkdir(parents=True, exist_ok=True)
    future_predictions.to_csv(output_dir / "total_goals_group_forecast.csv", index=False)
    evaluation.to_csv(output_dir / "total_goals_model_evaluation.csv", index=False)
    group_standings.to_csv(output_dir / "total_goals_projected_group_standings.csv", index=False)
    knockout_input.to_csv(knockout_input_path, index=False)
    knockout_predictions.to_csv(output_dir / "total_goals_knockout_forecast.csv", index=False)
    rating_report.to_csv(output_dir / "total_goals_team_ratings.csv", index=False)

    paths = []
    for path in [
        plot_total_goal_forecasts(output_dir, future_predictions),
        plot_win_odds(output_dir, future_predictions),
        plot_knockout_totals(output_dir, knockout_predictions),
        plot_evaluation(output_dir, evaluation),
    ]:
        if path is not None:
            paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Total-goals-first World Cup forecasting model with recent-data weighting."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_world_cup"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_total_goals_model"))
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--lookback-years", type=float, default=4.0)
    parser.add_argument("--half-life-days", type=float, default=365.0)
    parser.add_argument("--current-world-cup-weight", type=float, default=2.75)
    parser.add_argument("--min-knockout-total-goals", type=float, default=2.35)
    parser.add_argument("--knockout-input", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TotalGoalsConfig(
        lookback_years=args.lookback_years,
        half_life_days=args.half_life_days,
        current_world_cup_weight=args.current_world_cup_weight,
        min_knockout_total_goals=args.min_knockout_total_goals,
    )
    refresh = not args.no_refresh

    historical = load_historical_results(args.cache_dir, refresh=refresh)
    current = load_current_world_cup(args.cache_dir, refresh=refresh, historical=historical)
    group_matches = current[current.apply(is_group_stage_match, axis=1)].copy()
    knockout_layout = current[current.apply(is_knockout_match, axis=1)].copy()
    known_group_matches = group_matches[group_matches.apply(is_known_team_fixture, axis=1)].copy()
    completed = known_group_matches[known_group_matches["home_score"].notna()].copy()
    future_group = known_group_matches[known_group_matches["home_score"].isna()].copy()
    if completed.empty:
        raise RuntimeError("No completed current World Cup matches were available.")

    as_of = completed["date"].max() + pd.Timedelta(days=1)
    model = fit_total_goals_model(historical, completed, as_of, config)
    future_predictions = make_predictions(model, future_group)
    evaluation = rolling_evaluate_completed(historical, completed, config)

    group_standings = project_group_standings(
        group_matches,
        future_predictions,
        model.ratings,
    )
    knockout_input_path = args.knockout_input or (
        args.output_dir / "total_goals_knockout_input_template.csv"
    )
    knockout_input = build_knockout_input_template(knockout_layout, knockout_input_path)
    knockout_predictions = forecast_knockout_phase(
        model,
        knockout_layout,
        knockout_input,
        group_standings,
    )
    rating_report = ratings_report(model)

    visual_paths = write_outputs(
        args.output_dir,
        future_predictions,
        evaluation,
        group_standings,
        knockout_input,
        knockout_input_path,
        knockout_predictions,
        rating_report,
    )

    print("Total-goals-first model complete")
    print(f"Lookback years: {config.lookback_years:g}")
    print(f"Half-life days: {config.half_life_days:g}")
    print(f"Current World Cup weight: {config.current_world_cup_weight:g}")
    print(f"Recent weighted global total goals: {model.global_total_goals:.2f}")
    print(f"Current tournament weighted total goals: {model.current_tournament_total_goals:.2f}")
    print(f"Future group fixtures forecasted: {len(future_predictions)}")
    print(f"Knockout fixtures forecasted: {len(knockout_predictions)}")
    print(
        "Rolling completed-match evaluation: "
        f"outcome={evaluation['outcome_correct'].mean():.1%}, "
        f"exact={evaluation['exact_score_correct'].mean():.1%}, "
        f"total-goals MAE={evaluation['total_goals_absolute_error'].mean():.2f}"
    )
    print()
    print("Next forecasts:")
    print(
        future_predictions[
            [
                "date",
                "home_team",
                "away_team",
                "predicted_home_score",
                "predicted_away_score",
                "expected_total_goals",
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print()
    for path in [
        args.output_dir / "total_goals_group_forecast.csv",
        args.output_dir / "total_goals_model_evaluation.csv",
        args.output_dir / "total_goals_projected_group_standings.csv",
        knockout_input_path,
        args.output_dir / "total_goals_knockout_forecast.csv",
        args.output_dir / "total_goals_team_ratings.csv",
        *visual_paths,
    ]:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
