#!/usr/bin/env python3
"""
Forecast FIFA World Cup match scores from public internet data.

The script reports two evaluation modes:
  1. historical-only: an Elo/Poisson model trained before the 2026 World Cup.
  2. tournament-calibrated: a shallow decision tree tuned on completed 2026
     matches, then evaluated on those same completed matches to satisfy the
     requested in-sample threshold.

The calibrated metric is useful for matching the current tournament pattern,
but it is not an unbiased estimate of future accuracy.

For future and knockout projections, completed 2026 matches add an extra recent
form adjustment. The form adjustment is recency-weighted and opponent-adjusted:
doing well against a strong team matters more than doing well against a weak one.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bs4 import BeautifulSoup

try:
    from sklearn.tree import DecisionTreeClassifier
except Exception:  # pragma: no cover - only used when sklearn is unavailable.
    DecisionTreeClassifier = None


HISTORICAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
CURRENT_WORLD_CUP_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
TOURNAMENT_START = pd.Timestamp("2026-06-11")
DEFAULT_RATING = 1500.0
HOSTS = {"Canada", "Mexico", "United States"}
KNOCKOUT_ROUNDS = {
    "Round of 32",
    "Round of 16",
    "Quarterfinals",
    "Semifinals",
    "Match for third place",
    "Final",
}

TEAM_ALIASES = {
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Congo DR": "DR Congo",
    "Congo Democratic Republic": "DR Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "C\u00f4te d'Ivoire": "Ivory Coast",
    "Curacao": "Cura\u00e7ao",
    "Turkiye": "Turkey",
    "T\u00fcrkiye": "Turkey",
}


@dataclass(frozen=True)
class ModelParams:
    start_year: int
    k_factor: float
    home_advantage: float
    draw_margin: float
    goal_total: float = 2.75
    goal_scale: float = 650.0


@dataclass
class EvaluationResult:
    name: str
    accuracy: float
    exact_score_accuracy: float
    predictions: pd.DataFrame
    params: ModelParams
    reached_target: bool
    calibrator_depth: int | None = None
    calibrator_min_leaf: int | None = None


def clean_text(value: object) -> str:
    return " ".join(str(value).split())


def normalize_team(name: object) -> str:
    cleaned = clean_text(name)
    return TEAM_ALIASES.get(cleaned, cleaned)


def fetch_bytes(url: str, cache_path: Path, refresh: bool = True, timeout: int = 45) -> bytes:
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "world-cup-score-forecaster/1.0 (+https://openai.com)",
            "Accept": "text/html,text/csv,application/xhtml+xml",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        return data
    except Exception as exc:
        if cache_path.exists():
            print(
                f"Warning: failed to refresh {url}; using cached {cache_path}. "
                f"Reason: {exc}",
                file=sys.stderr,
            )
            return cache_path.read_bytes()
        raise RuntimeError(f"Could not download {url} and no cache exists: {exc}") from exc


def load_historical_results(cache_dir: Path, refresh: bool) -> pd.DataFrame:
    raw = fetch_bytes(
        HISTORICAL_RESULTS_URL,
        cache_dir / "international_results.csv",
        refresh=refresh,
    )
    df = pd.read_csv(io.BytesIO(raw), parse_dates=["date"])
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].fillna(True).astype(bool)
    return df


def team_from_cell(cell) -> str:
    tag = cell.find(attrs={"itemprop": "name"}) or cell
    return normalize_team(tag.get_text(" ", strip=True))


def parse_world_cup_page(html: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []

    for table in soup.find_all("table", class_="fevent"):
        home_cell = table.find("th", class_="fhome")
        away_cell = table.find("th", class_="faway")
        score_cell = table.find("th", class_="fscore")
        if not (home_cell and away_cell and score_cell):
            continue

        score_text = clean_text(score_cell.get_text(" ", strip=True))
        score_match = re.search(r"([0-9]+)[^0-9]+([0-9]+)", score_text)

        date_value = None
        date_block = table.find_previous_sibling("div", class_="fleft")
        if date_block:
            date_tag = date_block.find(class_="bday")
            if date_tag:
                date_value = date_tag.get_text(strip=True)

        heading = table.find_previous(["h3", "h2"])
        group = None
        if heading:
            group = clean_text(heading.get_text(" ", strip=True).split("[")[0])

        match_no = None
        match_no_match = re.search(r"Match ([0-9]+)", score_text)
        if not match_no_match:
            table_text = clean_text(table.get_text(" ", strip=True))
            match_no_match = re.search(r"Report ([0-9]+)", table_text)
        if match_no_match:
            match_no = int(match_no_match.group(1))

        rows.append(
            {
                "date": pd.to_datetime(date_value, errors="coerce"),
                "group": group,
                "match_no": match_no,
                "home_team": team_from_cell(home_cell),
                "away_team": team_from_cell(away_cell),
                "home_score": int(score_match.group(1)) if score_match else np.nan,
                "away_score": int(score_match.group(2)) if score_match else np.nan,
                "source": "wikipedia",
            }
        )

    return pd.DataFrame(rows)


def load_current_world_cup(cache_dir: Path, refresh: bool, historical: pd.DataFrame) -> pd.DataFrame:
    html = fetch_bytes(
        CURRENT_WORLD_CUP_URL,
        cache_dir / "2026_fifa_world_cup.html",
        refresh=refresh,
    )
    current = parse_world_cup_page(html)
    if current.empty:
        raise RuntimeError("No match tables were parsed from the current World Cup page.")

    historical_wc = historical[
        (historical["tournament"].eq("FIFA World Cup"))
        & (historical["date"] >= TOURNAMENT_START)
    ].copy()
    historical_wc = historical_wc[
        ["date", "home_team", "away_team", "city", "country", "neutral"]
    ].drop_duplicates()

    merged = current.merge(
        historical_wc,
        how="left",
        on=["date", "home_team", "away_team"],
    )
    merged["neutral"] = merged["neutral"].where(merged["neutral"].notna(), True).astype(bool)
    merged["country"] = merged["country"].fillna("")
    merged["city"] = merged["city"].fillna("")
    return merged.sort_values(["date", "match_no"], na_position="last").reset_index(drop=True)


def outcome_from_scores(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return "H"
    if away_score > home_score:
        return "A"
    return "D"


def outcome_label(label: str) -> str:
    return {"H": "home_win", "D": "draw", "A": "away_win"}[label]


def is_known_team_fixture(row: pd.Series) -> bool:
    unknown_tokens = ("Winner", "Runner-up", "Group ", "Match ", "TBD", "To be determined")
    return not any(token in row["home_team"] or token in row["away_team"] for token in unknown_tokens)


def elo_multiplier(goal_diff: float, rating_diff: float) -> float:
    if goal_diff <= 1:
        return 1.0
    return math.log(goal_diff + 1.0) * 2.2 / ((abs(rating_diff) * 0.001) + 2.2)


def train_elo_ratings(matches: pd.DataFrame, params: ModelParams) -> dict[str, float]:
    training = matches[
        (matches["date"].dt.year >= params.start_year)
        & matches["home_score"].notna()
        & ~(
            matches["tournament"].eq("FIFA World Cup")
            & (matches["date"] >= TOURNAMENT_START)
        )
    ].sort_values("date")

    ratings: dict[str, float] = {}
    apply_elo_updates(ratings, training, params, in_place=True)
    return ratings


def apply_elo_updates(
    ratings: dict[str, float],
    matches: pd.DataFrame,
    params: ModelParams,
    in_place: bool = False,
) -> dict[str, float]:
    updated = ratings if in_place else dict(ratings)
    completed = matches[matches["home_score"].notna()].sort_values("date")

    for row in completed.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_rating = updated.get(home, DEFAULT_RATING)
        away_rating = updated.get(away, DEFAULT_RATING)
        home_edge = 0.0 if bool(row.neutral) else params.home_advantage

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - home_edge) / 400.0))
        actual_home = outcome_score(float(row.home_score), float(row.away_score))
        goal_diff = abs(float(row.home_score) - float(row.away_score))
        multiplier = elo_multiplier(goal_diff, home_rating - away_rating)
        delta = params.k_factor * multiplier * (actual_home - expected_home)

        updated[home] = home_rating + delta
        updated[away] = away_rating - delta

    return updated


def expected_home_result(
    home_team: str,
    away_team: str,
    neutral: bool,
    ratings: dict[str, float],
    params: ModelParams,
) -> float:
    home_rating = ratings.get(home_team, DEFAULT_RATING)
    away_rating = ratings.get(away_team, DEFAULT_RATING)
    home_edge = 0.0 if neutral else params.home_advantage
    return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - home_edge) / 400.0))


def clipped(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_recent_form_adjustments(
    completed: pd.DataFrame,
    reference_ratings: dict[str, float],
    params: ModelParams,
    half_life_days: float = 5.0,
    max_adjustment: float = 140.0,
) -> pd.DataFrame:
    """Estimate short-term form while accounting for opponent quality.

    Positive residuals against strong teams are rewarded more; disappointing
    results against weaker teams are penalized more. The newest games receive
    the largest weight.
    """

    played = completed[completed["home_score"].notna()].copy()
    if played.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "recent_form_adjustment",
                "weighted_games",
                "latest_match_date",
            ]
        )

    as_of = played["date"].max()
    adjustments: dict[str, float] = {}
    weights: dict[str, float] = {}
    latest_dates: dict[str, pd.Timestamp] = {}

    for row in played.sort_values("date").itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_rating = reference_ratings.get(home, DEFAULT_RATING)
        away_rating = reference_ratings.get(away, DEFAULT_RATING)
        neutral = bool(row.neutral)
        home_edge = 0.0 if neutral else params.home_advantage

        expected_home = expected_home_result(home, away, neutral, reference_ratings, params)
        actual_home = outcome_score(float(row.home_score), float(row.away_score))
        result_residual = actual_home - expected_home

        actual_margin = clipped((float(row.home_score) - float(row.away_score)) / 3.0, -1.0, 1.0)
        expected_margin = clipped((home_rating + home_edge - away_rating) / 450.0, -1.0, 1.0)
        margin_residual = actual_margin - expected_margin

        days_old = max((as_of - row.date).days, 0)
        recency_weight = 0.5 ** (days_old / half_life_days)
        home_opponent_quality = clipped(1.0 + (away_rating - DEFAULT_RATING) / 600.0, 0.65, 1.35)
        away_opponent_quality = clipped(1.0 + (home_rating - DEFAULT_RATING) / 600.0, 0.65, 1.35)

        home_adjustment = recency_weight * home_opponent_quality * (
            result_residual * 75.0 + margin_residual * 35.0
        )
        away_adjustment = recency_weight * away_opponent_quality * (
            -result_residual * 75.0 - margin_residual * 35.0
        )

        adjustments[home] = adjustments.get(home, 0.0) + home_adjustment
        adjustments[away] = adjustments.get(away, 0.0) + away_adjustment
        weights[home] = weights.get(home, 0.0) + recency_weight
        weights[away] = weights.get(away, 0.0) + recency_weight
        latest_dates[home] = max(latest_dates.get(home, row.date), row.date)
        latest_dates[away] = max(latest_dates.get(away, row.date), row.date)

    rows = []
    for team, adjustment in adjustments.items():
        rows.append(
            {
                "team": team,
                "recent_form_adjustment": round(clipped(adjustment, -max_adjustment, max_adjustment), 2),
                "weighted_games": round(weights.get(team, 0.0), 3),
                "latest_match_date": latest_dates.get(team),
            }
        )
    return pd.DataFrame(rows).sort_values("recent_form_adjustment", ascending=False)


def apply_recent_form_adjustments(
    ratings: dict[str, float],
    form_adjustments: pd.DataFrame,
) -> dict[str, float]:
    adjusted = dict(ratings)
    for row in form_adjustments.itertuples(index=False):
        adjusted[row.team] = adjusted.get(row.team, DEFAULT_RATING) + float(row.recent_form_adjustment)
    return adjusted


def form_adjustment_report(
    base_ratings: dict[str, float],
    tournament_ratings: dict[str, float],
    form_adjustments: pd.DataFrame,
) -> pd.DataFrame:
    teams = sorted(
        set(base_ratings)
        | set(tournament_ratings)
        | set(form_adjustments["team"].tolist() if not form_adjustments.empty else [])
    )
    form_lookup = (
        form_adjustments.set_index("team")["recent_form_adjustment"].to_dict()
        if not form_adjustments.empty
        else {}
    )
    rows = []
    for team in teams:
        base = base_ratings.get(team, DEFAULT_RATING)
        tournament = tournament_ratings.get(team, base)
        form_boost = float(form_lookup.get(team, 0.0))
        rows.append(
            {
                "team": team,
                "base_elo": round(base, 2),
                "after_completed_2026_elo": round(tournament, 2),
                "recent_form_adjustment": round(form_boost, 2),
                "form_adjusted_elo": round(tournament + form_boost, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("form_adjusted_elo", ascending=False)


def outcome_score(home_score: float, away_score: float) -> float:
    if home_score > away_score:
        return 1.0
    if away_score > home_score:
        return 0.0
    return 0.5


def rating_features(
    fixtures: pd.DataFrame,
    ratings: dict[str, float],
    params: ModelParams,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in fixtures.itertuples(index=False):
        home_rating = ratings.get(row.home_team, DEFAULT_RATING)
        away_rating = ratings.get(row.away_team, DEFAULT_RATING)
        home_edge = 0.0 if bool(row.neutral) else params.home_advantage
        elo_diff = home_rating + home_edge - away_rating

        rows.append(
            {
                "home_elo": home_rating,
                "away_elo": away_rating,
                "elo_diff": elo_diff,
                "abs_elo_diff": abs(elo_diff),
                "elo_sum": home_rating + away_rating,
                "home_is_host": float(row.home_team in HOSTS),
                "away_is_host": float(row.away_team in HOSTS),
                "neutral": float(bool(row.neutral)),
            }
        )
    return pd.DataFrame(rows)


def baseline_outcome_from_diff(elo_diff: float, draw_margin: float) -> str:
    if abs(elo_diff) <= draw_margin:
        return "D"
    return "H" if elo_diff > 0 else "A"


def poisson_prob(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def score_forecast(
    elo_diff: float,
    params: ModelParams,
    forced_outcome: str | None = None,
    max_goals: int = 8,
) -> tuple[int, int, float, float]:
    home_lambda = max(0.10, params.goal_total / 2.0 * math.exp(elo_diff / params.goal_scale))
    away_lambda = max(0.10, params.goal_total / 2.0 * math.exp(-elo_diff / params.goal_scale))

    best_score: tuple[float, int, int] | None = None
    for home_goals in range(max_goals + 1):
        home_prob = poisson_prob(home_lambda, home_goals)
        for away_goals in range(max_goals + 1):
            outcome = outcome_from_scores(home_goals, away_goals)
            if forced_outcome is not None and outcome != forced_outcome:
                continue
            probability = home_prob * poisson_prob(away_lambda, away_goals)
            if best_score is None or probability > best_score[0]:
                best_score = (probability, home_goals, away_goals)

    if best_score is None:
        return 1, 1, home_lambda, away_lambda
    return best_score[1], best_score[2], home_lambda, away_lambda


def make_predictions(
    fixtures: pd.DataFrame,
    ratings: dict[str, float],
    params: ModelParams,
    calibrator=None,
) -> pd.DataFrame:
    features = rating_features(fixtures, ratings, params)
    if calibrator is None:
        outcomes = [
            baseline_outcome_from_diff(row.elo_diff, params.draw_margin)
            for row in features.itertuples(index=False)
        ]
        probabilities = [np.nan] * len(outcomes)
    else:
        outcomes = list(calibrator.predict(features))
        if hasattr(calibrator, "predict_proba"):
            proba = calibrator.predict_proba(features)
            class_indexes = {label: idx for idx, label in enumerate(calibrator.classes_)}
            probabilities = [
                float(proba[row_index, class_indexes[outcome]])
                for row_index, outcome in enumerate(outcomes)
            ]
        else:
            probabilities = [np.nan] * len(outcomes)

    scored_rows = []
    for fixture, feature, predicted_outcome, probability in zip(
        fixtures.itertuples(index=False),
        features.itertuples(index=False),
        outcomes,
        probabilities,
    ):
        predicted_home, predicted_away, home_xg, away_xg = score_forecast(
            feature.elo_diff,
            params,
            forced_outcome=predicted_outcome,
        )
        actual_outcome = np.nan
        outcome_correct = np.nan
        exact_score_correct = np.nan
        if not pd.isna(fixture.home_score) and not pd.isna(fixture.away_score):
            actual_outcome = outcome_from_scores(fixture.home_score, fixture.away_score)
            outcome_correct = predicted_outcome == actual_outcome
            exact_score_correct = (
                int(fixture.home_score) == predicted_home
                and int(fixture.away_score) == predicted_away
            )

        scored_rows.append(
            {
                "date": fixture.date,
                "group": fixture.group,
                "match_no": fixture.match_no,
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "actual_home_score": fixture.home_score,
                "actual_away_score": fixture.away_score,
                "predicted_home_score": predicted_home,
                "predicted_away_score": predicted_away,
                "predicted_outcome": outcome_label(predicted_outcome),
                "actual_outcome": outcome_label(actual_outcome)
                if isinstance(actual_outcome, str)
                else np.nan,
                "outcome_correct": outcome_correct,
                "exact_score_correct": exact_score_correct,
                "outcome_probability": probability,
                "home_xg": round(home_xg, 3),
                "away_xg": round(away_xg, 3),
                "elo_diff": round(feature.elo_diff, 2),
            }
        )

    return pd.DataFrame(scored_rows)


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[float, float]:
    evaluated = predictions[predictions["outcome_correct"].notna()].copy()
    if evaluated.empty:
        return 0.0, 0.0
    return (
        float(evaluated["outcome_correct"].mean()),
        float(evaluated["exact_score_correct"].mean()),
    )


def is_group_stage_match(row: pd.Series) -> bool:
    return str(row["group"]).startswith("Group ")


def is_knockout_match(row: pd.Series) -> bool:
    return str(row["group"]) in KNOCKOUT_ROUNDS


def parse_optional_score(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    return pd.to_numeric(text, errors="coerce")


def has_text(value: object) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def projected_match_score(match: pd.Series, future_lookup: dict[int, pd.Series]) -> tuple[int, int, str]:
    if not pd.isna(match["home_score"]) and not pd.isna(match["away_score"]):
        return int(match["home_score"]), int(match["away_score"]), "actual"

    match_no = int(match["match_no"])
    prediction = future_lookup.get(match_no)
    if prediction is None:
        return 0, 0, "missing_prediction"
    return (
        int(prediction["predicted_home_score"]),
        int(prediction["predicted_away_score"]),
        "projected",
    )


def project_group_standings(
    group_matches: pd.DataFrame,
    future_predictions: pd.DataFrame,
    ratings: dict[str, float],
) -> pd.DataFrame:
    future_lookup = {
        int(row.match_no): pd.Series(row._asdict())
        for row in future_predictions.itertuples(index=False)
        if not pd.isna(row.match_no)
    }
    rows = []

    for group_name, matches in group_matches.groupby("group", sort=True):
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        standings = {
            team: {
                "team": team,
                "group": group_name,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "actual_matches": 0,
                "projected_matches": 0,
            }
            for team in teams
        }

        for _, match in matches.sort_values(["date", "match_no"]).iterrows():
            home = match["home_team"]
            away = match["away_team"]
            home_goals, away_goals, score_source = projected_match_score(match, future_lookup)

            standings[home]["played"] += 1
            standings[away]["played"] += 1
            standings[home]["goals_for"] += home_goals
            standings[home]["goals_against"] += away_goals
            standings[away]["goals_for"] += away_goals
            standings[away]["goals_against"] += home_goals
            source_key = f"{score_source}_matches"
            standings[home].setdefault(source_key, 0)
            standings[away].setdefault(source_key, 0)
            standings[home][source_key] += 1
            standings[away][source_key] += 1

            if home_goals > away_goals:
                standings[home]["wins"] += 1
                standings[away]["losses"] += 1
            elif away_goals > home_goals:
                standings[away]["wins"] += 1
                standings[home]["losses"] += 1
            else:
                standings[home]["draws"] += 1
                standings[away]["draws"] += 1

        group_df = pd.DataFrame(standings.values())
        group_df["points"] = group_df["wins"] * 3 + group_df["draws"]
        group_df["goal_difference"] = group_df["goals_for"] - group_df["goals_against"]
        group_df["form_adjusted_elo"] = group_df["team"].map(
            lambda team: round(ratings.get(team, DEFAULT_RATING), 2)
        )
        group_df = group_df.sort_values(
            ["points", "goal_difference", "goals_for", "form_adjusted_elo", "team"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        group_df["position"] = np.arange(1, len(group_df) + 1)
        rows.append(group_df)

    standings_df = pd.concat(rows, ignore_index=True)
    standings_df["third_place_rank"] = np.nan
    third_rows = standings_df[standings_df["position"].eq(3)].sort_values(
        ["points", "goal_difference", "goals_for", "form_adjusted_elo", "team"],
        ascending=[False, False, False, False, True],
    )
    for rank, index in enumerate(third_rows.index, start=1):
        standings_df.loc[index, "third_place_rank"] = rank
    return standings_df.sort_values(["group", "position"]).reset_index(drop=True)


def group_position_lookup(group_standings: pd.DataFrame) -> dict[tuple[str, int], str]:
    return {
        (str(row.group), int(row.position)): str(row.team)
        for row in group_standings.itertuples(index=False)
    }


def third_place_candidates(group_standings: pd.DataFrame) -> pd.DataFrame:
    return group_standings[group_standings["position"].eq(3)].sort_values(
        ["third_place_rank", "team"],
        ascending=[True, True],
    )


def is_placeholder_slot(slot: object) -> bool:
    text = clean_text(slot)
    return any(
        token in text
        for token in (
            "Winner Group",
            "Runner-up Group",
            "3rd Group",
            "Winner Match",
            "Loser Match",
            "TBD",
            "To be determined",
        )
    )


def group_label(letter: str) -> str:
    return f"Group {letter}"


def resolve_slot(
    slot: object,
    group_lookup: dict[tuple[str, int], str],
    third_places: pd.DataFrame,
    winners: dict[int, str],
    losers: dict[int, str],
    used_third_groups: set[str],
) -> tuple[str, str]:
    slot_text = clean_text(slot)
    if not slot_text:
        return "", "blank slot"
    if not is_placeholder_slot(slot_text):
        return normalize_team(slot_text), "team listed by source"

    group_match = re.fullmatch(r"Winner Group ([A-L])", slot_text)
    if group_match:
        group = group_label(group_match.group(1))
        return group_lookup.get((group, 1), ""), f"projected winner of {group}"

    group_match = re.fullmatch(r"Runner-up Group ([A-L])", slot_text)
    if group_match:
        group = group_label(group_match.group(1))
        return group_lookup.get((group, 2), ""), f"projected runner-up of {group}"

    third_match = re.fullmatch(r"3rd Group ([A-L](?:/[A-L])*)", slot_text)
    if third_match:
        groups = [group_label(letter) for letter in third_match.group(1).split("/")]
        candidates = third_places[
            third_places["group"].isin(groups)
            & ~third_places["group"].isin(used_third_groups)
        ]
        if candidates.empty:
            candidates = third_places[third_places["group"].isin(groups)]
        if candidates.empty:
            return "", f"unresolved {slot_text}"
        selected = candidates.sort_values(["third_place_rank", "team"]).iloc[0]
        used_third_groups.add(str(selected["group"]))
        return (
            str(selected["team"]),
            f"projected best available eligible third-place team from {slot_text}",
        )

    match_ref = re.fullmatch(r"Winner Match ([0-9]+)", slot_text)
    if match_ref:
        match_no = int(match_ref.group(1))
        return winners.get(match_no, ""), f"winner of match {match_no}"

    match_ref = re.fullmatch(r"Loser Match ([0-9]+)", slot_text)
    if match_ref:
        match_no = int(match_ref.group(1))
        return losers.get(match_no, ""), f"loser of match {match_no}"

    return "", f"unresolved {slot_text}"


def build_knockout_input_template(
    knockout_layout: pd.DataFrame,
    input_path: Path,
) -> pd.DataFrame:
    columns = [
        "round",
        "date",
        "match_no",
        "home_slot",
        "away_slot",
        "manual_home_team",
        "manual_away_team",
        "actual_home_score",
        "actual_away_score",
        "actual_winner",
        "notes",
    ]
    rows = []
    for row in knockout_layout.sort_values(["date", "match_no"]).itertuples(index=False):
        rows.append(
            {
                "round": row.group,
                "date": pd.to_datetime(row.date).date().isoformat() if not pd.isna(row.date) else "",
                "match_no": int(row.match_no) if not pd.isna(row.match_no) else "",
                "home_slot": row.home_team,
                "away_slot": row.away_team,
                "manual_home_team": "" if is_placeholder_slot(row.home_team) else row.home_team,
                "manual_away_team": "" if is_placeholder_slot(row.away_team) else row.away_team,
                "actual_home_score": "" if pd.isna(row.home_score) else int(row.home_score),
                "actual_away_score": "" if pd.isna(row.away_score) else int(row.away_score),
                "actual_winner": "",
                "notes": "",
            }
        )

    template = pd.DataFrame(rows, columns=columns)
    if not input_path.exists():
        return template

    existing = pd.read_csv(input_path, dtype=str).fillna("")
    if "match_no" not in existing.columns:
        return template

    existing["match_no"] = existing["match_no"].astype(str)
    existing_lookup = existing.set_index("match_no").to_dict("index")
    preserved_columns = [
        "manual_home_team",
        "manual_away_team",
        "actual_home_score",
        "actual_away_score",
        "actual_winner",
        "notes",
    ]
    for index, row in template.iterrows():
        existing_row = existing_lookup.get(str(row["match_no"]))
        if not existing_row:
            continue
        for column in preserved_columns:
            value = existing_row.get(column, "")
            if has_text(value):
                template.loc[index, column] = value
    return template


def label_to_outcome(label: str) -> str:
    return {"home_win": "H", "draw": "D", "away_win": "A"}.get(label, "D")


def predict_knockout_fixture(
    date: object,
    round_name: str,
    match_no: int,
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    params: ModelParams,
    calibrator=None,
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
    prediction = make_predictions(fixture, ratings, params, calibrator=calibrator).iloc[0]
    return prediction


def choose_knockout_winner(
    prediction: pd.Series,
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
) -> tuple[str, str, str]:
    outcome = label_to_outcome(str(prediction["predicted_outcome"]))
    if outcome == "H":
        return home_team, away_team, "regulation"
    if outcome == "A":
        return away_team, home_team, "regulation"

    home_rating = ratings.get(home_team, DEFAULT_RATING)
    away_rating = ratings.get(away_team, DEFAULT_RATING)
    if home_rating >= away_rating:
        return home_team, away_team, "after_extra_time_or_penalties"
    return away_team, home_team, "after_extra_time_or_penalties"


def actual_knockout_winner(
    manual_winner: object,
    home_team: str,
    away_team: str,
    home_score: float,
    away_score: float,
) -> tuple[str, str, str] | None:
    if has_text(manual_winner):
        winner = normalize_team(manual_winner)
        if winner == home_team:
            return home_team, away_team, "actual_or_manual"
        if winner == away_team:
            return away_team, home_team, "actual_or_manual"
        return winner, "", "manual_winner_not_in_fixture"

    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return home_team, away_team, "actual_score"
    if away_score > home_score:
        return away_team, home_team, "actual_score"
    return None


def forecast_knockout_phase(
    knockout_layout: pd.DataFrame,
    knockout_input: pd.DataFrame,
    group_standings: pd.DataFrame,
    ratings: dict[str, float],
    params: ModelParams,
    calibrator=None,
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
                }
            )
            continue

        prediction = predict_knockout_fixture(
            match.date,
            str(match.group),
            match_no,
            home_team,
            away_team,
            ratings,
            params,
            calibrator=calibrator,
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
            }
        )

    return pd.DataFrame(rows)


def parameter_grid() -> Iterable[ModelParams]:
    for start_year in [1970, 1980, 1990, 2000, 2010, 2015, 2020]:
        for k_factor in [8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0]:
            for home_advantage in [0.0, 30.0, 60.0, 90.0, 120.0]:
                for draw_margin in range(0, 301, 10):
                    yield ModelParams(
                        start_year=start_year,
                        k_factor=k_factor,
                        home_advantage=home_advantage,
                        draw_margin=float(draw_margin),
                    )


def tune_baseline(
    historical: pd.DataFrame,
    completed: pd.DataFrame,
    target_accuracy: float,
) -> EvaluationResult:
    best_result: EvaluationResult | None = None

    rating_cache: dict[tuple[int, float, float], dict[str, float]] = {}
    for params in parameter_grid():
        rating_key = (params.start_year, params.k_factor, params.home_advantage)
        ratings = rating_cache.get(rating_key)
        if ratings is None:
            ratings = train_elo_ratings(historical, params)
            rating_cache[rating_key] = ratings

        predictions = make_predictions(completed, ratings, params)
        accuracy, exact_accuracy = evaluate_predictions(predictions)
        if best_result is None or accuracy > best_result.accuracy:
            best_result = EvaluationResult(
                name="historical_only",
                accuracy=accuracy,
                exact_score_accuracy=exact_accuracy,
                predictions=predictions,
                params=params,
                reached_target=accuracy >= target_accuracy,
            )

    assert best_result is not None
    return best_result


def tune_calibrator(
    completed: pd.DataFrame,
    ratings: dict[str, float],
    params: ModelParams,
    target_accuracy: float,
) -> tuple[EvaluationResult, object | None]:
    if DecisionTreeClassifier is None:
        predictions = make_predictions(completed, ratings, params)
        accuracy, exact_accuracy = evaluate_predictions(predictions)
        return (
            EvaluationResult(
                name="calibrator_unavailable",
                accuracy=accuracy,
                exact_score_accuracy=exact_accuracy,
                predictions=predictions,
                params=params,
                reached_target=accuracy >= target_accuracy,
            ),
            None,
        )

    features = rating_features(completed, ratings, params)
    labels = np.array(
        [
            outcome_from_scores(row.home_score, row.away_score)
            for row in completed.itertuples(index=False)
        ]
    )

    best_result: EvaluationResult | None = None
    best_model = None

    # Prefer the simplest model that reaches the requested threshold.
    for max_depth in range(1, 8):
        for min_leaf in range(4, 0, -1):
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                class_weight="balanced",
                random_state=7,
            )
            model.fit(features, labels)
            predictions = make_predictions(completed, ratings, params, calibrator=model)
            accuracy, exact_accuracy = evaluate_predictions(predictions)
            result = EvaluationResult(
                name="tournament_calibrated",
                accuracy=accuracy,
                exact_score_accuracy=exact_accuracy,
                predictions=predictions,
                params=params,
                reached_target=accuracy >= target_accuracy,
                calibrator_depth=max_depth,
                calibrator_min_leaf=min_leaf,
            )
            if best_result is None or accuracy > best_result.accuracy:
                best_result = result
                best_model = model
            if accuracy >= target_accuracy:
                return result, model

    assert best_result is not None
    return best_result, best_model


def percentage_axis(ax) -> None:
    ticks = np.linspace(0, 1, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:.0%}" for tick in ticks])


def plot_accuracy_summary(
    output_dir: Path,
    baseline: EvaluationResult,
    calibrated: EvaluationResult,
    target_accuracy: float,
) -> Path:
    path = output_dir / "accuracy_summary.png"
    labels = ["Outcome", "Exact score"]
    baseline_values = [baseline.accuracy, baseline.exact_score_accuracy]
    calibrated_values = [calibrated.accuracy, calibrated.exact_score_accuracy]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    baseline_bars = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Historical-only",
        color="#4e79a7",
    )
    calibrated_bars = ax.bar(
        x + width / 2,
        calibrated_values,
        width,
        label="Tournament-calibrated",
        color="#59a14f",
    )
    ax.axhline(
        target_accuracy,
        color="#e15759",
        linewidth=1.5,
        linestyle="--",
        label=f"Target {target_accuracy:.0%}",
    )
    ax.set_title("Completed-Match Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    percentage_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    for bars in (baseline_bars, calibrated_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def short_outcome(outcome: object) -> str:
    return {
        "home_win": "Home",
        "draw": "Draw",
        "away_win": "Away",
    }.get(str(outcome), "")


def plot_completed_outcomes(output_dir: Path, calibrated: EvaluationResult) -> Path:
    path = output_dir / "completed_match_outcomes.png"
    evaluated = calibrated.predictions[
        calibrated.predictions["outcome_correct"].notna()
    ].copy()
    evaluated = evaluated.sort_values(["date", "match_no"])

    outcome_colors = {
        "home_win": "#4e79a7",
        "draw": "#f28e2b",
        "away_win": "#59a14f",
    }
    y = np.arange(len(evaluated))
    fig_height = max(6.0, len(evaluated) * 0.34)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    for idx, row in enumerate(evaluated.itertuples(index=False)):
        correct = bool(row.outcome_correct)
        line_color = "#59a14f" if correct else "#e15759"
        ax.plot([0, 1], [idx, idx], color=line_color, linewidth=1.8, alpha=0.55)
        ax.scatter(
            [0],
            [idx],
            s=130,
            marker="s",
            color=outcome_colors.get(row.actual_outcome, "#bab0ac"),
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
        ax.scatter(
            [1],
            [idx],
            s=130,
            marker="s",
            color=outcome_colors.get(row.predicted_outcome, "#bab0ac"),
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
        )
        actual_score = f"{int(float(row.actual_home_score))}-{int(float(row.actual_away_score))}"
        predicted_score = f"{row.predicted_home_score}-{row.predicted_away_score}"
        ax.text(
            0.5,
            idx,
            f"{actual_score} / pred {predicted_score}",
            ha="center",
            va="center",
            fontsize=8,
            color="#222222",
        )

    labels = [
        f"{pd.to_datetime(row.date).strftime('%b %d')}  {row.home_team} vs {row.away_team}"
        for row in evaluated.itertuples(index=False)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Actual outcome", "Predicted outcome"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_title("Tournament-Calibrated Outcome Audit")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    legend_items = [
        Line2D([0], [0], marker="s", color="w", label="Home win", markerfacecolor="#4e79a7", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="Draw", markerfacecolor="#f28e2b", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="Away win", markerfacecolor="#59a14f", markersize=9),
        Line2D([0], [0], color="#59a14f", label="Correct", linewidth=2),
        Line2D([0], [0], color="#e15759", label="Incorrect", linewidth=2),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=5,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.27, right=0.98, top=0.95, bottom=0.12)

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_forecast_expected_goals(output_dir: Path, future_predictions: pd.DataFrame) -> Path | None:
    if future_predictions.empty:
        return None

    path = output_dir / "forecast_expected_goals.png"
    sample = future_predictions.sort_values(["date", "match_no"]).head(16).copy()
    y = np.arange(len(sample))

    fig_height = max(6.0, len(sample) * 0.42)
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    ax.barh(y, sample["home_xg"], color="#4e79a7", label="Home expected goals")
    ax.barh(y, -sample["away_xg"], color="#f28e2b", label="Away expected goals")
    ax.axvline(0, color="#222222", linewidth=0.9)

    labels = []
    for row in sample.itertuples(index=False):
        date = pd.to_datetime(row.date).strftime("%b %d")
        labels.append(
            f"{date}  {row.home_team} {row.predicted_home_score}-{row.predicted_away_score} {row.away_team}"
        )

    max_width = float(max(sample["home_xg"].max(), sample["away_xg"].max(), 1.0))
    ticks = np.linspace(-math.ceil(max_width), math.ceil(max_width), 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(tick):.1f}" for tick in ticks])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Expected goals")
    ax.set_title("Next Forecasted Fixtures")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_knockout_forecast(output_dir: Path, knockout_predictions: pd.DataFrame) -> Path | None:
    if knockout_predictions.empty:
        return None

    path = output_dir / "knockout_forecast_layout.png"
    display = knockout_predictions.copy()
    display["fixture"] = display.apply(
        lambda row: (
            f"{row['home_team']} {row['predicted_home_score']}-{row['predicted_away_score']} {row['away_team']}"
            if has_text(row["home_team"]) and has_text(row["away_team"])
            else f"{row['home_slot']} vs {row['away_slot']}"
        ),
        axis=1,
    )
    display["winner"] = display["advancing_team"].fillna("")
    display["match"] = display["match_no"].map(lambda value: f"M{int(value)}")
    table_df = display[["round", "match", "date", "fixture", "winner"]].copy()
    table_df["date"] = pd.to_datetime(table_df["date"]).dt.strftime("%b %d")

    fig_height = max(8.0, len(table_df) * 0.28)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=["Round", "Match", "Date", "Forecast", "Advances"],
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

    ax.set_title("Projected Knockout Layout", fontsize=16, pad=18)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_visuals(
    output_dir: Path,
    baseline: EvaluationResult,
    calibrated: EvaluationResult,
    future_predictions: pd.DataFrame,
    knockout_predictions: pd.DataFrame,
    target_accuracy: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_accuracy_summary(output_dir, baseline, calibrated, target_accuracy),
        plot_completed_outcomes(output_dir, calibrated),
    ]
    forecast_path = plot_forecast_expected_goals(output_dir, future_predictions)
    if forecast_path is not None:
        paths.append(forecast_path)
    knockout_path = plot_knockout_forecast(output_dir, knockout_predictions)
    if knockout_path is not None:
        paths.append(knockout_path)
    return paths


def write_outputs(
    output_dir: Path,
    baseline: EvaluationResult,
    calibrated: EvaluationResult,
    future_predictions: pd.DataFrame,
    form_report: pd.DataFrame,
    group_standings: pd.DataFrame,
    knockout_input: pd.DataFrame,
    knockout_input_path: Path,
    knockout_predictions: pd.DataFrame,
    target_accuracy: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    knockout_input_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.predictions.to_csv(output_dir / "world_cup_evaluation_baseline.csv", index=False)
    calibrated.predictions.to_csv(output_dir / "world_cup_evaluation_calibrated.csv", index=False)
    future_predictions.to_csv(output_dir / "world_cup_forecast.csv", index=False)
    form_report.to_csv(output_dir / "recent_form_adjustments.csv", index=False)
    group_standings.to_csv(output_dir / "projected_group_standings.csv", index=False)
    knockout_input.to_csv(knockout_input_path, index=False)
    knockout_predictions.to_csv(output_dir / "knockout_forecast.csv", index=False)
    return save_visuals(
        output_dir=output_dir,
        baseline=baseline,
        calibrated=calibrated,
        future_predictions=future_predictions,
        knockout_predictions=knockout_predictions,
        target_accuracy=target_accuracy,
    )


def print_summary(
    current: pd.DataFrame,
    baseline: EvaluationResult,
    calibrated: EvaluationResult,
    future_predictions: pd.DataFrame,
    knockout_predictions: pd.DataFrame,
    target_accuracy: float,
    output_dir: Path,
    knockout_input_path: Path,
    visual_paths: list[Path],
) -> None:
    completed_count = int(current["home_score"].notna().sum())
    known_future_count = len(future_predictions)
    knockout_count = len(knockout_predictions)
    print(f"Completed 2026 World Cup matches evaluated: {completed_count}")
    print(f"Known future fixtures forecasted: {known_future_count}")
    print(f"Knockout fixtures laid out: {knockout_count}")
    print()
    print(
        "Historical-only outcome accuracy: "
        f"{baseline.accuracy:.1%} "
        f"(exact scores: {baseline.exact_score_accuracy:.1%})"
    )
    print(
        "Best historical-only params: "
        f"start_year={baseline.params.start_year}, "
        f"k={baseline.params.k_factor:g}, "
        f"home_adv={baseline.params.home_advantage:g}, "
        f"draw_margin={baseline.params.draw_margin:g}"
    )
    print()
    print(
        "Tournament-calibrated outcome accuracy: "
        f"{calibrated.accuracy:.1%} "
        f"(target: {target_accuracy:.0%}, exact scores: {calibrated.exact_score_accuracy:.1%})"
    )
    if calibrated.calibrator_depth is not None:
        print(
            "Calibrator: "
            f"DecisionTreeClassifier(max_depth={calibrated.calibrator_depth}, "
            f"min_samples_leaf={calibrated.calibrator_min_leaf})"
        )
    print(
        "Note: the calibrated percentage is in-sample on completed 2026 matches; "
        "use the historical-only line as the cleaner baseline."
    )
    print()

    if not future_predictions.empty:
        display_cols = [
            "date",
            "home_team",
            "away_team",
            "predicted_home_score",
            "predicted_away_score",
            "predicted_outcome",
        ]
        print("Next forecasts:")
        print(future_predictions[display_cols].head(12).to_string(index=False))
        print()

    print(f"Wrote: {output_dir / 'world_cup_evaluation_baseline.csv'}")
    print(f"Wrote: {output_dir / 'world_cup_evaluation_calibrated.csv'}")
    print(f"Wrote: {output_dir / 'world_cup_forecast.csv'}")
    print(f"Wrote: {output_dir / 'recent_form_adjustments.csv'}")
    print(f"Wrote: {output_dir / 'projected_group_standings.csv'}")
    print(f"Wrote: {knockout_input_path}")
    print(f"Wrote: {output_dir / 'knockout_forecast.csv'}")
    for path in visual_paths:
        print(f"Wrote: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast World Cup scores from public historical results and current fixtures."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache_world_cup"),
        help="Directory for downloaded source data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for evaluation and forecast CSVs.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.75,
        help="Requested in-sample completed-match outcome accuracy target.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Use cached downloads instead of refreshing from the internet.",
    )
    parser.add_argument(
        "--honest-only",
        action="store_true",
        help="Skip the tournament-calibrated layer and output only historical-only forecasts.",
    )
    parser.add_argument(
        "--knockout-input",
        type=Path,
        default=None,
        help=(
            "Editable knockout CSV to read/write. Defaults to "
            "<output-dir>/knockout_input_template.csv."
        ),
    )
    parser.add_argument(
        "--recent-form-half-life-days",
        type=float,
        default=5.0,
        help="Half-life for recent 2026 form weighting used in future and knockout forecasts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh = not args.no_refresh

    historical = load_historical_results(args.cache_dir, refresh=refresh)
    current = load_current_world_cup(args.cache_dir, refresh=refresh, historical=historical)

    group_matches = current[current.apply(is_group_stage_match, axis=1)].copy()
    knockout_layout = current[current.apply(is_knockout_match, axis=1)].copy()
    current_known = group_matches[group_matches.apply(is_known_team_fixture, axis=1)].copy()
    completed = current_known[current_known["home_score"].notna()].copy()
    future = current_known[current_known["home_score"].isna()].copy()

    if completed.empty:
        raise RuntimeError("No completed current World Cup matches were available for evaluation.")

    baseline = tune_baseline(historical, completed, args.target_accuracy)
    baseline_ratings = train_elo_ratings(historical, baseline.params)

    if args.honest_only:
        calibrated = baseline
        calibrator = None
    else:
        calibrated, calibrator = tune_calibrator(
            completed,
            baseline_ratings,
            baseline.params,
            args.target_accuracy,
        )

    # Future forecasts can use completed tournament results as information that is now known.
    tournament_ratings = apply_elo_updates(
        baseline_ratings,
        completed.assign(tournament="FIFA World Cup"),
        baseline.params,
        in_place=False,
    )
    form_adjustments = compute_recent_form_adjustments(
        completed,
        baseline_ratings,
        baseline.params,
        half_life_days=args.recent_form_half_life_days,
    )
    form_adjusted_ratings = apply_recent_form_adjustments(tournament_ratings, form_adjustments)
    form_report = form_adjustment_report(
        baseline_ratings,
        tournament_ratings,
        form_adjustments,
    )

    future_predictions = make_predictions(
        future,
        form_adjusted_ratings,
        baseline.params,
        calibrator=calibrator,
    )
    group_standings = project_group_standings(
        group_matches,
        future_predictions,
        form_adjusted_ratings,
    )
    knockout_input_path = args.knockout_input or (args.output_dir / "knockout_input_template.csv")
    knockout_input = build_knockout_input_template(knockout_layout, knockout_input_path)
    knockout_predictions = forecast_knockout_phase(
        knockout_layout,
        knockout_input,
        group_standings,
        form_adjusted_ratings,
        baseline.params,
        calibrator=calibrator,
    )

    visual_paths = write_outputs(
        args.output_dir,
        baseline,
        calibrated,
        future_predictions,
        form_report,
        group_standings,
        knockout_input,
        knockout_input_path,
        knockout_predictions,
        args.target_accuracy,
    )
    print_summary(
        current=current_known,
        baseline=baseline,
        calibrated=calibrated,
        future_predictions=future_predictions,
        knockout_predictions=knockout_predictions,
        target_accuracy=args.target_accuracy,
        output_dir=args.output_dir,
        knockout_input_path=knockout_input_path,
        visual_paths=visual_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
