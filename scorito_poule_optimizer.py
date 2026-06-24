#!/usr/bin/env python3
"""
Consolidated Scorito poule optimizer.

This script combines the exact-score match model, projected knockout bracket,
country-prediction logic, and phase-specific top-scorer model into one advice
sheet for Scorito.

The model follows the official Scorito WK 2026 rules:
  * match toto/exact points increase by round
  * top-scorer goals are worth much more for keepers/defenders
  * country-prediction milestones are scored separately
  * top scorers can be changed every phase

The extra 2026 Round of 32 has its own multiplier between the group stage and
Round of 16.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATCH_POINTS = {
    "Group": {"label": "Groepsfase", "toto": 30, "exact": 45},
    "Round of 32": {"label": "Laatste 32", "toto": 60, "exact": 90},
    "Round of 16": {"label": "Achtste finale", "toto": 90, "exact": 135},
    "Quarterfinals": {"label": "Kwartfinale", "toto": 120, "exact": 180},
    "Quarter-finals": {"label": "Kwartfinale", "toto": 120, "exact": 180},
    "Semifinals": {"label": "Halve finale", "toto": 150, "exact": 225},
    "Semi-finals": {"label": "Halve finale", "toto": 150, "exact": 225},
    "Match for third place": {"label": "Troostfinale/finale multiplier", "toto": 180, "exact": 270},
    "Final": {"label": "Finale", "toto": 180, "exact": 270},
}

TOPSCORER_POINTS = {
    "Keeper / Verdediger": {
        "Group": 64,
        "Round of 32": 96,
        "Round of 16": 96,
        "Quarterfinals": 128,
        "Quarter-finals": 128,
        "Semifinals": 160,
        "Semi-finals": 160,
        "Match for third place": 192,
        "Final": 192,
    },
    "Middenvelder": {
        "Group": 32,
        "Round of 32": 48,
        "Round of 16": 48,
        "Quarterfinals": 64,
        "Quarter-finals": 64,
        "Semifinals": 80,
        "Semi-finals": 80,
        "Match for third place": 96,
        "Final": 96,
    },
    "Aanvaller": {
        "Group": 16,
        "Round of 32": 24,
        "Round of 16": 24,
        "Quarterfinals": 32,
        "Quarter-finals": 32,
        "Semifinals": 40,
        "Semi-finals": 40,
        "Match for third place": 48,
        "Final": 48,
    },
}

ROUND_ORDER = {
    "Group": 0,
    "Round of 32": 1,
    "Round of 16": 2,
    "Quarterfinals": 3,
    "Quarter-finals": 3,
    "Semifinals": 4,
    "Semi-finals": 4,
    "Match for third place": 5,
    "Final": 6,
}

COUNTRY_POINTS = {
    "Door groepsfase": 50,
    "Juiste groepspositie 1/2": 75,
    "Door naar kwartfinale": 30,
    "Door naar halve finale": 60,
    "Door naar finale": 120,
    "Wereldkampioen": 250,
}


def run_if_needed(command: list[str], required_paths: list[Path], force: bool) -> None:
    if not force and all(path.exists() for path in required_paths):
        return
    subprocess.run(command, check=True)


def ensure_source_outputs(args: argparse.Namespace) -> None:
    exact_required = [
        args.exact_score_dir / "exact_score_model_forecast.csv",
        args.exact_score_dir / "exact_score_knockout_forecast.csv",
        args.exact_score_dir / "exact_score_projected_group_standings.csv",
    ]
    exact_command = [
        sys.executable,
        "world_cup_exact_score_forecaster.py",
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(args.exact_score_dir),
    ]
    if args.no_refresh:
        exact_command.append("--no-refresh")
    run_if_needed(exact_command, exact_required, args.refresh_sources)

    scorer_required = [
        args.topscorer_dir / "scorito_candidate_round_model.csv",
        args.topscorer_dir / "scorito_topscorer_recommendations.csv",
    ]
    scorer_command = [
        sys.executable,
        "scorito_knockout_topscorer_picker.py",
        "--cache-dir",
        str(args.cache_dir),
        "--exact-score-dir",
        str(args.exact_score_dir),
        "--output-dir",
        str(args.topscorer_dir),
        "--simulations",
        str(args.simulations),
    ]
    if args.no_refresh:
        scorer_command.append("--no-refresh")
    run_if_needed(scorer_command, scorer_required, args.refresh_sources)


def outcome_from_score(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


def poisson_prob(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def score_probabilities(
    home_xg: float,
    away_xg: float,
    predicted_home: int,
    predicted_away: int,
    max_goals: int = 10,
) -> tuple[float, float]:
    target_outcome = outcome_from_score(predicted_home, predicted_away)
    exact_probability = 0.0
    outcome_probability = 0.0
    home_xg = max(float(home_xg), 0.05)
    away_xg = max(float(away_xg), 0.05)

    for home_goals in range(max_goals + 1):
        home_probability = poisson_prob(home_xg, home_goals)
        for away_goals in range(max_goals + 1):
            probability = home_probability * poisson_prob(away_xg, away_goals)
            if home_goals == predicted_home and away_goals == predicted_away:
                exact_probability = probability
            if outcome_from_score(home_goals, away_goals) == target_outcome:
                outcome_probability += probability

    return exact_probability, outcome_probability


def round_key(round_name: object) -> str:
    text = str(round_name)
    if text.startswith("Group"):
        return "Group"
    return text


def match_points_for_round(round_name: object) -> dict[str, object]:
    return MATCH_POINTS.get(round_key(round_name), MATCH_POINTS["Round of 16"])


def build_match_advice(
    future_group: pd.DataFrame,
    knockout: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not future_group.empty:
        group = future_group.copy()
        group["round"] = group["group"]
        group["home_slot"] = group["home_team"]
        group["away_slot"] = group["away_team"]
        group["advancing_team"] = ""
        frames.append(group)
    if not knockout.empty:
        frames.append(knockout.copy())
    if not frames:
        return pd.DataFrame()

    matches = pd.concat(frames, ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    for row in matches.itertuples(index=False):
        home_team = getattr(row, "home_team", "")
        away_team = getattr(row, "away_team", "")
        if not str(home_team).strip() or not str(away_team).strip():
            continue
        predicted_home = int(
            float(
                getattr(
                    row,
                    "elo_poisson_predicted_home_score",
                    getattr(row, "predicted_home_score"),
                )
            )
        )
        predicted_away = int(
            float(
                getattr(
                    row,
                    "elo_poisson_predicted_away_score",
                    getattr(row, "predicted_away_score"),
                )
            )
        )
        home_xg = float(getattr(row, "home_xg", predicted_home + 0.3) or predicted_home + 0.3)
        away_xg = float(getattr(row, "away_xg", predicted_away + 0.3) or predicted_away + 0.3)
        points = match_points_for_round(getattr(row, "round"))
        exact_prob, outcome_prob = score_probabilities(
            home_xg,
            away_xg,
            predicted_home,
            predicted_away,
        )
        expected_points = (
            exact_prob * float(points["exact"])
            + max(outcome_prob - exact_prob, 0.0) * float(points["toto"])
        )
        rows.append(
            {
                "date": getattr(row, "date", ""),
                "round": getattr(row, "round", ""),
                "round_scoring_label": points["label"],
                "match_no": getattr(row, "match_no", np.nan),
                "home_team": home_team,
                "away_team": away_team,
                "recommended_score": f"{predicted_home}-{predicted_away}",
                "predicted_outcome": outcome_from_score(predicted_home, predicted_away),
                "advancing_team_if_draw_or_win": getattr(row, "advancing_team", ""),
                "toto_points": points["toto"],
                "exact_points": points["exact"],
                "estimated_exact_probability": exact_prob,
                "estimated_toto_probability": outcome_prob,
                "expected_match_points": expected_points,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "note": "Penalty shootouts ignored; score is modeled through 120 minutes.",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["round_order"] = result["round"].map(lambda value: ROUND_ORDER.get(round_key(value), 99))
    return result.sort_values(["date", "round_order", "match_no"]).reset_index(drop=True)


def group_advancement_teams(group_standings: pd.DataFrame) -> set[str]:
    standings = group_standings.copy()
    top_two = standings[standings["position"].isin([1, 2])]["team"]
    best_thirds = standings[
        standings["position"].eq(3)
        & pd.to_numeric(standings["third_place_rank"], errors="coerce").le(8)
    ]["team"]
    return set(top_two.astype(str)) | set(best_thirds.astype(str))


def build_country_advice(
    group_standings: pd.DataFrame,
    knockout: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    advancement = group_advancement_teams(group_standings)

    for row in group_standings.itertuples(index=False):
        if str(row.team) in advancement:
            rows.append(
                {
                    "category": "Landen",
                    "prediction_type": "Door groepsfase",
                    "team": row.team,
                    "group": row.group,
                    "prediction": "qualifies for knockouts",
                    "points": COUNTRY_POINTS["Door groepsfase"],
                    "confidence_proxy": float(row.points),
                    "reason": f"Projected position {int(row.position)}, points {int(row.points)}",
                }
            )
        if int(row.position) in [1, 2]:
            rows.append(
                {
                    "category": "Landen",
                    "prediction_type": "Juiste groepspositie 1/2",
                    "team": row.team,
                    "group": row.group,
                    "prediction": f"{int(row.position)}e in {row.group}",
                    "points": COUNTRY_POINTS["Juiste groepspositie 1/2"],
                    "confidence_proxy": float(row.goal_difference),
                    "reason": f"Projected GD {int(row.goal_difference)}, goals for {int(row.goals_for)}",
                }
            )

    if knockout.empty:
        return pd.DataFrame(rows)

    milestone_map = {
        "Round of 16": "Door naar kwartfinale",
        "Quarterfinals": "Door naar halve finale",
        "Quarter-finals": "Door naar halve finale",
        "Semifinals": "Door naar finale",
        "Semi-finals": "Door naar finale",
        "Final": "Wereldkampioen",
    }
    for row in knockout.itertuples(index=False):
        round_name = str(row.round)
        prediction_type = milestone_map.get(round_name)
        if prediction_type is None:
            continue
        team = str(row.advancing_team)
        if not team.strip():
            continue
        rows.append(
            {
                "category": "Landen",
                "prediction_type": prediction_type,
                "team": team,
                "group": "",
                "prediction": (
                    "world champion"
                    if prediction_type == "Wereldkampioen"
                    else f"wins {round_name}"
                ),
                "points": COUNTRY_POINTS[prediction_type],
                "confidence_proxy": abs(float(row.elo_diff)) if not pd.isna(row.elo_diff) else 0.0,
                "reason": f"{row.home_team} {row.predicted_home_score}-{row.predicted_away_score} {row.away_team}",
            }
        )

    return pd.DataFrame(rows)


def phase_label(round_name: object) -> str:
    text = str(round_name)
    labels = {
        "Round of 32": "Fase 2 - Round of 32 / 1e knockout",
        "Round of 16": "Fase 2b - Achtste finale",
        "Quarterfinals": "Fase 3 - Kwartfinale",
        "Quarter-finals": "Fase 3 - Kwartfinale",
        "Semifinals": "Fase 4 - Halve finale",
        "Semi-finals": "Fase 4 - Halve finale",
        "Match for third place": "Fase 5 - Troostfinale",
        "Final": "Fase 6 - Finale",
    }
    return labels.get(text, text)


def differential_multiplier(position: str) -> float:
    if position == "Keeper / Verdediger":
        return 1.40
    if position == "Middenvelder":
        return 1.15
    return 0.95


def select_phase_players(
    phase_df: pd.DataFrame,
    picks_per_phase: int,
    pool_size: str,
) -> pd.DataFrame:
    ranked = phase_df.sort_values(
        ["expected_round_points", "expected_round_goals"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked["pure_points_rank"] = np.arange(1, len(ranked) + 1)
    ranked["differential_score"] = ranked.apply(
        lambda row: row["expected_round_points"]
        * differential_multiplier(str(row["scorito_position"])),
        axis=1,
    )

    if pool_size == "small":
        selected_indexes = list(ranked.head(picks_per_phase).index)
    elif pool_size == "medium":
        core_count = max(picks_per_phase - 1, 1)
        selected_indexes = list(ranked.head(core_count).index)
        pool = ranked.iloc[core_count : min(len(ranked), core_count + 8)]
        if not pool.empty:
            selected_indexes.extend(
                pool.sort_values("differential_score", ascending=False)
                .head(picks_per_phase - len(selected_indexes))
                .index.tolist()
            )
    else:
        core_count = max(picks_per_phase - 2, 1)
        selected_indexes = list(ranked.head(core_count).index)
        pool = ranked.iloc[core_count : min(len(ranked), core_count + 12)]
        if not pool.empty:
            selected_indexes.extend(
                pool.sort_values("differential_score", ascending=False)
                .head(picks_per_phase - len(selected_indexes))
                .index.tolist()
            )

    selected = ranked.loc[selected_indexes].copy()
    selected["selected_for_pool_size"] = pool_size
    selected["phase_pick_rank"] = np.arange(1, len(selected) + 1)
    selected["pick_style"] = selected["pure_points_rank"].map(
        lambda rank: "core_expected_points" if rank <= picks_per_phase else "differential_upside"
    )
    return selected


def build_topscorer_phase_advice(
    candidate_rounds: pd.DataFrame,
    picks_per_phase: int,
    pool_size: str,
) -> pd.DataFrame:
    if candidate_rounds.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    work = candidate_rounds.copy()
    work["phase"] = work["round"].map(phase_label)
    work["round_order"] = work["round"].map(lambda value: ROUND_ORDER.get(round_key(value), 99))
    for (_, round_name), phase_df in work.groupby(["round_order", "round"], sort=True):
        phase_selected = select_phase_players(phase_df, picks_per_phase, pool_size)
        rows.append(phase_selected)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if result.empty:
        return result
    columns = [
        "phase",
        "round",
        "phase_pick_rank",
        "pick_style",
        "player",
        "team",
        "opponent",
        "scorito_position",
        "group_goals",
        "expected_round_goals",
        "points_per_goal",
        "expected_round_points",
        "team_xg",
        "team_elo_diff",
        "pure_points_rank",
        "differential_score",
        "selected_for_pool_size",
        "round_order",
    ]
    result = result[columns].sort_values(["round_order", "phase_pick_rank"]).reset_index(drop=True)
    return result.drop(columns=["round_order"])


def build_master_sheet(
    match_advice: pd.DataFrame,
    country_advice: pd.DataFrame,
    top_advice: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not match_advice.empty:
        for row in match_advice.head(20).itertuples(index=False):
            rows.append(
                {
                    "section": "Wedstrijden",
                    "priority": float(row.expected_match_points),
                    "advice": f"{row.home_team} - {row.away_team}: {row.recommended_score}",
                    "points_context": f"EV {row.expected_match_points:.1f}; exact {row.exact_points}; toto {row.toto_points}",
                    "reason": f"{row.round_scoring_label}, exact p~{row.estimated_exact_probability:.1%}, toto p~{row.estimated_toto_probability:.1%}",
                }
            )
    if not top_advice.empty:
        first_round_order = top_advice["round"].map(lambda value: ROUND_ORDER.get(round_key(value), 99)).min()
        current_phase = top_advice[
            top_advice["round"].map(lambda value: ROUND_ORDER.get(round_key(value), 99))
            .eq(first_round_order)
        ]
        for row in current_phase.itertuples(index=False):
            rows.append(
                {
                    "section": "Topscorers huidige fase",
                    "priority": float(row.expected_round_points),
                    "advice": f"{row.player} ({row.team}, {row.scorito_position})",
                    "points_context": f"{row.points_per_goal} pnt/goal; EV {row.expected_round_points:.1f}",
                    "reason": f"{row.pick_style}; vs {row.opponent}; team xG {row.team_xg:.2f}",
                }
            )
    if not country_advice.empty:
        for row in country_advice.sort_values(["points", "confidence_proxy"], ascending=False).head(20).itertuples(index=False):
            rows.append(
                {
                    "section": "Landen",
                    "priority": float(row.points),
                    "advice": f"{row.prediction_type}: {row.team}",
                    "points_context": f"{int(row.points)} pnt",
                    "reason": row.reason,
                }
            )
    return pd.DataFrame(rows).sort_values(["section", "priority"], ascending=[True, False])


def plot_summary(output_dir: Path, top_advice: pd.DataFrame, country_advice: pd.DataFrame) -> Path | None:
    if top_advice.empty and country_advice.empty:
        return None
    path = output_dir / "scorito_poule_summary.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    if top_advice.empty:
        axes[0].axis("off")
    else:
        first_round_order = top_advice["round"].map(lambda value: ROUND_ORDER.get(round_key(value), 99)).min()
        current_phase = top_advice[
            top_advice["round"].map(lambda value: ROUND_ORDER.get(round_key(value), 99)).eq(first_round_order)
        ].copy().iloc[::-1]
        labels = current_phase.apply(lambda row: f"{row['player']} ({row['team']})", axis=1)
        colors = current_phase["scorito_position"].map(
            {
                "Keeper / Verdediger": "#4e79a7",
                "Middenvelder": "#59a14f",
                "Aanvaller": "#f28e2b",
            }
        ).fillna("#bab0ac")
        axes[0].barh(labels, current_phase["expected_round_points"], color=colors)
        axes[0].set_title("Topscorers: current phase")
        axes[0].set_xlabel("Expected phase points")

    if country_advice.empty:
        axes[1].axis("off")
    else:
        milestones = country_advice[
            country_advice["prediction_type"].isin(
                ["Door naar finale", "Wereldkampioen", "Door naar halve finale"]
            )
        ].copy()
        milestones = milestones.sort_values(["points", "confidence_proxy"], ascending=False).head(10).iloc[::-1]
        labels = milestones.apply(lambda row: f"{row['prediction_type']}: {row['team']}", axis=1)
        axes[1].barh(labels, milestones["points"], color="#6f4e7c")
        axes[1].set_title("Country milestone picks")
        axes[1].set_xlabel("Scorito points if correct")

    fig.subplots_adjust(left=0.22, right=0.98, top=0.90, bottom=0.12, wspace=0.45)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_outputs(
    output_dir: Path,
    match_advice: pd.DataFrame,
    country_advice: pd.DataFrame,
    top_advice: pd.DataFrame,
    master: pd.DataFrame,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "scorito_match_picks.csv",
        output_dir / "scorito_country_picks.csv",
        output_dir / "scorito_topscorers_by_phase.csv",
        output_dir / "scorito_master_advice.csv",
    ]
    match_advice.to_csv(paths[0], index=False)
    country_advice.to_csv(paths[1], index=False)
    top_advice.to_csv(paths[2], index=False)
    master.to_csv(paths[3], index=False)
    figure = plot_summary(output_dir, top_advice, country_advice)
    if figure is not None:
        paths.append(figure)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Scorito poule advice sheet.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_world_cup"))
    parser.add_argument("--exact-score-dir", type=Path, default=Path("outputs_exact_score_model"))
    parser.add_argument("--topscorer-dir", type=Path, default=Path("outputs_scorito_topscorers"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_scorito_poule"))
    parser.add_argument(
        "--pool-size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Small=max expected points; medium/large add more differential defender/midfielder upside.",
    )
    parser.add_argument(
        "--topscorers-per-phase",
        type=int,
        default=4,
        help="Scorito top-scorer picks per phase according to the supplied rules.",
    )
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--refresh-sources", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_source_outputs(args)

    future_group = pd.read_csv(args.exact_score_dir / "exact_score_model_forecast.csv")
    knockout = pd.read_csv(args.exact_score_dir / "exact_score_knockout_forecast.csv")
    group_standings = pd.read_csv(args.exact_score_dir / "exact_score_projected_group_standings.csv")
    candidate_rounds = pd.read_csv(args.topscorer_dir / "scorito_candidate_round_model.csv")

    match_advice = build_match_advice(future_group, knockout)
    country_advice = build_country_advice(group_standings, knockout)
    top_advice = build_topscorer_phase_advice(
        candidate_rounds,
        picks_per_phase=args.topscorers_per_phase,
        pool_size=args.pool_size,
    )
    master = build_master_sheet(match_advice, country_advice, top_advice)
    output_paths = write_outputs(args.output_dir, match_advice, country_advice, top_advice, master)

    print("Scorito poule optimizer complete")
    print(f"Pool-size strategy: {args.pool_size}")
    print(f"Match picks: {len(match_advice)}")
    print(f"Country picks: {len(country_advice)}")
    print(f"Topscorer phase picks: {len(top_advice)}")
    if not top_advice.empty:
        first_order = top_advice["round"].map(lambda value: ROUND_ORDER.get(round_key(value), 99)).min()
        current = top_advice[
            top_advice["round"].map(lambda value: ROUND_ORDER.get(round_key(value), 99)).eq(first_order)
        ]
        print("Current phase top-scorer picks:")
        for row in current.itertuples(index=False):
            print(
                f"{int(row.phase_pick_rank)}. {row.player} ({row.team}, {row.scorito_position}) "
                f"- {row.expected_round_points:.1f} expected points"
            )
    champion = country_advice[country_advice["prediction_type"].eq("Wereldkampioen")]
    if not champion.empty:
        print(f"Champion pick: {champion.iloc[0]['team']}")
    for path in output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
