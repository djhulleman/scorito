#!/usr/bin/env python3
"""
Walk-forward backtest for the Scorito knockout advice model.

For each previous tournament:
  1. hold that tournament out of the historical priors
  2. use only its group-stage goals/results as the visible information
  3. make phase-by-phase top-scorer picks and knockout match picks
  4. score those picks against actual knockout goals/results

This is not a perfect recreation of all Scorito inputs, because real player
availability and official Scorito positions are not public in these datasets.
The script reports position coverage and defaults unknown positions to attacker,
which is conservative for defender/keeper upside.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scorito_knockout_topscorer_picker import (
    POSITION_SHARE_PRIOR,
    ROUND_ORDER,
    SCORITO_POINTS,
    TOURNAMENT_SOURCES,
    TournamentSource,
    bucket_group_goals,
    historical_carryover_patterns,
    historical_team_patterns,
    load_historical_tournaments,
    load_player_positions,
    scorito_points_for_round,
)


MATCH_POINTS = {
    "Round of 32": {"toto": 90, "exact": 135},
    "Round of 16": {"toto": 90, "exact": 135},
    "Quarterfinals": {"toto": 120, "exact": 180},
    "Quarter-finals": {"toto": 120, "exact": 180},
    "Semifinals": {"toto": 150, "exact": 225},
    "Semi-finals": {"toto": 150, "exact": 225},
    "Match for third place": {"toto": 180, "exact": 270},
    "Final": {"toto": 180, "exact": 270},
}


def outcome_from_score(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


def poisson_prob(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def clipped(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def top_goal_scorers_from_group(goals: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    group_goals = goals[goals["stage_type"].eq("group")].copy()
    if group_goals.empty:
        return pd.DataFrame()

    players = group_goals.groupby(["player", "player_url", "team"], as_index=False).agg(
        group_goals=("player", "size"),
        group_goal_matches=("match_key", "nunique"),
    )
    team_rows = team_stats_from_group(matches)
    return players.merge(team_rows, how="left", on="team").fillna(
        {"team_completed_group_goals": 0, "team_group_matches": 0}
    )


def team_stats_from_group(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_matches = matches[
        matches["stage_type"].eq("group")
        & matches["home_score"].notna()
        & matches["away_score"].notna()
    ]
    for row in group_matches.itertuples(index=False):
        rows.append(
            {
                "team": row.home_team,
                "goals_for": float(row.home_score),
                "goals_against": float(row.away_score),
            }
        )
        rows.append(
            {
                "team": row.away_team,
                "goals_for": float(row.away_score),
                "goals_against": float(row.home_score),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "team",
                "team_completed_group_goals",
                "team_group_goals_against",
                "team_group_matches",
                "group_goals_for_per_match",
                "group_goals_against_per_match",
            ]
        )
    stats = pd.DataFrame(rows).groupby("team", as_index=False).agg(
        team_completed_group_goals=("goals_for", "sum"),
        team_group_goals_against=("goals_against", "sum"),
        team_group_matches=("goals_for", "size"),
    )
    stats["group_goals_for_per_match"] = (
        stats["team_completed_group_goals"] / stats["team_group_matches"]
    )
    stats["group_goals_against_per_match"] = (
        stats["team_group_goals_against"] / stats["team_group_matches"]
    )
    return stats


def global_knockout_team_goals(train_matches: pd.DataFrame) -> float:
    knockout = train_matches[
        train_matches["stage_type"].eq("knockout")
        & train_matches["home_score"].notna()
        & train_matches["away_score"].notna()
    ]
    if knockout.empty:
        return 1.15
    return float((knockout["home_score"].sum() + knockout["away_score"].sum()) / (2 * len(knockout)))


def expected_team_goals(
    team: str,
    opponent: str,
    group_stats: pd.DataFrame,
    global_knockout_gpm: float,
) -> float:
    lookup = group_stats.set_index("team").to_dict("index")
    team_row = lookup.get(team, {})
    opp_row = lookup.get(opponent, {})
    team_attack = float(team_row.get("group_goals_for_per_match", global_knockout_gpm))
    opponent_defense = float(opp_row.get("group_goals_against_per_match", global_knockout_gpm))
    team_margin = team_attack - float(team_row.get("group_goals_against_per_match", global_knockout_gpm))
    opp_margin = float(opp_row.get("group_goals_for_per_match", global_knockout_gpm)) - opponent_defense
    edge_boost = 0.10 * (team_margin - opp_margin)
    estimate = 0.45 * team_attack + 0.30 * opponent_defense + 0.25 * global_knockout_gpm + edge_boost
    return round(clipped(estimate, 0.25, 2.85), 3)


def knockout_round_matches(matches: pd.DataFrame, round_name: str) -> pd.DataFrame:
    return matches[
        matches["stage_type"].eq("knockout")
        & matches["round"].eq(round_name)
        & matches["home_score"].notna()
        & matches["away_score"].notna()
    ].copy()


def ordered_knockout_rounds(matches: pd.DataFrame) -> list[str]:
    rounds = (
        matches[matches["stage_type"].eq("knockout")]["round"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return sorted(rounds, key=lambda value: ROUND_ORDER.get(str(value), 99))


def build_phase_candidate_rows(
    tournament: str,
    round_name: str,
    round_matches: pd.DataFrame,
    group_players: pd.DataFrame,
    positions: pd.DataFrame,
    carryover_patterns: pd.DataFrame,
    group_stats: pd.DataFrame,
    global_knockout_gpm: float,
) -> pd.DataFrame:
    if group_players.empty or round_matches.empty:
        return pd.DataFrame()

    teams = set(round_matches["home_team"].astype(str)) | set(round_matches["away_team"].astype(str))
    candidates = group_players[group_players["team"].isin(teams)].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["group_goal_bucket"] = candidates["group_goals"].map(bucket_group_goals)
    boost_lookup = carryover_patterns.set_index("group_goal_bucket")["carryover_boost"].to_dict()
    candidates["carryover_boost"] = candidates["group_goal_bucket"].map(boost_lookup).fillna(1.0)
    candidates = candidates.merge(positions, how="left", on=["player", "player_url"])
    candidates["scorito_position"] = candidates["scorito_position"].fillna("Aanvaller")
    candidates["raw_position"] = candidates["raw_position"].fillna("")
    candidates["position_known"] = candidates["raw_position"].astype(str).str.len().gt(0)
    candidates["position_share_prior"] = candidates["scorito_position"].map(POSITION_SHARE_PRIOR).fillna(
        POSITION_SHARE_PRIOR["Aanvaller"]
    )
    candidates["player_goal_share"] = (
        candidates["group_goals"] + 2.0 * candidates["position_share_prior"]
    ) / (candidates["team_completed_group_goals"] + 2.0)
    candidates["player_goal_share"] = candidates["player_goal_share"].clip(0.015, 0.75)
    candidates["multi_match_scorer_boost"] = (
        1.0 + 0.10 * (candidates["group_goal_matches"].clip(lower=1) - 1)
    ).clip(1.0, 1.25)

    fixture_rows = []
    for match in round_matches.itertuples(index=False):
        home_xg = expected_team_goals(
            str(match.home_team),
            str(match.away_team),
            group_stats,
            global_knockout_gpm,
        )
        away_xg = expected_team_goals(
            str(match.away_team),
            str(match.home_team),
            group_stats,
            global_knockout_gpm,
        )
        fixture_rows.append(
            {
                "team": str(match.home_team),
                "opponent": str(match.away_team),
                "team_xg": home_xg,
                "opponent_xg": away_xg,
            }
        )
        fixture_rows.append(
            {
                "team": str(match.away_team),
                "opponent": str(match.home_team),
                "team_xg": away_xg,
                "opponent_xg": home_xg,
            }
        )
    fixtures = pd.DataFrame(fixture_rows)

    rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        fixture = fixtures[fixtures["team"].eq(candidate.team)]
        if fixture.empty:
            continue
        fixture_row = fixture.iloc[0]
        expected_goals = (
            float(fixture_row["team_xg"])
            * float(candidate.player_goal_share)
            * float(candidate.carryover_boost)
            * float(candidate.multi_match_scorer_boost)
        )
        points_per_goal = scorito_points_for_round(str(candidate.scorito_position), round_name)
        rows.append(
            {
                "tournament": tournament,
                "round": round_name,
                "player": candidate.player,
                "player_url": candidate.player_url,
                "team": candidate.team,
                "opponent": fixture_row["opponent"],
                "scorito_position": candidate.scorito_position,
                "raw_position": candidate.raw_position,
                "position_known": bool(candidate.position_known),
                "group_goals": int(candidate.group_goals),
                "group_goal_matches": int(candidate.group_goal_matches),
                "player_goal_share": float(candidate.player_goal_share),
                "carryover_boost": float(candidate.carryover_boost),
                "team_xg": float(fixture_row["team_xg"]),
                "expected_round_goals": expected_goals,
                "points_per_goal": points_per_goal,
                "expected_round_points": expected_goals * points_per_goal,
            }
        )
    return pd.DataFrame(rows)


def differential_multiplier(position: str) -> float:
    if position == "Keeper / Verdediger":
        return 1.40
    if position == "Middenvelder":
        return 1.15
    return 0.95


def select_topscorer_picks(
    candidates: pd.DataFrame,
    picks_per_phase: int,
    pool_size: str,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ranked = candidates.sort_values(
        ["expected_round_points", "expected_round_goals"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked["pure_points_rank"] = np.arange(1, len(ranked) + 1)
    ranked["differential_score"] = ranked.apply(
        lambda row: row["expected_round_points"] * differential_multiplier(row["scorito_position"]),
        axis=1,
    )
    if pool_size == "small":
        indexes = ranked.head(picks_per_phase).index.tolist()
    else:
        core = max(picks_per_phase - (1 if pool_size == "medium" else 2), 1)
        indexes = ranked.head(core).index.tolist()
        pool = ranked.iloc[core : core + (8 if pool_size == "medium" else 12)]
        remaining = picks_per_phase - len(indexes)
        if remaining > 0 and not pool.empty:
            indexes.extend(pool.sort_values("differential_score", ascending=False).head(remaining).index.tolist())
    picks = ranked.loc[indexes].copy()
    picks["pick_rank"] = np.arange(1, len(picks) + 1)
    picks["pick_style"] = picks["pure_points_rank"].map(
        lambda rank: "core_expected_points" if rank <= picks_per_phase else "differential_upside"
    )
    return picks


def actual_phase_points(
    goals: pd.DataFrame,
    round_name: str,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    phase_goals = goals[
        goals["stage_type"].eq("knockout")
        & goals["round"].eq(round_name)
    ].copy()
    if phase_goals.empty:
        return pd.DataFrame(columns=["player", "player_url", "team", "actual_goals", "actual_points"])
    actual = phase_goals.groupby(["player", "player_url", "team"], as_index=False).size()
    actual = actual.rename(columns={"size": "actual_goals"})
    actual = actual.merge(positions, how="left", on=["player", "player_url"])
    actual["scorito_position"] = actual["scorito_position"].fillna("Aanvaller")
    actual["actual_points"] = actual.apply(
        lambda row: int(row["actual_goals"]) * scorito_points_for_round(row["scorito_position"], round_name),
        axis=1,
    )
    return actual


def score_topscorer_picks(
    picks: pd.DataFrame,
    actual: pd.DataFrame,
    candidates: pd.DataFrame,
    picks_per_phase: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if picks.empty:
        return picks, {
            "selected_points": 0.0,
            "oracle_candidate_points": 0.0,
            "oracle_overall_points": 0.0,
            "selected_goals": 0.0,
            "scoring_picks": 0.0,
            "candidate_count": 0.0,
            "position_known_rate": 0.0,
        }

    actual_lookup = actual.set_index(["player", "player_url", "team"]).to_dict("index") if not actual.empty else {}
    scored = picks.copy()
    actual_goals = []
    actual_points = []
    for row in scored.itertuples(index=False):
        lookup = actual_lookup.get((row.player, row.player_url, row.team), {})
        actual_goals.append(int(lookup.get("actual_goals", 0)))
        actual_points.append(float(lookup.get("actual_points", 0)))
    scored["actual_goals"] = actual_goals
    scored["actual_points"] = actual_points

    candidate_keys = set(zip(candidates["player"], candidates["player_url"], candidates["team"]))
    actual_candidates = actual[
        actual.apply(lambda row: (row["player"], row["player_url"], row["team"]) in candidate_keys, axis=1)
    ].copy()
    oracle_candidate = (
        actual_candidates.sort_values("actual_points", ascending=False).head(picks_per_phase)["actual_points"].sum()
        if not actual_candidates.empty
        else 0.0
    )
    oracle_overall = (
        actual.sort_values("actual_points", ascending=False).head(picks_per_phase)["actual_points"].sum()
        if not actual.empty
        else 0.0
    )
    summary = {
        "selected_points": float(scored["actual_points"].sum()),
        "oracle_candidate_points": float(oracle_candidate),
        "oracle_overall_points": float(oracle_overall),
        "selected_goals": float(scored["actual_goals"].sum()),
        "scoring_picks": float((scored["actual_goals"] > 0).sum()),
        "candidate_count": float(len(candidates)),
        "position_known_rate": float(candidates["position_known"].mean()) if "position_known" in candidates else 0.0,
    }
    return scored, summary


def most_likely_score(home_xg: float, away_xg: float, max_goals: int = 7) -> tuple[int, int]:
    best = (-1.0, 0, 0)
    for home_goals in range(max_goals + 1):
        home_prob = poisson_prob(home_xg, home_goals)
        for away_goals in range(max_goals + 1):
            probability = home_prob * poisson_prob(away_xg, away_goals)
            if probability > best[0]:
                best = (probability, home_goals, away_goals)
    return best[1], best[2]


def match_points_for_round(round_name: str) -> dict[str, int]:
    return MATCH_POINTS.get(round_name, MATCH_POINTS["Round of 16"])


def backtest_match_picks(
    tournament: str,
    matches: pd.DataFrame,
    group_stats: pd.DataFrame,
    global_knockout_gpm: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    knockout = matches[
        matches["stage_type"].eq("knockout")
        & matches["home_score"].notna()
        & matches["away_score"].notna()
    ].copy()
    for row in knockout.sort_values(["date", "round"]).itertuples(index=False):
        home_xg = expected_team_goals(str(row.home_team), str(row.away_team), group_stats, global_knockout_gpm)
        away_xg = expected_team_goals(str(row.away_team), str(row.home_team), group_stats, global_knockout_gpm)
        pred_home, pred_away = most_likely_score(home_xg, away_xg)
        actual_home = int(float(row.home_score))
        actual_away = int(float(row.away_score))
        exact = pred_home == actual_home and pred_away == actual_away
        toto = outcome_from_score(pred_home, pred_away) == outcome_from_score(actual_home, actual_away)
        points = match_points_for_round(str(row.round))
        scored_points = points["exact"] if exact else (points["toto"] if toto else 0)
        rows.append(
            {
                "tournament": tournament,
                "round": row.round,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "predicted_score": f"{pred_home}-{pred_away}",
                "actual_score": f"{actual_home}-{actual_away}",
                "predicted_outcome": outcome_from_score(pred_home, pred_away),
                "actual_outcome": outcome_from_score(actual_home, actual_away),
                "exact_correct": exact,
                "toto_correct": toto,
                "scorito_match_points": scored_points,
                "max_exact_points": points["exact"],
                "home_xg": home_xg,
                "away_xg": away_xg,
            }
        )
    return pd.DataFrame(rows)


def backtest_tournament(
    tournament: str,
    all_matches: pd.DataFrame,
    all_goals: pd.DataFrame,
    positions: pd.DataFrame,
    picks_per_phase: int,
    pool_size: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_matches = all_matches[all_matches["tournament"].eq(tournament)].copy()
    target_goals = all_goals[all_goals["tournament"].eq(tournament)].copy()
    train_matches = all_matches[~all_matches["tournament"].eq(tournament)].copy()
    train_goals = all_goals[~all_goals["tournament"].eq(tournament)].copy()
    carryover = historical_carryover_patterns(train_goals)
    global_gpm = global_knockout_team_goals(train_matches)
    group_stats = team_stats_from_group(target_matches)
    group_players = top_goal_scorers_from_group(target_goals, target_matches)

    pick_frames: list[pd.DataFrame] = []
    phase_summaries: list[dict[str, object]] = []
    for round_name in ordered_knockout_rounds(target_matches):
        round_matches = knockout_round_matches(target_matches, round_name)
        candidates = build_phase_candidate_rows(
            tournament,
            round_name,
            round_matches,
            group_players,
            positions,
            carryover,
            group_stats,
            global_gpm,
        )
        picks = select_topscorer_picks(candidates, picks_per_phase, pool_size)
        actual = actual_phase_points(target_goals, round_name, positions)
        scored, summary = score_topscorer_picks(picks, actual, candidates, picks_per_phase)
        if not scored.empty:
            pick_frames.append(scored)
        phase_summaries.append(
            {
                "tournament": tournament,
                "round": round_name,
                "picks": int(len(scored)),
                **summary,
                "candidate_efficiency": (
                    summary["selected_points"] / summary["oracle_candidate_points"]
                    if summary["oracle_candidate_points"] > 0
                    else np.nan
                ),
                "overall_efficiency": (
                    summary["selected_points"] / summary["oracle_overall_points"]
                    if summary["oracle_overall_points"] > 0
                    else np.nan
                ),
            }
        )

    match_picks = backtest_match_picks(tournament, target_matches, group_stats, global_gpm)
    picks_df = pd.concat(pick_frames, ignore_index=True) if pick_frames else pd.DataFrame()
    phase_df = pd.DataFrame(phase_summaries)
    return picks_df, phase_df, match_picks


def load_or_fetch_positions(
    all_goals: pd.DataFrame,
    cache_dir: Path,
    refresh: bool,
    fetch_positions: bool,
) -> pd.DataFrame:
    players = (
        all_goals[["player", "player_url"]]
        .drop_duplicates()
        .sort_values("player")
        .reset_index(drop=True)
    )
    if fetch_positions:
        return load_player_positions(players, cache_dir, refresh=refresh, overrides_path=None)

    cache_path = cache_dir / "topscorer_player_positions.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path).fillna("")
        return players.merge(cached, how="left", on=["player", "player_url"]).fillna(
            {"raw_position": "", "scorito_position": "Aanvaller"}
        )
    players["raw_position"] = ""
    players["scorito_position"] = "Aanvaller"
    return players


def tournament_summary(
    picks: pd.DataFrame,
    phases: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    tournament_rows: list[dict[str, object]] = []
    tournaments = sorted(set(phases["tournament"].dropna()) | set(matches["tournament"].dropna()))
    for tournament in tournaments:
        phase_rows = phases[phases["tournament"].eq(tournament)]
        match_rows = matches[matches["tournament"].eq(tournament)]
        tournament_rows.append(
            {
                "tournament": tournament,
                "topscorer_points": float(phase_rows["selected_points"].sum()),
                "oracle_candidate_points": float(phase_rows["oracle_candidate_points"].sum()),
                "oracle_overall_points": float(phase_rows["oracle_overall_points"].sum()),
                "topscorer_candidate_efficiency": (
                    float(phase_rows["selected_points"].sum())
                    / float(phase_rows["oracle_candidate_points"].sum())
                    if float(phase_rows["oracle_candidate_points"].sum()) > 0
                    else np.nan
                ),
                "topscorer_overall_efficiency": (
                    float(phase_rows["selected_points"].sum())
                    / float(phase_rows["oracle_overall_points"].sum())
                    if float(phase_rows["oracle_overall_points"].sum()) > 0
                    else np.nan
                ),
                "topscorer_goals": float(phase_rows["selected_goals"].sum()),
                "topscorer_scoring_picks": float(phase_rows["scoring_picks"].sum()),
                "match_points": float(match_rows["scorito_match_points"].sum()) if not match_rows.empty else 0.0,
                "match_exact_accuracy": float(match_rows["exact_correct"].mean()) if not match_rows.empty else np.nan,
                "match_toto_accuracy": float(match_rows["toto_correct"].mean()) if not match_rows.empty else np.nan,
                "match_count": int(len(match_rows)),
                "position_known_rate": float(phase_rows["position_known_rate"].mean()) if not phase_rows.empty else 0.0,
            }
        )
    return pd.DataFrame(tournament_rows)


def overall_summary(tournaments: pd.DataFrame, phases: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tournaments_backtested": int(len(tournaments)),
                "topscorer_total_points": float(tournaments["topscorer_points"].sum()),
                "topscorer_candidate_efficiency": (
                    float(tournaments["topscorer_points"].sum())
                    / float(tournaments["oracle_candidate_points"].sum())
                    if float(tournaments["oracle_candidate_points"].sum()) > 0
                    else np.nan
                ),
                "topscorer_overall_efficiency": (
                    float(tournaments["topscorer_points"].sum())
                    / float(tournaments["oracle_overall_points"].sum())
                    if float(tournaments["oracle_overall_points"].sum()) > 0
                    else np.nan
                ),
                "topscorer_scoring_pick_rate": (
                    float(phases["scoring_picks"].sum()) / float(phases["picks"].sum())
                    if not phases.empty and float(phases["picks"].sum()) > 0
                    else np.nan
                ),
                "match_total_points": float(matches["scorito_match_points"].sum()) if not matches.empty else 0.0,
                "match_exact_accuracy": float(matches["exact_correct"].mean()) if not matches.empty else np.nan,
                "match_toto_accuracy": float(matches["toto_correct"].mean()) if not matches.empty else np.nan,
                "matches_backtested": int(len(matches)),
                "average_position_known_rate": float(phases["position_known_rate"].mean()) if not phases.empty else 0.0,
            }
        ]
    )


def plot_backtest(output_dir: Path, tournaments: pd.DataFrame) -> Path | None:
    if tournaments.empty:
        return None
    path = output_dir / "scorito_historical_backtest_summary.png"
    display = tournaments.sort_values("tournament")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(display["tournament"], display["topscorer_points"], color="#4e79a7")
    axes[0].set_title("Top-scorer points by tournament")
    axes[0].set_xlabel("Scorito points")
    axes[1].barh(display["tournament"], display["match_toto_accuracy"], color="#59a14f")
    axes[1].set_title("Knockout toto accuracy by tournament")
    axes[1].set_xlabel("Toto accuracy")
    axes[1].set_xlim(0, 1)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.12, wspace=0.38)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_outputs(
    output_dir: Path,
    picks: pd.DataFrame,
    phases: pd.DataFrame,
    matches: pd.DataFrame,
    tournaments: pd.DataFrame,
    overall: pd.DataFrame,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "backtest_topscorer_picks.csv",
        output_dir / "backtest_topscorer_phase_summary.csv",
        output_dir / "backtest_match_picks.csv",
        output_dir / "backtest_tournament_summary.csv",
        output_dir / "backtest_overall_summary.csv",
    ]
    picks.to_csv(paths[0], index=False)
    phases.to_csv(paths[1], index=False)
    matches.to_csv(paths[2], index=False)
    tournaments.to_csv(paths[3], index=False)
    overall.to_csv(paths[4], index=False)
    figure = plot_backtest(output_dir, tournaments)
    if figure is not None:
        paths.append(figure)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest Scorito knockout advice.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_world_cup"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_scorito_backtest"))
    parser.add_argument(
        "--pool-size",
        choices=["small", "medium", "large"],
        default="medium",
    )
    parser.add_argument("--topscorers-per-phase", type=int, default=4)
    parser.add_argument(
        "--fetch-positions",
        action="store_true",
        help="Fetch missing Wikipedia player positions. Slower but closer to Scorito scoring.",
    )
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument(
        "--historical-source-limit",
        type=int,
        default=None,
        help="Debug option: only backtest first N configured historical sources.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refresh = not args.no_refresh
    all_matches, all_goals = load_historical_tournaments(
        args.cache_dir,
        refresh=refresh,
        max_sources=args.historical_source_limit,
    )
    if all_matches.empty or all_goals.empty:
        raise RuntimeError("No historical matches/goals were available for backtesting.")

    positions = load_or_fetch_positions(
        all_goals,
        args.cache_dir,
        refresh=refresh,
        fetch_positions=args.fetch_positions,
    )

    configured_sources = [TournamentSource(*source).name for source in TOURNAMENT_SOURCES]
    if args.historical_source_limit is not None:
        configured_sources = configured_sources[: args.historical_source_limit]
    tournaments = [name for name in configured_sources if name in set(all_matches["tournament"])]

    pick_frames: list[pd.DataFrame] = []
    phase_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    for tournament in tournaments:
        picks, phases, matches = backtest_tournament(
            tournament,
            all_matches,
            all_goals,
            positions,
            picks_per_phase=args.topscorers_per_phase,
            pool_size=args.pool_size,
        )
        if not picks.empty:
            pick_frames.append(picks)
        if not phases.empty:
            phase_frames.append(phases)
        if not matches.empty:
            match_frames.append(matches)

    picks_df = pd.concat(pick_frames, ignore_index=True) if pick_frames else pd.DataFrame()
    phases_df = pd.concat(phase_frames, ignore_index=True) if phase_frames else pd.DataFrame()
    matches_df = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    tournament_df = tournament_summary(picks_df, phases_df, matches_df)
    overall_df = overall_summary(tournament_df, phases_df, matches_df)
    output_paths = write_outputs(
        args.output_dir,
        picks_df,
        phases_df,
        matches_df,
        tournament_df,
        overall_df,
    )

    overall = overall_df.iloc[0]
    print("Scorito historical backtest complete")
    print(f"Tournaments backtested: {int(overall.tournaments_backtested)}")
    print(
        "Top-scorer efficiency vs candidate oracle: "
        f"{overall.topscorer_candidate_efficiency:.1%}"
    )
    print(
        "Top-scorer efficiency vs all-scorer oracle: "
        f"{overall.topscorer_overall_efficiency:.1%}"
    )
    print(f"Top-scorer scoring pick rate: {overall.topscorer_scoring_pick_rate:.1%}")
    print(f"Knockout match exact accuracy: {overall.match_exact_accuracy:.1%}")
    print(f"Knockout match toto accuracy: {overall.match_toto_accuracy:.1%}")
    print(f"Average known-position rate: {overall.average_position_known_rate:.1%}")
    for path in output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
