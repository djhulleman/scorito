#!/usr/bin/env python3
"""
Pick Scorito-style knockout top-scorer candidates.

The model is intentionally practical:
  * current group-phase goals come from the 2026 World Cup match boxes
  * historical carryover priors come from previous World Cup and Euro pages
  * projected knockout opportunity comes from world_cup_exact_score_forecaster.py
  * player position is inferred from Wikipedia player pages, with a CSV override hook

It ranks the five players with the best simulated chance of finishing in the
top five by Scorito knockout goal points. Because Scorito gives defenders and
keepers more points per goal, this is not the same as selecting the five most
likely raw goal scorers.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from world_cup_score_forecaster import (
    CURRENT_WORLD_CUP_URL,
    clean_text,
    fetch_bytes,
    normalize_team,
)


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


WIKI_BASE_URL = "https://en.wikipedia.org"

TOURNAMENT_SOURCES = [
    ("FIFA World Cup", 2010, "https://en.wikipedia.org/wiki/2010_FIFA_World_Cup"),
    ("FIFA World Cup", 2014, "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup"),
    ("FIFA World Cup", 2018, "https://en.wikipedia.org/wiki/2018_FIFA_World_Cup"),
    ("FIFA World Cup", 2022, "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup"),
    ("UEFA Euro", 2012, "https://en.wikipedia.org/wiki/UEFA_Euro_2012"),
    ("UEFA Euro", 2016, "https://en.wikipedia.org/wiki/UEFA_Euro_2016"),
    ("UEFA Euro", 2020, "https://en.wikipedia.org/wiki/UEFA_Euro_2020"),
    ("UEFA Euro", 2024, "https://en.wikipedia.org/wiki/UEFA_Euro_2024"),
]

SCORITO_POINTS = {
    "Keeper / Verdediger": {
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

POSITION_SHARE_PRIOR = {
    "Keeper / Verdediger": 0.07,
    "Middenvelder": 0.17,
    "Aanvaller": 0.28,
}

POSITION_SHARE_CAP = {
    "Keeper / Verdediger": 0.18,
    "Middenvelder": 0.38,
    "Aanvaller": 0.62,
}

ROUND_ORDER = {
    "Round of 32": 1,
    "Round of 16": 2,
    "Quarterfinals": 3,
    "Quarter-finals": 3,
    "Semifinals": 4,
    "Semi-finals": 4,
    "Match for third place": 5,
    "Final": 6,
}

CURRENT_GROUP_LETTERS = "ABCDEFGHIJKL"
EXPECTED_2026_GROUP_MATCHES = 72
OPPONENT_DEFENSE_FACTOR_MIN = 0.90
OPPONENT_DEFENSE_FACTOR_MAX = 1.10


@dataclass(frozen=True)
class TournamentSource:
    competition: str
    year: int
    url: str

    @property
    def name(self) -> str:
        return f"{self.competition} {self.year}"


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def cache_path_for_url(cache_dir: Path, url: str) -> Path:
    if url.rstrip("/") == CURRENT_WORLD_CUP_URL.rstrip("/"):
        return cache_dir / "2026_fifa_world_cup.html"
    parsed = urllib.parse.urlparse(url)
    name = safe_filename(parsed.path.strip("/") or parsed.netloc)
    return cache_dir / "topscorer_sources" / f"{name}.html"


def fetch_html(url: str, cache_dir: Path, refresh: bool) -> bytes:
    return fetch_bytes(url, cache_path_for_url(cache_dir, url), refresh=refresh)


def historical_urls_for_source(source: TournamentSource) -> list[tuple[str, str]]:
    urls = [("main", source.url)]
    if source.competition == "FIFA World Cup":
        for letter in "ABCDEFGH":
            urls.append(
                (
                    f"group_{letter}",
                    f"https://en.wikipedia.org/wiki/{source.year}_FIFA_World_Cup_Group_{letter}",
                )
            )
    elif source.competition == "UEFA Euro":
        group_count = 6 if source.year >= 2016 else 4
        for letter in "ABCDEF"[:group_count]:
            urls.append(
                (
                    f"group_{letter}",
                    f"https://en.wikipedia.org/wiki/UEFA_Euro_{source.year}_Group_{letter}",
                )
            )
    return urls


def current_2026_group_urls() -> list[tuple[str, str]]:
    return [
        (
            f"group_{letter}",
            f"https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{letter}",
        )
        for letter in CURRENT_GROUP_LETTERS
    ]


def wiki_url(path_or_url: object) -> str:
    text = str(path_or_url or "")
    if text.startswith("http"):
        return text
    if text.startswith("//"):
        return f"https:{text}"
    if text.startswith("/"):
        return f"{WIKI_BASE_URL}{text}"
    return text


def team_from_cell(cell) -> str:
    tag = cell.find(attrs={"itemprop": "name"}) or cell
    return normalize_team(tag.get_text(" ", strip=True))


def previous_round_heading(table) -> str:
    heading = table.find_previous(["h3", "h2"])
    if heading is None:
        return ""
    text = clean_text(heading.get_text(" ", strip=True))
    return re.sub(r"\[.*?\]", "", text).strip()


def classify_stage(round_name: str) -> str:
    text = clean_text(round_name)
    if text.startswith("Group ") or text == "Group stage":
        return "group"
    if any(
        token in text
        for token in (
            "Round of",
            "Quarter",
            "Semi",
            "Final",
            "third place",
            "Third place",
        )
    ):
        return "knockout"
    return "other"


def parse_score(score_text: str) -> tuple[float, float]:
    match = re.search(r"([0-9]+)\s*[–-]\s*([0-9]+)", score_text)
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def parse_match_date(table) -> pd.Timestamp:
    box = table.find_parent("div", class_="footballbox")
    date_tag = None
    if box is not None:
        date_tag = box.find(class_="bday") or box.find(class_="dtstart")
    if date_tag is None:
        date_block = table.find_previous_sibling("div", class_="fleft")
        if date_block is not None:
            date_tag = date_block.find(class_="bday") or date_block.find(class_="dtstart")
    return pd.to_datetime(date_tag.get_text(strip=True), errors="coerce") if date_tag else pd.NaT


def first_player_link(goal_item) -> tuple[str, str]:
    for link in goal_item.find_all("a", href=True):
        href = str(link.get("href"))
        if href.startswith("/wiki/"):
            player_name = clean_text(link.get("title") or link.get_text(" ", strip=True))
            player_name = re.sub(r"\s+\([^)]*\)$", "", player_name).strip()
            return player_name, wiki_url(href)
    return clean_text(goal_item.get_text(" ", strip=True)), ""


def is_own_goal_text(text: str) -> bool:
    lowered = text.lower()
    return "o.g." in lowered or "own goal" in lowered


def goal_count_from_item(goal_item, text: str) -> int:
    icons = goal_item.find_all("span", class_="fb-goal")
    minute_marks = re.findall(r"\b[0-9]+(?:\+[0-9]+)?'", text)
    return max(1, len(icons), len(minute_marks))


def parse_goal_items(
    table,
    side_class: str,
    team: str,
    match_key: str,
    tournament: str,
    competition: str,
    year: int,
    round_name: str,
    stage_type: str,
    date: pd.Timestamp,
) -> list[dict[str, object]]:
    cell = table.find("td", class_=side_class)
    if cell is None:
        return []

    rows: list[dict[str, object]] = []
    for item in cell.find_all("li"):
        text = clean_text(item.get_text(" ", strip=True))
        if not text or is_own_goal_text(text):
            continue
        player, player_url = first_player_link(item)
        if not player:
            continue
        goals = goal_count_from_item(item, text)
        for _ in range(goals):
            rows.append(
                {
                    "match_key": match_key,
                    "tournament": tournament,
                    "competition": competition,
                    "year": year,
                    "round": round_name,
                    "stage_type": stage_type,
                    "date": date,
                    "team": team,
                    "player": player,
                    "player_url": player_url,
                    "goal_text": text,
                }
            )
    return rows


def parse_tournament_page(
    html: bytes,
    source: TournamentSource,
    page_label: str = "main",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    matches: list[dict[str, object]] = []
    goals: list[dict[str, object]] = []

    for index, table in enumerate(soup.find_all("table", class_="fevent"), start=1):
        home_cell = table.find("th", class_="fhome")
        away_cell = table.find("th", class_="faway")
        score_cell = table.find("th", class_="fscore")
        if not (home_cell and away_cell and score_cell):
            continue

        home_team = team_from_cell(home_cell)
        away_team = team_from_cell(away_cell)
        score_text = clean_text(score_cell.get_text(" ", strip=True))
        home_score, away_score = parse_score(score_text)
        round_name = previous_round_heading(table)
        if page_label.startswith("group_"):
            group_letter = page_label.split("_", maxsplit=1)[1].upper()
            round_name = f"Group {group_letter}"
            stage_type = "group"
        else:
            stage_type = classify_stage(round_name)
        date = parse_match_date(table)
        match_key = f"{source.name}-{page_label}-{index}"

        matches.append(
            {
                "match_key": match_key,
                "tournament": source.name,
                "competition": source.competition,
                "year": source.year,
                "round": round_name,
                "stage_type": stage_type,
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
        goals.extend(
            parse_goal_items(
                table,
                "fhgoal",
                home_team,
                match_key,
                source.name,
                source.competition,
                source.year,
                round_name,
                stage_type,
                date,
            )
        )
        goals.extend(
            parse_goal_items(
                table,
                "fagoal",
                away_team,
                match_key,
                source.name,
                source.competition,
                source.year,
                round_name,
                stage_type,
                date,
            )
        )

    return pd.DataFrame(matches), pd.DataFrame(goals)


def load_historical_tournaments(
    cache_dir: Path,
    refresh: bool,
    max_sources: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_frames: list[pd.DataFrame] = []
    goal_frames: list[pd.DataFrame] = []
    sources = [TournamentSource(*source) for source in TOURNAMENT_SOURCES]
    if max_sources is not None:
        sources = sources[:max_sources]

    for source in sources:
        pages = historical_urls_for_source(source)
        main_label, main_url = pages[0]
        html = fetch_html(main_url, cache_dir, refresh=refresh)
        matches, goals = parse_tournament_page(html, source, page_label=main_label)
        match_frames.append(matches)
        goal_frames.append(goals)

        main_has_group_stage = not matches[matches["stage_type"].eq("group")].empty
        if main_has_group_stage:
            continue

        for page_label, url in pages[1:]:
            html = fetch_html(url, cache_dir, refresh=refresh)
            matches, goals = parse_tournament_page(html, source, page_label=page_label)
            match_frames.append(matches)
            goal_frames.append(goals)

    all_matches = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    all_goals = pd.concat(goal_frames, ignore_index=True) if goal_frames else pd.DataFrame()
    if not all_matches.empty:
        all_matches = all_matches.drop_duplicates(
            subset=[
                "tournament",
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ],
            keep="first",
        ).reset_index(drop=True)
    return all_matches, all_goals


def bucket_group_goals(goals: float) -> str:
    if goals >= 4:
        return "4+"
    if goals >= 3:
        return "3"
    if goals >= 2:
        return "2"
    return "1"


def historical_carryover_patterns(goals: pd.DataFrame) -> pd.DataFrame:
    if goals.empty:
        return fallback_carryover_patterns()

    group = (
        goals[goals["stage_type"].eq("group")]
        .groupby(["tournament", "player", "team"], as_index=False)
        .size()
        .rename(columns={"size": "group_goals"})
    )
    knockout = (
        goals[goals["stage_type"].eq("knockout")]
        .groupby(["tournament", "player", "team"], as_index=False)
        .size()
        .rename(columns={"size": "knockout_goals"})
    )
    if group.empty:
        return fallback_carryover_patterns()

    players = group.merge(
        knockout,
        how="left",
        on=["tournament", "player", "team"],
    )
    players["knockout_goals"] = players["knockout_goals"].fillna(0)
    players["group_goal_bucket"] = players["group_goals"].map(bucket_group_goals)
    global_avg = max(float(players["knockout_goals"].mean()), 0.05)

    rows = []
    for bucket, bucket_df in players.groupby("group_goal_bucket", sort=False):
        n = len(bucket_df)
        smoothed_avg = (
            float(bucket_df["knockout_goals"].sum()) + global_avg * 10.0
        ) / (n + 10.0)
        rows.append(
            {
                "group_goal_bucket": bucket,
                "historical_players": n,
                "avg_knockout_goals": float(bucket_df["knockout_goals"].mean()),
                "smoothed_avg_knockout_goals": smoothed_avg,
                "knockout_goal_hit_rate": float((bucket_df["knockout_goals"] > 0).mean()),
                "carryover_boost": float(np.clip(smoothed_avg / global_avg, 0.65, 1.65)),
            }
        )

    patterns = pd.DataFrame(rows)
    needed = {"1", "2", "3", "4+"}
    existing = set(patterns["group_goal_bucket"])
    for bucket in sorted(needed - existing):
        patterns = pd.concat(
            [
                patterns,
                pd.DataFrame(
                    [
                        {
                            "group_goal_bucket": bucket,
                            "historical_players": 0,
                            "avg_knockout_goals": global_avg,
                            "smoothed_avg_knockout_goals": global_avg,
                            "knockout_goal_hit_rate": 0.0,
                            "carryover_boost": 1.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return patterns.sort_values("group_goal_bucket").reset_index(drop=True)


def fallback_carryover_patterns() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_goal_bucket": "1",
                "historical_players": 0,
                "avg_knockout_goals": 0.20,
                "smoothed_avg_knockout_goals": 0.20,
                "knockout_goal_hit_rate": 0.18,
                "carryover_boost": 0.90,
            },
            {
                "group_goal_bucket": "2",
                "historical_players": 0,
                "avg_knockout_goals": 0.35,
                "smoothed_avg_knockout_goals": 0.35,
                "knockout_goal_hit_rate": 0.27,
                "carryover_boost": 1.10,
            },
            {
                "group_goal_bucket": "3",
                "historical_players": 0,
                "avg_knockout_goals": 0.48,
                "smoothed_avg_knockout_goals": 0.48,
                "knockout_goal_hit_rate": 0.34,
                "carryover_boost": 1.25,
            },
            {
                "group_goal_bucket": "4+",
                "historical_players": 0,
                "avg_knockout_goals": 0.60,
                "smoothed_avg_knockout_goals": 0.60,
                "knockout_goal_hit_rate": 0.42,
                "carryover_boost": 1.40,
            },
        ]
    )


def team_goal_rows(matches: pd.DataFrame, stage_type: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    played = matches[
        matches["stage_type"].eq(stage_type)
        & matches["home_score"].notna()
        & matches["away_score"].notna()
    ].copy()
    for row in played.itertuples(index=False):
        rows.append(
            {
                "tournament": row.tournament,
                "team": row.home_team,
                "goals_for": float(row.home_score),
                "goals_against": float(row.away_score),
            }
        )
        rows.append(
            {
                "tournament": row.tournament,
                "team": row.away_team,
                "goals_for": float(row.away_score),
                "goals_against": float(row.home_score),
            }
        )
    return pd.DataFrame(rows)


def historical_team_patterns(matches: pd.DataFrame) -> pd.DataFrame:
    group_rows = team_goal_rows(matches, "group")
    knockout_rows = team_goal_rows(matches, "knockout")
    if group_rows.empty or knockout_rows.empty:
        return pd.DataFrame()

    group = group_rows.groupby(["tournament", "team"], as_index=False).agg(
        group_goals=("goals_for", "sum"),
        group_matches=("goals_for", "size"),
    )
    knockout = knockout_rows.groupby(["tournament", "team"], as_index=False).agg(
        knockout_goals=("goals_for", "sum"),
        knockout_matches=("goals_for", "size"),
    )
    teams = group.merge(knockout, how="inner", on=["tournament", "team"])
    teams["group_goals_per_match"] = teams["group_goals"] / teams["group_matches"]
    teams["knockout_goals_per_match"] = teams["knockout_goals"] / teams["knockout_matches"]
    return teams.sort_values(
        ["tournament", "knockout_goals_per_match"],
        ascending=[True, False],
    )


def normalize_scorito_position(raw_position: object) -> str:
    text = clean_text(raw_position).lower()
    if not text:
        return "Aanvaller"
    if "goalkeeper" in text or "defender" in text or "centre-back" in text or "center-back" in text:
        return "Keeper / Verdediger"
    if "full-back" in text or "wing-back" in text or re.search(r"\bback\b", text):
        return "Keeper / Verdediger"
    if "forward" in text or "striker" in text or "winger" in text:
        return "Aanvaller"
    if "midfielder" in text or "midfield" in text:
        return "Middenvelder"
    return "Aanvaller"


def parse_position_from_player_page(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table", class_="infobox"):
        for row in table.find_all("tr"):
            header = row.find("th")
            value = row.find("td")
            if not header or not value:
                continue
            header_text = clean_text(header.get_text(" ", strip=True)).lower()
            if "position" in header_text:
                return clean_text(value.get_text(" ", strip=True))
    return ""


def load_position_overrides(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    overrides = pd.read_csv(path).fillna("")
    result: dict[str, str] = {}
    for row in overrides.itertuples(index=False):
        player = clean_text(getattr(row, "player", ""))
        position = clean_text(getattr(row, "scorito_position", ""))
        if player and position:
            result[player] = position
    return result


def load_player_positions(
    players: pd.DataFrame,
    cache_dir: Path,
    refresh: bool,
    overrides_path: Path | None,
) -> pd.DataFrame:
    position_cache_path = cache_dir / "topscorer_player_positions.csv"
    if position_cache_path.exists():
        cached = pd.read_csv(position_cache_path).fillna("")
    else:
        cached = pd.DataFrame(columns=["player", "player_url", "raw_position", "scorito_position"])

    cached_lookup = {
        (str(row.player), str(row.player_url)): {
            "raw_position": str(row.raw_position),
            "scorito_position": str(row.scorito_position),
        }
        for row in cached.itertuples(index=False)
    }
    overrides = load_position_overrides(overrides_path)
    rows: list[dict[str, object]] = []

    unique_players = players[["player", "player_url"]].drop_duplicates()
    for row in unique_players.itertuples(index=False):
        player = clean_text(row.player)
        player_url = wiki_url(row.player_url)
        cached_row = cached_lookup.get((player, player_url), {})
        raw_position = cached_row.get("raw_position", "")
        scorito_position = cached_row.get("scorito_position", "")

        if player in overrides:
            scorito_position = overrides[player]
        elif (refresh or not raw_position) and player_url:
            try:
                html = fetch_html(player_url, cache_dir, refresh=refresh)
                raw_position = parse_position_from_player_page(html)
                scorito_position = normalize_scorito_position(raw_position)
            except Exception:
                scorito_position = scorito_position or normalize_scorito_position(raw_position)
        else:
            scorito_position = scorito_position or normalize_scorito_position(raw_position)

        rows.append(
            {
                "player": player,
                "player_url": player_url,
                "raw_position": raw_position,
                "scorito_position": scorito_position,
            }
        )

    positions = pd.DataFrame(rows)
    position_cache_path.parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(position_cache_path, index=False)
    return positions


def ensure_exact_score_outputs(
    exact_score_dir: Path,
    cache_dir: Path,
    refresh: bool,
    force_run: bool,
) -> None:
    forecast_path = exact_score_dir / "exact_score_knockout_forecast.csv"
    standings_path = exact_score_dir / "exact_score_projected_group_standings.csv"
    if forecast_path.exists() and standings_path.exists() and not force_run:
        return

    command = [
        sys.executable,
        "world_cup_exact_score_forecaster.py",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(exact_score_dir),
    ]
    if not refresh:
        command.append("--no-refresh")
    subprocess.run(command, check=True)


def load_current_2026_data(cache_dir: Path, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    html = fetch_html(CURRENT_WORLD_CUP_URL, cache_dir, refresh=refresh)
    source = TournamentSource("FIFA World Cup", 2026, CURRENT_WORLD_CUP_URL)
    main_matches, main_goals = parse_tournament_page(html, source)
    group_match_count = int(main_matches["stage_type"].eq("group").sum()) if not main_matches.empty else 0
    if group_match_count >= EXPECTED_2026_GROUP_MATCHES:
        return main_matches, main_goals

    group_match_frames: list[pd.DataFrame] = []
    group_goal_frames: list[pd.DataFrame] = []
    for page_label, url in current_2026_group_urls():
        try:
            group_html = fetch_html(url, cache_dir, refresh=refresh)
        except Exception as exc:
            print(f"Warning: failed to load {url}; skipping group subpage. Reason: {exc}", file=sys.stderr)
            continue
        matches, goals = parse_tournament_page(group_html, source, page_label=page_label)
        if not matches.empty:
            group_match_frames.append(matches)
        if not goals.empty:
            group_goal_frames.append(goals)

    if not group_match_frames:
        return main_matches, main_goals

    group_matches = pd.concat(group_match_frames, ignore_index=True)
    group_goals = pd.concat(group_goal_frames, ignore_index=True) if group_goal_frames else pd.DataFrame()
    non_group_matches = (
        main_matches[~main_matches["stage_type"].eq("group")].copy()
        if not main_matches.empty
        else pd.DataFrame()
    )
    non_group_goals = (
        main_goals[~main_goals["stage_type"].eq("group")].copy()
        if not main_goals.empty
        else pd.DataFrame()
    )
    all_matches = pd.concat([non_group_matches, group_matches], ignore_index=True)
    all_goals = pd.concat([non_group_goals, group_goals], ignore_index=True)
    return all_matches, all_goals


def current_group_player_table(
    current_goals: pd.DataFrame,
    current_matches: pd.DataFrame,
) -> pd.DataFrame:
    group_goals = current_goals[current_goals["stage_type"].eq("group")].copy()
    if group_goals.empty:
        return pd.DataFrame()

    players = group_goals.groupby(["player", "player_url", "team"], as_index=False).agg(
        group_goals=("player", "size"),
        group_goal_matches=("match_key", "nunique"),
        latest_group_goal_date=("date", "max"),
    )

    team_rows = team_goal_rows(current_matches, "group")
    team_group = team_rows.groupby("team", as_index=False).agg(
        team_completed_group_goals=("goals_for", "sum"),
        team_completed_group_matches=("goals_for", "size"),
    )
    return players.merge(team_group, how="left", on="team").fillna(
        {"team_completed_group_goals": 0, "team_completed_group_matches": 0}
    )


def current_group_defensive_profiles(current_matches: pd.DataFrame) -> pd.DataFrame:
    played_group = current_matches[
        current_matches["stage_type"].eq("group")
        & current_matches["home_score"].notna()
        & current_matches["away_score"].notna()
    ].copy()
    if played_group.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "group_goals_against",
                "group_defensive_matches",
                "group_goals_against_per_match",
                "group_clean_sheet_rate",
                "opponent_defense_factor",
            ]
        )

    rows: list[dict[str, object]] = []
    for row in played_group.itertuples(index=False):
        rows.append(
            {
                "team": row.home_team,
                "goals_against": float(row.away_score),
                "clean_sheet": 1.0 if float(row.away_score) == 0.0 else 0.0,
            }
        )
        rows.append(
            {
                "team": row.away_team,
                "goals_against": float(row.home_score),
                "clean_sheet": 1.0 if float(row.home_score) == 0.0 else 0.0,
            }
        )

    team_rows = pd.DataFrame(rows)
    avg_goals_against = float(team_rows["goals_against"].mean())
    avg_clean_sheet_rate = float(team_rows["clean_sheet"].mean())
    profiles = team_rows.groupby("team", as_index=False).agg(
        group_goals_against=("goals_against", "sum"),
        group_defensive_matches=("goals_against", "size"),
        group_clean_sheets=("clean_sheet", "sum"),
    )
    profiles["group_goals_against_per_match"] = (
        profiles["group_goals_against"] / profiles["group_defensive_matches"]
    )
    profiles["group_clean_sheet_rate"] = (
        profiles["group_clean_sheets"] / profiles["group_defensive_matches"]
    )

    shrink = profiles["group_defensive_matches"] / (profiles["group_defensive_matches"] + 2.0)
    adjusted_goals_against = avg_goals_against + shrink * (
        profiles["group_goals_against_per_match"] - avg_goals_against
    )
    adjusted_clean_sheet_rate = avg_clean_sheet_rate + shrink * (
        profiles["group_clean_sheet_rate"] - avg_clean_sheet_rate
    )
    profiles["opponent_defense_factor"] = (
        1.0
        + 0.08 * (adjusted_goals_against - avg_goals_against)
        - 0.05 * (adjusted_clean_sheet_rate - avg_clean_sheet_rate)
    ).clip(OPPONENT_DEFENSE_FACTOR_MIN, OPPONENT_DEFENSE_FACTOR_MAX)
    return profiles


def first_knockout_round_name(knockout_predictions: pd.DataFrame) -> str:
    if knockout_predictions.empty:
        return ""
    order = knockout_predictions["round"].map(lambda value: ROUND_ORDER.get(str(value), 99))
    return str(knockout_predictions.loc[order.idxmin(), "round"])


def team_knockout_path(knockout_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in knockout_predictions.itertuples(index=False):
        round_name = str(row.round)
        if not clean_text(row.home_team) or not clean_text(row.away_team):
            continue
        home_xg = float(row.home_xg) if not pd.isna(row.home_xg) else 0.0
        away_xg = float(row.away_xg) if not pd.isna(row.away_xg) else 0.0
        elo_diff = float(row.elo_diff) if not pd.isna(row.elo_diff) else 0.0
        rows.append(
            {
                "team": normalize_team(row.home_team),
                "round": round_name,
                "round_order": ROUND_ORDER.get(round_name, 99),
                "opponent": normalize_team(row.away_team),
                "team_xg": home_xg,
                "opponent_xg": away_xg,
                "team_elo_diff": elo_diff,
                "advances": normalize_team(row.advancing_team) == normalize_team(row.home_team),
            }
        )
        rows.append(
            {
                "team": normalize_team(row.away_team),
                "round": round_name,
                "round_order": ROUND_ORDER.get(round_name, 99),
                "opponent": normalize_team(row.home_team),
                "team_xg": away_xg,
                "opponent_xg": home_xg,
                "team_elo_diff": -elo_diff,
                "advances": normalize_team(row.advancing_team) == normalize_team(row.away_team),
            }
        )
    return pd.DataFrame(rows)


def scorito_points_for_round(position: str, round_name: str) -> int:
    return int(SCORITO_POINTS.get(position, SCORITO_POINTS["Aanvaller"]).get(round_name, 24))


def build_candidate_round_rows(
    players: pd.DataFrame,
    positions: pd.DataFrame,
    path: pd.DataFrame,
    carryover_patterns: pd.DataFrame,
    team_defense_profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if players.empty or path.empty:
        return pd.DataFrame()

    first_round = first_knockout_round_name_from_path(path)
    first_round_teams = set(path[path["round"].eq(first_round)]["team"]) if first_round else set(path["team"])
    candidates = players[players["team"].isin(first_round_teams)].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["group_goal_bucket"] = candidates["group_goals"].map(bucket_group_goals)
    boost_lookup = carryover_patterns.set_index("group_goal_bucket")["carryover_boost"].to_dict()
    candidates["carryover_boost"] = candidates["group_goal_bucket"].map(boost_lookup).fillna(1.0)
    candidates = candidates.merge(positions, how="left", on=["player", "player_url"])
    candidates["scorito_position"] = candidates["scorito_position"].fillna("Aanvaller")
    candidates["raw_position"] = candidates["raw_position"].fillna("")
    candidates["position_share_prior"] = candidates["scorito_position"].map(POSITION_SHARE_PRIOR).fillna(
        POSITION_SHARE_PRIOR["Aanvaller"]
    )
    candidates["position_share_cap"] = candidates["scorito_position"].map(POSITION_SHARE_CAP).fillna(
        POSITION_SHARE_CAP["Aanvaller"]
    )
    prior_strength = 2.0
    candidates["player_goal_share_uncapped"] = (
        candidates["group_goals"] + prior_strength * candidates["position_share_prior"]
    ) / (candidates["team_completed_group_goals"] + prior_strength)
    candidates["player_goal_share"] = candidates["player_goal_share_uncapped"].clip(0.015)
    candidates["player_goal_share"] = np.minimum(
        candidates["player_goal_share"],
        candidates["position_share_cap"],
    )
    candidates["multi_match_scorer_boost"] = (
        1.0 + 0.10 * (candidates["group_goal_matches"].clip(lower=1) - 1)
    ).clip(1.0, 1.25)

    if team_defense_profiles is None or team_defense_profiles.empty:
        defense_lookup: dict[str, dict[str, object]] = {}
        avg_goals_against_per_match = 0.0
        avg_clean_sheet_rate = 0.0
    else:
        defense = team_defense_profiles.copy()
        for column in [
            "group_goals_against",
            "group_defensive_matches",
            "group_goals_against_per_match",
            "group_clean_sheet_rate",
            "opponent_defense_factor",
        ]:
            defense[column] = pd.to_numeric(defense[column], errors="coerce")
        defensive_matches = defense["group_defensive_matches"].fillna(0.0)
        if float(defensive_matches.sum()) > 0:
            avg_goals_against_per_match = float(
                defense["group_goals_against"].fillna(0.0).sum() / defensive_matches.sum()
            )
            avg_clean_sheet_rate = float(
                (
                    defense["group_clean_sheet_rate"].fillna(0.0)
                    * defensive_matches
                ).sum()
                / defensive_matches.sum()
            )
        else:
            avg_goals_against_per_match = 0.0
            avg_clean_sheet_rate = 0.0
        defense_lookup = defense.set_index("team").to_dict("index")

    default_defense = {
        "group_goals_against_per_match": avg_goals_against_per_match,
        "group_clean_sheet_rate": avg_clean_sheet_rate,
        "opponent_defense_factor": 1.0,
    }

    rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        team_path = path[path["team"].eq(candidate.team)].sort_values("round_order")
        for path_row in team_path.itertuples(index=False):
            raw_team_xg = float(path_row.team_xg)
            team_elo_diff = float(path_row.team_elo_diff)
            opponent_profile = defense_lookup.get(str(path_row.opponent), default_defense)
            opponent_goals_against_per_match = float(
                opponent_profile.get("group_goals_against_per_match", avg_goals_against_per_match)
            )
            opponent_clean_sheet_rate = float(
                opponent_profile.get("group_clean_sheet_rate", avg_clean_sheet_rate)
            )
            opponent_defense_factor = float(opponent_profile.get("opponent_defense_factor", 1.0))
            adjusted_team_xg = raw_team_xg * opponent_defense_factor
            attacking_opportunity_score = (
                adjusted_team_xg
                + 0.0015 * team_elo_diff
                + 0.10 * (opponent_goals_against_per_match - avg_goals_against_per_match)
                - 0.05 * (opponent_clean_sheet_rate - avg_clean_sheet_rate)
            )
            expected_goals = (
                adjusted_team_xg
                * float(candidate.player_goal_share)
                * float(candidate.carryover_boost)
                * float(candidate.multi_match_scorer_boost)
            )
            points_per_goal = scorito_points_for_round(
                str(candidate.scorito_position),
                str(path_row.round),
            )
            rows.append(
                {
                    "player": candidate.player,
                    "player_url": candidate.player_url,
                    "team": candidate.team,
                    "raw_position": candidate.raw_position,
                    "scorito_position": candidate.scorito_position,
                    "round": path_row.round,
                    "round_order": path_row.round_order,
                    "opponent": path_row.opponent,
                    "team_xg": raw_team_xg,
                    "adjusted_team_xg": adjusted_team_xg,
                    "opponent_xg": path_row.opponent_xg,
                    "team_elo_diff": team_elo_diff,
                    "opponent_goals_against_per_match": opponent_goals_against_per_match,
                    "opponent_clean_sheet_rate": opponent_clean_sheet_rate,
                    "opponent_defense_factor": opponent_defense_factor,
                    "attacking_opportunity_score": attacking_opportunity_score,
                    "group_goals": candidate.group_goals,
                    "group_goal_matches": candidate.group_goal_matches,
                    "team_completed_group_goals": candidate.team_completed_group_goals,
                    "player_goal_share_uncapped": candidate.player_goal_share_uncapped,
                    "position_share_cap": candidate.position_share_cap,
                    "player_goal_share": candidate.player_goal_share,
                    "carryover_boost": candidate.carryover_boost,
                    "multi_match_scorer_boost": candidate.multi_match_scorer_boost,
                    "expected_round_goals": expected_goals,
                    "points_per_goal": points_per_goal,
                    "expected_round_points": expected_goals * points_per_goal,
                }
            )
    return pd.DataFrame(rows)


def first_knockout_round_name_from_path(path: pd.DataFrame) -> str:
    if path.empty:
        return ""
    ordered = path.sort_values("round_order")
    return str(ordered.iloc[0]["round"])


def simulate_top5_probabilities(
    round_rows: pd.DataFrame,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    if round_rows.empty:
        return pd.DataFrame()

    player_keys = (
        round_rows[["player", "player_url", "team"]]
        .drop_duplicates()
        .sort_values(["team", "player"])
        .reset_index(drop=True)
    )
    round_keys = sorted(round_rows["round"].unique(), key=lambda value: ROUND_ORDER.get(str(value), 99))
    player_index = {
        (row.player, row.player_url, row.team): idx
        for idx, row in enumerate(player_keys.itertuples(index=False))
    }
    round_index = {round_name: idx for idx, round_name in enumerate(round_keys)}

    means = np.zeros((len(player_keys), len(round_keys)), dtype=float)
    points = np.zeros((len(player_keys), len(round_keys)), dtype=float)
    for row in round_rows.itertuples(index=False):
        pidx = player_index[(row.player, row.player_url, row.team)]
        ridx = round_index[row.round]
        means[pidx, ridx] = float(row.expected_round_goals)
        points[pidx, ridx] = float(row.points_per_goal)

    rng = np.random.default_rng(seed)
    top5_points_counts = np.zeros(len(player_keys), dtype=int)
    top5_goal_counts = np.zeros(len(player_keys), dtype=int)
    expected_points = (means * points).sum(axis=1)
    expected_goals = means.sum(axis=1)

    batch_size = min(5000, max(1000, simulations))
    remaining = simulations
    while remaining > 0:
        batch = min(batch_size, remaining)
        sampled_goals = rng.poisson(lam=means, size=(batch, len(player_keys), len(round_keys)))
        sampled_points = (sampled_goals * points).sum(axis=2)
        sampled_raw_goals = sampled_goals.sum(axis=2)
        top_points = np.argpartition(-sampled_points, kth=min(4, len(player_keys) - 1), axis=1)[
            :, : min(5, len(player_keys))
        ]
        top_goals = np.argpartition(-sampled_raw_goals, kth=min(4, len(player_keys) - 1), axis=1)[
            :, : min(5, len(player_keys))
        ]
        for row in top_points:
            top5_points_counts[row] += 1
        for row in top_goals:
            top5_goal_counts[row] += 1
        remaining -= batch

    result = player_keys.copy()
    result["expected_scorito_points"] = expected_points
    result["expected_knockout_goals"] = expected_goals
    result["top5_scorito_probability"] = top5_points_counts / simulations
    result["top5_raw_goal_probability"] = top5_goal_counts / simulations
    return result


def summarize_candidates(round_rows: pd.DataFrame, simulation: pd.DataFrame) -> pd.DataFrame:
    if round_rows.empty or simulation.empty:
        return pd.DataFrame()

    summary = round_rows.groupby(["player", "player_url", "team"], as_index=False).agg(
        scorito_position=("scorito_position", "first"),
        raw_position=("raw_position", "first"),
        group_goals=("group_goals", "first"),
        group_goal_matches=("group_goal_matches", "first"),
        team_completed_group_goals=("team_completed_group_goals", "first"),
        player_goal_share_uncapped=("player_goal_share_uncapped", "first"),
        position_share_cap=("position_share_cap", "first"),
        player_goal_share=("player_goal_share", "first"),
        carryover_boost=("carryover_boost", "first"),
        projected_knockout_matches=("round", "nunique"),
        path_team_xg=("team_xg", "sum"),
        path_adjusted_team_xg=("adjusted_team_xg", "sum"),
        avg_path_ease_xg=("adjusted_team_xg", lambda values: float(values.mean())),
        avg_path_elo_diff=("team_elo_diff", "mean"),
        avg_opponent_goals_against_per_match=("opponent_goals_against_per_match", "mean"),
        avg_opponent_clean_sheet_rate=("opponent_clean_sheet_rate", "mean"),
        avg_opponent_defense_factor=("opponent_defense_factor", "mean"),
        path_ease_score=("attacking_opportunity_score", "mean"),
        expected_scorito_points=("expected_round_points", "sum"),
        expected_knockout_goals=("expected_round_goals", "sum"),
    )
    opponents = round_rows.groupby(["player", "player_url", "team"])["opponent"].apply(
        lambda values: " -> ".join(values.astype(str).tolist())
    )
    max_round = round_rows.sort_values("round_order").groupby(["player", "player_url", "team"])[
        "round"
    ].last()
    summary = summary.merge(
        opponents.rename("projected_opponent_path").reset_index(),
        how="left",
        on=["player", "player_url", "team"],
    )
    summary = summary.merge(
        max_round.rename("projected_last_round").reset_index(),
        how="left",
        on=["player", "player_url", "team"],
    )
    summary = summary.drop(columns=["expected_scorito_points", "expected_knockout_goals"]).merge(
        simulation,
        how="left",
        on=["player", "player_url", "team"],
    )
    summary["_top5_probability_sort"] = summary["top5_scorito_probability"].round(3)
    summary["_expected_points_sort"] = summary["expected_scorito_points"].round(3)
    summary["_expected_goals_sort"] = summary["expected_knockout_goals"].round(3)
    summary = summary.sort_values(
        [
            "_top5_probability_sort",
            "_expected_points_sort",
            "_expected_goals_sort",
            "path_ease_score",
            "avg_opponent_defense_factor",
            "path_adjusted_team_xg",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    summary = summary.drop(
        columns=["_top5_probability_sort", "_expected_points_sort", "_expected_goals_sort"]
    )
    summary["recommended_rank"] = np.arange(1, len(summary) + 1)
    summary["selection_reason"] = summary.apply(selection_reason, axis=1)
    return summary


def selection_reason(row: pd.Series) -> str:
    path_xg = row.get("path_adjusted_team_xg", row.get("path_team_xg", 0.0))
    return (
        f"{int(row['group_goals'])} group goals, "
        f"{row['scorito_position']}, "
        f"{int(row['projected_knockout_matches'])} projected knockout matches, "
        f"{path_xg:.1f} adjusted team xG path"
    )


def plot_recommendations(output_dir: Path, recommendations: pd.DataFrame) -> Path | None:
    if recommendations.empty:
        return None

    path = output_dir / "scorito_topscorer_recommendations.png"
    top = recommendations.head(15).copy().iloc[::-1]
    labels = top.apply(lambda row: f"{row['player']} ({row['team']})", axis=1)
    colors = top["scorito_position"].map(
        {
            "Keeper / Verdediger": "#4e79a7",
            "Middenvelder": "#59a14f",
            "Aanvaller": "#f28e2b",
        }
    ).fillna("#bab0ac")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, top["expected_scorito_points"], color=colors)
    ax.set_xlabel("Expected Scorito knockout points from goals")
    ax.set_title("Top Scorito Knockout Top-Scorer Picks")
    for idx, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["expected_scorito_points"] + 0.5,
            idx,
            f"{row['top5_scorito_probability']:.1%}",
            va="center",
            fontsize=8,
        )
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.31, right=0.96, top=0.90, bottom=0.12)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_outputs(
    output_dir: Path,
    recommendations: pd.DataFrame,
    round_rows: pd.DataFrame,
    current_group_scorers: pd.DataFrame,
    current_team_defense: pd.DataFrame,
    historical_matches: pd.DataFrame,
    historical_goals: pd.DataFrame,
    carryover_patterns: pd.DataFrame,
    team_patterns: pd.DataFrame,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "scorito_topscorer_recommendations.csv",
        output_dir / "scorito_top_5_picks.csv",
        output_dir / "scorito_candidate_round_model.csv",
        output_dir / "current_group_scorers.csv",
        output_dir / "current_team_defense_profiles.csv",
        output_dir / "historical_carryover_patterns.csv",
        output_dir / "historical_team_patterns.csv",
        output_dir / "historical_goal_events.csv",
    ]
    recommendations.to_csv(paths[0], index=False)
    recommendations.head(5).to_csv(paths[1], index=False)
    round_rows.to_csv(paths[2], index=False)
    current_group_scorers.to_csv(paths[3], index=False)
    current_team_defense.to_csv(paths[4], index=False)
    carryover_patterns.to_csv(paths[5], index=False)
    team_patterns.to_csv(paths[6], index=False)
    historical_goals.to_csv(paths[7], index=False)
    historical_matches.to_csv(output_dir / "historical_match_results.csv", index=False)
    figure_path = plot_recommendations(output_dir, recommendations)
    if figure_path is not None:
        paths.append(figure_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick five Scorito knockout top-scorer candidates using group form and bracket path."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_world_cup"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_scorito_topscorers"))
    parser.add_argument(
        "--exact-score-dir",
        type=Path,
        default=Path("outputs_exact_score_model"),
        help="Directory containing exact_score_knockout_forecast.csv.",
    )
    parser.add_argument(
        "--position-overrides",
        type=Path,
        default=None,
        help="Optional CSV with columns player,scorito_position to override inferred positions.",
    )
    parser.add_argument(
        "--refresh-exact-score",
        action="store_true",
        help="Run world_cup_exact_score_forecaster.py before ranking players.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Use cached Wikipedia pages instead of refreshing source pages.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=20000,
        help="Monte Carlo simulations for top-five probabilities.",
    )
    parser.add_argument(
        "--historical-source-limit",
        type=int,
        default=None,
        help="Debug option: only use the first N historical tournament pages.",
    )
    return parser.parse_args()


def main() -> int:
    configure_output_encoding()
    args = parse_args()
    refresh = not args.no_refresh

    ensure_exact_score_outputs(
        args.exact_score_dir,
        args.cache_dir,
        refresh=refresh,
        force_run=args.refresh_exact_score,
    )
    knockout_path = args.exact_score_dir / "exact_score_knockout_forecast.csv"
    knockout_predictions = pd.read_csv(knockout_path).fillna("")
    path = team_knockout_path(knockout_predictions)

    current_matches, current_goals = load_current_2026_data(args.cache_dir, refresh=refresh)
    current_group_scorers = current_group_player_table(current_goals, current_matches)
    if current_group_scorers.empty:
        raise RuntimeError("No current group-stage scorers were parsed from the World Cup page.")
    current_team_defense = current_group_defensive_profiles(current_matches)

    historical_matches, historical_goals = load_historical_tournaments(
        args.cache_dir,
        refresh=refresh,
        max_sources=args.historical_source_limit,
    )
    carryover_patterns = historical_carryover_patterns(historical_goals)
    team_patterns = historical_team_patterns(historical_matches)

    knockout_teams = set(path["team"]) if not path.empty else set()
    position_candidates = current_group_scorers[current_group_scorers["team"].isin(knockout_teams)]
    positions = load_player_positions(
        position_candidates,
        args.cache_dir,
        refresh=refresh,
        overrides_path=args.position_overrides,
    )

    round_rows = build_candidate_round_rows(
        current_group_scorers,
        positions,
        path,
        carryover_patterns,
        current_team_defense,
    )
    if round_rows.empty:
        raise RuntimeError("No knockout candidate player-round rows could be built.")

    simulation = simulate_top5_probabilities(
        round_rows,
        simulations=max(args.simulations, 1000),
        seed=7,
    )
    recommendations = summarize_candidates(round_rows, simulation)
    output_paths = write_outputs(
        args.output_dir,
        recommendations,
        round_rows,
        current_group_scorers,
        current_team_defense,
        historical_matches,
        historical_goals,
        carryover_patterns,
        team_patterns,
    )

    top5 = recommendations.head(5)
    print("Scorito knockout top-scorer picker complete")
    print(
        "Current group matches parsed: "
        f"{int(current_matches['stage_type'].eq('group').sum())} "
        f"({int((current_matches['stage_type'].eq('group') & current_matches['home_score'].notna() & current_matches['away_score'].notna()).sum())} with scores)"
    )
    print(f"Current group scorers parsed: {len(current_group_scorers)}")
    print(f"Historical goal events parsed: {len(historical_goals)}")
    print(f"Candidate player-round rows: {len(round_rows)}")
    print("Recommended top 5:")
    for row in top5.itertuples(index=False):
        print(
            f"{int(row.recommended_rank)}. {row.player} ({row.team}, {row.scorito_position}) "
            f"- top-5 chance {row.top5_scorito_probability:.1%}, "
            f"expected points {row.expected_scorito_points:.1f}"
        )
    for path_item in output_paths:
        print(f"Wrote: {path_item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
