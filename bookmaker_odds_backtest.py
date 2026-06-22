#!/usr/bin/env python3
"""
Benchmark the knockout match-pick model against bookmaker-market baselines.

The primary comparison is:
  * our historical knockout match model
  * the 1X2 market favorite from bookmaker average odds
  * naive common-score baselines
  * a random historical-scoreline baseline

Odds can be supplied with --odds-csv or fetched from OddsPortal with
--fetch-oddsportal. OddsPortal currently returns encrypted archive payloads;
the fetcher decodes the public page payload and caches normalized odds rows so
future runs can use --no-refresh.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scorito_historical_backtest import (
    backtest_tournament,
    load_or_fetch_positions,
    match_points_for_round,
    outcome_from_score,
)
from scorito_knockout_topscorer_picker import TOURNAMENT_SOURCES, TournamentSource, load_historical_tournaments
from world_cup_score_forecaster import normalize_team


ODDSPORTAL_DECRYPTION_KEY = "J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k"
ODDSPORTAL_DECRYPTION_SALT = "5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d"

ODDSPORTAL_SOURCES = [
    ("FIFA World Cup 2010", "https://www.oddsportal.com/football/world/world-cup-2010/results/"),
    ("FIFA World Cup 2014", "https://www.oddsportal.com/football/world/world-cup-2014/results/"),
    ("FIFA World Cup 2018", "https://www.oddsportal.com/football/world/world-cup-2018/results/"),
    ("FIFA World Cup 2022", "https://www.oddsportal.com/football/world/world-cup-2022/results/"),
    ("UEFA Euro 2012", "https://www.oddsportal.com/football/europe/euro-2012/results/"),
    ("UEFA Euro 2016", "https://www.oddsportal.com/football/europe/euro-2016/results/"),
    ("UEFA Euro 2020", "https://www.oddsportal.com/football/europe/euro-2020/results/"),
    ("UEFA Euro 2024", "https://www.oddsportal.com/football/europe/euro-2024/results/"),
]

ODDS_COLUMN_SETS = [
    ("home_odds", "draw_odds", "away_odds"),
    ("market_home_odds", "market_draw_odds", "market_away_odds"),
    ("AvgH", "AvgD", "AvgA"),
    ("MaxH", "MaxD", "MaxA"),
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
]

COMMON_HOME_SCORE = (1, 0)
COMMON_DRAW_SCORE = (1, 1)
COMMON_AWAY_SCORE = (0, 1)
RANDOM_FALLBACK_SCORES = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (0, 0)]


@dataclass(frozen=True)
class OddsPortalRequest:
    tournament: str
    page_url: str
    tournament_id: int
    encoded_tournament_id: str
    odds_url: str
    url_part_tz: str
    url_part_qs: str
    bookiehash: str
    use_premium: str


def clean_date(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce").normalize()


def parse_score(value: object) -> tuple[int, int]:
    text = str(value).strip()
    if "-" in text:
        left, right = text.split("-", maxsplit=1)
    elif ":" in text:
        left, right = text.split(":", maxsplit=1)
    else:
        raise ValueError(f"Cannot parse score: {value!r}")
    return int(float(left)), int(float(right))


def score_text(home: int, away: int) -> str:
    return f"{int(home)}-{int(away)}"


def common_score_for_outcome(outcome: str) -> tuple[int, int]:
    if outcome == "home_win":
        return COMMON_HOME_SCORE
    if outcome == "away_win":
        return COMMON_AWAY_SCORE
    return COMMON_DRAW_SCORE


def scorito_points(round_name: object, exact: bool, toto: bool) -> int:
    points = match_points_for_round(str(round_name))
    if exact:
        return int(points["exact"])
    if toto:
        return int(points["toto"])
    return 0


def score_prediction_frame(
    matches: pd.DataFrame,
    home_col: str,
    away_col: str,
    prefix: str,
    eligible_mask: pd.Series | None = None,
) -> pd.DataFrame:
    scored = matches.copy()
    if eligible_mask is None:
        eligible_mask = pd.Series(True, index=scored.index)
    else:
        eligible_mask = eligible_mask.reindex(scored.index).fillna(False).astype(bool)

    exact_values: list[object] = []
    toto_values: list[object] = []
    points_values: list[object] = []
    outcome_values: list[object] = []
    score_values: list[object] = []

    for idx, row in scored.iterrows():
        if not bool(eligible_mask.loc[idx]) or pd.isna(row.get(home_col)) or pd.isna(row.get(away_col)):
            exact_values.append(pd.NA)
            toto_values.append(pd.NA)
            points_values.append(pd.NA)
            outcome_values.append(pd.NA)
            score_values.append(pd.NA)
            continue
        pred_home = int(row[home_col])
        pred_away = int(row[away_col])
        actual_home, actual_away = parse_score(row["actual_score"])
        predicted_outcome = outcome_from_score(pred_home, pred_away)
        actual_outcome = outcome_from_score(actual_home, actual_away)
        exact = pred_home == actual_home and pred_away == actual_away
        toto = predicted_outcome == actual_outcome
        exact_values.append(exact)
        toto_values.append(toto)
        points_values.append(scorito_points(row["round"], exact, toto))
        outcome_values.append(predicted_outcome)
        score_values.append(score_text(pred_home, pred_away))

    scored[f"{prefix}_predicted_score"] = score_values
    scored[f"{prefix}_predicted_outcome"] = outcome_values
    scored[f"{prefix}_exact_correct"] = exact_values
    scored[f"{prefix}_toto_correct"] = toto_values
    scored[f"{prefix}_scorito_points"] = points_values
    return scored


def normalize_model_picks(raw: pd.DataFrame) -> pd.DataFrame:
    picks = raw.copy()
    picks["date"] = picks["date"].map(clean_date)
    picks["home_team"] = picks["home_team"].map(normalize_team)
    picks["away_team"] = picks["away_team"].map(normalize_team)
    picks[["model_home_score", "model_away_score"]] = picks["predicted_score"].apply(
        lambda value: pd.Series(parse_score(value))
    )
    picks[["actual_home_score", "actual_away_score"]] = picks["actual_score"].apply(
        lambda value: pd.Series(parse_score(value))
    )
    picks["actual_outcome"] = [
        outcome_from_score(home, away)
        for home, away in zip(picks["actual_home_score"], picks["actual_away_score"])
    ]
    picks = score_prediction_frame(picks, "model_home_score", "model_away_score", "model")
    return picks


def configured_tournament_names(limit: int | None = None) -> list[str]:
    sources = [TournamentSource(*source).name for source in TOURNAMENT_SOURCES]
    if limit is not None:
        sources = sources[:limit]
    return sources


def build_model_picks(
    cache_dir: Path,
    refresh: bool,
    pool_size: str,
    topscorers_per_phase: int,
    historical_source_limit: int | None,
) -> pd.DataFrame:
    all_matches, all_goals = load_historical_tournaments(
        cache_dir,
        refresh=refresh,
        max_sources=historical_source_limit,
    )
    if all_matches.empty or all_goals.empty:
        raise RuntimeError("No historical match data available to build model picks.")

    positions = load_or_fetch_positions(
        all_goals,
        cache_dir,
        refresh=refresh,
        fetch_positions=False,
    )
    tournaments = [
        name
        for name in configured_tournament_names(historical_source_limit)
        if name in set(all_matches["tournament"])
    ]

    frames: list[pd.DataFrame] = []
    for tournament in tournaments:
        _, _, match_picks = backtest_tournament(
            tournament,
            all_matches,
            all_goals,
            positions,
            picks_per_phase=topscorers_per_phase,
            pool_size=pool_size,
        )
        if not match_picks.empty:
            frames.append(match_picks)
    if not frames:
        raise RuntimeError("No historical match picks could be built.")
    return pd.concat(frames, ignore_index=True)


def load_or_build_model_picks(args: argparse.Namespace) -> pd.DataFrame:
    model_path = args.model_picks_csv
    if model_path.exists() and not args.rebuild_model_picks:
        return normalize_model_picks(pd.read_csv(model_path))

    raw = build_model_picks(
        cache_dir=args.cache_dir,
        refresh=not args.no_refresh,
        pool_size=args.pool_size,
        topscorers_per_phase=args.topscorers_per_phase,
        historical_source_limit=args.historical_source_limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "rebuilt_model_match_picks.csv", index=False)
    return normalize_model_picks(raw)


def make_opener() -> urllib.request.OpenerDirector:
    cookie_jar = CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))


def fetch_text(
    opener: urllib.request.OpenerDirector,
    url: str,
    cache_path: Path | None,
    refresh: bool,
    referer: str | None = None,
    accept: str = "text/html,application/xhtml+xml,application/json,text/plain,*/*",
    timeout: int = 45,
) -> str:
    if cache_path is not None and cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")

    headers = {
        "User-Agent": "Mozilla/5.0 bookmaker-odds-backtest/1.0",
        "Accept": accept,
        "X-Requested-With": "XMLHttpRequest",
    }
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        if cache_path is not None and cache_path.exists():
            print(f"Warning: failed to refresh {url}; using cached {cache_path}. Reason: {exc}", file=sys.stderr)
            return cache_path.read_text(encoding="utf-8")
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(payload, encoding="utf-8")
    return payload


def parse_embedded_json_parse(script_text: str) -> dict[str, object]:
    match = re.search(r'JSON\.parse\("(.+?)"\)', script_text, flags=re.S)
    if not match:
        raise ValueError("Could not find JSON.parse payload in OddsPortal user-data response.")
    escaped = match.group(1)
    try:
        decoded = json.loads(f'"{escaped}"')
    except json.JSONDecodeError:
        decoded = escaped.encode("utf-8").decode("unicode_escape")
    return json.loads(decoded)


def find_required(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find {label} in OddsPortal page.")
    return html.unescape(match.group(1)).replace("\\/", "/")


def parse_oddsportal_page(
    opener: urllib.request.OpenerDirector,
    tournament: str,
    page_url: str,
    cache_dir: Path,
    refresh: bool,
) -> OddsPortalRequest:
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", tournament).strip("_").lower()
    page_html = fetch_text(
        opener,
        page_url,
        cache_dir / "oddsportal" / f"{safe_name}_page.html",
        refresh=refresh,
        referer=page_url,
    )
    encoded = find_required(r'encodedTurnamentId&quot;:&quot;([^&]+)&quot;', page_html, "encoded tournament id")
    tournament_id = int(find_required(r'tournamentId&quot;:(\d+)', page_html, "tournament id"))
    odds_url = find_required(
        r'oddsRequest&quot;:\{&quot;url&quot;:&quot;([^&]+)&quot;',
        page_html,
        "odds request URL",
    )
    url_part_tz = find_required(r'urlPartTz&quot;:([^,&]+)', page_html, "odds request timezone part")
    url_part_qs = find_required(r'urlPartQs&quot;:&quot;([^&]+)&quot;', page_html, "odds request query part")

    user_data_path = re.search(rf'/ajax-user-data/t/{tournament_id}/\?[^"\']+', page_html)
    if not user_data_path:
        raise ValueError(f"Could not find OddsPortal user-data URL for {tournament}.")
    user_data_url = urllib.parse.urljoin(page_url, user_data_path.group(0))
    user_data = fetch_text(
        opener,
        user_data_url,
        cache_dir / "oddsportal" / f"{safe_name}_user_data.js",
        refresh=refresh,
        referer=page_url,
        accept="application/javascript,text/javascript,text/plain,*/*",
    )
    config = parse_embedded_json_parse(user_data)
    return OddsPortalRequest(
        tournament=tournament,
        page_url=page_url,
        tournament_id=tournament_id,
        encoded_tournament_id=encoded,
        odds_url=odds_url,
        url_part_tz=str(url_part_tz).strip().strip('"'),
        url_part_qs=url_part_qs,
        bookiehash=str(config["bookiehash"]),
        use_premium=str(config["usePremium"]),
    )


def decrypt_oddsportal_payload(payload: str) -> dict[str, object]:
    try:
        from cryptography.hazmat.primitives import hashes, padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except Exception as exc:
        raise RuntimeError(
            "OddsPortal fetch requires the 'cryptography' package. "
            "Install requirements.txt or pass --odds-csv instead."
        ) from exc

    raw = base64.b64decode(payload)
    encrypted_b64, iv_hex = raw.split(b":", maxsplit=1)
    encrypted = base64.b64decode(encrypted_b64)
    iv = bytes.fromhex(iv_hex.decode("ascii"))
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=ODDSPORTAL_DECRYPTION_SALT.encode("utf-8"),
        iterations=1000,
    )
    key = kdf.derive(ODDSPORTAL_DECRYPTION_KEY.encode("utf-8"))
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(encrypted) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    decoded = unpadder.update(padded) + unpadder.finalize()
    if decoded.startswith(b"\x1f\x8b"):
        decoded = gzip.decompress(decoded)
    return json.loads(decoded.decode("utf-8"))


def oddsportal_archive_url(request: OddsPortalRequest, page: int) -> str:
    base = urllib.parse.urljoin(
        request.page_url,
        f"{request.odds_url}{request.bookiehash}/{request.use_premium}/{request.url_part_tz}/",
    )
    if page > 1:
        base = urllib.parse.urljoin(base + "/", f"page/{page}/")
    return f"{base}{request.url_part_qs}{int(time.time() * 1000)}"


def parse_oddsportal_rows(rows: Iterable[dict[str, object]], source_hint: str) -> pd.DataFrame:
    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        odds = row.get("odds") or []
        if not isinstance(odds, list) or len(odds) < 3:
            continue
        cols = str(row.get("cols", "1|X|2")).split("|")
        market = [
            odd
            for odd in odds
            if isinstance(odd, dict)
            and int(float(odd.get("bettingTypeId", 1) or 1)) == 1
            and int(float(odd.get("scopeId", 2) or 2)) == 2
        ]
        if len(market) < 3:
            market = [odd for odd in odds if isinstance(odd, dict)][:3]
        else:
            market = market[:3]
        if len(market) < 3:
            continue

        values: dict[str, float] = {}
        for column, odd in zip(cols[:3], market):
            value = odd.get("avgOdds", odd.get("maxOdds"))
            if value is None or value == "":
                value = odd.get("maxOdds")
            try:
                values[column] = float(value)
            except (TypeError, ValueError):
                values[column] = math.nan

        if not {"1", "X", "2"}.issubset(values) or any(math.isnan(values[key]) for key in ["1", "X", "2"]):
            continue

        tournament_name = normalize_oddsportal_tournament_name(
            str(row.get("tournament-name") or row.get("tournament_name") or source_hint)
        )
        timestamp = row.get("date-start-timestamp", row.get("date-start-base"))
        date = pd.to_datetime(float(timestamp), unit="s", utc=True).tz_convert(None).normalize() if timestamp else pd.NaT
        parsed_rows.append(
            {
                "source": "oddsportal",
                "event_id": row.get("id"),
                "tournament": tournament_name,
                "date": date,
                "home_team": normalize_team(row.get("home-name", "")),
                "away_team": normalize_team(row.get("away-name", "")),
                "home_odds": values["1"],
                "draw_odds": values["X"],
                "away_odds": values["2"],
                "bookmakers_count": market[0].get("cntActive") if market else pd.NA,
                "source_tournament": row.get("tournament-name", source_hint),
            }
        )
    if not parsed_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(parsed_rows)
    return frame.drop_duplicates(
        subset=["event_id", "tournament", "date", "home_team", "away_team"],
        keep="first",
    ).reset_index(drop=True)


def normalize_oddsportal_tournament_name(value: str) -> str:
    text = " ".join(str(value).split())
    year_match = re.search(r"(20\d{2}|19\d{2})", text)
    year = year_match.group(1) if year_match else ""
    if "World Cup" in text:
        return f"FIFA World Cup {year}".strip()
    if "Euro" in text or "European Championship" in text:
        return f"UEFA Euro {year}".strip()
    return text


def fetch_oddsportal_odds(
    cache_dir: Path,
    output_dir: Path,
    refresh: bool,
    max_pages: int | None,
) -> pd.DataFrame:
    opener = make_opener()
    frames: list[pd.DataFrame] = []
    raw_cache_dir = cache_dir / "oddsportal"
    normalized_cache = raw_cache_dir / "normalized_oddsportal_1x2.csv"
    if normalized_cache.exists() and not refresh:
        cached = pd.read_csv(normalized_cache)
        cached["date"] = cached["date"].map(clean_date)
        return cached

    for tournament, page_url in ODDSPORTAL_SOURCES:
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", tournament).strip("_").lower()
        try:
            request = parse_oddsportal_page(opener, tournament, page_url, cache_dir, refresh=refresh)
        except Exception as exc:
            print(f"Warning: skipping OddsPortal source {tournament}: {exc}", file=sys.stderr)
            continue

        page = 1
        total = None
        one_page = None
        while True:
            archive_url = oddsportal_archive_url(request, page)
            payload_path = raw_cache_dir / f"{safe_name}_archive_page_{page}.txt"
            try:
                payload = fetch_text(
                    opener,
                    archive_url,
                    payload_path,
                    refresh=refresh,
                    referer=page_url,
                    accept="application/json,text/plain,*/*",
                )
                data = decrypt_oddsportal_payload(payload)
            except Exception as exc:
                print(f"Warning: stopping {tournament} odds fetch at page {page}: {exc}", file=sys.stderr)
                break

            details = data.get("d", {}) if isinstance(data, dict) else {}
            rows = details.get("rows", [])
            if rows:
                parsed = parse_oddsportal_rows(rows, tournament)
                if not parsed.empty:
                    frames.append(parsed)
            total = int(details.get("total") or total or 0)
            one_page = int(details.get("onePage") or one_page or len(rows) or 50)

            fetched_rows = page * one_page
            if not rows or (total and fetched_rows >= total):
                break
            page += 1
            if max_pages is not None and page > max_pages:
                break

    odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not odds.empty:
        odds = odds.drop_duplicates(
            subset=["tournament", "date", "home_team", "away_team"],
            keep="first",
        ).reset_index(drop=True)
        raw_cache_dir.mkdir(parents=True, exist_ok=True)
        odds.to_csv(normalized_cache, index=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        odds.to_csv(output_dir / "bookmaker_oddsportal_1x2_odds.csv", index=False)
    return odds


def column_case_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(column).lower(): str(column) for column in columns}


def first_existing(columns: dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        found = columns.get(candidate.lower())
        if found:
            return found
    return None


def load_manual_odds(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    columns = column_case_map(raw.columns)
    tournament_col = first_existing(columns, ["tournament", "competition"])
    date_col = first_existing(columns, ["date", "Date"])
    home_col = first_existing(columns, ["home_team", "HomeTeam", "home", "Home"])
    away_col = first_existing(columns, ["away_team", "AwayTeam", "away", "Away"])
    if not all([tournament_col, date_col, home_col, away_col]):
        raise ValueError(
            "Manual odds CSV needs tournament/date/home_team/away_team columns "
            "(case-insensitive aliases are accepted)."
        )

    odds_columns = None
    for home_odds, draw_odds, away_odds in ODDS_COLUMN_SETS:
        found = (
            columns.get(home_odds.lower()),
            columns.get(draw_odds.lower()),
            columns.get(away_odds.lower()),
        )
        if all(found):
            odds_columns = found
            break
    if odds_columns is None:
        raise ValueError(
            "Manual odds CSV needs odds columns such as home_odds/draw_odds/away_odds, "
            "AvgH/AvgD/AvgA, B365H/B365D/B365A, or PSH/PSD/PSA."
        )

    source_col = first_existing(columns, ["source"])
    event_col = first_existing(columns, ["event_id", "id"])
    rows = pd.DataFrame(
        {
            "source": raw[source_col] if source_col else "manual_csv",
            "event_id": raw[event_col] if event_col else pd.NA,
            "tournament": raw[tournament_col].astype(str).map(normalize_oddsportal_tournament_name),
            "date": raw[date_col].map(clean_date),
            "home_team": raw[home_col].map(normalize_team),
            "away_team": raw[away_col].map(normalize_team),
            "home_odds": pd.to_numeric(raw[odds_columns[0]], errors="coerce"),
            "draw_odds": pd.to_numeric(raw[odds_columns[1]], errors="coerce"),
            "away_odds": pd.to_numeric(raw[odds_columns[2]], errors="coerce"),
            "bookmakers_count": pd.NA,
            "source_tournament": raw[tournament_col],
        }
    )
    return rows.dropna(subset=["date", "home_odds", "draw_odds", "away_odds"]).reset_index(drop=True)


def load_odds(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if args.odds_csv is not None:
        frames.append(load_manual_odds(args.odds_csv))
    if args.fetch_oddsportal:
        frames.append(
            fetch_oddsportal_odds(
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                refresh=not args.no_refresh,
                max_pages=args.oddsportal_max_pages,
            )
        )
    if not frames:
        cached = args.cache_dir / "oddsportal" / "normalized_oddsportal_1x2.csv"
        if cached.exists():
            frame = pd.read_csv(cached)
            frame["date"] = frame["date"].map(clean_date)
            frames.append(frame)
    odds = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if odds.empty:
        return odds
    odds["date"] = odds["date"].map(clean_date)
    odds["home_team"] = odds["home_team"].map(normalize_team)
    odds["away_team"] = odds["away_team"].map(normalize_team)
    return odds.drop_duplicates(
        subset=["tournament", "date", "home_team", "away_team"],
        keep="first",
    ).reset_index(drop=True)


def find_odds_for_match(match: pd.Series, odds: pd.DataFrame, max_date_diff_days: int = 2) -> dict[str, object]:
    if odds.empty:
        return {}
    tournament_odds = odds[odds["tournament"].eq(match["tournament"])].copy()
    if tournament_odds.empty:
        return {}

    home = normalize_team(match["home_team"])
    away = normalize_team(match["away_team"])
    match_date = clean_date(match["date"])
    candidates = tournament_odds[
        (
            tournament_odds["home_team"].eq(home)
            & tournament_odds["away_team"].eq(away)
        )
        | (
            tournament_odds["home_team"].eq(away)
            & tournament_odds["away_team"].eq(home)
        )
    ].copy()
    if candidates.empty:
        return {}
    candidates["date_diff"] = (candidates["date"] - match_date).abs().dt.days
    candidates = candidates[candidates["date_diff"].le(max_date_diff_days)].sort_values("date_diff")
    if candidates.empty:
        return {}
    row = candidates.iloc[0]
    swapped = row["home_team"] == away and row["away_team"] == home
    if swapped:
        home_odds = float(row["away_odds"])
        away_odds = float(row["home_odds"])
    else:
        home_odds = float(row["home_odds"])
        away_odds = float(row["away_odds"])
    return {
        "odds_source": row.get("source", ""),
        "odds_event_id": row.get("event_id", pd.NA),
        "odds_date": row["date"],
        "odds_home_team": row["home_team"],
        "odds_away_team": row["away_team"],
        "odds_swapped": bool(swapped),
        "home_odds": home_odds,
        "draw_odds": float(row["draw_odds"]),
        "away_odds": away_odds,
        "bookmakers_count": row.get("bookmakers_count", pd.NA),
        "odds_date_diff_days": int(row["date_diff"]),
    }


def attach_market_odds(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, match in matches.iterrows():
        row = match.to_dict()
        row.update(find_odds_for_match(match, odds))
        rows.append(row)
    merged = pd.DataFrame(rows)
    for column in ["home_odds", "draw_odds", "away_odds"]:
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce")
    has_odds = merged[["home_odds", "draw_odds", "away_odds"]].notna().all(axis=1)
    merged["has_market_odds"] = has_odds
    for column, probability_column in [
        ("home_odds", "market_home_implied"),
        ("draw_odds", "market_draw_implied"),
        ("away_odds", "market_away_implied"),
    ]:
        merged[probability_column] = np.where(has_odds, 1.0 / merged[column], np.nan)
    overround = (
        merged["market_home_implied"] + merged["market_draw_implied"] + merged["market_away_implied"]
    )
    for column in ["market_home_implied", "market_draw_implied", "market_away_implied"]:
        merged[column] = np.where(has_odds, merged[column] / overround, np.nan)
    merged["market_overround"] = np.where(has_odds, overround, np.nan)

    market_outcomes = []
    market_home_scores = []
    market_away_scores = []
    for _, row in merged.iterrows():
        if not bool(row["has_market_odds"]):
            market_outcomes.append(pd.NA)
            market_home_scores.append(pd.NA)
            market_away_scores.append(pd.NA)
            continue
        odds_values = {
            "home_win": float(row["home_odds"]),
            "draw": float(row["draw_odds"]),
            "away_win": float(row["away_odds"]),
        }
        outcome = min(odds_values, key=odds_values.get)
        home_score, away_score = common_score_for_outcome(outcome)
        market_outcomes.append(outcome)
        market_home_scores.append(home_score)
        market_away_scores.append(away_score)
    merged["market_home_score"] = market_home_scores
    merged["market_away_score"] = market_away_scores
    merged["market_favorite_outcome"] = market_outcomes
    merged = score_prediction_frame(
        merged,
        "market_home_score",
        "market_away_score",
        "market",
        eligible_mask=merged["has_market_odds"],
    )
    return merged


def add_common_score_baseline(matches: pd.DataFrame) -> pd.DataFrame:
    output = matches.copy()
    output["common_home_score"] = COMMON_HOME_SCORE[0]
    output["common_away_score"] = COMMON_HOME_SCORE[1]
    return score_prediction_frame(output, "common_home_score", "common_away_score", "common")


def strategy_summary(
    matches: pd.DataFrame,
    strategy: str,
    exact_col: str,
    toto_col: str,
    points_col: str,
    eligible_mask: pd.Series | None = None,
    notes: str = "",
) -> dict[str, object]:
    if eligible_mask is None:
        eligible_mask = matches[exact_col].notna()
    else:
        eligible_mask = eligible_mask.reindex(matches.index).fillna(False).astype(bool) & matches[exact_col].notna()
    subset = matches[eligible_mask]
    return {
        "strategy": strategy,
        "matches_evaluated": int(len(subset)),
        "exact_accuracy": float(subset[exact_col].mean()) if not subset.empty else np.nan,
        "toto_accuracy": float(subset[toto_col].mean()) if not subset.empty else np.nan,
        "scorito_points": float(subset[points_col].sum()) if not subset.empty else 0.0,
        "points_per_match": float(subset[points_col].mean()) if not subset.empty else np.nan,
        "market_coverage_rate": float(matches["has_market_odds"].mean()) if "has_market_odds" in matches else 0.0,
        "notes": notes,
    }


def tournament_strategy_summary(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tournament, group in matches.groupby("tournament", sort=True):
        market_mask = group["has_market_odds"].fillna(False).astype(bool)
        strategies = [
            ("model_all", "model_exact_correct", "model_toto_correct", "model_scorito_points", None),
            ("common_home_1_0_all", "common_exact_correct", "common_toto_correct", "common_scorito_points", None),
        ]
        if market_mask.any():
            strategies.extend(
                [
                    ("model_market_covered", "model_exact_correct", "model_toto_correct", "model_scorito_points", market_mask),
                    ("market_favorite_common_score", "market_exact_correct", "market_toto_correct", "market_scorito_points", market_mask),
                    ("common_home_1_0_market_covered", "common_exact_correct", "common_toto_correct", "common_scorito_points", market_mask),
                ]
            )
        for strategy, exact_col, toto_col, points_col, mask in strategies:
            summary = strategy_summary(group, strategy, exact_col, toto_col, points_col, mask)
            summary["tournament"] = tournament
            rows.append(summary)
    return pd.DataFrame(rows)


def random_baseline(
    matches: pd.DataFrame,
    trials: int,
    seed: int,
) -> pd.DataFrame:
    ordered = matches.sort_values(["date", "tournament", "round", "home_team", "away_team"]).reset_index(drop=True)
    rng = random.Random(seed)
    summaries: list[dict[str, object]] = []
    for trial in range(trials):
        historical_scores = RANDOM_FALLBACK_SCORES.copy()
        exact_count = 0
        toto_count = 0
        points = 0
        for row in ordered.itertuples(index=False):
            pred_home, pred_away = rng.choice(historical_scores)
            actual_home, actual_away = parse_score(row.actual_score)
            exact = pred_home == actual_home and pred_away == actual_away
            toto = outcome_from_score(pred_home, pred_away) == outcome_from_score(actual_home, actual_away)
            exact_count += int(exact)
            toto_count += int(toto)
            points += scorito_points(row.round, exact, toto)
            historical_scores.append((actual_home, actual_away))
        count = len(ordered)
        summaries.append(
            {
                "trial": trial + 1,
                "matches_evaluated": count,
                "exact_accuracy": exact_count / count if count else np.nan,
                "toto_accuracy": toto_count / count if count else np.nan,
                "scorito_points": points,
                "points_per_match": points / count if count else np.nan,
            }
        )
    return pd.DataFrame(summaries)


def random_summary_row(random_trials: pd.DataFrame) -> dict[str, object]:
    if random_trials.empty:
        return {
            "strategy": "random_historical_scoreline",
            "matches_evaluated": 0,
            "exact_accuracy": np.nan,
            "toto_accuracy": np.nan,
            "scorito_points": 0.0,
            "points_per_match": np.nan,
            "market_coverage_rate": np.nan,
            "notes": "No random trials run.",
        }
    return {
        "strategy": "random_historical_scoreline",
        "matches_evaluated": int(random_trials["matches_evaluated"].iloc[0]),
        "exact_accuracy": float(random_trials["exact_accuracy"].mean()),
        "toto_accuracy": float(random_trials["toto_accuracy"].mean()),
        "scorito_points": float(random_trials["scorito_points"].mean()),
        "points_per_match": float(random_trials["points_per_match"].mean()),
        "market_coverage_rate": np.nan,
        "notes": (
            "Mean over random trials. "
            f"Points p05={random_trials['scorito_points'].quantile(0.05):.0f}, "
            f"p95={random_trials['scorito_points'].quantile(0.95):.0f}."
        ),
    }


def overall_summary(matches: pd.DataFrame, random_trials: pd.DataFrame) -> pd.DataFrame:
    market_mask = matches["has_market_odds"].fillna(False).astype(bool)
    rows = [
        strategy_summary(
            matches,
            "model_all",
            "model_exact_correct",
            "model_toto_correct",
            "model_scorito_points",
            notes="Current historical knockout score heuristic on all backtested matches.",
        ),
        strategy_summary(
            matches,
            "common_home_1_0_all",
            "common_exact_correct",
            "common_toto_correct",
            "common_scorito_points",
            notes="Naive common-score baseline: every listed home team wins 1-0.",
        ),
        random_summary_row(random_trials),
    ]
    if market_mask.any():
        rows.extend(
            [
                strategy_summary(
                    matches,
                    "model_market_covered",
                    "model_exact_correct",
                    "model_toto_correct",
                    "model_scorito_points",
                    eligible_mask=market_mask,
                    notes="Model restricted to matches with bookmaker odds.",
                ),
                strategy_summary(
                    matches,
                    "market_favorite_common_score",
                    "market_exact_correct",
                    "market_toto_correct",
                    "market_scorito_points",
                    eligible_mask=market_mask,
                    notes="Lowest average 1X2 odds; exact score set to 1-0, 1-1, or 0-1.",
                ),
                strategy_summary(
                    matches,
                    "common_home_1_0_market_covered",
                    "common_exact_correct",
                    "common_toto_correct",
                    "common_scorito_points",
                    eligible_mask=market_mask,
                    notes="Naive 1-0 home baseline restricted to matches with bookmaker odds.",
                ),
            ]
        )
    else:
        rows.append(
            {
                "strategy": "market_favorite_common_score",
                "matches_evaluated": 0,
                "exact_accuracy": np.nan,
                "toto_accuracy": np.nan,
                "scorito_points": 0.0,
                "points_per_match": np.nan,
                "market_coverage_rate": 0.0,
                "notes": "No bookmaker odds loaded. Use --fetch-oddsportal or --odds-csv.",
            }
        )
    return pd.DataFrame(rows)


def plot_summary(output_dir: Path, summary: pd.DataFrame) -> Path | None:
    if summary.empty:
        return None
    display = summary[summary["matches_evaluated"].gt(0)].copy()
    if display.empty:
        return None
    order = [
        "model_all",
        "model_market_covered",
        "market_favorite_common_score",
        "common_home_1_0_all",
        "common_home_1_0_market_covered",
        "random_historical_scoreline",
    ]
    display["sort_order"] = display["strategy"].map({name: idx for idx, name in enumerate(order)}).fillna(99)
    display = display.sort_values("sort_order")
    labels = display["strategy"].str.replace("_", " ", regex=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    colors = ["#4e79a7" if "model" in strategy else "#f28e2b" if "market" in strategy else "#59a14f" for strategy in display["strategy"]]

    axes[0].barh(labels, display["toto_accuracy"], color=colors)
    axes[0].set_title("Outcome accuracy")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Toto accuracy")

    axes[1].barh(labels, display["exact_accuracy"], color=colors)
    axes[1].set_title("Exact-score accuracy")
    axes[1].set_xlim(0, max(0.20, float(display["exact_accuracy"].max(skipna=True) or 0.0) * 1.25))
    axes[1].set_xlabel("Exact accuracy")

    axes[2].barh(labels, display["points_per_match"], color=colors)
    axes[2].set_title("Scorito match points")
    axes[2].set_xlabel("Points per match")

    fig.subplots_adjust(left=0.26, right=0.98, top=0.88, bottom=0.12, wspace=0.35)
    path = output_dir / "bookmaker_odds_backtest_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_outputs(
    output_dir: Path,
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    summary: pd.DataFrame,
    tournament_summary: pd.DataFrame,
    random_trials: pd.DataFrame,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "bookmaker_backtest_by_match.csv",
        output_dir / "bookmaker_backtest_summary.csv",
        output_dir / "bookmaker_backtest_by_tournament.csv",
        output_dir / "bookmaker_random_baseline_trials.csv",
    ]
    matches.to_csv(paths[0], index=False)
    summary.to_csv(paths[1], index=False)
    tournament_summary.to_csv(paths[2], index=False)
    random_trials.to_csv(paths[3], index=False)
    if not odds.empty:
        odds_path = output_dir / "bookmaker_1x2_odds_used.csv"
        odds.to_csv(odds_path, index=False)
        paths.append(odds_path)
    figure = plot_summary(output_dir, summary)
    if figure is not None:
        paths.append(figure)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model picks with bookmaker-market baselines.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_world_cup"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_bookmaker_backtest"))
    parser.add_argument(
        "--model-picks-csv",
        type=Path,
        default=Path("outputs_scorito_backtest") / "backtest_match_picks.csv",
        help="Existing historical knockout model picks. Rebuilt only if missing or --rebuild-model-picks is used.",
    )
    parser.add_argument("--rebuild-model-picks", action="store_true")
    parser.add_argument("--odds-csv", type=Path, default=None, help="Manual bookmaker odds CSV.")
    parser.add_argument("--fetch-oddsportal", action="store_true", help="Fetch and cache OddsPortal 1X2 average odds.")
    parser.add_argument("--oddsportal-max-pages", type=int, default=None, help="Debug cap for fetched OddsPortal pages.")
    parser.add_argument("--no-refresh", action="store_true", help="Use cached model/odds source files where possible.")
    parser.add_argument("--pool-size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--topscorers-per-phase", type=int, default=4)
    parser.add_argument("--historical-source-limit", type=int, default=None)
    parser.add_argument("--random-trials", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260622)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_matches = load_or_build_model_picks(args)
    odds = load_odds(args)
    compared = attach_market_odds(model_matches, odds)
    compared = add_common_score_baseline(compared)
    random_trials = random_baseline(compared, trials=args.random_trials, seed=args.random_seed)
    summary = overall_summary(compared, random_trials)
    by_tournament = tournament_strategy_summary(compared)
    output_paths = write_outputs(args.output_dir, compared, odds, summary, by_tournament, random_trials)

    model_row = summary[summary["strategy"].eq("model_all")].iloc[0]
    market_rows = summary[summary["strategy"].eq("market_favorite_common_score")]
    market_row = market_rows.iloc[0] if not market_rows.empty else None
    print("Bookmaker odds backtest complete")
    print(f"Matches backtested: {int(model_row.matches_evaluated)}")
    print(f"Model exact accuracy: {model_row.exact_accuracy:.1%}")
    print(f"Model toto accuracy: {model_row.toto_accuracy:.1%}")
    print(f"Market odds coverage: {compared['has_market_odds'].mean():.1%}")
    if market_row is not None and int(market_row.matches_evaluated) > 0:
        print(f"Market favorite toto accuracy: {market_row.toto_accuracy:.1%}")
        print(f"Market favorite exact-score proxy accuracy: {market_row.exact_accuracy:.1%}")
    else:
        print("Market favorite benchmark not scored because no bookmaker odds were loaded.")
    for path in output_paths:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
