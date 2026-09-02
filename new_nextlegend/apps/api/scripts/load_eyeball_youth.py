from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PIPELINE_CANDIDATES = [
    Path("/jobs/pipeline"),
    *(parent / "jobs" / "pipeline" for parent in Path(__file__).resolve().parents),
]
PIPELINE_ROOT = next((path for path in PIPELINE_CANDIDATES if path.exists()), None)
if PIPELINE_ROOT and str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from settings import settings  # noqa: E402
from pipeline import scoring_v2  # noqa: E402


CURRENT_SEASON = 2027


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS youth_player_rankings (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'eyeball',
    provider_player_id TEXT NOT NULL,
    source_row_hash TEXT NOT NULL,
    provider_player_url TEXT,
    season INT NOT NULL,
    calendar TEXT,
    is_current_season BOOLEAN NOT NULL DEFAULT FALSE,
    country_code TEXT,
    first_name TEXT,
    last_name TEXT,
    display_name TEXT NOT NULL,
    birth_year INT,
    birth_date TEXT,
    age INT,
    age_category TEXT,
    championship TEXT,
    club_name TEXT,
    team_name TEXT,
    team_level INT,
    position TEXT,
    primary_position TEXT,
    position_group TEXT,
    strong_foot TEXT,
    height_cm DOUBLE PRECISION,
    weight_kg DOUBLE PRECISION,
    games_count DOUBLE PRECISION,
    minutes_played DOUBLE PRECISION,
    rating DOUBLE PRECISION,
    score DOUBLE PRECISION,
    score_raw DOUBLE PRECISION,
    score_percentile_global DOUBLE PRECISION,
    score_percentile_age_category DOUBLE PRECISION,
    score_percentile_birth_year DOUBLE PRECISION,
    score_percentile_championship DOUBLE PRECISION,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    metric_percentiles JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    imported_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(provider, season, source_row_hash)
);

CREATE INDEX IF NOT EXISTS youth_player_rankings_score_idx
    ON youth_player_rankings(season, score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS youth_player_rankings_context_idx
    ON youth_player_rankings(season, championship, age_category, birth_year, position_group);
CREATE INDEX IF NOT EXISTS youth_player_rankings_player_idx
    ON youth_player_rankings(provider_player_id);
CREATE INDEX IF NOT EXISTS youth_player_rankings_search_idx
    ON youth_player_rankings(LOWER(display_name), LOWER(club_name), LOWER(championship));
ALTER TABLE youth_player_rankings ADD COLUMN IF NOT EXISTS calendar TEXT;
ALTER TABLE youth_player_rankings ADD COLUMN IF NOT EXISTS is_current_season BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE youth_player_rankings ADD COLUMN IF NOT EXISTS birth_date TEXT;
"""


METRIC_COLUMNS = {
    "rating": "rating_value",
    "minutes_played": "minutesPlayed",
    "games_count": "gamesCount",
    "goals_per_90": "goalsScored",
    "assists_per_90": "assists_value",
    "total_goals": "totalGoals",
    "total_assists": "totalAssists",
    "shots_per_90": "shots_value",
    "shots_on_target_per_90": "shotsOnTarget_value",
    "shots_inside_area_per_90": "shotsInsidePa_value",
    "shots_outside_area_per_90": "shotsOutsidePa_value",
    "key_passes_per_90": "keyPasses_value",
    "passes_success_per_90": "passes_success_value",
    "passes_total_per_90": "passes_total_value",
    "passes_accuracy_pct": "passesAccuracy_value",
    "forward_passes_success_per_90": "passesForward_success_value",
    "forward_passes_total_per_90": "passesForward_total_value",
    "crosses_success_per_90": "crosses_success_value",
    "crosses_total_per_90": "crosses_total_value",
    "takeons_success_per_90": "takeons_success_value",
    "takeons_total_per_90": "takeons_total_value",
    "tackles_success_per_90": "tackles_success_value",
    "tackles_total_per_90": "tackles_total_value",
    "aerial_duels_success_per_90": "aerialDuels_success_value",
    "aerial_duels_total_per_90": "aerialDuels_total_value",
    "recoveries_per_90": "recoveries_value",
    "interceptions_per_90": "interceptions_value",
    "clearances_per_90": "clearances_value",
    "blocks_per_90": "blocks_value",
    "goals_conceded_per_90": "goalsConceded_value",
    "catches_per_90": "catches_value",
    "punches_per_90": "punches_value",
    "goal_kicks_success_per_90": "goalsKicks_success_value",
    "goal_kicks_total_per_90": "goalsKicks_total_value",
    "aerial_clearances_success_per_90": "aerialClearances_success_value",
    "aerial_clearances_total_per_90": "aerialClearances_total_value",
}

LOWER_IS_BETTER = {"goals_conceded_per_90"} | scoring_v2.LOWER_IS_BETTER_PERCENTILE_METRICS


def clean_text(value: Any) -> str | None:
    text_value = str(value or "").strip()
    return re.sub(r"\s+", " ", text_value) or None


def parse_float(value: Any) -> float | None:
    raw = clean_text(value)
    if raw is None:
        return None
    cleaned = raw.replace("%", "").replace(",", ".").strip()
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]
    if cleaned.lower() in {"nan", "none", "null", "-"}:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def parse_birth_year(value: Any) -> int | None:
    cleaned = clean_text(value)
    if not cleaned:
        return None
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", cleaned)
    if year_match:
        return int(year_match.group(1))
    return parse_int(cleaned)


def normalize_name(first_name: str | None, last_name: str | None) -> str:
    return clean_text(" ".join([first_name or "", last_name or ""])) or "Unknown player"


def position_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        token.strip().upper().replace("*", "")
        for token in re.split(r"[,/]", value)
        if token.strip()
    ]


def primary_position(value: str | None) -> str | None:
    if not value:
        return None
    starred = re.search(r"([A-Za-z]+)\*", value)
    if starred:
        return starred.group(1).upper()
    tokens = position_tokens(value)
    return tokens[0] if tokens else None


def position_group(position: str | None) -> str:
    primary = primary_position(position)
    if primary in {"G", "GK", "GB"}:
        return "Goalkeepers"
    if primary == "DC":
        return "Centre Backs"
    if primary in {"DD", "DG", "DLD", "DLG"}:
        return "Fullbacks"
    if primary == "MDC":
        return "Defensive Midfielders"
    if primary == "MC":
        return "Central Midfielders"
    if primary == "MOC":
        return "Attacking Midfielders"
    if primary in {"MD", "MG", "AD", "AG"}:
        return "Wingers"
    if primary in {"ATT", "BU"}:
        return "Forwards"
    tokens = set(position_tokens(position))
    if tokens & {"G", "GK", "GB"}:
        return "Goalkeepers"
    if tokens & {"ATT", "BU"}:
        return "Forwards"
    if tokens & {"MD", "MG", "AD", "AG"}:
        return "Wingers"
    if tokens & {"DD", "DG", "DLD", "DLG"}:
        return "Fullbacks"
    return "Central Midfielders"


def wyscout_position(position: str | None) -> str | None:
    primary = primary_position(position)
    mapping = {
        "G": "GK",
        "GB": "GK",
        "GK": "GK",
        "DC": "CB",
        "DD": "RB",
        "DLD": "RB",
        "DG": "LB",
        "DLG": "LB",
        "MDC": "DMF",
        "MC": "CMF",
        "MOC": "AMF",
        "MD": "RW",
        "AD": "RW",
        "MG": "LW",
        "AG": "LW",
        "ATT": "CF",
        "BU": "CF",
    }
    if primary in mapping:
        return mapping[primary]
    for token in position_tokens(position):
        if token in mapping:
            return mapping[token]
    return None


def age_category(value: str | None, birth_year: int | None, season: int) -> str | None:
    cleaned = clean_text(value)
    if cleaned:
        return cleaned.upper().replace(", ", "/").replace(",", "/")
    if birth_year:
        return f"U{max(1, season - birth_year)}"
    return None


def birth_date_label(raw: dict[str, Any]) -> str | None:
    return clean_text(
        raw.get("birthDate")
        or raw.get("birthdate")
        or raw.get("dateOfBirth")
        or raw.get("birthday")
    )


def normalize_competition_level(team_name: str | None, club_name: str | None) -> str | None:
    team = clean_text(team_name)
    club = clean_text(club_name)
    if not team:
        return None
    if club and team.upper() == club.upper():
        return None
    levels: list[str] = []
    for item in re.split(r"[/,]", team):
        normalized = item.strip().upper()
        normalized = (
            normalized.replace("É", "E")
            .replace("È", "E")
            .replace("Ê", "E")
            .replace("À", "A")
        )
        normalized = normalized.replace("RÉGIONAL", "REGIONAL")
        regional_match = re.fullmatch(r"REGIONAL\s+([1-5])", normalized)
        departmental_match = re.fullmatch(r"DEPART[EA]MENTAL(?:E)?\s+([1-5])", normalized)
        national_match = re.fullmatch(r"NATIONAL\s+([1-5])", normalized)
        if regional_match:
            level = f"R{regional_match.group(1)}"
        elif departmental_match:
            level = f"D{departmental_match.group(1)}"
        elif national_match:
            level = f"N{national_match.group(1)}"
        elif normalized in {"NATIONAUX"}:
            level = "NATIONAL"
        elif re.fullmatch(r"N[1-5]", normalized):
            level = normalized
        elif re.fullmatch(r"R[1-5]", normalized):
            level = normalized
        elif re.fullmatch(r"D[1-5]", normalized):
            level = normalized
        elif normalized in {
            "NATIONAL",
            "NATIONAL TEAM",
            "ELITE",
            "RESERVE",
            "RESERVE PRO",
            "SENIOR",
            "SENIOR RESERVE",
            "PREMIERE DIVISION",
            "PRO LEAGUE 1",
            "CHALLENGER PRO LEAGUE",
            "REGIONAL",
            "LIGUE",
        }:
            level = normalized
        elif re.fullmatch(r"LIGUE\s+[1-3]", normalized):
            level = normalized
        elif re.fullmatch(r"ELITE\s+[1-3]", normalized):
            level = normalized
        else:
            continue
        if level not in levels:
            levels.append(level)
    return "/".join(levels) if levels else None


def championship_label(raw: dict[str, Any], resolved_age_category: str | None) -> str:
    level = normalize_competition_level(raw.get("teamName"), raw.get("clubName"))
    age = clean_text(resolved_age_category)
    if level:
        if age and re.match(r"^U\d+", age) and not level.startswith(age):
            return f"{age} {level}"
        return level
    if age:
        if re.match(r"^U\d+", age) and clean_text(raw.get("teamName")):
            return f"{age} NATIONAL"
        return age
    return "Unknown competition"


def calendar_label(season: int) -> str:
    return f"{season - 1}/{season}"


def source_hash(row: dict[str, Any], season: int) -> str:
    parts = [
        "eyeball",
        str(season),
        clean_text(row.get("id")) or "",
        clean_text(row.get("clubName")) or "",
        clean_text(row.get("teamName")) or "",
        clean_text(row.get("teamAgeGroup")) or "",
        clean_text(row.get("position")) or "",
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def ratio_percent(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator * 100


def weighted_proxy(parts: list[tuple[float | None, float]]) -> float | None:
    values = [(value, weight) for value, weight in parts if value is not None]
    if not values:
        return None
    weight_sum = sum(weight for _, weight in values)
    if weight_sum <= 0:
        return None
    return sum(float(value) * weight for value, weight in values) / weight_sum


def add_wyscout_metric_proxies(metrics: dict[str, float | None]) -> dict[str, float | None]:
    """
    Keep Eyeball raw metrics, then add Wyscout-compatible metric keys used by
    the production scoring_v2 model. Missing provider concepts are represented
    by conservative semantic proxies, not by zeroes.
    """
    enriched = dict(metrics)
    enriched["matches_played"] = metrics.get("games_count")
    enriched["goals"] = metrics.get("total_goals")
    enriched["assists"] = metrics.get("total_assists")
    enriched["passes_per_90"] = metrics.get("passes_total_per_90")
    enriched["accurate_passes_per_90"] = metrics.get("passes_success_per_90")
    enriched["accurate_passes_percent"] = metrics.get("passes_accuracy_pct")
    enriched["progressive_passes_per_90"] = metrics.get("forward_passes_total_per_90")
    enriched["accurate_progressive_passes_percent"] = ratio_percent(
        metrics.get("forward_passes_success_per_90"),
        metrics.get("forward_passes_total_per_90"),
    )
    enriched["passes_to_final_third_per_90"] = metrics.get("forward_passes_success_per_90")
    enriched["passes_to_penalty_area_per_90"] = weighted_proxy(
        [
            (metrics.get("key_passes_per_90"), 0.65),
            (metrics.get("crosses_success_per_90"), 0.35),
        ]
    )
    enriched["crosses_per_90"] = metrics.get("crosses_total_per_90")
    enriched["accurate_crosses_percent"] = ratio_percent(
        metrics.get("crosses_success_per_90"),
        metrics.get("crosses_total_per_90"),
    )
    enriched["dribbles_per_90"] = metrics.get("takeons_total_per_90")
    enriched["successful_dribbles_percent"] = ratio_percent(
        metrics.get("takeons_success_per_90"),
        metrics.get("takeons_total_per_90"),
    )
    enriched["progressive_runs_per_90"] = metrics.get("takeons_success_per_90")
    enriched["shots_on_target_percent"] = ratio_percent(
        metrics.get("shots_on_target_per_90"),
        metrics.get("shots_per_90"),
    )
    enriched["goal_conversion_rate"] = ratio_percent(
        metrics.get("goals_per_90"),
        metrics.get("shots_per_90"),
    )
    enriched["touches_in_penalty_area_per_90"] = metrics.get("shots_inside_area_per_90")
    enriched["xg_per_90"] = weighted_proxy(
        [
            (metrics.get("shots_inside_area_per_90"), 0.50),
            (metrics.get("shots_on_target_per_90"), 0.35),
            (metrics.get("shots_per_90"), 0.15),
        ]
    )
    enriched["xa_per_90"] = weighted_proxy(
        [
            (metrics.get("key_passes_per_90"), 0.55),
            (metrics.get("assists_per_90"), 0.30),
            (metrics.get("crosses_success_per_90"), 0.15),
        ]
    )
    enriched["shot_assists_per_90"] = metrics.get("key_passes_per_90")
    enriched["smart_passes_per_90"] = metrics.get("key_passes_per_90")
    enriched["through_passes_per_90"] = metrics.get("key_passes_per_90")
    enriched["deep_completions_per_90"] = metrics.get("key_passes_per_90")
    enriched["successful_attacks_per_90"] = weighted_proxy(
        [
            (metrics.get("takeons_success_per_90"), 0.55),
            (metrics.get("key_passes_per_90"), 0.30),
            (metrics.get("crosses_success_per_90"), 0.15),
        ]
    )
    enriched["successful_def_actions_per_90"] = weighted_proxy(
        [
            (metrics.get("tackles_success_per_90"), 0.45),
            (metrics.get("interceptions_per_90"), 0.30),
            (metrics.get("clearances_per_90"), 0.15),
            (metrics.get("blocks_per_90"), 0.10),
        ]
    )
    enriched["def_duels_per_90"] = metrics.get("tackles_total_per_90")
    enriched["def_duels_won_percent"] = ratio_percent(
        metrics.get("tackles_success_per_90"),
        metrics.get("tackles_total_per_90"),
    )
    enriched["aerial_duels_per_90"] = metrics.get("aerial_duels_total_per_90")
    enriched["aerial_duels_won_percent"] = ratio_percent(
        metrics.get("aerial_duels_success_per_90"),
        metrics.get("aerial_duels_total_per_90"),
    )
    enriched["interceptions_padj"] = metrics.get("interceptions_per_90")
    enriched["blocked_shots_per_90"] = metrics.get("blocks_per_90")
    enriched["save_percent"] = ratio_percent(
        weighted_proxy([(metrics.get("catches_per_90"), 0.65), (metrics.get("punches_per_90"), 0.35)]),
        weighted_proxy(
            [
                (metrics.get("catches_per_90"), 0.65),
                (metrics.get("punches_per_90"), 0.35),
                (metrics.get("goals_conceded_per_90"), 1.0),
            ]
        ),
    )
    enriched["goals_prevented_per_90"] = (
        -metrics["goals_conceded_per_90"] if metrics.get("goals_conceded_per_90") is not None else None
    )
    enriched["aerial_duels_gk_per_90"] = metrics.get("aerial_clearances_total_per_90")
    enriched["exits_per_90"] = weighted_proxy(
        [(metrics.get("catches_per_90"), 0.65), (metrics.get("punches_per_90"), 0.35)]
    )
    enriched["successful_goal_kicks_per_90"] = metrics.get("goal_kicks_success_per_90")
    enriched["goal_kicks_per_90"] = metrics.get("goal_kicks_total_per_90")
    enriched["accurate_long_passes_percent"] = enriched.get("accurate_progressive_passes_percent") or metrics.get("passes_accuracy_pct")
    return enriched


def percentile_maps(rows: list[dict[str, Any]], metrics: list[str], key_func) -> dict[tuple[int, str], float]:
    values_by_group: dict[tuple[Any, str], list[float]] = defaultdict(list)
    for row in rows:
        group = key_func(row)
        if group is None:
            continue
        for metric in metrics:
            value = row["metrics"].get(metric)
            if value is not None:
                values_by_group[(group, metric)].append(value)
    sorted_values = {key: sorted(values) for key, values in values_by_group.items() if values}
    output: dict[tuple[int, str], float] = {}
    for index, row in enumerate(rows):
        group = key_func(row)
        if group is None:
            continue
        for metric in metrics:
            values = sorted_values.get((group, metric))
            value = row["metrics"].get(metric)
            if not values or value is None:
                continue
            pct = bisect_right(values, value) / len(values) * 100
            if metric in LOWER_IS_BETTER:
                pct = 100 - pct
            output[(index, metric)] = round(max(0.0, min(100.0, pct)), 4)
    return output


def score_percentiles(rows: list[dict[str, Any]], key_func, score_key: str = "score") -> dict[int, float]:
    values_by_group: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        group = key_func(row)
        if group is not None and row.get(score_key) is not None:
            values_by_group[group].append(row[score_key])
    sorted_values = {key: sorted(values) for key, values in values_by_group.items() if values}
    output: dict[int, float] = {}
    for index, row in enumerate(rows):
        group = key_func(row)
        values = sorted_values.get(group)
        score = row.get(score_key)
        if not values or score is None:
            continue
        output[index] = round(bisect_right(values, score) / len(values) * 100, 4)
    return output


def build_rows(path: Path, season: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            first_name = clean_text(raw.get("firstName"))
            last_name = clean_text(raw.get("lastName"))
            birth_year = parse_birth_year(raw.get("birthday"))
            birth_date = birth_date_label(raw)
            position = clean_text(raw.get("position"))
            resolved_age_category = age_category(raw.get("teamAgeGroup"), birth_year, season)
            metrics = {
                metric: parse_float(raw.get(source))
                for metric, source in METRIC_COLUMNS.items()
            }
            if metrics.get("passes_accuracy_pct") is not None and metrics["passes_accuracy_pct"] <= 1:
                metrics["passes_accuracy_pct"] *= 100
            metrics = add_wyscout_metric_proxies(metrics)
            row = {
                "provider": "eyeball",
                "provider_player_id": clean_text(raw.get("id")) or "",
                "source_row_hash": source_hash(raw, season),
                "provider_player_url": clean_text(raw.get("player_url")),
                "season": season,
                "calendar": calendar_label(season),
                "is_current_season": season == CURRENT_SEASON,
                "country_code": clean_text(raw.get("countryCode")),
                "first_name": first_name,
                "last_name": last_name,
                "display_name": normalize_name(first_name, last_name),
                "birth_year": birth_year,
                "birth_date": birth_date,
                "age": season - birth_year if birth_year else None,
                "age_category": resolved_age_category,
                "championship": championship_label(raw, resolved_age_category),
                "club_name": clean_text(raw.get("clubName")),
                "team_name": clean_text(raw.get("teamName")),
                "team_level": parse_int(raw.get("teamLevel")),
                "position": position,
                "primary_position": primary_position(position),
                "position_group": position_group(position),
                "wyscout_position": wyscout_position(position),
                "strong_foot": clean_text(raw.get("strongFoot")),
                "height_cm": parse_float(raw.get("height")),
                "weight_kg": parse_float(raw.get("weight")),
                "games_count": parse_float(raw.get("gamesCount")),
                "minutes_played": parse_float(raw.get("minutesPlayed")),
                "rating": metrics.get("rating"),
                "metrics": metrics,
                "raw_payload": raw,
            }
            if row["provider_player_id"]:
                rows.append(row)
    return rows


def apply_scores(rows: list[dict[str, Any]]) -> None:
    metric_keys = sorted({key for row in rows for key in (row.get("metrics") or {}).keys()})
    global_metric_pcts = percentile_maps(rows, metric_keys, lambda row: row["position_group"])
    age_metric_pcts = percentile_maps(rows, metric_keys, lambda row: (row["position_group"], row["age_category"]))
    birth_metric_pcts = percentile_maps(rows, metric_keys, lambda row: (row["position_group"], row["birth_year"]))
    champ_metric_pcts = percentile_maps(rows, metric_keys, lambda row: (row["position_group"], row["championship"]))

    scoring_records = []
    for row in rows:
        record = {
            "player_id": row["provider_player_id"],
            "competition_name": row["championship"],
            "calendar": row["calendar"],
            "team_in_selected_period": row["club_name"],
            "position": row.get("wyscout_position"),
            "second_position": None,
            "minutes_played": row.get("minutes_played"),
            "matches_played": row.get("games_count"),
        }
        record.update(row.get("metrics") or {})
        scoring_records.append(record)

    scoring_df = pd.DataFrame(scoring_records)
    scoring_output = scoring_v2.score_dataframe(scoring_df)
    breakdown = scoring_output.get("score_breakdown", pd.DataFrame())

    for index, row in enumerate(rows):
        if not breakdown.empty and index in breakdown.index:
            local_score = breakdown.at[index, "local_score"] if "local_score" in breakdown.columns else None
            final_score = breakdown.at[index, "final_score"] if "final_score" in breakdown.columns else None
            row["score_raw"] = round(float(local_score), 4) if pd.notna(local_score) else None
            row["score"] = round(float(final_score), 4) if pd.notna(final_score) else None
            row["metrics"]["scoring_model_version"] = scoring_v2.SCORE_VERSION
            row["metrics"]["scoring_position_group"] = (
                str(breakdown.at[index, "position_group"]) if "position_group" in breakdown.columns and pd.notna(breakdown.at[index, "position_group"]) else None
            )
            for key in (
                "metric_score",
                "team_strength_z",
                "minutes_confidence",
                "competition_modifier",
                "club_strength_modifier",
                "minutes_regularity_modifier",
                "competition_cap",
                "production_bonus",
            ):
                if key in breakdown.columns and pd.notna(breakdown.at[index, key]):
                    row["metrics"][key] = round(float(breakdown.at[index, key]), 4)
        else:
            row["score_raw"] = None
            row["score"] = None
        row["metric_percentiles"] = {
            metric: {
                "global_position": global_metric_pcts.get((index, metric)),
                "age_category": age_metric_pcts.get((index, metric)),
                "birth_year": birth_metric_pcts.get((index, metric)),
                "championship": champ_metric_pcts.get((index, metric)),
                "lower_is_better": metric in LOWER_IS_BETTER,
            }
            for metric in metric_keys
            if global_metric_pcts.get((index, metric)) is not None
        }

    global_score_pcts = score_percentiles(rows, lambda row: row["position_group"], "score")
    age_score_pcts = score_percentiles(rows, lambda row: (row["position_group"], row["age_category"]), "score")
    birth_score_pcts = score_percentiles(rows, lambda row: (row["position_group"], row["birth_year"]), "score")
    champ_score_pcts = score_percentiles(rows, lambda row: (row["position_group"], row["championship"]), "score")
    for index, row in enumerate(rows):
        row["score_percentile_global"] = global_score_pcts.get(index)
        row["score_percentile_age_category"] = age_score_pcts.get(index)
        row["score_percentile_birth_year"] = birth_score_pcts.get(index)
        row["score_percentile_championship"] = champ_score_pcts.get(index)


def insert_rows(
    rows: list[dict[str, Any]],
    season: int,
    replace: bool,
    force_historical_refresh: bool,
) -> None:
    engine = create_engine(settings.database_url, future=True, connect_args={"prepare_threshold": 0})
    with engine.begin() as conn:
        for statement in [part.strip() for part in SCHEMA_SQL.split(";") if part.strip()]:
            conn.execute(text(statement))
        existing_rows = conn.execute(
            text("SELECT COUNT(*) FROM youth_player_rankings WHERE provider = 'eyeball' AND season = :season"),
            {"season": season},
        ).scalar_one()
        if season != CURRENT_SEASON and existing_rows and not force_historical_refresh:
            raise SystemExit(
                "Historical Eyeball seasons are locked once imported. "
                "Use --force-historical-refresh only when the scoring algorithm intentionally changes."
            )
        if replace:
            conn.execute(
                text("DELETE FROM youth_player_rankings WHERE provider = 'eyeball' AND season = :season"),
                {"season": season},
            )
        sql = text(
            """
            INSERT INTO youth_player_rankings (
              provider, provider_player_id, source_row_hash, provider_player_url,
              season, calendar, is_current_season, country_code, first_name, last_name, display_name,
              birth_year, birth_date, age, age_category, championship, club_name, team_name,
              team_level, position, primary_position, position_group, strong_foot,
              height_cm, weight_kg, games_count, minutes_played, rating,
              score, score_raw, score_percentile_global, score_percentile_age_category,
              score_percentile_birth_year, score_percentile_championship,
              metrics, metric_percentiles, raw_payload, updated_at
            ) VALUES (
              :provider, :provider_player_id, :source_row_hash, :provider_player_url,
              :season, :calendar, :is_current_season, :country_code, :first_name, :last_name, :display_name,
              :birth_year, :birth_date, :age, :age_category, :championship, :club_name, :team_name,
              :team_level, :position, :primary_position, :position_group, :strong_foot,
              :height_cm, :weight_kg, :games_count, :minutes_played, :rating,
              :score, :score_raw, :score_percentile_global, :score_percentile_age_category,
              :score_percentile_birth_year, :score_percentile_championship,
              CAST(:metrics AS JSONB), CAST(:metric_percentiles AS JSONB), CAST(:raw_payload AS JSONB), NOW()
            )
            ON CONFLICT (provider, season, source_row_hash) DO UPDATE SET
              provider_player_url = EXCLUDED.provider_player_url,
              calendar = EXCLUDED.calendar,
              is_current_season = EXCLUDED.is_current_season,
              country_code = EXCLUDED.country_code,
              first_name = EXCLUDED.first_name,
              last_name = EXCLUDED.last_name,
              display_name = EXCLUDED.display_name,
              birth_year = EXCLUDED.birth_year,
              birth_date = EXCLUDED.birth_date,
              age = EXCLUDED.age,
              age_category = EXCLUDED.age_category,
              championship = EXCLUDED.championship,
              club_name = EXCLUDED.club_name,
              team_name = EXCLUDED.team_name,
              team_level = EXCLUDED.team_level,
              position = EXCLUDED.position,
              primary_position = EXCLUDED.primary_position,
              position_group = EXCLUDED.position_group,
              strong_foot = EXCLUDED.strong_foot,
              height_cm = EXCLUDED.height_cm,
              weight_kg = EXCLUDED.weight_kg,
              games_count = EXCLUDED.games_count,
              minutes_played = EXCLUDED.minutes_played,
              rating = EXCLUDED.rating,
              score = EXCLUDED.score,
              score_raw = EXCLUDED.score_raw,
              score_percentile_global = EXCLUDED.score_percentile_global,
              score_percentile_age_category = EXCLUDED.score_percentile_age_category,
              score_percentile_birth_year = EXCLUDED.score_percentile_birth_year,
              score_percentile_championship = EXCLUDED.score_percentile_championship,
              metrics = EXCLUDED.metrics,
              metric_percentiles = EXCLUDED.metric_percentiles,
              raw_payload = EXCLUDED.raw_payload,
              updated_at = NOW()
            """
        )
        payloads = [
            {
                **row,
                "metrics": json.dumps(row["metrics"], ensure_ascii=False),
                "metric_percentiles": json.dumps(row["metric_percentiles"], ensure_ascii=False),
                "raw_payload": json.dumps(row["raw_payload"], ensure_ascii=False),
            }
            for row in rows
        ]
        conn.execute(sql, payloads)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load Eyeball youth CSV into calculated youth_player_rankings.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument("--replace", action="store_true", help="Delete existing rows for the season before loading.")
    parser.add_argument("--no-replace", action="store_true", help="Deprecated compatibility flag. Upsert without delete is now the default.")
    parser.add_argument(
        "--force-historical-refresh",
        action="store_true",
        help="Allow replacing/upserting an already imported historical season after a scoring algorithm change.",
    )
    args = parser.parse_args()

    rows = build_rows(args.csv_path, args.season)
    if not rows:
        raise SystemExit("No rows found in Eyeball CSV.")
    apply_scores(rows)
    insert_rows(
        rows,
        args.season,
        replace=args.replace and not args.no_replace,
        force_historical_refresh=args.force_historical_refresh,
    )
    print(
        json.dumps(
            {
                "table": "youth_player_rankings",
                "rows": len(rows),
                "season": args.season,
                "distinct_provider_players": len({row["provider_player_id"] for row in rows}),
                "position_groups": sorted({row["position_group"] for row in rows}),
                "championships": sorted({row["championship"] for row in rows if row["championship"]}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
