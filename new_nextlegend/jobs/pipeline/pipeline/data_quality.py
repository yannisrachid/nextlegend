from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def analyze_club_player_counts(
    fact: pd.DataFrame,
    *,
    min_players: Optional[int] = None,
    max_alerts: Optional[int] = None,
) -> list[dict[str, object]]:
    """Return low-roster warnings for each competition/calendar/club slice."""
    if fact.empty:
        return []
    required = {"competition_name", "calendar", "team_in_selected_period", "wyscout_id"}
    if not required.issubset(fact.columns):
        return []

    min_players = min_players if min_players is not None else _env_int("DATA_QUALITY_MIN_CLUB_PLAYERS", 15)
    max_alerts = max_alerts if max_alerts is not None else _env_int("DATA_QUALITY_MAX_ALERTS", 80)
    if min_players <= 0:
        return []

    source = fact.copy()
    source["team_in_selected_period"] = source["team_in_selected_period"].astype("string").str.strip()
    source = source[source["team_in_selected_period"].notna() & (source["team_in_selected_period"] != "")]
    counts = (
        source.groupby(["competition_name", "calendar", "team_in_selected_period"], dropna=False)["wyscout_id"]
        .nunique()
        .reset_index(name="player_count")
    )
    low_counts = counts[counts["player_count"] < min_players].sort_values(
        ["player_count", "competition_name", "team_in_selected_period"],
        kind="stable",
    )

    warnings: list[dict[str, object]] = []
    for row in low_counts.head(max(0, max_alerts)).itertuples(index=False):
        warnings.append(
            {
                "competition_name": str(row.competition_name),
                "calendar": str(row.calendar),
                "club": str(row.team_in_selected_period),
                "player_count": int(row.player_count),
                "min_players": int(min_players),
            }
        )
    return warnings


def log_data_quality_warnings(warnings: list[dict[str, object]]) -> None:
    if not warnings:
        print("[DATA-QUALITY] club player-count check passed")
        return
    print(f"[DATA-QUALITY][WARN] low club player-count slices={len(warnings)}")
    for item in warnings:
        print(
            "[DATA-QUALITY][WARN] "
            f"{item['competition_name']} | {item['calendar']} | {item['club']}: "
            f"{item['player_count']}/{item['min_players']} players"
        )
