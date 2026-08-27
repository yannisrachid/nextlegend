from __future__ import annotations

import datetime as dt
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _split_env_list(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.replace(",", " ").split() if part.strip()]


def _missing_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return len(frame)
    series = frame[column]
    missing = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        missing = missing | series.astype("string").str.strip().eq("").fillna(False)
    return int(missing.sum())


def _required_columns(frame: pd.DataFrame, columns: list[str], label: str) -> list[str]:
    return [f"{label}.{column}" for column in columns if column not in frame.columns]


def validate_artifacts(
    *,
    raw: pd.DataFrame,
    artifacts: dict[str, pd.DataFrame],
    strict: Optional[bool] = None,
) -> dict[str, object]:
    """
    Validate data quality/freshness before persistence.

    Fatal checks protect DB integrity. Warning checks are kept in the report and
    pipeline log but do not block ingestion.
    """
    strict = _env_bool("DATA_QUALITY_STRICT", True) if strict is None else strict
    failures: list[str] = []
    warnings: list[str] = []

    fact = artifacts.get("player_seasons", pd.DataFrame())
    metrics = artifacts.get("player_metrics", pd.DataFrame())
    role_scores = artifacts.get("role_scores", pd.DataFrame())
    similarity = artifacts.get("player_similarity", pd.DataFrame())

    raw_required = [
        "player",
        "competition_name",
        "calendar",
        "team_in_selected_period",
        "position",
        "minutes_played",
        "matches_played",
    ]
    fact_required = [
        "wyscout_id",
        "competition_name",
        "calendar",
        "team_in_selected_period",
        "position",
        "minutes_played",
        "matches_played",
        "global_score_adjusted",
        "assigned_role",
    ]
    failures.extend(_required_columns(raw, raw_required, "raw"))
    failures.extend(_required_columns(fact, fact_required, "player_seasons"))

    min_rows = _env_int("DATA_QUALITY_MIN_ROWS", 1)
    if len(raw) < min_rows:
        failures.append(f"raw row count {len(raw)} below DATA_QUALITY_MIN_ROWS={min_rows}")
    if fact.empty:
        failures.append("player_seasons artifact is empty")
    if metrics.empty:
        failures.append("player_metrics artifact is empty")
    if role_scores.empty:
        failures.append("role_scores artifact is empty")

    key_cols = ["wyscout_id", "competition_name", "calendar", "team_in_selected_period"]
    if all(column in fact.columns for column in key_cols):
        missing_key = {column: _missing_count(fact, column) for column in key_cols}
        for column, count in missing_key.items():
            if count:
                failures.append(f"player_seasons key column {column} has {count} missing values")
        duplicate_rows = int(fact.duplicated(subset=key_cols, keep=False).sum())
        if duplicate_rows:
            failures.append(f"player_seasons natural key duplicate rows={duplicate_rows}")

    if "global_score_adjusted" in fact.columns:
        score = pd.to_numeric(fact["global_score_adjusted"], errors="coerce")
        score_null = int(score.isna().sum())
        missing_position = _missing_count(fact, "position") if "position" in fact.columns else 0
        tolerated_null_scores = missing_position
        if score_null > tolerated_null_scores:
            failures.append(
                f"global_score_adjusted has {score_null} null values, tolerated={tolerated_null_scores}"
            )
        below = int((score < 50).sum())
        above = int((score > 99).sum())
        if below or above:
            failures.append(f"global_score_adjusted outside [50,99]: below={below} above={above}")

    max_missing_position_rate = _env_float("DATA_QUALITY_MAX_MISSING_POSITION_RATE", 0.001)
    if "position" in fact.columns and len(fact):
        missing_position_rate = _missing_count(fact, "position") / len(fact)
        if missing_position_rate > max_missing_position_rate:
            failures.append(
                "missing position rate "
                f"{missing_position_rate:.4%} above DATA_QUALITY_MAX_MISSING_POSITION_RATE={max_missing_position_rate:.4%}"
            )

    expected_calendars = _split_env_list("DATA_FRESHNESS_EXPECT_CALENDARS")
    if expected_calendars:
        observed = set(raw.get("calendar", pd.Series(dtype="object")).dropna().astype(str).str.strip())
        missing = [calendar for calendar in expected_calendars if calendar not in observed]
        if missing:
            failures.append(f"expected calendars missing from input: {missing}")

    max_age_hours = _env_float("DATA_FRESHNESS_MAX_INPUT_AGE_HOURS", 0)
    source_mtime = os.getenv("DATA_FRESHNESS_INPUT_MTIME", "").strip()
    if max_age_hours > 0 and source_mtime:
        try:
            mtime = dt.datetime.fromtimestamp(float(source_mtime), tz=dt.timezone.utc)
            age_hours = (dt.datetime.now(tz=dt.timezone.utc) - mtime).total_seconds() / 3600
            if age_hours > max_age_hours:
                failures.append(
                    f"input file age {age_hours:.2f}h above DATA_FRESHNESS_MAX_INPUT_AGE_HOURS={max_age_hours}"
                )
        except ValueError:
            warnings.append(f"invalid DATA_FRESHNESS_INPUT_MTIME={source_mtime}")

    expected_topk = _env_int("DATA_QUALITY_SIM_TOPK", _env_int("SIM_TOPK", 10))
    if expected_topk > 0:
        if similarity.empty:
            warnings.append("player_similarity artifact is empty")
        else:
            sim_required = ["player_a_id", "player_b_id", "competition_a", "calendar_a", "profile", "similarity"]
            failures.extend(_required_columns(similarity, sim_required, "player_similarity"))
            group_cols = [column for column in ["player_a_id", "competition_a", "calendar_a", "team_a", "profile"] if column in similarity.columns]
            if group_cols:
                per_seed = similarity.groupby(group_cols, dropna=False).size()
                over = int((per_seed > expected_topk).sum())
                if over:
                    failures.append(f"player_similarity has {over} seeds above topk={expected_topk}")
            edge_cols = [column for column in ["player_a_id", "player_b_id", "competition_a", "competition_b", "calendar_a", "calendar_b", "profile"] if column in similarity.columns]
            if edge_cols:
                duplicate_edges = int(similarity.duplicated(subset=edge_cols).sum())
                if duplicate_edges:
                    failures.append(f"player_similarity duplicate edges={duplicate_edges}")

    club_warnings = analyze_club_player_counts(fact)
    warnings.extend(
        f"low club player count: {item['competition_name']} | {item['calendar']} | {item['club']} | {item['player_count']}/{item['min_players']}"
        for item in club_warnings
    )

    report = {
        "strict": strict,
        "rows": {
            "raw": int(len(raw)),
            "player_seasons": int(len(fact)),
            "player_metrics": int(len(metrics)),
            "role_scores": int(len(role_scores)),
            "player_similarity": int(len(similarity)),
        },
        "failures": failures,
        "warnings": warnings,
    }

    print(
        "[DATA-QUALITY] "
        f"raw={report['rows']['raw']} player_seasons={report['rows']['player_seasons']} "
        f"metrics={report['rows']['player_metrics']} role_scores={report['rows']['role_scores']} "
        f"similarity={report['rows']['player_similarity']} failures={len(failures)} warnings={len(warnings)}"
    )
    for failure in failures[:50]:
        print(f"[DATA-QUALITY][FAIL] {failure}")
    for warning in warnings[:50]:
        print(f"[DATA-QUALITY][WARN] {warning}")

    if strict and failures:
        raise ValueError("Data quality validation failed")
    return report
