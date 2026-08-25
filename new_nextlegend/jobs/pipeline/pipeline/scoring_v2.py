from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCORE_VERSION = "position_groups_v2_2"
REFERENCE_LEVEL = "France. Ligue 1"
LOWER_IS_BETTER_PERCENTILE_METRICS = {
    "fouls_per_90",
    "yellow_cards_per_90",
    "red_cards_per_90",
    "goals_conceded_per_90",
    "shots_against_per_90",
    "xg_against_per_90",
}

REPORT_ANALYSIS_METRICS = {
    "accurate_crosses_percent",
    "accurate_long_passes_percent",
    "accurate_passes_percent",
    "accurate_progressive_passes_percent",
    "aerial_duels_gk_per_90",
    "aerial_duels_per_90",
    "aerial_duels_won_percent",
    "accelerations_per_90",
    "assists",
    "assists_per_90",
    "backward_passes_per_90",
    "blocked_shots_per_90",
    "crosses_per_90",
    "def_duels_per_90",
    "def_duels_won_percent",
    "dribbles_per_90",
    "exits_per_90",
    "forward_passes_per_90",
    "fouls_per_90",
    "fouls_suffered_per_90",
    "goal_conversion_rate",
    "goals",
    "goals_conceded_per_90",
    "goals_per_90",
    "goals_prevented_per_90",
    "headed_goals",
    "headed_goals_per_90",
    "interceptions_padj",
    "interceptions_per_90",
    "key_passes_per_90",
    "lateral_passes_per_90",
    "long_passes_per_90",
    "non_penalty_goals_per_90",
    "offensive_duels_per_90",
    "offensive_duels_won_percent",
    "passes_per_90",
    "passes_to_final_third_per_90",
    "passes_to_penalty_area_per_90",
    "progressive_passes_per_90",
    "progressive_runs_per_90",
    "red_cards_per_90",
    "save_percent",
    "shot_assists_per_90",
    "shots_against_per_90",
    "shots_on_target_percent",
    "shots_per_90",
    "sliding_tackles_per_90",
    "smart_passes_per_90",
    "successful_def_actions_per_90",
    "successful_dribbles_percent",
    "touches_in_penalty_area_per_90",
    "xa",
    "xa_per_90",
    "xg",
    "xg_against_per_90",
    "xg_per_90",
    "yellow_cards_per_90",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    weight: float
    metrics: tuple[str, ...]
    family: str = "efficiency"
    lower_is_better: bool = False


@dataclass(frozen=True)
class PositionGroup:
    key: str
    display_name: str
    positions: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]
    elite_traits: tuple[str, ...]
    critical_weaknesses: tuple[str, ...]


def m(key: str, weight: float, family: str = "efficiency", lower_is_better: bool = False) -> MetricSpec:
    return MetricSpec(key=key, weight=weight, metrics=(key,), family=family, lower_is_better=lower_is_better)


def composite(key: str, metrics: Iterable[str], weight: float, family: str = "efficiency") -> MetricSpec:
    return MetricSpec(key=key, weight=weight, metrics=tuple(metrics), family=family)


POSITION_GROUPS: tuple[PositionGroup, ...] = (
    PositionGroup(
        key="goalkeepers",
        display_name="Goalkeepers",
        positions=("GK",),
        metrics=(
            m("goals_prevented_per_90", 25),
            m("def_duels_won_percent", 20),
            m("save_percent", 18),
            composite("aerial_gk_command", ("aerial_duels_won_percent", "aerial_duels_gk_per_90"), 15),
            composite("passing_security_volume", ("accurate_passes_percent", "passes_per_90"), 12, "build_up_volume"),
            composite("progressive_passing_security", ("accurate_progressive_passes_percent", "progressive_passes_per_90"), 10, "build_up_volume"),
        ),
        elite_traits=("goals_prevented_per_90", "save_percent", "aerial_gk_command"),
        critical_weaknesses=("goals_prevented_per_90", "save_percent"),
    ),
    PositionGroup(
        key="centre_backs",
        display_name="Centre Backs",
        positions=("CB", "LCB", "RCB"),
        metrics=(
            m("def_duels_won_percent", 20),
            m("aerial_duels_won_percent", 16),
            m("interceptions_per_90", 10, "defensive_volume"),
            m("successful_def_actions_per_90", 10, "defensive_volume"),
            m("def_duels_per_90", 5, "defensive_volume"),
            m("aerial_duels_per_90", 4, "defensive_volume"),
            m("progressive_passes_per_90", 11, "build_up_volume"),
            m("accurate_progressive_passes_percent", 11),
            m("passes_to_final_third_per_90", 5, "build_up_volume"),
            m("accurate_long_passes_percent", 5),
            m("progressive_runs_per_90", 3, "build_up_volume"),
        ),
        elite_traits=("def_duels_won_percent", "aerial_duels_won_percent", "progressive_passes_per_90"),
        critical_weaknesses=("def_duels_won_percent",),
    ),
    PositionGroup(
        key="left_backs",
        display_name="Left Backs",
        positions=("LB", "LWB"),
        metrics=(
            m("def_duels_won_percent", 14),
            m("successful_def_actions_per_90", 8, "defensive_volume"),
            m("successful_attacks_per_90", 8, "attack_volume"),
            m("interceptions_per_90", 8, "defensive_volume"),
            m("progressive_runs_per_90", 12, "attack_volume"),
            m("successful_dribbles_percent", 6),
            m("progressive_passes_per_90", 10, "build_up_volume"),
            m("passes_to_final_third_per_90", 8, "build_up_volume"),
            m("passes_to_penalty_area_per_90", 6, "attack_volume"),
            m("xa_per_90", 8, "attack_volume"),
            m("accurate_crosses_percent", 8),
            m("crosses_per_90", 4, "attack_volume"),
        ),
        elite_traits=("progressive_runs_per_90", "xa_per_90", "def_duels_won_percent"),
        critical_weaknesses=("def_duels_won_percent",),
    ),
    PositionGroup(
        key="right_backs",
        display_name="Right Backs",
        positions=("RB", "RWB"),
        metrics=(
            m("def_duels_won_percent", 14),
            m("successful_def_actions_per_90", 8, "defensive_volume"),
            m("successful_attacks_per_90", 8, "attack_volume"),
            m("interceptions_per_90", 8, "defensive_volume"),
            m("progressive_runs_per_90", 12, "attack_volume"),
            m("successful_dribbles_percent", 6),
            m("progressive_passes_per_90", 10, "build_up_volume"),
            m("passes_to_final_third_per_90", 8, "build_up_volume"),
            m("passes_to_penalty_area_per_90", 6, "attack_volume"),
            m("xa_per_90", 8, "attack_volume"),
            m("accurate_crosses_percent", 8),
            m("crosses_per_90", 4, "attack_volume"),
        ),
        elite_traits=("progressive_runs_per_90", "xa_per_90", "def_duels_won_percent"),
        critical_weaknesses=("def_duels_won_percent",),
    ),
    PositionGroup(
        key="defensive_midfielders",
        display_name="Defensive Midfielders",
        positions=("DMF", "LDMF", "RDMF"),
        metrics=(
            m("successful_def_actions_per_90", 14, "defensive_volume"),
            m("interceptions_per_90", 12, "defensive_volume"),
            m("def_duels_won_percent", 14),
            m("def_duels_per_90", 5, "defensive_volume"),
            m("aerial_duels_won_percent", 5),
            m("passes_per_90", 6, "build_up_volume"),
            m("accurate_passes_percent", 8),
            m("progressive_passes_per_90", 14, "build_up_volume"),
            m("accurate_progressive_passes_percent", 10),
            m("passes_to_final_third_per_90", 8, "build_up_volume"),
            m("progressive_runs_per_90", 4, "build_up_volume"),
        ),
        elite_traits=("interceptions_per_90", "progressive_passes_per_90", "def_duels_won_percent"),
        critical_weaknesses=("def_duels_won_percent", "accurate_passes_percent"),
    ),
    PositionGroup(
        key="central_midfielders",
        display_name="Central Midfielders",
        positions=("CMF", "LCMF", "RCMF"),
        metrics=(
            m("progressive_passes_per_90", 14, "build_up_volume"),
            m("accurate_progressive_passes_percent", 8),
            m("passes_to_final_third_per_90", 10, "build_up_volume"),
            m("progressive_runs_per_90", 10, "build_up_volume"),
            m("successful_dribbles_percent", 5),
            m("successful_def_actions_per_90", 8, "defensive_volume"),
            m("def_duels_won_percent", 7),
            m("xa_per_90", 10, "attack_volume"),
            m("key_passes_per_90", 8, "attack_volume"),
            m("xg_per_90", 6, "attack_volume"),
            m("goals_per_90", 4, "attack_volume"),
            m("accurate_passes_percent", 6),
            m("passes_to_penalty_area_per_90", 4, "attack_volume"),
        ),
        elite_traits=("progressive_passes_per_90", "xa_per_90", "successful_def_actions_per_90"),
        critical_weaknesses=(),
    ),
    PositionGroup(
        key="attacking_midfielders",
        display_name="Attacking Midfielders",
        positions=("AMF", "LAMF", "RAMF"),
        metrics=(
            m("xa_per_90", 16, "attack_volume"),
            m("key_passes_per_90", 12, "attack_volume"),
            m("smart_passes_per_90", 10, "attack_volume"),
            m("through_passes_per_90", 8, "attack_volume"),
            m("deep_completions_per_90", 8, "attack_volume"),
            m("passes_to_penalty_area_per_90", 10, "attack_volume"),
            m("progressive_passes_per_90", 8, "build_up_volume"),
            m("progressive_runs_per_90", 6, "attack_volume"),
            m("touches_in_penalty_area_per_90", 8, "attack_volume"),
            m("xg_per_90", 7, "attack_volume"),
            m("goals_per_90", 4, "attack_volume"),
            m("successful_dribbles_percent", 3),
        ),
        elite_traits=("xa_per_90", "key_passes_per_90", "smart_passes_per_90"),
        critical_weaknesses=("xa_per_90", "xg_per_90"),
    ),
    PositionGroup(
        key="left_wingers",
        display_name="Left Wingers",
        positions=("LW", "LWF"),
        metrics=(
            m("progressive_runs_per_90", 14, "attack_volume"),
            m("dribbles_per_90", 10, "attack_volume"),
            m("successful_dribbles_percent", 10),
            m("xa_per_90", 12, "attack_volume"),
            m("key_passes_per_90", 8, "attack_volume"),
            m("passes_to_penalty_area_per_90", 6, "attack_volume"),
            m("touches_in_penalty_area_per_90", 10, "attack_volume"),
            m("xg_per_90", 10, "attack_volume"),
            m("goals_per_90", 7, "attack_volume"),
            m("accurate_crosses_percent", 5),
            m("crosses_per_90", 3, "attack_volume"),
            m("successful_def_actions_per_90", 3, "defensive_volume"),
            m("shots_per_90", 2, "attack_volume"),
        ),
        elite_traits=("xg_per_90", "xa_per_90", "progressive_runs_per_90"),
        critical_weaknesses=("xg_per_90", "xa_per_90"),
    ),
    PositionGroup(
        key="right_wingers",
        display_name="Right Wingers",
        positions=("RW", "RWF"),
        metrics=(
            m("progressive_runs_per_90", 14, "attack_volume"),
            m("dribbles_per_90", 10, "attack_volume"),
            m("successful_dribbles_percent", 10),
            m("xa_per_90", 12, "attack_volume"),
            m("key_passes_per_90", 8, "attack_volume"),
            m("passes_to_penalty_area_per_90", 6, "attack_volume"),
            m("touches_in_penalty_area_per_90", 10, "attack_volume"),
            m("xg_per_90", 10, "attack_volume"),
            m("goals_per_90", 7, "attack_volume"),
            m("accurate_crosses_percent", 5),
            m("crosses_per_90", 3, "attack_volume"),
            m("successful_def_actions_per_90", 3, "defensive_volume"),
            m("shots_per_90", 2, "attack_volume"),
        ),
        elite_traits=("xg_per_90", "xa_per_90", "progressive_runs_per_90"),
        critical_weaknesses=("xg_per_90", "xa_per_90"),
    ),
    PositionGroup(
        key="centre_forwards",
        display_name="Centre Forwards",
        positions=("CF",),
        metrics=(
            m("xg_per_90", 22, "attack_volume"),
            m("goals_per_90", 14, "attack_volume"),
            m("shots_per_90", 6, "attack_volume"),
            m("shots_on_target_percent", 6),
            m("goal_conversion_rate", 6),
            m("touches_in_penalty_area_per_90", 14, "attack_volume"),
            m("aerial_duels_won_percent", 5),
            m("aerial_duels_per_90", 4, "attack_volume"),
            m("passes_received_per_90", 8, "attack_volume"),
            m("long_passes_received_per_90", 3, "attack_volume"),
            m("xa_per_90", 5, "attack_volume"),
            m("key_passes_per_90", 3, "attack_volume"),
            m("progressive_runs_per_90", 4, "attack_volume"),
        ),
        elite_traits=("xg_per_90", "touches_in_penalty_area_per_90", "goals_per_90"),
        critical_weaknesses=("xg_per_90",),
    ),
)

POSITION_GROUP_BY_KEY = {group.key: group for group in POSITION_GROUPS}
POSITION_TO_GROUP = {
    position: group.key
    for group in POSITION_GROUPS
    for position in group.positions
}

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "shots_on_target_per_90": ("shots_on_target_per_90", "shots_on_target"),
    "def_duels_won_percent": ("def_duels_won_percent", "Defensive duels won (%)"),
    "aerial_duels_won_percent": ("aerial_duels_won_percent", "Aerial duels won (%)"),
    "accurate_passes_percent": ("accurate_passes_percent", "Accurate passes (%)"),
    "accurate_progressive_passes_percent": (
        "accurate_progressive_passes_percent",
        "Acc. progressive passes (%)",
    ),
    "successful_attacks_per_90": ("successful_attacks_per_90", "successful attacks per 90"),
}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"-": np.nan, "": np.nan}), errors="coerce")


def _value(df: pd.DataFrame, key: str) -> pd.Series:
    for col in METRIC_ALIASES.get(key, (key,)):
        if col in df.columns:
            return _coerce_numeric(df[col])
    return pd.Series(np.nan, index=df.index, dtype=float)


def _positions_for_row(primary: object, secondary: object) -> set[str]:
    positions: set[str] = set()
    for raw in (primary, secondary):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        for token in str(raw).split(","):
            token = token.strip()
            if token:
                positions.add(token)
    return positions


def assign_position_groups(df: pd.DataFrame) -> pd.Series:
    groups = []
    for primary, secondary in zip(df.get("position", ""), df.get("second_position", "")):
        positions = _positions_for_row(primary, secondary)
        group_key = next((POSITION_TO_GROUP[pos] for pos in positions if pos in POSITION_TO_GROUP), None)
        groups.append(group_key)
    return pd.Series(groups, index=df.index, dtype="object")


def minute_confidence(minutes: pd.Series) -> pd.Series:
    numeric = _coerce_numeric(minutes).fillna(0)
    confidence = pd.Series(0.0, index=minutes.index, dtype=float)
    confidence.loc[(numeric >= 90) & (numeric <= 179)] = 0.60
    confidence.loc[(numeric >= 180) & (numeric <= 449)] = 0.70
    confidence.loc[(numeric >= 450) & (numeric <= 899)] = 0.80
    confidence.loc[(numeric >= 900) & (numeric <= 1349)] = 0.90
    confidence.loc[(numeric >= 1350) & (numeric <= 1799)] = 0.96
    confidence.loc[numeric >= 1800] = 1.00
    return confidence


def context_minutes_confidence(df: pd.DataFrame, minutes: pd.Series) -> pd.Series:
    """
    Confidence must work in August/September too.
    A player with 260 minutes when the competition max is 270 has strong early
    evidence, even if 260 minutes is not enough in an absolute full-season view.
    """
    numeric = _coerce_numeric(minutes).fillna(0)
    confidence = minute_confidence(numeric)
    if "competition_name" not in df.columns or "calendar" not in df.columns:
        return confidence

    context = pd.DataFrame(
        {
            "competition": df["competition_name"].astype(str),
            "calendar": df["calendar"].astype(str),
            "minutes": numeric,
        },
        index=df.index,
    )
    max_minutes = context.groupby(["competition", "calendar"])["minutes"].transform("max").fillna(0)
    ratio = numeric / max_minutes.where(max_minutes > 0, np.nan)

    relative = pd.Series(np.nan, index=df.index, dtype=float)
    relative.loc[ratio >= 0.30] = 0.68
    relative.loc[ratio >= 0.50] = 0.78
    relative.loc[ratio >= 0.70] = 0.86
    relative.loc[ratio >= 0.85] = 0.92

    cap = pd.Series(0.0, index=df.index, dtype=float)
    cap.loc[max_minutes >= 90] = 0.75
    cap.loc[max_minutes >= 180] = 0.84
    cap.loc[max_minutes >= 270] = 0.90
    cap.loc[max_minutes >= 450] = 0.96
    cap.loc[max_minutes >= 900] = 1.00

    early_season_confidence = relative.clip(upper=cap)
    return pd.concat([confidence, early_season_confidence], axis=1).max(axis=1).fillna(confidence)


def _robust_z(values: pd.Series, mask: pd.Series) -> pd.Series:
    numeric = _coerce_numeric(values)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    sample = numeric.loc[mask & numeric.notna()]
    if len(sample) < 8:
        result.loc[numeric.notna()] = 0.0
        return result
    low, high = sample.quantile([0.02, 0.98])
    clipped = numeric.clip(lower=low, upper=high)
    sample = clipped.loc[mask & clipped.notna()]
    median = sample.median()
    mad = (sample - median).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = sample.std(ddof=0)
    if not np.isfinite(scale) or scale < 1e-9:
        result.loc[numeric.notna()] = 0.0
        return result
    result.loc[numeric.notna()] = (clipped.loc[numeric.notna()] - median) / scale
    return result.clip(lower=-3.0, upper=3.0)


def _z_to_score(z: pd.Series) -> pd.Series:
    return 100.0 / (1.0 + np.exp(-z / 1.15))


def _metric_score(values: pd.Series, group_mask: pd.Series, lower_is_better: bool = False) -> pd.Series:
    z = _robust_z(values, group_mask)
    if lower_is_better:
        z = -z
    return _z_to_score(z)


def _team_strength_z(df: pd.DataFrame) -> pd.Series:
    team_col = "team_in_selected_period" if "team_in_selected_period" in df.columns else "team" if "team" in df.columns else None
    if not team_col or "competition_name" not in df.columns or "calendar" not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    attack_cols = ["xg_per_90", "shots_per_90", "touches_in_penalty_area_per_90", "passes_per_90", "accurate_passes_percent"]
    parts = []
    for col in attack_cols:
        if col in df.columns:
            parts.append(_metric_score(_value(df, col), pd.Series(True, index=df.index)))
    if not parts:
        return pd.Series(0.0, index=df.index, dtype=float)
    player_attack = pd.concat(parts, axis=1).mean(axis=1)
    tmp = pd.DataFrame(
        {
            "competition": df["competition_name"].astype(str),
            "calendar": df["calendar"].astype(str),
            "team": df[team_col].astype(str),
            "attack": player_attack,
            "minutes": _coerce_numeric(df.get("minutes_played", pd.Series(0, index=df.index))).fillna(0),
        },
        index=df.index,
    )
    tmp["weighted"] = tmp["attack"] * tmp["minutes"].clip(lower=1)
    team = (
        tmp.groupby(["competition", "calendar", "team"], dropna=False)
        .agg(weighted=("weighted", "sum"), minutes=("minutes", "sum"))
        .reset_index()
    )
    team["strength"] = team["weighted"] / team["minutes"].clip(lower=1)
    strength = tmp[["competition", "calendar", "team"]].merge(
        team[["competition", "calendar", "team", "strength"]],
        on=["competition", "calendar", "team"],
        how="left",
    )["strength"]
    strength.index = df.index
    z = strength.groupby(tmp["competition"] + "||" + tmp["calendar"]).transform(
        lambda s: _robust_z(s, pd.Series(True, index=s.index))
    )
    return z.fillna(0.0).clip(lower=-2.5, upper=2.5)


def _context_adjusted_z(metric_score: pd.Series, family: str, team_strength_z: pd.Series) -> pd.Series:
    clipped_score = metric_score.clip(lower=1e-6, upper=99.999999)
    z = 1.15 * np.log(clipped_score / (100.0 - clipped_score))
    if family == "attack_volume":
        z = z - 0.15 * team_strength_z
    elif family == "build_up_volume":
        z = z - 0.20 * team_strength_z
    elif family == "defensive_volume":
        z = z + 0.20 * team_strength_z
    return z.clip(lower=-3.0, upper=3.0)


def _load_league_coefficients() -> dict[str, tuple[float, float, float]]:
    candidate_paths = []
    env_path = os.getenv("MERCATO_LEAGUE_LEVELS_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.extend(
        [
            Path("/config/mercato_league_levels.json"),
            Path("/helpers/csv/mercato_league_levels.json"),
        ]
    )
    here = Path(__file__).resolve()
    candidate_paths.extend(parent / "apps" / "api" / "helpers" / "mercato_league_levels.json" for parent in here.parents)
    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result = {}
    for item in data.get("exact_overrides", []) or []:
        competition = str(item.get("competition") or "").strip()
        if not competition:
            continue
        coefficient = float(item.get("coefficient", 0.75) or 0.75)
        cap = float(item.get("cap", 84) or 84)
        slope = min(1.03, max(0.50, 0.55 + 0.45 * coefficient))
        shift = min(2.0, max(-8.0, (coefficient - 0.90) * 10.0))
        if competition == REFERENCE_LEVEL:
            slope, shift = 1.0, 0.0
        result[competition] = (slope, shift, cap)
    return result


def _league_translate(local_score: pd.Series, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    coefficients = _load_league_coefficients()
    competition = df.get("competition_name", pd.Series("", index=df.index)).astype(str)
    slope = competition.map(lambda c: coefficients.get(c, (0.85, -2.0, 88.0))[0]).astype(float)
    shift = competition.map(lambda c: coefficients.get(c, (0.85, -2.0, 88.0))[1]).astype(float)
    cap = competition.map(lambda c: coefficients.get(c, (0.85, -2.0, 88.0))[2]).astype(float)
    translated = 50.0 + slope * (local_score - 50.0) + shift
    translated = translated.clip(lower=0.0).where(translated <= cap, cap)
    return translated.clip(upper=99.0), slope, shift, cap


def _production_bonus(df: pd.DataFrame) -> pd.Series:
    goals = _value(df, "goals").fillna(0).clip(lower=0)
    assists = _value(df, "assists").fillna(0).clip(lower=0)
    goals_per_90 = _value(df, "goals_per_90").fillna(0).clip(lower=0)
    assists_per_90 = _value(df, "assists_per_90").fillna(0).clip(lower=0)
    production_per_90 = goals_per_90 + assists_per_90

    volume_bonus = np.log1p(goals) * 0.80 + np.log1p(assists) * 0.75
    rate_score = _metric_score(production_per_90, pd.Series(True, index=df.index))
    rate_bonus = ((rate_score - 55.0) / 14.0).clip(lower=0.0, upper=2.5)
    return (volume_bonus + rate_bonus).clip(lower=0.0, upper=7.0)


def _competition_modifier(slope: pd.Series, shift: pd.Series) -> pd.Series:
    modifier = (slope.fillna(0.85) - 0.85) * 6.0 + shift.fillna(-2.0) * 0.25
    return modifier.clip(lower=-2.0, upper=3.0)


def _final_competition_cap(slope: pd.Series, shift: pd.Series) -> pd.Series:
    cap = 82.0 + (slope.fillna(0.85) - 0.75) * 70.0 + shift.fillna(-2.0) * 1.50
    return cap.clip(lower=78.0, upper=99.0)


def _club_strength_modifier(team_strength_z: pd.Series) -> pd.Series:
    return (team_strength_z.fillna(0.0) * 0.90).clip(lower=-1.5, upper=3.0)


def _minutes_regularity_modifier(confidence: pd.Series) -> pd.Series:
    return ((confidence.fillna(0.0) - 0.70) * 6.0).clip(lower=-1.0, upper=2.0)


def _scout_calibrated_score(projected_score: pd.Series) -> pd.Series:
    return 50.0 + (projected_score - 45.0) * 1.15


def _season_sort_key(value: object) -> int:
    text = str(value or "")
    years = [int(item) for item in pd.Series([text]).str.findall(r"20\d{2}").iloc[0]]
    if len(years) >= 2:
        return years[-1] * 10000 + years[0]
    if len(years) == 1:
        return years[0] * 10000 + years[0]
    return 0


def _previous_season_score(df: pd.DataFrame, final_score: pd.Series) -> pd.Series:
    if "player_id" not in df.columns or "calendar" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    season_key = df["calendar"].map(_season_sort_key)
    source = pd.DataFrame(
        {
            "player_id": df["player_id"].astype(str),
            "season_key": season_key,
            "score": final_score,
        },
        index=df.index,
    ).dropna(subset=["score"])
    if source.empty:
        return pd.Series(np.nan, index=df.index, dtype=float)

    season_scores = (
        source.groupby(["player_id", "season_key"], as_index=False)["score"]
        .max()
        .sort_values(["player_id", "season_key"])
    )
    season_scores["previous_season_score"] = season_scores.groupby("player_id")["score"].shift(1)

    lookup = pd.DataFrame(
        {
            "player_id": df["player_id"].astype(str),
            "season_key": season_key,
            "_idx": df.index,
        }
    )
    merged = lookup.merge(
        season_scores[["player_id", "season_key", "previous_season_score"]],
        on=["player_id", "season_key"],
        how="left",
    )
    result = pd.Series(np.nan, index=df.index, dtype=float)
    result.loc[merged["_idx"]] = pd.to_numeric(merged["previous_season_score"], errors="coerce").to_numpy()
    return result


def _build_role_scores(
    df: pd.DataFrame,
    group_names: pd.Series,
    local_score: pd.Series,
    final_score: pd.Series,
) -> pd.DataFrame:
    valid = group_names.notna() & final_score.notna()
    if not valid.any():
        return pd.DataFrame()
    competition_key = df.get("competition_name", pd.Series("", index=df.index)).astype(str)
    calendar_key = df.get("calendar", pd.Series("", index=df.index)).astype(str)
    pct_league = final_score.groupby(competition_key + "||" + calendar_key).rank(pct=True) * 100.0
    pct_global = final_score.rank(pct=True) * 100.0

    team = (
        df["team_in_selected_period"]
        if "team_in_selected_period" in df.columns
        else df["team"]
        if "team" in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    records = []
    for idx in df.index[valid]:
        records.append(
            {
                "wyscout_id": df.at[idx, "player_id"] if "player_id" in df.columns else None,
                "competition_name": df.at[idx, "competition_name"] if "competition_name" in df.columns else None,
                "calendar": df.at[idx, "calendar"] if "calendar" in df.columns else None,
                "team_in_selected_period": team.loc[idx],
                "profile": group_names.loc[idx],
                "raw_score": float(local_score.loc[idx]) if pd.notna(local_score.loc[idx]) else np.nan,
                "pct_league": float(pct_league.loc[idx]) if pd.notna(pct_league.loc[idx]) else np.nan,
                "pct_global": float(pct_global.loc[idx]) if pd.notna(pct_global.loc[idx]) else np.nan,
                "pct_global_adjusted": float(final_score.loc[idx]) if pd.notna(final_score.loc[idx]) else np.nan,
            }
        )
    return pd.DataFrame(records)


def _profile_modifiers(group: PositionGroup, metric_scores: dict[str, pd.Series], group_mask: pd.Series, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    bonus = pd.Series(0.0, index=df.index, dtype=float)
    penalty = pd.Series(0.0, index=df.index, dtype=float)
    for trait in group.elite_traits:
        score = metric_scores.get(trait)
        if score is None:
            continue
        threshold = score.loc[group_mask].quantile(0.95)
        bonus.loc[group_mask & (score >= threshold)] += 1.5
    for weakness in group.critical_weaknesses:
        score = metric_scores.get(weakness)
        if score is None:
            continue
        p20 = score.loc[group_mask].quantile(0.20)
        p10 = score.loc[group_mask].quantile(0.10)
        penalty.loc[group_mask & (score <= p20)] -= 1.5
        penalty.loc[group_mask & (score <= p10)] -= 1.5
    discipline = pd.Series(0.0, index=df.index, dtype=float)
    discipline_metrics = (
        ("fouls_per_90", -1.0),
        ("yellow_cards_per_90", -1.0),
        ("red_cards_per_90", -1.5),
    )
    for key, penalty_value in discipline_metrics:
        values = _value(df, key)
        sample = values.loc[group_mask & values.notna()]
        if len(sample) < 20:
            continue
        p90 = sample.quantile(0.90)
        p95 = sample.quantile(0.95)
        discipline.loc[group_mask & (values >= p90)] += penalty_value
        discipline.loc[group_mask & (values >= p95)] += penalty_value
    return bonus.clip(upper=5.0), penalty.clip(lower=-7.0), discipline.clip(lower=-3.0)


def score_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    if df.empty:
        return {
            "position_group": pd.Series(dtype="object"),
            "position_group_key": pd.Series(dtype="object"),
            "score_breakdown": pd.DataFrame(),
            "role_scores": pd.DataFrame(),
        }

    group_keys = assign_position_groups(df)
    group_names = group_keys.map(lambda key: POSITION_GROUP_BY_KEY[key].display_name if key in POSITION_GROUP_BY_KEY else None)
    team_strength = _team_strength_z(df)
    minutes = _coerce_numeric(df.get("minutes_played", pd.Series(0, index=df.index))).fillna(0)
    confidence = context_minutes_confidence(df, minutes)

    metric_score_cache: dict[tuple[str, str, bool], pd.Series] = {}
    breakdown = pd.DataFrame(index=df.index)
    breakdown["scoring_model_version"] = SCORE_VERSION
    breakdown["position_group_key"] = group_keys
    breakdown["position_group"] = group_names
    breakdown["team_strength_z"] = team_strength
    breakdown["minutes_confidence"] = confidence

    final_score = pd.Series(np.nan, index=df.index, dtype=float)
    local_score_out = pd.Series(np.nan, index=df.index, dtype=float)
    metric_score_out = pd.Series(np.nan, index=df.index, dtype=float)
    bonus_out = pd.Series(0.0, index=df.index, dtype=float)
    penalty_out = pd.Series(0.0, index=df.index, dtype=float)
    discipline_out = pd.Series(0.0, index=df.index, dtype=float)
    league_slope_out = pd.Series(np.nan, index=df.index, dtype=float)
    league_shift_out = pd.Series(np.nan, index=df.index, dtype=float)
    competition_cap_out = pd.Series(np.nan, index=df.index, dtype=float)
    projected_score_out = pd.Series(np.nan, index=df.index, dtype=float)
    competition_modifier_out = pd.Series(0.0, index=df.index, dtype=float)
    club_modifier_out = pd.Series(0.0, index=df.index, dtype=float)
    minutes_modifier_out = pd.Series(0.0, index=df.index, dtype=float)
    production_bonus_out = _production_bonus(df)

    for group in POSITION_GROUPS:
        group_mask = group_keys == group.key
        if not group_mask.any():
            continue

        weighted_components = []
        weight_sum = 0.0
        group_metric_scores: dict[str, pd.Series] = {}
        for spec in group.metrics:
            component_scores = []
            for metric_key in spec.metrics:
                cache_key = (group.key, metric_key, spec.lower_is_better)
                if cache_key not in metric_score_cache:
                    metric_score_cache[cache_key] = _metric_score(
                        _value(df, metric_key),
                        group_mask,
                        lower_is_better=spec.lower_is_better,
                    )
                component_scores.append(metric_score_cache[cache_key])
                group_metric_scores[metric_key] = metric_score_cache[cache_key]
            if not component_scores:
                continue
            component = pd.concat(component_scores, axis=1).mean(axis=1, skipna=True)
            adj_z = _context_adjusted_z(component, spec.family, team_strength)
            # Missing source metrics should be neutral, not a hidden zero-score.
            adjusted_component = _z_to_score(adj_z).where(component.notna(), 50.0)
            weighted_components.append(adjusted_component * float(spec.weight))
            weight_sum += float(spec.weight)

        if not weighted_components or weight_sum <= 0:
            continue

        metric_score = pd.concat(weighted_components, axis=1).sum(axis=1) / weight_sum
        bonus, penalty, discipline = _profile_modifiers(group, group_metric_scores, group_mask, df)
        local_score = (metric_score + bonus + penalty).clip(lower=0.0, upper=100.0)
        reliable_score = 50.0 + confidence * (local_score - 50.0)
        translated, slope, shift, _cap = _league_translate(reliable_score, df)
        competition_modifier = _competition_modifier(slope, shift)
        competition_cap = _final_competition_cap(slope, shift)
        club_modifier = _club_strength_modifier(team_strength)
        minutes_modifier = _minutes_regularity_modifier(confidence)
        calibrated_score = _scout_calibrated_score(translated)
        group_final = (
            calibrated_score
            + competition_modifier
            + club_modifier
            + minutes_modifier
            + production_bonus_out
            + discipline
        ).clip(lower=50.0).where(lambda s: s <= competition_cap, competition_cap)

        final_score.loc[group_mask] = group_final.loc[group_mask]
        local_score_out.loc[group_mask] = local_score.loc[group_mask]
        metric_score_out.loc[group_mask] = metric_score.loc[group_mask]
        bonus_out.loc[group_mask] = bonus.loc[group_mask]
        penalty_out.loc[group_mask] = penalty.loc[group_mask]
        discipline_out.loc[group_mask] = discipline.loc[group_mask]
        league_slope_out.loc[group_mask] = slope.loc[group_mask]
        league_shift_out.loc[group_mask] = shift.loc[group_mask]
        competition_cap_out.loc[group_mask] = competition_cap.loc[group_mask]
        projected_score_out.loc[group_mask] = translated.loc[group_mask]
        competition_modifier_out.loc[group_mask] = competition_modifier.loc[group_mask]
        club_modifier_out.loc[group_mask] = club_modifier.loc[group_mask]
        minutes_modifier_out.loc[group_mask] = minutes_modifier.loc[group_mask]

    previous_score = _previous_season_score(df, final_score)
    previous_modifier = ((previous_score - 70.0) * 0.12).clip(lower=-1.0, upper=4.0)
    final_score = (final_score + previous_modifier.fillna(0.0)).clip(lower=50.0)
    final_score = final_score.where(final_score <= competition_cap_out, competition_cap_out)
    role_scores = _build_role_scores(df, group_names, local_score_out, final_score)

    breakdown["metric_score"] = metric_score_out
    breakdown["profile_bonus"] = bonus_out
    breakdown["profile_penalty"] = penalty_out
    breakdown["discipline_modifier"] = discipline_out
    breakdown["production_bonus"] = production_bonus_out.where(group_names.notna(), np.nan)
    breakdown["local_score"] = local_score_out
    breakdown["projected_score"] = projected_score_out
    breakdown["league_slope"] = league_slope_out
    breakdown["league_shift"] = league_shift_out
    breakdown["competition_cap"] = competition_cap_out
    breakdown["competition_modifier"] = competition_modifier_out.where(group_names.notna(), np.nan)
    breakdown["club_strength_modifier"] = club_modifier_out.where(group_names.notna(), np.nan)
    breakdown["minutes_regularity_modifier"] = minutes_modifier_out.where(group_names.notna(), np.nan)
    breakdown["previous_season_score"] = previous_score
    breakdown["previous_season_modifier"] = previous_modifier
    breakdown["final_score"] = final_score

    return {
        "position_group": group_names,
        "position_group_key": group_keys,
        "score_breakdown": breakdown,
        "role_scores": role_scores,
    }


def scoring_metric_columns() -> list[str]:
    keys: set[str] = set()
    for group in POSITION_GROUPS:
        for spec in group.metrics:
            keys.update(spec.metrics)
    keys.update({"minutes_played", "matches_played"})
    return sorted(keys)


def report_metric_columns() -> list[str]:
    keys = set(scoring_metric_columns())
    keys.update(REPORT_ANALYSIS_METRICS)
    return sorted(keys)


def as_legacy_profiles() -> dict[str, dict[str, object]]:
    """
    Compatibility adapter for the existing similarity/reporting code paths.
    The v2 model has position groups, not roles, but those callers still expect
    profile dictionaries with weights and eligible position groups.
    """
    profiles: dict[str, dict[str, object]] = {}
    for group in POSITION_GROUPS:
        weights: dict[str, float] = {}
        for spec in group.metrics:
            per_metric_weight = float(spec.weight) / max(1, len(spec.metrics))
            for metric_key in spec.metrics:
                weights[metric_key] = weights.get(metric_key, 0.0) + per_metric_weight
        profiles[group.display_name] = {
            "position_groups": list(group.positions),
            "description": f"{group.display_name} position-group v2 scoring profile.",
            "min_minutes": 90,
            "lower_is_better": [],
            "weights": weights,
        }
    return profiles
