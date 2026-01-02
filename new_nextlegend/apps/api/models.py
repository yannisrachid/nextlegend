from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Competition(BaseModel):
    id: int
    name: str
    strength_factor: Optional[float] = None


class Player(BaseModel):
    id: int
    wyscout_id: str
    name: str
    country: Optional[str] = None
    foot: Optional[str] = None
    tm_id: Optional[str] = None
    tm_profile_url: Optional[str] = None


class PlayerSeason(BaseModel):
    id: int
    player_id: int
    competition_id: int
    season_id: int
    club_id: Optional[int] = None
    calendar: Optional[str] = None
    team_in_selected_period: Optional[str] = None
    position: Optional[str] = None
    second_position: Optional[str] = None
    minutes_played: Optional[float] = None
    matches_played: Optional[float] = None
    assigned_role: Optional[str] = None
    assigned_role_pct_league: Optional[float] = None
    assigned_role_pct_global: Optional[float] = None
    global_score_adjusted: Optional[float] = None
    league_strength_factor: Optional[float] = None


class RankingRow(BaseModel):
    player_season_id: int
    player_id: int
    name: str
    competition_name: str
    calendar: str
    team: Optional[str]
    position: Optional[str]
    assigned_role: Optional[str]
    minutes_played: Optional[float]
    matches_played: Optional[float] = None
    global_score_adjusted: Optional[float]
    age: Optional[float] = None
    assigned_role_pct_league: Optional[float] = None
    assigned_role_pct_global: Optional[float] = None
    tm_id: Optional[str] = None
    tm_profile_url: Optional[str] = None
    tm_fields: Optional[dict[str, Optional[float | str]]] = None


class Report(BaseModel):
    player: RankingRow
    metrics: dict[str, Optional[float]]
    raw_metrics: dict[str, Optional[float]]
    radar_metrics: list[str]
    tm_fields: dict[str, Optional[float | str]]
    role_scores: list["RoleScore"]
    summary: dict[str, Optional[float]]


class SimilarityRow(BaseModel):
    player_b_id: int
    player_b_name: str
    team: Optional[str] = None
    competition_name: Optional[str] = None
    calendar: Optional[str] = None
    profile: Optional[str] = None
    similarity: Optional[float] = None
    global_score_adjusted: Optional[float] = None
    assigned_role_pct_league: Optional[float] = None
    assigned_role_pct_global: Optional[float] = None
    age: Optional[float] = None
    tm_fields: Optional[dict[str, Optional[float | str]]] = None


class RoleScore(BaseModel):
    profile: str
    pct_league: Optional[float] = None
    pct_global: Optional[float] = None
    pct_global_adjusted: Optional[float] = None
    raw_score: Optional[float] = None


class RankingPage(BaseModel):
    items: list[RankingRow]
    total: int
    offset: int
    limit: int
