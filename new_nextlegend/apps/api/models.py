from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

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


class ReportSeasonOption(BaseModel):
    player_season_id: int
    calendar: Optional[str] = None
    competition_name: Optional[str] = None
    team: Optional[str] = None
    minutes_played: Optional[float] = None
    global_score_adjusted: Optional[float] = None


class ScoreHistoryPoint(BaseModel):
    player_season_id: int
    calendar: str
    competition_name: Optional[str] = None
    team: Optional[str] = None
    minutes_played: Optional[float] = None
    global_score_adjusted: Optional[float] = None


class Report(BaseModel):
    player: RankingRow
    metrics: dict[str, Optional[float]]
    raw_metrics: dict[str, Optional[float]]
    radar_metrics: list[str]
    tm_fields: dict[str, Optional[float | str]]
    role_scores: list["RoleScore"]
    summary: dict[str, Optional[float]]
    available_seasons: list[ReportSeasonOption] = []
    score_history: list[ScoreHistoryPoint] = []
    similarities_enabled: bool = False
    current_season_label: str = "2025/2026"


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


class AIOverrides(BaseModel):
    league: Optional[str] = None
    season: Optional[str] = None
    role: Optional[str] = None
    position: Optional[str] = None
    max_age: Optional[int] = None
    min_minutes: Optional[int] = None


class AIScoutRequest(BaseModel):
    prompt: str
    overrides: Optional[AIOverrides] = None
    language: Optional[str] = "auto"
    limit: Optional[int] = 30


class AIScoutResponse(BaseModel):
    filters: dict[str, Any]
    shortlist: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    usage: Optional[dict[str, Any]] = None


class AIPlayerReportRequest(BaseModel):
    player_id: int
    player_season_id: Optional[int] = None
    prompt: str
    language: Optional[str] = "auto"


class AIPlayerReportResponse(BaseModel):
    player_id: int
    report: str
    context: dict[str, Any]
    usage: Optional[dict[str, Any]] = None


class AIUsageResponse(BaseModel):
    user_id: str
    conversation_id: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: Optional[str] = None


class AIConversationCreate(BaseModel):
    user_id: str
    title: Optional[str] = None
    mode: Optional[str] = "scout"


class AIConversation(BaseModel):
    id: int
    user_id: str
    title: Optional[str] = None
    mode: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AIConversationList(BaseModel):
    items: list[AIConversation]


class AIMessage(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    payload: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None


class AIConversationDetail(BaseModel):
    conversation: AIConversation
    messages: list[AIMessage]


class AIMessageCreate(BaseModel):
    user_id: str
    prompt: str
    mode: Optional[str] = None
    player_id: Optional[int] = None
    player_season_id: Optional[int] = None
    season: Optional[str] = None
    language: Optional[str] = "auto"


class AIMessageResponse(BaseModel):
    conversation: AIConversation
    user_message: AIMessage
    assistant_message: AIMessage


class AIConversationUpdate(BaseModel):
    user_id: str
    title: Optional[str] = None
