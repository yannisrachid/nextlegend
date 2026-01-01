"""LangChain-based helpers for the NextLegend AI Assistant page."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
MAX_SCOUT_ROWS = 50

load_dotenv()


class PlayerFilters(BaseModel):
    """Structured filters derived from the user request."""

    league: Optional[str] = Field(None, description="League or competition name to restrict.")
    role: Optional[str] = Field(None, description="Target role (assigned_role) if specified.")
    position: Optional[str] = Field(None, description="Positional constraint if relevant.")
    max_age: Optional[int] = Field(None, description="Maximum player age.")
    min_minutes: Optional[int] = Field(None, description="Minimum minutes played.")
    min_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping metric name -> minimum acceptable value or percentile.",
    )
    sort_by: Optional[str] = Field(None, description="Metric used to rank the filtered players.")
    limit: int = Field(30, description="Maximum number of rows to keep (cap to 50).")

    @root_validator(allow_reuse=True)
    def _clamp_values(cls, values: dict) -> dict:
        limit = values.get("limit") or 30
        values["limit"] = int(min(max(limit, 1), MAX_SCOUT_ROWS))
        return values


class ScoutCandidate(BaseModel):
    player_id: Optional[str] = Field(None, description="Player identifier if available.")
    player_name: str = Field(..., description="Display name of the player.")
    priority: int = Field(..., ge=1, le=3, description="1=top target, 2=shortlist, 3=monitor.")
    reason: str = Field(..., description="Short justification tied to the stats provided.")
    role_summary: str = Field(..., description="Concise description of the player's profile.")


class ScoutResponse(BaseModel):
    candidates: List[ScoutCandidate]


def get_llm(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.25,
    max_output_tokens: int = 900,
) -> ChatOpenAI:
    """Create a ChatOpenAI client configured for deterministic, structured outputs."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        api_key=api_key,
    )


def detect_language(text: str) -> str:
    """Crude language detection between French and English."""

    lowered = text.lower()
    french_markers = [" le ", " la ", " les ", " des ", " je ", " cherche", "attaquant", "buteur"]
    accented = re.search(r"[àâçéèêëîïôûùüÿœ]", lowered) is not None
    if accented or any(marker in lowered for marker in french_markers):
        return "fr"
    return "en"


def build_column_catalog(df: pd.DataFrame, extra_metrics: Optional[Iterable[str]] = None) -> str:
    """Return a lightweight description of the columns available to the agent."""

    preferred_columns = [
        "competition_name",
        "league",
        "assigned_role",
        "position",
        "age",
        "minutes_played",
        "minutes",
        "global_score_adjusted",
        "assigned_role_pct_global",
        "summary_finishing",
        "summary_creation",
        "summary_defense",
        "summary_technique",
        "summary_construction",
        "summary_aerial",
        "goals_per_90",
        "xg_per_90",
        "xa_per_90",
        "assists_per_90",
    ]
    if extra_metrics:
        preferred_columns.extend(list(extra_metrics))

    descriptions: List[str] = []
    for column in preferred_columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        desc = f"{column}: min={series.min():.2f}, median={series.median():.2f}, max={series.max():.2f}"
        descriptions.append(desc)
        if len(descriptions) >= 20:
            break
    return "; ".join(descriptions)


def run_data_scientist(
    user_text: str,
    column_catalog: str,
    overrides: dict,
    llm: Optional[ChatOpenAI] = None,
) -> PlayerFilters:
    """Translate free text into structured filters using the Data Scientist agent."""

    agent = llm or get_llm()
    structured_llm = agent.with_structured_output(PlayerFilters)
    overrides_text = ", ".join(
        f"{key}={value}" for key, value in overrides.items() if value not in (None, "", "None")
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a football data scientist. Convert the user's scouting brief into "
                    "structured filters over the player dataset. Only propose numeric thresholds "
                    "for metrics present in the catalog. Never ask to see the full dataset. "
                    "Use conservative limits (cap 50 results)."
                ),
            ),
            (
                "system",
                "Available columns summary: {column_catalog}. Explicit overrides already set: {overrides_text}",
            ),
            ("human", "{user_text}"),
        ]
    )
    chain = prompt | structured_llm
    return chain.invoke(
        {
            "user_text": user_text,
            "column_catalog": column_catalog,
            "overrides_text": overrides_text or "none",
        }
    )


def _get_first_available(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def filter_candidates(df: pd.DataFrame, filters: PlayerFilters) -> pd.DataFrame:
    """Apply structured filters to the dataset safely."""

    working = df.copy()
    player_col = _get_first_available(working, ("player", "player_name"))
    league_col = _get_first_available(working, ("competition_name", "league"))
    minutes_col = _get_first_available(working, ("minutes_played", "minutes"))
    position_col = _get_first_available(working, ("position",))
    role_col = _get_first_available(working, ("assigned_role", "role"))
    age_col = _get_first_available(working, ("age",))

    if filters.league and league_col:
        working = working[working[league_col].astype(str).str.lower() == str(filters.league).lower()]

    if filters.role and role_col:
        working = working[working[role_col].astype(str).str.lower() == str(filters.role).lower()]

    if filters.position and position_col:
        working = working[
            working[position_col].astype(str).str.lower().str.contains(str(filters.position).lower())
        ]

    if filters.max_age is not None and age_col:
        working = working[pd.to_numeric(working[age_col], errors="coerce") <= float(filters.max_age)]

    if filters.min_minutes is not None and minutes_col:
        working = working[pd.to_numeric(working[minutes_col], errors="coerce") >= float(filters.min_minutes)]

    for metric, threshold in filters.min_metrics.items():
        if metric not in working.columns:
            continue
        working = working[pd.to_numeric(working[metric], errors="coerce") >= float(threshold)]

    # Sorting preference
    candidate_sorts = [
        filters.sort_by,
        "global_score_adjusted",
        "assigned_role_pct_global",
        "assigned_role_pct_league",
        "summary_finishing",
        "minutes_played",
        "minutes",
    ]
    sort_col = next((col for col in candidate_sorts if col and col in working.columns), None)
    if sort_col:
        working = working.sort_values(by=sort_col, ascending=False)

    if filters.limit:
        working = working.head(filters.limit)

    if player_col:
        working = working.dropna(subset=[player_col])

    return working


def select_payload_columns(df: pd.DataFrame) -> list[str]:
    """Pick a compact list of columns to send to the LLM."""

    ordered = [
        "player",
        "player_name",
        "player_id",
        "team_in_selected_period",
        "team",
        "competition_name",
        "league",
        "assigned_role",
        "position",
        "age",
        "minutes_played",
        "minutes",
        "global_score_adjusted",
        "summary_finishing",
        "summary_creation",
        "summary_defense",
        "summary_technique",
        "summary_construction",
        "summary_aerial",
        "goals_per_90",
        "xg_per_90",
        "xa_per_90",
        "assists_per_90",
    ]
    return [col for col in ordered if col in df.columns]


def _frame_to_payload(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    payload_rows: list[dict] = []
    for _, row in df.iterrows():
        item = {col: row.get(col, None) for col in columns}
        # Normalise NaN to None for clean JSON
        payload_rows.append(
            {k: (None if pd.isna(v) else v) for k, v in item.items()}
        )
    return payload_rows


def run_scout_agent(
    user_text: str,
    players: list[dict],
    *,
    language: str = "en",
    llm: Optional[ChatOpenAI] = None,
) -> ScoutResponse:
    """Rank the filtered players using the Scout agent."""

    agent = llm or get_llm(temperature=0.35)
    structured_llm = agent.with_structured_output(ScoutResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a professional football scout. Based on the shortlist below, "
                    "prioritise players aligned with the brief. Return JSON only, with "
                    "priorities 1 (top target), 2 (shortlist), 3 (monitor). "
                    "Write reasons in concise {language}."
                ),
            ),
            (
                "system",
                "Shortlist (compact JSON, never request more data): {players_json}",
            ),
            ("human", "{user_text}"),
        ]
    )
    chain = prompt | structured_llm
    return chain.invoke(
        {
            "user_text": user_text,
            "players_json": json.dumps(players, ensure_ascii=False),
            "language": "French" if language == "fr" else "English",
        }
    )


def run_player_agent(
    user_text: str,
    player: dict,
    *,
    language: str = "en",
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """Generate a written report for a single player."""

    agent = llm or get_llm(temperature=0.4, max_output_tokens=1200)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a player agent preparing a scouting brief. "
                    "Write in polished {language}, 2–3 paragraphs, using the stats provided. "
                    "Keep a factual, concise tone and avoid inventing numbers."
                ),
            ),
            ("system", "Player context (JSON): {player_json}"),
            ("human", "{user_text}"),
        ]
    )
    chain = prompt | agent
    response = chain.invoke(
        {
            "user_text": user_text,
            "player_json": json.dumps(player, ensure_ascii=False),
            "language": "French" if language == "fr" else "English",
        }
    )
    return response.content


def prepare_scout_payload(df: pd.DataFrame) -> list[dict]:
    columns = select_payload_columns(df)
    return _frame_to_payload(df, columns)


def summarise_player_row(row: pd.Series) -> dict:
    """Extract a compact, serialisable dict for the player report agent."""

    preferred_cols = select_payload_columns(pd.DataFrame([row]))
    summary = {col: row.get(col) for col in preferred_cols if col in row.index}
    return {k: (None if pd.isna(v) else v) for k, v in summary.items()}
