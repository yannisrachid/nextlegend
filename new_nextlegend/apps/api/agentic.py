from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_openai import ChatOpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session

DEFAULT_MODEL = "gpt-4o"
MAX_SCOUT_ROWS = 50
CACHE_TTL_SECONDS = 10 * 60

_PROJECT_ROOT = Path(__file__).resolve().parent
METRIC_HINTS_PATH = _PROJECT_ROOT / "helpers" / "ai_metric_hints.json"
PROMPTS_PATH = _PROJECT_ROOT / "helpers" / "ai_prompts.json"
POSITION_GLOSSARY_PATH = _PROJECT_ROOT / "helpers" / "position_glossary.json"

_COLUMN_STATS_CACHE: dict[str, object] = {"timestamp": 0.0, "catalog": ""}

load_dotenv()


class PlayerFilters(BaseModel):
    league: Optional[str] = Field(None, description="Competition or league name.")
    role: Optional[str] = Field(None, description="Assigned role.")
    position: Optional[str] = Field(None, description="Position or positional family.")
    max_age: Optional[int] = Field(None, description="Maximum age.")
    min_minutes: Optional[int] = Field(None, description="Minimum minutes played.")
    min_league_strength: Optional[float] = Field(
        None,
        description="Minimum league strength factor (top leagues).",
    )
    min_minutes_ratio: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Minimum share of max minutes played in league/season.",
    )
    min_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metric -> minimum threshold (per-90, percent, or raw).",
    )
    sort_by: Optional[str] = Field(None, description="Metric to rank by.")
    limit: int = Field(30, description="Max shortlist size, capped.")

    @root_validator(allow_reuse=True)
    def _clamp_limit(cls, values: dict) -> dict:
        limit = values.get("limit") or 30
        values["limit"] = int(min(max(limit, 1), MAX_SCOUT_ROWS))
        return values


class ScoutCandidate(BaseModel):
    player_id: Optional[int] = Field(None, description="Player id from shortlist.")
    player_name: str = Field(..., description="Player display name.")
    priority: int = Field(..., ge=1, le=3, description="1=top target, 2=shortlist, 3=monitor.")
    reason: str = Field(..., description="Concise justification.")
    role_summary: str = Field(..., description="Short role description.")


class ScoutResponse(BaseModel):
    candidates: List[ScoutCandidate]


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _normalize_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKD", text)
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9\\s]", " ", cleaned.lower()).strip()


def _load_position_glossary() -> list[dict]:
    payload = _load_json_file(POSITION_GLOSSARY_PATH)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    return [entry for entry in entries if isinstance(entry, dict)]


@lru_cache(maxsize=1)
def _position_alias_map() -> dict[str, list[str]]:
    entries = _load_position_glossary()
    alias_map: dict[str, set[str]] = {}

    for entry in entries:
        codes = [str(item) for item in entry.get("codes", []) if item]
        synonyms = [str(item) for item in entry.get("synonyms", []) if item]
        canonical = str(entry.get("canonical") or (codes[0] if codes else "")).strip()
        tokens = {canonical, *codes, *synonyms}
        for token in tokens:
            if not token:
                continue
            norm = _normalize_text(token)
            if not norm:
                continue
            alias_map.setdefault(norm, set()).update(tokens)
            alias_map[norm].update(codes)

    for token, synonyms in {
        "winger": {"winger", "rw", "lw", "rm", "lm", "rwf", "lwf", "wide"},
        "goalkeeper": {"goalkeeper", "gk", "keeper"},
        "midfielder": {"midfielder", "mf", "cm", "dm", "am", "midfield"},
        "defender": {"defender", "df", "cb", "rb", "lb", "centre back"},
        "striker": {"striker", "forward", "st", "cf", "fw"},
    }.items():
        norm = _normalize_text(token)
        alias_map.setdefault(norm, set()).update(synonyms)

    return {key: sorted(values) for key, values in alias_map.items()}


def build_position_glossary_text() -> str:
    entries = _load_position_glossary()
    parts = []
    for entry in entries:
        code = entry.get("canonical") or entry.get("codes", [None])[0]
        if not code:
            continue
        fr = entry.get("label_fr") or ""
        en = entry.get("label_en") or ""
        synonyms = entry.get("synonyms", [])
        snippet = f"{code}: {fr} / {en}"
        if synonyms:
            snippet += f" (synonyms: {', '.join(synonyms[:6])})"
        parts.append(snippet)
    return "; ".join(parts[:20])


def resolve_position_from_text(text: str) -> Optional[str]:
    entries = _load_position_glossary()
    if not entries:
        return None
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return None
    for entry in entries:
        canonical = entry.get("canonical") or (entry.get("codes") or [None])[0]
        if not canonical:
            continue
        terms = entry.get("synonyms", []) + entry.get("codes", [])
        terms = sorted({str(t) for t in terms if t}, key=len, reverse=True)
        for term in terms:
            if _normalize_text(term) in normalized_text:
                return str(canonical)
    return None


def _metric_hints() -> list[str]:
    payload = _load_json_file(METRIC_HINTS_PATH)
    raw = payload.get("core_metrics", []) if isinstance(payload, dict) else payload
    return [str(item) for item in raw or []]


def _prompt_templates() -> dict[str, str]:
    payload = _load_json_file(PROMPTS_PATH)
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items() if isinstance(v, str)}


def get_llm(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.25,
    max_output_tokens: int = 900,
) -> ChatOpenAI:
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
    lowered = text.lower()
    french_markers = [" le ", " la ", " les ", " des ", " je ", " cherche", "attaquant", "buteur"]
    accented = re.search(r"[àâçéèêëîïôûùüÿœ]", lowered) is not None
    if accented or any(marker in lowered for marker in french_markers):
        return "fr"
    return "en"


def extract_requested_count(text: str, *, max_value: int = 15) -> Optional[int]:
    lowered = text.lower()
    keyword_patterns = [
        r"\b(\d{1,2})\b\s*(?:joueurs|profils|players|options|candidats|targets)",
        r"(?:propose|donne|liste|list|suggest)[^0-9]{0,20}\b(\d{1,2})\b",
    ]
    for pattern in keyword_patterns:
        match = re.search(pattern, lowered)
        if match:
            value = int(match.group(1))
            if 1 <= value <= max_value:
                return value

    word_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "un": 1,
        "une": 1,
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "six": 6,
        "sept": 7,
        "huit": 8,
        "neuf": 9,
        "dix": 10,
    }
    for word, value in word_map.items():
        if re.search(
            rf"\b{re.escape(word)}\b\s*(?:joueurs|profils|players|options|candidats|targets)?",
            lowered,
        ):
            if 1 <= value <= max_value:
                return value

    numbers = [int(n) for n in re.findall(r"\b(\d{1,2})\b", lowered)]
    for value in numbers:
        if 1 <= value <= max_value:
            return value
    return None


def _get_table_columns(session: Session, table_name: str) -> set[str]:
    rows = session.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :table"
        ),
        {"table": table_name},
    ).fetchall()
    return {row[0] for row in rows}


def _expand_patterns(value: Optional[str]) -> list[str]:
    if not value:
        return []
    key = _normalize_text(value)
    alias_map = _position_alias_map()
    patterns = set()
    if key:
        patterns.update(alias_map.get(key, []))
    if not patterns:
        patterns.add(value.strip())
    return [f"%{item}%" for item in patterns if item]


def _column_stats(session: Session, table: str, column: str, alias: str) -> Optional[str]:
    try:
        sql = text(
            f"""
            SELECT
              MIN({alias}."{column}") AS min_val,
              percentile_cont(0.5) WITHIN GROUP (ORDER BY {alias}."{column}") AS median_val,
              MAX({alias}."{column}") AS max_val
            FROM {table} {alias}
            WHERE {alias}."{column}" IS NOT NULL
            """
        )
        row = session.execute(sql).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return f"{column}: min={row[0]:.2f}, median={row[1]:.2f}, max={row[2]:.2f}"


def build_column_catalog(session: Session) -> str:
    now = time.time()
    cached = _COLUMN_STATS_CACHE
    if cached.get("catalog") and now - float(cached.get("timestamp", 0.0)) < CACHE_TTL_SECONDS:
        return str(cached["catalog"])

    metrics = _metric_hints()
    metric_cols = _get_table_columns(session, "player_metrics")
    season_cols = _get_table_columns(session, "player_seasons")
    catalog_entries: list[str] = []

    for metric in metrics:
        if metric in metric_cols:
            stat = _column_stats(session, "player_metrics", metric, "pm")
            if stat:
                catalog_entries.append(stat)

    for metric in [
        "minutes_played",
        "global_score_adjusted",
        "assigned_role_pct_global",
        "league_strength_factor",
    ]:
        if metric in season_cols:
            stat = _column_stats(session, "player_seasons", metric, "ps")
            if stat:
                catalog_entries.append(stat)

    base_cols = ["competition_name", "assigned_role", "position", "age", "league_strength_factor"]
    catalog_entries.append(f"filterable_fields: {', '.join(base_cols)}")

    catalog = "; ".join(catalog_entries[:25])
    _COLUMN_STATS_CACHE["timestamp"] = now
    _COLUMN_STATS_CACHE["catalog"] = catalog
    return catalog


def run_data_scientist(
    user_text: str,
    column_catalog: str,
    overrides: dict,
    llm: Optional[ChatOpenAI] = None,
) -> PlayerFilters:
    agent = llm or get_llm()
    structured_llm = agent.with_structured_output(PlayerFilters)
    overrides_text = ", ".join(
        f"{key}={value}" for key, value in overrides.items() if value not in (None, "", "None")
    )
    prompts = _prompt_templates()
    system_prompt = prompts.get(
        "data_scientist_system",
        (
            "You are a football data scientist. Convert the scouting brief into structured filters. "
            "Only propose metric thresholds if they are listed in the catalog. "
            "Never ask to see raw tables. Keep results under 50."
        ),
    )
    position_glossary = build_position_glossary_text()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "system",
                "Available columns summary: {column_catalog}. Explicit overrides: {overrides_text}",
            ),
            ("system", "Positions glossary: {position_glossary}"),
            ("human", "{user_text}"),
        ]
    )
    chain = prompt | structured_llm
    return chain.invoke(
        {
            "user_text": user_text,
            "column_catalog": column_catalog,
            "overrides_text": overrides_text or "none",
            "position_glossary": position_glossary or "not provided",
        }
    )


def filter_candidates(session: Session, filters: PlayerFilters) -> list[dict]:
    metric_cols = _get_table_columns(session, "player_metrics")
    season_cols = _get_table_columns(session, "player_seasons")
    clauses = []
    params: Dict[str, object] = {}

    if filters.league:
        league_value = str(filters.league).strip()
        if "*" in league_value or "%" in league_value:
            clauses.append("c.name ILIKE :league_pattern")
            params["league_pattern"] = league_value.replace("*", "%")
        else:
            clauses.append("LOWER(c.name) = LOWER(:league)")
            params["league"] = league_value
    if filters.role:
        role_patterns = _expand_patterns(filters.role) or [f"%{filters.role}%"]
        role_clauses = []
        for idx, pattern in enumerate(role_patterns):
            key = f"role_{idx}"
            params[key] = pattern
            role_clauses.append(f"ps.assigned_role ILIKE :{key}")
        clauses.append("(" + " OR ".join(role_clauses) + ")")
    if filters.position:
        position_patterns = _expand_patterns(filters.position) or [f"%{filters.position}%"]
        position_clauses = []
        for idx, pattern in enumerate(position_patterns):
            key = f"pos_{idx}"
            params[key] = pattern
            position_clauses.append(
                f"(ps.position ILIKE :{key} OR ps.second_position ILIKE :{key})"
            )
        clauses.append("(" + " OR ".join(position_clauses) + ")")
    if filters.max_age is not None:
        clauses.append("pm.age <= :max_age")
        params["max_age"] = filters.max_age
    if filters.min_minutes is not None:
        clauses.append("ps.minutes_played >= :min_minutes")
        params["min_minutes"] = filters.min_minutes
    if filters.min_league_strength is not None:
        clauses.append("ps.league_strength_factor >= :min_league_strength")
        params["min_league_strength"] = filters.min_league_strength
    if filters.min_minutes_ratio is not None:
        clauses.append(
            """
            ps.minutes_played >= :min_minutes_ratio * (
              SELECT MAX(ps2.minutes_played)
              FROM player_seasons ps2
              WHERE ps2.competition_id = ps.competition_id
                AND ps2.season_id = ps.season_id
            )
            """
        )
        params["min_minutes_ratio"] = filters.min_minutes_ratio

    metric_filters = []
    for metric, threshold in filters.min_metrics.items():
        if metric not in metric_cols:
            continue
        key = f"metric_{metric}"
        metric_filters.append(f'pm."{metric}" >= :{key}')
        params[key] = threshold
    clauses.extend(metric_filters)

    where_clause = ""
    if clauses:
        where_clause = "WHERE " + " AND ".join(clauses)

    select_metrics = [metric for metric in _metric_hints() if metric in metric_cols]
    tm_columns = [col for col in season_cols if col.startswith("tm_")]
    select_parts = [
        "ps.id AS player_season_id",
        "p.id AS player_id",
        "p.name AS player_name",
        "p.tm_id AS tm_id",
        "p.tm_profile_url AS tm_profile_url",
        "c.name AS competition_name",
        "ps.calendar AS calendar",
        "ps.team_in_selected_period AS team",
        "ps.position AS position",
        "ps.assigned_role AS assigned_role",
        "ps.minutes_played AS minutes_played",
        "ps.global_score_adjusted AS global_score_adjusted",
        "ps.assigned_role_pct_league AS assigned_role_pct_league",
        "ps.assigned_role_pct_global AS assigned_role_pct_global",
        "pm.age AS age",
    ]
    select_parts.extend([f'pm."{metric}" AS "{metric}"' for metric in select_metrics])
    select_parts.extend([f'ps."{col}" AS "{col}"' for col in tm_columns])

    sort_col = None
    if filters.sort_by:
        if filters.sort_by in metric_cols:
            sort_col = f'pm."{filters.sort_by}"'
        elif filters.sort_by in season_cols:
            sort_col = f'ps."{filters.sort_by}"'
    if not sort_col:
        sort_col = "ps.global_score_adjusted" if "global_score_adjusted" in season_cols else None

    order_clause = f"ORDER BY {sort_col} DESC NULLS LAST" if sort_col else ""

    sql = f"""
        SELECT {", ".join(select_parts)}
        FROM player_seasons ps
        JOIN players p ON p.id = ps.player_id
        JOIN competitions c ON c.id = ps.competition_id
        LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
        {where_clause}
        {order_clause}
        LIMIT :limit
    """
    params["limit"] = filters.limit
    rows = session.execute(text(sql), params).fetchall()
    return [dict(row._mapping) for row in rows]


def select_payload_columns(rows: Iterable[dict]) -> list[str]:
    base = [
        "player_id",
        "player_name",
        "competition_name",
        "team",
        "position",
        "assigned_role",
        "age",
        "minutes_played",
        "global_score_adjusted",
        "accelerations_per_90",
        "progressive_runs_per_90",
        "dribbles_per_90",
        "successful_dribbles_percent",
        "crosses_per_90",
        "accurate_crosses_percent",
        "deep_crosses_per_90",
    ]
    if not rows:
        return base
    available = set(rows[0].keys())
    return [col for col in base if col in available]


def prepare_scout_payload(rows: list[dict]) -> list[dict]:
    columns = select_payload_columns(rows)
    payload: list[dict] = []
    for row in rows:
        payload.append({col: row.get(col) for col in columns})
    return payload


def run_scout_agent(
    user_text: str,
    players: list[dict],
    *,
    language: str = "en",
    llm: Optional[ChatOpenAI] = None,
) -> ScoutResponse:
    agent = llm or get_llm(temperature=0.35)
    structured_llm = agent.with_structured_output(ScoutResponse)
    prompts = _prompt_templates()
    system_prompt = prompts.get(
        "scout_system",
        (
            "You are a professional football scout. Based on the shortlist, "
            "prioritise players aligned with the brief. Return JSON only."
        ),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "Shortlist (JSON): {players_json}"),
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
    player_context: dict,
    *,
    language: str = "en",
    llm: Optional[ChatOpenAI] = None,
) -> str:
    agent = llm or get_llm(temperature=0.4, max_output_tokens=1200)
    prompts = _prompt_templates()
    system_prompt = prompts.get(
        "player_agent_system",
        (
            "You are a player agent preparing a scouting brief. Write 2-3 paragraphs, "
            "professional tone, avoid inventing numbers."
        ),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "Player context (JSON): {player_json}"),
            ("human", "{user_text}"),
        ]
    )
    chain = prompt | agent
    response = chain.invoke(
        {
            "user_text": user_text,
            "player_json": json.dumps(player_context, ensure_ascii=False),
            "language": "French" if language == "fr" else "English",
        }
    )
    return response.content
