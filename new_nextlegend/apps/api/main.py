from fastapi import FastAPI, Depends, Query, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session
from typing import Any, Optional, List
from pathlib import Path
import os
import secrets
import hashlib
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
import mimetypes
import uuid
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from settings import settings
from db import get_session, SessionLocal
from models import (
    AIConversation,
    AIConversationCreate,
    AIConversationDetail,
    AIConversationList,
    AIConversationUpdate,
    AIMessage,
    AIMessageCreate,
    AIMessageResponse,
    AIPlayerReportRequest,
    AIPlayerReportResponse,
    AIScoutRequest,
    AIScoutResponse,
    AIUsageResponse,
    RankingRow,
    Report,
    SimilarityRow,
    RankingPage,
    ReportSeasonOption,
    RoleScore,
    ScoreHistoryPoint,
    ScoreSnapshotMetric,
    ScoreSnapshotPoint,
    TransferHistoryItem,
)
from agentic import (
    PlayerFilters,
    build_column_catalog,
    detect_language,
    extract_requested_count,
    resolve_position_from_text,
    filter_candidates,
    get_llm,
    prepare_scout_payload,
    run_data_scientist,
    run_player_agent,
    run_scout_agent,
)
from mercato_logic import calculate_calibrated_level, clamp as logic_clamp, safe_float as logic_safe_float
from langchain.callbacks import get_openai_callback
import toml
import re
import unicodedata
import json
import csv
import threading
import io
import zipfile
import html
from bisect import bisect_right

app = FastAPI(title="NextLegend v2 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_PROXY_IMAGE_BYTES = 4 * 1024 * 1024


def _is_public_proxy_host(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True

_MERCATO_SCHEMA_LOCK = threading.Lock()
_MERCATO_SCHEMA_READY = False
_AGENCY_OPS_SCHEMA_LOCK = threading.Lock()
_AGENCY_OPS_SCHEMA_READY = False
_CRM_SCHEMA_LOCK = threading.Lock()
_CRM_SCHEMA_READY = False
_YOUTH_SCHEMA_LOCK = threading.Lock()
_YOUTH_SCHEMA_READY = False


def _auth_json_response(request: Request, detail: str, status_code: int = 401) -> JSONResponse:
    """
    Return auth errors with explicit CORS headers so browser clients can read the
    401 payload and trigger login redirect logic instead of generic CORS failures.
    """
    response = JSONResponse({"detail": detail}, status_code=status_code)
    origin = request.headers.get("origin")
    if origin and origin in settings.cors_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    return response


@app.middleware("http")
async def auth_middleware(request, call_next):
    public_paths = {
        "/",
        "/health",
        "/image-proxy",
        "/auth/login",
        "/auth/logout",
        "/auth/me",
        "/auth/password/forgot",
        "/auth/password/reset",
    }
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path in public_paths or request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
        return await call_next(request)
    session_id = request.cookies.get(AUTH_COOKIE_NAME)
    if not session_id:
        return _auth_json_response(request, "Authentication required", status_code=401)
    db_session = SessionLocal()
    try:
        _ensure_auth_schema(db_session)
        user = _get_session_user(db_session, session_id)
    finally:
        db_session.close()
    if not user:
        return _auth_json_response(request, "Invalid session", status_code=401)
    request.state.user = user
    return await call_next(request)

PROSPECT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prospects (
    id SERIAL PRIMARY KEY,
    player_id INT UNIQUE REFERENCES players(id) ON DELETE CASCADE,
    player_season_id INT REFERENCES player_seasons(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE prospects ADD COLUMN IF NOT EXISTS player_season_id INT REFERENCES player_seasons(id) ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS club_needs (
    id SERIAL PRIMARY KEY,
    club_id INT REFERENCES clubs(id),
    need_label TEXT NOT NULL,
    contact_name TEXT,
    contact_phone TEXT,
    assigned_user TEXT DEFAULT 'admin',
    priority_stage TEXT NOT NULL,
    sort_order INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS club_need_players (
    id SERIAL PRIMARY KEY,
    club_need_id INT REFERENCES club_needs(id) ON DELETE CASCADE,
    player_id INT REFERENCES players(id),
    sort_order INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(club_need_id, player_id)
);

CREATE INDEX IF NOT EXISTS prospects_player_id_idx ON prospects(player_id);
CREATE INDEX IF NOT EXISTS prospects_player_season_id_idx ON prospects(player_season_id);
CREATE INDEX IF NOT EXISTS club_needs_stage_order_idx ON club_needs(priority_stage, sort_order);
CREATE INDEX IF NOT EXISTS club_need_players_order_idx ON club_need_players(club_need_id, sort_order);
"""

AI_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ai_conversations (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,
    mode TEXT NOT NULL DEFAULT 'scout',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_messages (
    id SERIAL PRIMARY KEY,
    conversation_id INT REFERENCES ai_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ai_conversations_user_id_idx ON ai_conversations(user_id);
CREATE INDEX IF NOT EXISTS ai_messages_conversation_idx ON ai_messages(conversation_id);
"""

AUTH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS auth_users (
    username TEXT PRIMARY KEY,
    display_name TEXT,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    password_algo TEXT NOT NULL DEFAULT 'bcrypt',
    role TEXT DEFAULT 'user',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES auth_users(username) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS auth_sessions_user_id_idx ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS auth_users_email_idx ON auth_users(email);

CREATE TABLE IF NOT EXISTS auth_password_reset_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES auth_users(username) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS auth_password_reset_tokens_user_idx ON auth_password_reset_tokens(user_id);
CREATE INDEX IF NOT EXISTS auth_password_reset_tokens_expires_idx ON auth_password_reset_tokens(expires_at);
"""

MERCATO_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mercato_requests (
    id SERIAL PRIMARY KEY,
    club_id INT REFERENCES clubs(id),
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    assigned_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    season TEXT NOT NULL DEFAULT '2026',
    title TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'new',
    budget_min DOUBLE PRECISION,
    budget_max DOUBLE PRECISION,
    salary_max DOUBLE PRECISION,
    deal_type TEXT NOT NULL DEFAULT 'any',
    extra_info TEXT,
    archived_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mercato_needs (
    id SERIAL PRIMARY KEY,
    mercato_request_id INT NOT NULL REFERENCES mercato_requests(id) ON DELETE CASCADE,
    position TEXT,
    role TEXT,
    age_min INT,
    age_max INT,
    preferred_foot TEXT,
    height_min DOUBLE PRECISION,
    target_league_level TEXT,
    required_player_level DOUBLE PRECISION,
    nationality_preferences TEXT,
    contract_preferences TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mercato_candidates (
    id SERIAL PRIMARY KEY,
    mercato_need_id INT NOT NULL REFERENCES mercato_needs(id) ON DELETE CASCADE,
    player_id INT NOT NULL REFERENCES players(id),
    player_season_id INT,
    source TEXT NOT NULL DEFAULT 'manual',
    status TEXT NOT NULL DEFAULT 'suggested',
    match_score DOUBLE PRECISION,
    calibrated_player_level DOUBLE PRECISION,
    raw_player_level DOUBLE PRECISION,
    league_coefficient DOUBLE PRECISION,
    explanation_json JSONB,
    agent_note TEXT,
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(mercato_need_id, player_id)
);

CREATE TABLE IF NOT EXISTS mercato_candidate_events (
    id SERIAL PRIMARY KEY,
    mercato_candidate_id INT NOT NULL REFERENCES mercato_candidates(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT,
    note TEXT,
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS mercato_requests_club_status_idx ON mercato_requests(club_id, status);
CREATE INDEX IF NOT EXISTS mercato_requests_agent_idx ON mercato_requests(assigned_agent_id);
CREATE INDEX IF NOT EXISTS mercato_needs_request_idx ON mercato_needs(mercato_request_id);
CREATE INDEX IF NOT EXISTS mercato_candidates_need_status_idx ON mercato_candidates(mercato_need_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS mercato_requests_active_dedupe_idx ON mercato_requests (
    club_id,
    LOWER(BTRIM(title)),
    COALESCE(assigned_agent_id, ''),
    season
) WHERE archived_at IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS mercato_needs_dedupe_idx ON mercato_needs (
    mercato_request_id,
    COALESCE(LOWER(BTRIM(position)), ''),
    COALESCE(LOWER(BTRIM(role)), '')
);
"""

AGENCY_OPS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hq_priority_items (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    agent_name TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'planned',
    start_date DATE NOT NULL DEFAULT CURRENT_DATE,
    end_date DATE,
    color TEXT,
    related_page TEXT,
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    updated_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hq_calendar_events (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    event_type TEXT NOT NULL DEFAULT 'team',
    agent_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    start_date DATE NOT NULL,
    end_date DATE,
    location TEXT,
    color TEXT,
    related_page TEXT,
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    updated_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE hq_calendar_events ADD COLUMN IF NOT EXISTS event_type TEXT NOT NULL DEFAULT 'team';
ALTER TABLE hq_calendar_events ADD COLUMN IF NOT EXISTS agent_names JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE hq_calendar_events ADD COLUMN IF NOT EXISTS end_date DATE;
ALTER TABLE hq_calendar_events ADD COLUMN IF NOT EXISTS location TEXT;
ALTER TABLE hq_calendar_events ADD COLUMN IF NOT EXISTS color TEXT;
ALTER TABLE hq_calendar_events ADD COLUMN IF NOT EXISTS related_page TEXT;
ALTER TABLE hq_calendar_events ADD COLUMN IF NOT EXISTS updated_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL;
UPDATE hq_calendar_events SET end_date = start_date WHERE end_date IS NOT NULL AND end_date < start_date;
ALTER TABLE hq_calendar_events DROP CONSTRAINT IF EXISTS hq_calendar_events_valid_date_range;
ALTER TABLE hq_calendar_events
ADD CONSTRAINT hq_calendar_events_valid_date_range
CHECK (end_date IS NULL OR end_date >= start_date);

CREATE TABLE IF NOT EXISTS hd_players (
    id SERIAL PRIMARY KEY,
    player_id INT REFERENCES players(id) ON DELETE SET NULL,
    display_name TEXT NOT NULL,
    position TEXT,
    current_club TEXT,
    contract_expiry DATE,
    current_club_situation TEXT,
    plan TEXT,
    priority TEXT NOT NULL DEFAULT 'B',
    demanded_transfer_fee DOUBLE PRECISION,
    next_step TEXT,
    assigned_agent TEXT,
    photo_url TEXT,
    birth_date DATE,
    player_phone TEXT,
    player_email TEXT,
    entourage_phone TEXT,
    entourage_email TEXT,
    season_objectives TEXT,
    eyeball_url TEXT,
    transfermarkt_url TEXT,
    is_young_player BOOLEAN NOT NULL DEFAULT FALSE,
    manual_performance JSONB NOT NULL DEFAULT '{}'::jsonb,
    contract_status TEXT,
    mandate_status TEXT,
    medical_status TEXT,
    market_notes TEXT,
    scouting_notes TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    updated_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS player_phone TEXT;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS player_email TEXT;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS entourage_phone TEXT;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS entourage_email TEXT;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS season_objectives TEXT;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS eyeball_url TEXT;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS transfermarkt_url TEXT;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS is_young_player BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS manual_performance JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE hd_players ADD COLUMN IF NOT EXISTS birth_date DATE;

CREATE TABLE IF NOT EXISTS hd_player_documents (
    id SERIAL PRIMARY KEY,
    hd_player_id INT REFERENCES hd_players(id) ON DELETE CASCADE,
    player_id INT REFERENCES players(id) ON DELETE SET NULL,
    document_type TEXT NOT NULL DEFAULT 'other',
    title TEXT NOT NULL,
    file_name TEXT,
    file_key TEXT,
    storage_url TEXT,
    content_type TEXT,
    size_bytes BIGINT,
    notes TEXT,
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hd_player_prospect_clubs (
    id SERIAL PRIMARY KEY,
    hd_player_id INT NOT NULL REFERENCES hd_players(id) ON DELETE CASCADE,
    club_id INT REFERENCES clubs(id) ON DELETE SET NULL,
    club_name TEXT NOT NULL,
    competition_name TEXT,
    status TEXT,
    offer TEXT,
    contact TEXT,
    notes TEXT,
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hd_player_manual_transfers (
    id SERIAL PRIMARY KEY,
    hd_player_id INT NOT NULL REFERENCES hd_players(id) ON DELETE CASCADE,
    transfer_date DATE,
    transfer_type TEXT,
    transfer_fee TEXT,
    team_in_name TEXT,
    team_out_name TEXT,
    league_name TEXT,
    notes TEXT,
    created_by_agent_id TEXT REFERENCES auth_users(username) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS player_transfer_history (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'transferts.xlsx',
    source_player_id INT,
    linked_player_id INT REFERENCES players(id) ON DELETE SET NULL,
    normalized_player_name TEXT,
    player_name TEXT NOT NULL,
    league_id INT,
    league_name TEXT,
    team_id_context INT,
    team_name_context TEXT,
    transfer_date DATE,
    transfer_type TEXT,
    transfer_fee TEXT,
    team_in_id INT,
    team_in_name TEXT,
    team_out_id INT,
    team_out_name TEXT,
    transfer_date_serial DOUBLE PRECISION,
    raw_payload JSONB,
    imported_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS hq_priority_items_agent_date_idx ON hq_priority_items(agent_name, start_date);
CREATE INDEX IF NOT EXISTS hq_calendar_events_date_idx ON hq_calendar_events(start_date, end_date);
CREATE INDEX IF NOT EXISTS hq_calendar_events_created_by_idx ON hq_calendar_events(created_by_agent_id);
CREATE INDEX IF NOT EXISTS hd_players_player_id_idx ON hd_players(player_id);
CREATE INDEX IF NOT EXISTS hd_players_agent_priority_idx ON hd_players(assigned_agent, priority);
CREATE UNIQUE INDEX IF NOT EXISTS hd_players_active_player_unique_idx ON hd_players(player_id)
WHERE player_id IS NOT NULL AND status <> 'archived';
CREATE UNIQUE INDEX IF NOT EXISTS hd_players_active_name_unique_idx ON hd_players(LOWER(BTRIM(display_name)))
WHERE status <> 'archived';
CREATE INDEX IF NOT EXISTS hd_player_documents_hd_player_idx ON hd_player_documents(hd_player_id);
CREATE INDEX IF NOT EXISTS hd_player_prospect_clubs_hd_player_idx ON hd_player_prospect_clubs(hd_player_id);
CREATE INDEX IF NOT EXISTS hd_player_manual_transfers_hd_player_idx ON hd_player_manual_transfers(hd_player_id);
CREATE INDEX IF NOT EXISTS player_transfer_history_source_player_idx ON player_transfer_history(source_player_id);
CREATE INDEX IF NOT EXISTS player_transfer_history_linked_player_idx ON player_transfer_history(linked_player_id);
CREATE INDEX IF NOT EXISTS player_transfer_history_normalized_name_idx ON player_transfer_history(normalized_player_name);
CREATE INDEX IF NOT EXISTS player_transfer_history_date_idx ON player_transfer_history(transfer_date DESC NULLS LAST);
CREATE UNIQUE INDEX IF NOT EXISTS player_transfer_history_unique_idx ON player_transfer_history (
    source,
    COALESCE(source_player_id, -1),
    player_name,
    COALESCE(transfer_date, DATE '1900-01-01'),
    COALESCE(transfer_type, ''),
    COALESCE(team_in_name, ''),
    COALESCE(team_out_name, ''),
    COALESCE(transfer_fee, '')
);
"""

CRM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crm_clubs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    logo TEXT,
    email TEXT,
    phone TEXT,
    website TEXT,
    source TEXT NOT NULL DEFAULT 'nextlegend',
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS crm_players (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    age INT NOT NULL DEFAULT 0,
    position TEXT NOT NULL,
    nationality TEXT NOT NULL,
    photo TEXT,
    email TEXT,
    phone TEXT,
    club_id TEXT NOT NULL REFERENCES crm_clubs(id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'nextlegend',
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS crm_contacts (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    role TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    type TEXT NOT NULL DEFAULT 'CLUB',
    notes TEXT,
    club_id TEXT REFERENCES crm_clubs(id) ON DELETE SET NULL,
    player_id TEXT REFERENCES crm_players(id) ON DELETE SET NULL,
    source TEXT NOT NULL DEFAULT 'nextlegend',
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT crm_contacts_type_check CHECK (type IN ('CLUB', 'PLAYER'))
);

CREATE TABLE IF NOT EXISTS crm_prospects (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL DEFAULT 'prequalification',
    notes TEXT,
    contact_id TEXT NOT NULL REFERENCES crm_contacts(id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'nextlegend',
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT crm_prospects_stage_check CHECK (stage IN ('prequalification', 'relance1', 'relance2', 'relance3'))
);

CREATE INDEX IF NOT EXISTS crm_clubs_search_idx ON crm_clubs (LOWER(name), LOWER(city), LOWER(country));
CREATE INDEX IF NOT EXISTS crm_players_club_idx ON crm_players (club_id);
CREATE INDEX IF NOT EXISTS crm_contacts_club_idx ON crm_contacts (club_id);
CREATE INDEX IF NOT EXISTS crm_contacts_player_idx ON crm_contacts (player_id);
CREATE INDEX IF NOT EXISTS crm_contacts_type_idx ON crm_contacts (type);
CREATE INDEX IF NOT EXISTS crm_prospects_contact_idx ON crm_prospects (contact_id);
CREATE INDEX IF NOT EXISTS crm_prospects_stage_idx ON crm_prospects (stage);
"""

YOUTH_SCHEMA_SQL = """
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

CREATE TABLE IF NOT EXISTS youth_prospects (
    id SERIAL PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'eyeball',
    season INT NOT NULL,
    source_row_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(provider, season, source_row_hash)
);

CREATE INDEX IF NOT EXISTS youth_prospects_lookup_idx
    ON youth_prospects(provider, season, source_row_hash);
"""


class CrmClubPayload(BaseModel):
    name: str
    city: str
    country: str
    logo: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None


class CrmPlayerPayload(BaseModel):
    first_name: str
    last_name: str = ""
    age: int = 0
    position: str
    nationality: str
    club_id: str
    photo: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class CrmContactPayload(BaseModel):
    first_name: str
    last_name: str
    role: str
    type: str = "CLUB"
    club_id: Optional[str] = None
    player_id: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None


class CrmProspectPayload(BaseModel):
    contact_id: str
    stage: str = "prequalification"
    notes: Optional[str] = None


AUTH_COOKIE_NAME = "nl_session"
DEFAULT_SESSION_DAYS = 365
PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
LEGACY_SHA256 = "sha256"
ADMIN_USERNAME = "yrachid"
CURRENT_SEASON_LABEL = os.getenv("CURRENT_SEASON_LABEL", "2025/2026")


def _season_bounds(label: Optional[str]) -> tuple[int, int]:
    if label is None:
        return (0, 0)
    raw = str(label).strip()
    if not raw:
        return (0, 0)

    full_range = re.search(r"(\d{4})\s*[/\-_]\s*(\d{4})", raw)
    if full_range:
        return (int(full_range.group(1)), int(full_range.group(2)))

    short_range = re.search(r"(\d{4})\s*[/\-_]\s*(\d{2})", raw)
    if short_range:
        start = int(short_range.group(1))
        end_short = int(short_range.group(2))
        end = (start // 100) * 100 + end_short
        if end < start:
            end += 100
        return (start, end)

    years = [int(item) for item in re.findall(r"\d{4}", raw)]
    if not years:
        return (0, 0)
    if len(years) == 1:
        return (years[0], years[0])
    return (years[0], years[-1])


def _season_sort_key_desc(label: Optional[str]) -> tuple[int, int, str]:
    start, end = _season_bounds(label)
    return (end, start, str(label or ""))


def _season_sort_key_asc(label: Optional[str]) -> tuple[int, int, str]:
    start, end = _season_bounds(label)
    return (start, end, str(label or ""))


def _season_filter_values(value: Optional[str]) -> list[str]:
    if not value:
        return []
    raw = str(value).strip()
    if not raw:
        return []
    labels = {raw}
    start, end = _season_bounds(raw)
    if start and end:
        labels.add(f"{start}/{end}")
        labels.add(f"{start}-{end}")
        labels.add(f"{start}/{str(end)[-2:]}")
        labels.add(str(start))
        labels.add(str(end))
    return [label for label in sorted(labels) if label]


def _apply_season_filter(
    sql: str,
    params: dict,
    season: Optional[str],
    *,
    column: str = "ps.calendar",
    param_prefix: str = "season",
) -> tuple[str, dict]:
    if not season:
        return sql, params
    season_values = _season_filter_values(season)
    if len(season_values) <= 1:
        sql += f" AND {column} = :{param_prefix}"
        params[param_prefix] = season_values[0] if season_values else season
    else:
        key = f"{param_prefix}_values"
        sql += f" AND {column} = ANY(:{key})"
        params[key] = season_values
    return sql, params


def _current_season_labels() -> list[str]:
    return _season_filter_values(CURRENT_SEASON_LABEL)


def _load_player_season_context(session: Session, player_id: int) -> list[dict]:
    rows = session.execute(
        text(
            """
            SELECT
              ps.id AS player_season_id,
              ps.calendar,
              ps.minutes_played,
              ps.global_score_adjusted,
              ps.team_in_selected_period AS team,
              c.name AS competition_name
            FROM player_seasons ps
            JOIN competitions c ON c.id = ps.competition_id
            WHERE ps.player_id = :player_id
            """
        ),
        {"player_id": player_id},
    ).fetchall()
    items = [_row_to_dict(row) for row in rows]
    items.sort(
        key=lambda item: (
            _season_sort_key_desc(item.get("calendar")),
            float(item.get("minutes_played") or -1),
            str(item.get("competition_name") or ""),
        ),
        reverse=True,
    )
    return items


def _build_score_history(season_items: list[dict]) -> list[ScoreHistoryPoint]:
    by_calendar: dict[str, dict] = {}
    for item in season_items:
        calendar = str(item.get("calendar") or "").strip()
        if not calendar:
            continue
        current_best = by_calendar.get(calendar)
        minutes = float(item.get("minutes_played") or -1)
        if current_best is None or minutes > float(current_best.get("minutes_played") or -1):
            by_calendar[calendar] = item

    history = [
        ScoreHistoryPoint(
            player_season_id=int(item["player_season_id"]),
            calendar=calendar,
            competition_name=item.get("competition_name"),
            team=item.get("team"),
            minutes_played=item.get("minutes_played"),
            global_score_adjusted=item.get("global_score_adjusted"),
        )
        for calendar, item in by_calendar.items()
    ]
    history.sort(key=lambda point: _season_sort_key_asc(point.calendar))
    return history


def _build_season_metric_history(session: Session, score_history: list[ScoreHistoryPoint]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for point in score_history:
        metrics_row = session.execute(
            text("SELECT * FROM player_metrics WHERE player_season_id = :psid"),
            {"psid": point.player_season_id},
        ).fetchone()
        metrics = _row_to_dict(metrics_row) if metrics_row else {}
        metrics.pop("player_season_id", None)
        metrics.pop("created_at", None)
        metrics.pop("updated_at", None)
        metrics = _hydrate_report_metric_percentiles(session, int(point.player_season_id), metrics)
        items.append(
            {
                "player_season_id": point.player_season_id,
                "calendar": point.calendar,
                "competition_name": point.competition_name,
                "team": point.team,
                "minutes_played": point.minutes_played,
                "global_score_adjusted": point.global_score_adjusted,
                "metrics": metrics,
            }
        )
    return items


def _load_score_snapshots(
    session: Session,
    *,
    player_id: int,
    player_season_id: Optional[int] = None,
) -> list[ScoreSnapshotPoint]:
    try:
        if not (
            _table_exists(session, "scoring_snapshot_runs")
            and _table_exists(session, "player_score_snapshots")
            and _table_exists(session, "player_metric_snapshots")
        ):
            return []
        sql = """
        SELECT
          ss.id AS score_snapshot_id,
          sr.snapshot_key,
          sr.snapshot_date,
          sr.snapshot_at,
          sr.season_label AS calendar,
          c.name AS competition_name,
          cl.name AS team,
          ss.position,
          ss.position_group,
          ss.minutes_played,
          ss.matches_played,
          ss.minutes_possible,
          ss.minutes_ratio,
          ss.global_score_adjusted,
          ss.assigned_role_pct_league,
          ss.assigned_role_pct_global,
          ss.league_strength_factor,
          ss.team_strength_z,
          ss.club_strength_modifier,
          ss.minutes_regularity_modifier,
          sr.scoring_model_version
        FROM player_score_snapshots ss
        JOIN scoring_snapshot_runs sr ON sr.id = ss.snapshot_run_id
        JOIN player_seasons ps ON ps.id = ss.player_season_id
        JOIN competitions c ON c.id = ss.competition_id
        LEFT JOIN clubs cl ON cl.id = ss.club_id
        WHERE ss.player_id = :player_id
        """
        params: dict[str, Any] = {"player_id": player_id}
        if player_season_id is not None:
            sql += " AND ss.player_season_id = :player_season_id"
            params["player_season_id"] = player_season_id
        sql += " ORDER BY sr.snapshot_date ASC, sr.snapshot_at ASC"
        rows = [_row_to_dict(row) for row in session.execute(text(sql), params).fetchall()]
        if not rows:
            return []

        snapshot_ids = [int(row["score_snapshot_id"]) for row in rows]
        metrics_sql = (
            text(
                """
                SELECT
                  score_snapshot_id,
                  metric_key,
                  raw_value,
                  percentile_global,
                  percentile_league,
                  metric_weight,
                  metric_family,
                  lower_is_better
                FROM player_metric_snapshots
                WHERE score_snapshot_id IN :snapshot_ids
                ORDER BY score_snapshot_id, metric_weight DESC NULLS LAST, metric_key ASC
                """
            )
            .bindparams(bindparam("snapshot_ids", expanding=True))
        )
        metric_rows = session.execute(metrics_sql, {"snapshot_ids": snapshot_ids}).fetchall()
        metrics_by_snapshot: dict[int, list[ScoreSnapshotMetric]] = {snapshot_id: [] for snapshot_id in snapshot_ids}
        for metric_row in metric_rows:
            item = _row_to_dict(metric_row)
            snapshot_id = int(item.pop("score_snapshot_id"))
            metrics_by_snapshot.setdefault(snapshot_id, []).append(ScoreSnapshotMetric(**item))

        snapshots = []
        for row in rows:
            snapshot_id = int(row["score_snapshot_id"])
            snapshots.append(
                ScoreSnapshotPoint(
                    **row,
                    metrics=metrics_by_snapshot.get(snapshot_id, []),
                )
            )
        return snapshots
    except Exception:
        session.rollback()
        return []


def _ensure_prospect_schema(session: Session) -> None:
    statements = [chunk.strip() for chunk in PROSPECT_SCHEMA_SQL.split(";") if chunk.strip()]
    for statement in statements:
        session.execute(text(statement))
    session.commit()


def _ensure_ai_schema(session: Session) -> None:
    statements = [chunk.strip() for chunk in AI_SCHEMA_SQL.split(";") if chunk.strip()]
    for statement in statements:
        session.execute(text(statement))
    session.commit()


def _ensure_auth_schema(session: Session) -> None:
    if not _table_exists(session, "auth_users"):
        _drop_orphan_type(session, "auth_users")
        session.execute(
            text(
                """
                CREATE TABLE auth_users (
                    username TEXT PRIMARY KEY,
                    display_name TEXT,
                    email TEXT UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_algo TEXT NOT NULL DEFAULT 'bcrypt',
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    last_login TIMESTAMPTZ
                )
                """
            )
        )
    if not _table_exists(session, "auth_sessions"):
        _drop_orphan_type(session, "auth_sessions")
        session.execute(
            text(
                """
                CREATE TABLE auth_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES auth_users(username) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_seen TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ
                )
                """
            )
        )
    if not _table_exists(session, "auth_password_reset_tokens"):
        _drop_orphan_type(session, "auth_password_reset_tokens")
        session.execute(
            text(
                """
                CREATE TABLE auth_password_reset_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES auth_users(username) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL,
                    used_at TIMESTAMPTZ
                )
                """
            )
        )
    if _table_exists(session, "auth_sessions"):
        session.execute(
            text("CREATE INDEX IF NOT EXISTS auth_sessions_user_id_idx ON auth_sessions(user_id)")
        )
    if _table_exists(session, "auth_users"):
        session.execute(
            text("CREATE INDEX IF NOT EXISTS auth_users_email_idx ON auth_users(email)")
        )
    if _table_exists(session, "auth_password_reset_tokens"):
        session.execute(
            text("CREATE INDEX IF NOT EXISTS auth_password_reset_tokens_user_idx ON auth_password_reset_tokens(user_id)")
        )
        session.execute(
            text("CREATE INDEX IF NOT EXISTS auth_password_reset_tokens_expires_idx ON auth_password_reset_tokens(expires_at)")
        )
    session.commit()
    _seed_auth_users(session)


def _ensure_mercato_schema(session: Session) -> None:
    global _MERCATO_SCHEMA_READY
    if _MERCATO_SCHEMA_READY:
        return
    with _MERCATO_SCHEMA_LOCK:
        if _MERCATO_SCHEMA_READY:
            return
        _ensure_auth_schema(session)
        for table_name in (
            "mercato_requests",
            "mercato_needs",
            "mercato_candidates",
            "mercato_candidate_events",
        ):
            if not _table_exists(session, table_name):
                _drop_orphan_type(session, table_name)
        statements = [chunk.strip() for chunk in MERCATO_SCHEMA_SQL.split(";") if chunk.strip()]
        for statement in statements:
            session.execute(text(statement))
        session.execute(
            text(
                "ALTER TABLE mercato_candidates "
                "ADD COLUMN IF NOT EXISTS player_season_id INT"
            )
        )
        session.execute(
            text(
                "ALTER TABLE mercato_candidates "
                "DROP CONSTRAINT IF EXISTS mercato_candidates_player_season_id_fkey"
            )
        )
        session.commit()
        _MERCATO_SCHEMA_READY = True


def _ensure_agency_ops_schema(session: Session) -> None:
    global _AGENCY_OPS_SCHEMA_READY
    if _AGENCY_OPS_SCHEMA_READY:
        return
    with _AGENCY_OPS_SCHEMA_LOCK:
        if _AGENCY_OPS_SCHEMA_READY:
            return
        _ensure_auth_schema(session)
        statements = [chunk.strip() for chunk in AGENCY_OPS_SCHEMA_SQL.split(";") if chunk.strip()]
        for statement in statements:
            session.execute(text(statement))
        session.commit()
        _AGENCY_OPS_SCHEMA_READY = True


def _ensure_crm_schema(session: Session) -> None:
    global _CRM_SCHEMA_READY
    if _CRM_SCHEMA_READY:
        return
    with _CRM_SCHEMA_LOCK:
        if _CRM_SCHEMA_READY:
            return
        statements = [chunk.strip() for chunk in CRM_SCHEMA_SQL.split(";") if chunk.strip()]
        for statement in statements:
            session.execute(text(statement))
        session.commit()
        _CRM_SCHEMA_READY = True


def _ensure_youth_schema(session: Session) -> None:
    global _YOUTH_SCHEMA_READY
    if _YOUTH_SCHEMA_READY:
        return
    with _YOUTH_SCHEMA_LOCK:
        if _YOUTH_SCHEMA_READY:
            return
        statements = [chunk.strip() for chunk in YOUTH_SCHEMA_SQL.split(";") if chunk.strip()]
        for statement in statements:
            session.execute(text(statement))
        session.commit()
        _YOUTH_SCHEMA_READY = True


def _hash_password(password: str, algo: str = "bcrypt") -> str:
    if algo == "bcrypt":
        return PASSWORD_CONTEXT.hash(password)
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _detect_hash_algo(password_hash: str) -> str:
    if password_hash.startswith("$2a$") or password_hash.startswith("$2b$") or password_hash.startswith("$2y$"):
        return "bcrypt"
    return LEGACY_SHA256


def _verify_password(password: str, password_hash: str, algo: Optional[str]) -> bool:
    resolved = algo or _detect_hash_algo(password_hash)
    if resolved == "bcrypt":
        if not _detect_hash_algo(password_hash) == "bcrypt":
            return _hash_password(password, LEGACY_SHA256) == password_hash
        try:
            return PASSWORD_CONTEXT.verify(password, password_hash)
        except Exception:
            return False
    if resolved == LEGACY_SHA256:
        return _hash_password(password, LEGACY_SHA256) == password_hash
    return False


def _load_bootstrap_users() -> list[dict[str, str]]:
    users: list[dict[str, str]] = []

    env_blob = os.getenv("AUTH_USERS_JSON") or os.getenv("AUTH_USERS")
    if env_blob:
        try:
            data = json.loads(env_blob)
            raw_users = data.get("users") if isinstance(data, dict) else data
            if isinstance(raw_users, list):
                for entry in raw_users:
                    if not isinstance(entry, dict):
                        continue
                    username = str(entry.get("username") or "").strip()
                    if not username:
                        continue
                    password_hash = entry.get("password_hash")
                    if not password_hash and entry.get("password"):
                        password_hash = _hash_password(str(entry.get("password")))
                    if not password_hash:
                        continue
                    algo = entry.get("password_algo")
                    if not algo and password_hash:
                        algo = _detect_hash_algo(str(password_hash))
                    if not algo:
                        algo = "bcrypt"
                    email = (entry.get("email") or "").strip() or None
                    users.append(
                        {
                            "username": username,
                            "display_name": entry.get("display_name") or username,
                            "email": email,
                            "password_hash": str(password_hash),
                            "password_algo": str(algo),
                        }
                    )
        except json.JSONDecodeError:
            pass

    if not users and os.getenv("AUTH_USERNAME"):
        username = os.getenv("AUTH_USERNAME", "").strip()
        password_hash = os.getenv("AUTH_PASSWORD_HASH")
        if not password_hash and os.getenv("AUTH_PASSWORD"):
            password_hash = _hash_password(os.getenv("AUTH_PASSWORD", ""))
        if username and password_hash:
            email = (os.getenv("AUTH_EMAIL") or "").strip() or None
            users.append(
                {
                    "username": username,
                    "display_name": os.getenv("AUTH_DISPLAY_NAME") or username,
                    "email": email,
                    "password_hash": str(password_hash),
                    "password_algo": _detect_hash_algo(str(password_hash)),
                }
            )

    if not users:
        candidates = [
            Path(__file__).resolve().parent.parent / "config" / "credentials.toml",
            Path(__file__).resolve().parent.parent.parent / "new_nextlegend" / "config" / "credentials.toml",
            Path(__file__).resolve().parent.parent.parent / "nextlegend" / "config" / "credentials.toml",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                data = toml.loads(path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            raw_users = data.get("users") if isinstance(data, dict) else None
            if not raw_users:
                continue
            for entry in raw_users:
                if not isinstance(entry, dict):
                    continue
                username = str(entry.get("username") or "").strip()
                if not username:
                    continue
                password_hash = entry.get("password_hash")
                if not password_hash and entry.get("password"):
                    password_hash = _hash_password(str(entry.get("password")))
                if not password_hash:
                    continue
                algo = entry.get("password_algo")
                if not algo and password_hash:
                    algo = _detect_hash_algo(str(password_hash))
                if not algo:
                    algo = "bcrypt"
                email = (entry.get("email") or "").strip() or None
                users.append(
                    {
                        "username": username,
                        "display_name": entry.get("display_name") or username,
                        "email": email,
                        "password_hash": str(password_hash),
                        "password_algo": str(algo),
                    }
                )
            if users:
                break

    return users


def _table_exists(session: Session, table_name: str) -> bool:
    return (
        session.execute(
            text("SELECT to_regclass(:name)"),
            {"name": f"public.{table_name}"},
        ).scalar()
        is not None
    )


def _drop_orphan_type(session: Session, type_name: str) -> None:
    if _table_exists(session, type_name):
        return
    type_exists = session.execute(
        text(
            """
            SELECT 1
            FROM pg_type
            WHERE typname = :type_name AND typtype = 'c'
            """
        ),
        {"type_name": type_name},
    ).fetchone()
    if type_exists:
        session.execute(text(f'DROP TYPE "{type_name}"'))


def _seed_auth_users(session: Session) -> None:
    has_users = session.execute(text("SELECT 1 FROM auth_users LIMIT 1")).fetchone()
    if has_users:
        admin_exists = session.execute(
            text("SELECT 1 FROM auth_users WHERE lower(username) = :admin"),
            {"admin": ADMIN_USERNAME},
        ).fetchone()
        if admin_exists:
            return
        seed_users = _load_bootstrap_users()
        admin_entry = next(
            (entry for entry in seed_users if entry["username"].strip().lower() == ADMIN_USERNAME),
            None,
        )
        if not admin_entry:
            return
        session.execute(
            text(
                """
                INSERT INTO auth_users (
                    username, display_name, email, password_hash, password_algo, role
                )
                VALUES (
                    :username, :display_name, :email, :password_hash, :password_algo, 'admin'
                )
                ON CONFLICT (username) DO NOTHING
                """
            ),
            admin_entry,
        )
        session.commit()
        return
    _sync_auth_users(session, replace_passwords=True)


def _sync_auth_users(session: Session, replace_passwords: bool) -> int:
    seed_users = _load_bootstrap_users()
    if not seed_users:
        return 0
    for entry in seed_users:
        role = "admin" if entry["username"].strip().lower() == ADMIN_USERNAME else "user"
        session.execute(
            text(
                """
                INSERT INTO auth_users (
                    username, display_name, email, password_hash, password_algo, role
                )
                VALUES (
                    :username, :display_name, :email, :password_hash, :password_algo, :role
                )
                ON CONFLICT (username) DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    email = COALESCE(EXCLUDED.email, auth_users.email),
                    password_hash = CASE
                        WHEN :replace_passwords THEN EXCLUDED.password_hash
                        ELSE auth_users.password_hash
                    END,
                    password_algo = CASE
                        WHEN :replace_passwords THEN EXCLUDED.password_algo
                        ELSE auth_users.password_algo
                    END,
                    role = CASE
                        WHEN EXCLUDED.username = :admin_username THEN 'admin'
                        ELSE 'user'
                    END,
                    updated_at = NOW()
                """
            ),
            {
                **entry,
                "role": role,
                "replace_passwords": replace_passwords,
                "admin_username": ADMIN_USERNAME,
            },
        )
    session.commit()
    return len(seed_users)


def _fetch_user(session: Session, identifier: str) -> Optional[dict[str, str]]:
    key = identifier.strip().lower()
    if not key:
        return None
    row = session.execute(
        text(
            """
            SELECT username, display_name, email, password_hash, password_algo, role
            FROM auth_users
            WHERE lower(username) = :key OR lower(email) = :key
            """
        ),
        {"key": key},
    ).fetchone()
    return _row_to_dict(row) if row else None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _frontend_base_url(request: Request) -> str:
    configured = os.getenv("FRONTEND_BASE_URL") or os.getenv("PUBLIC_APP_URL") or os.getenv("APP_URL")
    if configured:
        return configured.rstrip("/")
    origin = request.headers.get("origin")
    if origin:
        return origin.rstrip("/")
    return "http://localhost:3000"


def _smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM_EMAIL"))


def _send_password_reset_email(to_email: str, reset_url: str) -> bool:
    if not _smtp_configured():
        print(f"[auth] SMTP not configured. Password reset link for {to_email}: {reset_url}")
        return False

    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS")
    from_email = os.getenv("SMTP_FROM_EMAIL", "")
    from_name = os.getenv("SMTP_FROM_NAME", "Next Legend")
    use_ssl = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    message = EmailMessage()
    message["Subject"] = "Reset your Next Legend password"
    message["From"] = f"{from_name} <{from_email}>"
    message["To"] = to_email
    message.set_content(
        "\n".join(
            [
                "A password reset was requested for your Next Legend account.",
                "",
                f"Reset link: {reset_url}",
                "",
                "This link expires in 30 minutes. If you did not request this, you can ignore this email.",
            ]
        )
    )

    context = ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=12) as server:
            if username and password:
                server.login(username, password)
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=12) as server:
            if use_tls:
                server.starttls(context=context)
            if username and password:
                server.login(username, password)
            server.send_message(message)
    return True


def _authenticate_user(session: Session, username: str, password: str) -> Optional[dict[str, str]]:
    entry = _fetch_user(session, username)
    if not entry:
        return None
    if not _verify_password(password, entry["password_hash"], entry.get("password_algo")):
        return None
    if entry.get("password_algo") == LEGACY_SHA256:
        try:
            new_hash = _hash_password(password, "bcrypt")
        except Exception:
            new_hash = None
        if new_hash:
            session.execute(
                text(
                    """
                    UPDATE auth_users
                    SET password_hash = :password_hash,
                        password_algo = 'bcrypt',
                        updated_at = NOW()
                    WHERE username = :username
                    """
                ),
                {"password_hash": new_hash, "username": entry["username"]},
            )
    session.execute(
        text("UPDATE auth_users SET last_login = NOW() WHERE username = :username"),
        {"username": entry["username"]},
    )
    session.commit()
    return {
        "username": entry["username"],
        "display_name": entry.get("display_name") or entry["username"],
        "email": entry.get("email") or "",
        "role": entry.get("role") or "user",
    }


def _get_session_user(session: Session, session_id: str) -> Optional[dict[str, str]]:
    row = session.execute(
        text(
            """
            SELECT s.user_id, s.expires_at, u.display_name, u.email, u.role
            FROM auth_sessions s
            JOIN auth_users u ON u.username = s.user_id
            WHERE id = :session_id
            """
        ),
        {"session_id": session_id},
    ).fetchone()
    if not row:
        return None
    if row.expires_at and row.expires_at < datetime.now(timezone.utc):
        session.execute(
            text("DELETE FROM auth_sessions WHERE id = :session_id"),
            {"session_id": session_id},
        )
        session.commit()
        return None
    session.execute(
        text("UPDATE auth_sessions SET last_seen = NOW() WHERE id = :session_id"),
        {"session_id": session_id},
    )
    session.commit()
    return {
        "username": row.user_id,
        "display_name": row.display_name,
        "email": row.email,
        "role": row.role or "user",
    }


class ProspectToggle(BaseModel):
    player_id: int
    player_season_id: Optional[int] = None


class YouthProspectToggle(BaseModel):
    youth_id: int


class ClubNeedCreate(BaseModel):
    club_id: Optional[int] = None
    need_label: str
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    assigned_user: Optional[str] = "admin"
    priority_stage: str = "Priority 1"


class ClubNeedOrderItem(BaseModel):
    id: int
    priority_stage: str
    sort_order: int


class ClubNeedReorder(BaseModel):
    needs: list[ClubNeedOrderItem]


class ClubNeedPlayerAdd(BaseModel):
    player_id: int


class ClubNeedPlayerOrder(BaseModel):
    player_ids: list[int]


class HqPriorityItemPayload(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    agent_name: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    color: Optional[str] = None
    related_page: Optional[str] = None


class HqCalendarEventPayload(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    event_type: Optional[str] = "team"
    agent_names: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    color: Optional[str] = None
    related_page: Optional[str] = None


class HdPlayerPayload(BaseModel):
    player_id: Optional[int] = None
    display_name: Optional[str] = None
    position: Optional[str] = None
    current_club: Optional[str] = None
    contract_expiry: Optional[str] = None
    current_club_situation: Optional[str] = None
    plan: Optional[str] = None
    priority: Optional[str] = None
    demanded_transfer_fee: Optional[float] = None
    next_step: Optional[str] = None
    assigned_agent: Optional[str] = None
    photo_url: Optional[str] = None
    birth_date: Optional[str] = None
    player_phone: Optional[str] = None
    player_email: Optional[str] = None
    entourage_phone: Optional[str] = None
    entourage_email: Optional[str] = None
    season_objectives: Optional[str] = None
    eyeball_url: Optional[str] = None
    transfermarkt_url: Optional[str] = None
    is_young_player: Optional[bool] = None
    manual_performance: Optional[dict[str, Any]] = None
    contract_status: Optional[str] = None
    mandate_status: Optional[str] = None
    medical_status: Optional[str] = None
    market_notes: Optional[str] = None
    scouting_notes: Optional[str] = None
    status: Optional[str] = None


class HdPlayerDocumentPayload(BaseModel):
    document_type: str = "other"
    title: str
    file_name: Optional[str] = None
    file_key: Optional[str] = None
    storage_url: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    notes: Optional[str] = None


class HdPlayerProspectClubPayload(BaseModel):
    club_id: Optional[int] = None
    club_name: str
    competition_name: Optional[str] = None
    status: Optional[str] = None
    offer: Optional[str] = None
    contact: Optional[str] = None
    notes: Optional[str] = None


class HdPlayerManualTransferPayload(BaseModel):
    transfer_date: Optional[str] = None
    transfer_type: Optional[str] = None
    transfer_fee: Optional[str] = None
    team_in_name: Optional[str] = None
    team_out_name: Optional[str] = None
    league_name: Optional[str] = None
    notes: Optional[str] = None


class MercatoNeedPayload(BaseModel):
    id: Optional[int] = None
    position: Optional[str] = None
    role: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    preferred_foot: Optional[str] = None
    height_min: Optional[float] = None
    target_league_level: Optional[str] = None
    required_player_level: Optional[float] = None
    nationality_preferences: Optional[str] = None
    contract_preferences: Optional[str] = None
    notes: Optional[str] = None


class MercatoRequestCreate(BaseModel):
    club_id: Optional[int] = None
    assigned_agent_id: Optional[str] = None
    season: str = "2026"
    title: Optional[str] = None
    priority: str = "medium"
    status: str = "new"
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    salary_max: Optional[float] = None
    deal_type: str = "any"
    extra_info: Optional[str] = None
    need: MercatoNeedPayload


class MercatoRequestUpdate(BaseModel):
    club_id: Optional[int] = None
    assigned_agent_id: Optional[str] = None
    season: Optional[str] = None
    title: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    salary_max: Optional[float] = None
    deal_type: Optional[str] = None
    extra_info: Optional[str] = None
    need: Optional[MercatoNeedPayload] = None


class MercatoCandidateCreate(BaseModel):
    player_id: int
    player_season_id: Optional[int] = None
    source: str = "manual"
    status: str = "suggested"
    agent_note: Optional[str] = None


class MercatoCandidateUpdate(BaseModel):
    status: Optional[str] = None
    agent_note: Optional[str] = None


class MercatoCandidateStatus(BaseModel):
    status: str
    note: Optional[str] = None


class MercatoShortlistGenerate(BaseModel):
    competitions: list[str] = []
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    min_minutes: Optional[int] = None
    min_match_score: Optional[float] = None


class AuthLoginRequest(BaseModel):
    username: str
    password: str
    legacy_user_id: Optional[str] = None


class AuthUserResponse(BaseModel):
    username: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None


class AuthProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None


class AuthPasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class AuthPasswordForgotRequest(BaseModel):
    identifier: str


class AuthPasswordResetRequest(BaseModel):
    token: str
    new_password: str


class AdminUser(BaseModel):
    username: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: str
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class AdminUserList(BaseModel):
    items: list[AdminUser]


class AdminUserCreate(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = "user"


class AdminUserUpdate(BaseModel):
    password: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def root() -> dict:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/image-proxy")
def image_proxy(url: str = Query(..., min_length=8)) -> Response:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Only public HTTP(S) image URLs are supported")
    if not _is_public_proxy_host(parsed.hostname):
        raise HTTPException(status_code=400, detail="Image host is not allowed")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "NextLegendImageProxy/1.0",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=6) as response:
            content_type = response.headers.get("content-type", "").split(";")[0].lower()
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=415, detail="URL did not return an image")
            content = response.read(MAX_PROXY_IMAGE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail="Image could not be loaded") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail="Image could not be loaded") from exc

    if len(content) > MAX_PROXY_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")

    return Response(
        content,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/auth/login")
def auth_login(payload: AuthLoginRequest, session: Session = Depends(get_session)):
    _ensure_auth_schema(session)
    user = _authenticate_user(session, payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if payload.legacy_user_id and payload.legacy_user_id != user["username"]:
        _ensure_ai_schema(session)
        session.execute(
            text("UPDATE ai_conversations SET user_id = :new_id WHERE user_id = :old_id"),
            {"new_id": user["username"], "old_id": payload.legacy_user_id},
        )
    session_id = secrets.token_urlsafe(32)
    session_days = int(os.getenv("AUTH_SESSION_DAYS", DEFAULT_SESSION_DAYS))
    expires_at = datetime.now(timezone.utc) + timedelta(days=session_days)
    session.execute(
        text(
            """
            INSERT INTO auth_sessions (id, user_id, created_at, last_seen, expires_at)
            VALUES (:id, :user_id, NOW(), NOW(), :expires_at)
            """
        ),
        {"id": session_id, "user_id": user["username"], "expires_at": expires_at},
    )
    session.commit()
    response = JSONResponse({"user": user})
    secure_cookie = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        AUTH_COOKIE_NAME,
        session_id,
        httponly=True,
        samesite="lax",
        secure=secure_cookie,
        max_age=session_days * 24 * 60 * 60,
    )
    return response


@app.post("/auth/logout")
def auth_logout(request: Request, session: Session = Depends(get_session)):
    _ensure_auth_schema(session)
    session_id = request.cookies.get(AUTH_COOKIE_NAME)
    if session_id:
        session.execute(
            text("DELETE FROM auth_sessions WHERE id = :session_id"),
            {"session_id": session_id},
        )
        session.commit()
    response = JSONResponse({"logged_out": True})
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response


@app.get("/auth/me", response_model=AuthUserResponse)
def auth_me(request: Request, session: Session = Depends(get_session)):
    _ensure_auth_schema(session)
    session_id = request.cookies.get(AUTH_COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = _get_session_user(session, session_id)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session")
    return AuthUserResponse(
        username=user.get("username"),
        display_name=user.get("display_name") or user.get("username"),
        email=user.get("email"),
        role=user.get("role"),
    )


@app.get("/auth/profile", response_model=AuthUserResponse)
def auth_profile(request: Request, session: Session = Depends(get_session)):
    user = getattr(request.state, "user", None) or {}
    if not user.get("username"):
        raise HTTPException(status_code=401, detail="Authentication required")
    return AuthUserResponse(
        username=user.get("username"),
        display_name=user.get("display_name") or user.get("username"),
        email=user.get("email"),
        role=user.get("role"),
    )


@app.patch("/auth/profile", response_model=AuthUserResponse)
def auth_update_profile(
    payload: AuthProfileUpdate,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_auth_schema(session)
    user = getattr(request.state, "user", None) or {}
    username = user.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Authentication required")

    updates: dict[str, object] = {}
    if payload.display_name is not None:
        updates["display_name"] = payload.display_name.strip() or username
    if payload.email is not None:
        email = payload.email.strip().lower() if payload.email else None
        if email:
            existing = session.execute(
                text(
                    """
                    SELECT username
                    FROM auth_users
                    WHERE lower(email) = :email AND lower(username) <> :username
                    """
                ),
                {"email": email, "username": username.lower()},
            ).fetchone()
            if existing:
                raise HTTPException(status_code=400, detail="Email is already used")
        updates["email"] = email

    if not updates:
        return AuthUserResponse(
            username=username,
            display_name=user.get("display_name") or username,
            email=user.get("email"),
            role=user.get("role"),
        )

    set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
    row = session.execute(
        text(
            f"""
            UPDATE auth_users
            SET {set_clause}, updated_at = NOW()
            WHERE username = :username
            RETURNING username, display_name, email, role
            """
        ),
        {"username": username, **updates},
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return AuthUserResponse(**_row_to_dict(row))


@app.post("/auth/password/change")
def auth_change_password(
    payload: AuthPasswordChangeRequest,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_auth_schema(session)
    user = getattr(request.state, "user", None) or {}
    username = user.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Authentication required")
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must contain at least 8 characters")
    if len(payload.new_password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password exceeds 72 bytes")
    entry = _fetch_user(session, username)
    if not entry or not _verify_password(payload.current_password, entry["password_hash"], entry.get("password_algo")):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    session_id = request.cookies.get(AUTH_COOKIE_NAME)
    session.execute(
        text(
            """
            UPDATE auth_users
            SET password_hash = :password_hash,
                password_algo = 'bcrypt',
                updated_at = NOW()
            WHERE username = :username
            """
        ),
        {"username": username, "password_hash": _hash_password(payload.new_password, "bcrypt")},
    )
    session.execute(
        text(
            """
            DELETE FROM auth_sessions
            WHERE user_id = :username AND (:session_id IS NULL OR id <> :session_id)
            """
        ),
        {"username": username, "session_id": session_id},
    )
    session.commit()
    return {"updated": True}


@app.post("/auth/password/forgot")
def auth_forgot_password(
    payload: AuthPasswordForgotRequest,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_auth_schema(session)
    session.execute(
        text("DELETE FROM auth_password_reset_tokens WHERE used_at IS NOT NULL OR expires_at < NOW()")
    )
    identifier = payload.identifier.strip()
    user = _fetch_user(session, identifier) if identifier else None
    if user and user.get("email"):
        token = secrets.token_urlsafe(36)
        reset_url = f"{_frontend_base_url(request)}/login?reset_token={urllib.parse.quote(token)}"
        session.execute(
            text(
                """
                INSERT INTO auth_password_reset_tokens (token_hash, user_id, created_at, expires_at, used_at)
                VALUES (:token_hash, :user_id, NOW(), NOW() + INTERVAL '30 minutes', NULL)
                """
            ),
            {"token_hash": _token_hash(token), "user_id": user["username"]},
        )
        session.commit()
        try:
            _send_password_reset_email(user["email"], reset_url)
        except Exception as exc:
            print(f"[auth] Failed to send password reset email to {user['email']}: {exc}")
    else:
        session.commit()
    return {"sent": True}


@app.post("/auth/password/reset")
def auth_reset_password(payload: AuthPasswordResetRequest, session: Session = Depends(get_session)):
    _ensure_auth_schema(session)
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must contain at least 8 characters")
    if len(payload.new_password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password exceeds 72 bytes")
    row = session.execute(
        text(
            """
            SELECT token_hash, user_id
            FROM auth_password_reset_tokens
            WHERE token_hash = :token_hash
              AND used_at IS NULL
              AND expires_at >= NOW()
            """
        ),
        {"token_hash": _token_hash(payload.token)},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="Reset link is invalid or expired")
    session.execute(
        text(
            """
            UPDATE auth_users
            SET password_hash = :password_hash,
                password_algo = 'bcrypt',
                updated_at = NOW()
            WHERE username = :username
            """
        ),
        {"username": row.user_id, "password_hash": _hash_password(payload.new_password, "bcrypt")},
    )
    session.execute(
        text("UPDATE auth_password_reset_tokens SET used_at = NOW() WHERE token_hash = :token_hash"),
        {"token_hash": row.token_hash},
    )
    session.execute(
        text("DELETE FROM auth_sessions WHERE user_id = :username"),
        {"username": row.user_id},
    )
    session.commit()
    return {"updated": True}


def _require_admin(request: Request) -> None:
    user = getattr(request.state, "user", None) or {}
    if user.get("role") != "admin" or str(user.get("username") or "").strip().lower() != ADMIN_USERNAME:
        raise HTTPException(status_code=403, detail="Admin access required")


@app.get("/admin/users", response_model=AdminUserList)
def admin_users(request: Request, session: Session = Depends(get_session)):
    _ensure_auth_schema(session)
    _require_admin(request)
    rows = session.execute(
        text(
            """
            SELECT username, display_name, email, role, created_at, last_login
            FROM auth_users
            ORDER BY created_at DESC
            """
        )
    ).fetchall()
    items = [AdminUser(**_row_to_dict(row)) for row in rows]
    return AdminUserList(items=items)


@app.post("/admin/users", response_model=AdminUser)
def admin_create_user(
    payload: AdminUserCreate,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_auth_schema(session)
    _require_admin(request)
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username required")
    if len(payload.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password exceeds 72 bytes")
    password_hash = _hash_password(payload.password, "bcrypt")
    role = payload.role or "user"
    if username.lower() == ADMIN_USERNAME:
        role = "admin"
    elif role == "admin":
        raise HTTPException(status_code=400, detail="Only the primary admin can have admin role")
    row = session.execute(
        text(
            """
            INSERT INTO auth_users (username, display_name, email, password_hash, password_algo, role)
            VALUES (:username, :display_name, :email, :password_hash, 'bcrypt', :role)
            RETURNING username, display_name, email, role, created_at, last_login
            """
        ),
        {
            "username": username,
            "display_name": payload.display_name or username,
            "email": (payload.email or "").strip() or None,
            "password_hash": password_hash,
            "role": role,
        },
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create user")
    return AdminUser(**_row_to_dict(row))


@app.patch("/admin/users/{username}", response_model=AdminUser)
def admin_update_user(
    username: str,
    payload: AdminUserUpdate,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_auth_schema(session)
    _require_admin(request)
    target = username.strip()
    if not target:
        raise HTTPException(status_code=400, detail="Username required")
    updates: dict[str, object] = {}
    if payload.display_name is not None:
        updates["display_name"] = payload.display_name or target
    if payload.email is not None:
        updates["email"] = payload.email.strip() if payload.email else None
    if payload.role is not None:
        if target.lower() == ADMIN_USERNAME and payload.role != "admin":
            raise HTTPException(status_code=400, detail="Cannot remove admin role from primary admin")
        if target.lower() != ADMIN_USERNAME and payload.role == "admin":
            raise HTTPException(status_code=400, detail="Only the primary admin can have admin role")
        updates["role"] = payload.role
    if payload.password:
        if len(payload.password.encode("utf-8")) > 72:
            raise HTTPException(status_code=400, detail="Password exceeds 72 bytes")
        updates["password_hash"] = _hash_password(payload.password, "bcrypt")
        updates["password_algo"] = "bcrypt"
    if not updates:
        row = session.execute(
            text(
                """
                SELECT username, display_name, email, role, created_at, last_login
                FROM auth_users
                WHERE username = :username
                """
            ),
            {"username": target},
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return AdminUser(**_row_to_dict(row))
    set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
    row = session.execute(
        text(
            f"""
            UPDATE auth_users
            SET {set_clause}, updated_at = NOW()
            WHERE username = :username
            RETURNING username, display_name, email, role, created_at, last_login
            """
        ),
        {"username": target, **updates},
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return AdminUser(**_row_to_dict(row))


@app.delete("/admin/users/{username}")
def admin_delete_user(
    username: str,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_auth_schema(session)
    _require_admin(request)
    target = username.strip()
    if not target:
        raise HTTPException(status_code=400, detail="Username required")
    if target.lower() == ADMIN_USERNAME:
        raise HTTPException(status_code=400, detail="Cannot delete primary admin")
    row = session.execute(
        text("DELETE FROM auth_users WHERE username = :username RETURNING username"),
        {"username": target},
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": True}


@app.post("/admin/users/import")
def admin_import_users(
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_auth_schema(session)
    _require_admin(request)
    imported = _sync_auth_users(session, replace_passwords=True)
    return {"imported": imported}


@app.post("/ai/scout", response_model=AIScoutResponse)
def ai_scout(request: AIScoutRequest, session: Session = Depends(get_session)):
    overrides = request.overrides.model_dump() if request.overrides else {}
    if request.limit:
        overrides["limit"] = request.limit

    language = request.language or "auto"
    if language == "auto":
        language = detect_language(request.prompt)

    result = _run_scout_flow(
        session=session,
        prompt=request.prompt,
        overrides=overrides,
        language_override=language,
    )
    return AIScoutResponse(**result)


@app.post("/ai/player-report", response_model=AIPlayerReportResponse)
def ai_player_report(request: AIPlayerReportRequest, session: Session = Depends(get_session)):
    try:
        llm = get_llm()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OpenAI client error: {exc}") from exc

    report = player_report(
        request.player_id,
        player_season_id=request.player_season_id,
        session=session,
    )
    player_context = _build_player_context(report)

    language = request.language or "auto"
    if language == "auto":
        language = detect_language(request.prompt)

    model_name = _resolve_llm_model_name(llm)
    with get_openai_callback() as cb:
        report_text = run_player_agent(
            user_text=request.prompt,
            player_context=player_context,
            language=language,
            llm=llm,
        )
    usage_summary = _merge_usage_calls([_usage_from_callback(cb, model_name, "player_report")])
    return AIPlayerReportResponse(
        player_id=request.player_id,
        report=report_text,
        context=player_context,
        usage=usage_summary,
    )


@app.get("/ai/conversations", response_model=AIConversationList)
def ai_conversations(
    request: Request,
    user_id: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
    current_user_id = _current_user_id(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    rows = session.execute(
        text(
            """
            SELECT id, user_id, title, mode, created_at, updated_at
            FROM ai_conversations
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            """
        ),
        {"user_id": current_user_id},
    ).fetchall()
    items = [AIConversation(**_row_to_dict(row)) for row in rows]
    return AIConversationList(items=items)


@app.post("/ai/conversations", response_model=AIConversation)
def ai_conversation_create(
    payload: AIConversationCreate,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
    current_user_id = _current_user_id(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    row = session.execute(
        text(
            """
            INSERT INTO ai_conversations (user_id, title, mode)
            VALUES (:user_id, :title, :mode)
            RETURNING id, user_id, title, mode, created_at, updated_at
            """
        ),
        {
            "user_id": current_user_id,
            "title": payload.title,
            "mode": payload.mode or "scout",
        },
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
    return AIConversation(**_row_to_dict(row))


@app.get("/ai/users")
def ai_users(request: Request, session: Session = Depends(get_session)):
    _require_admin(request)
    _ensure_ai_schema(session)
    rows = session.execute(
        text(
            """
            SELECT user_id, COUNT(*) AS conversations, MAX(updated_at) AS last_updated
            FROM ai_conversations
            GROUP BY user_id
            ORDER BY last_updated DESC NULLS LAST
            """
        )
    ).fetchall()
    return [
        {
            "user_id": r.user_id,
            "conversations": r.conversations,
            "last_updated": r.last_updated,
        }
        for r in rows
    ]


@app.patch("/ai/conversations/{conversation_id}", response_model=AIConversation)
def ai_conversation_update(
    conversation_id: int,
    payload: AIConversationUpdate,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
    current_user_id = _current_user_id(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    row = session.execute(
        text(
            """
            UPDATE ai_conversations
            SET title = :title, updated_at = NOW()
            WHERE id = :conversation_id AND user_id = :user_id
            RETURNING id, user_id, title, mode, created_at, updated_at
            """
        ),
        {
            "conversation_id": conversation_id,
            "user_id": current_user_id,
            "title": payload.title,
        },
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return AIConversation(**_row_to_dict(row))


@app.get("/ai/conversations/{conversation_id}", response_model=AIConversationDetail)
def ai_conversation_detail(
    conversation_id: int,
    request: Request,
    user_id: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
    current_user_id = _current_user_id(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    convo = session.execute(
        text(
            """
            SELECT id, user_id, title, mode, created_at, updated_at
            FROM ai_conversations
            WHERE id = :conversation_id AND user_id = :user_id
            """
        ),
        {"conversation_id": conversation_id, "user_id": current_user_id},
    ).fetchone()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    rows = session.execute(
        text(
            """
            SELECT id, conversation_id, role, content, payload, created_at
            FROM ai_messages
            WHERE conversation_id = :conversation_id
            ORDER BY created_at ASC
            """
        ),
        {"conversation_id": conversation_id},
    ).fetchall()
    messages = [AIMessage(**_row_to_dict(row)) for row in rows]
    return AIConversationDetail(
        conversation=AIConversation(**_row_to_dict(convo)),
        messages=messages,
    )


@app.get("/ai/usage", response_model=AIUsageResponse)
def ai_usage(
    request: Request,
    user_id: str = Query(...),
    conversation_id: Optional[int] = Query(None),
    session: Session = Depends(get_session),
):
    _require_admin(request)
    _ensure_ai_schema(session)
    sql = """
    SELECT m.payload
    FROM ai_messages m
    JOIN ai_conversations c ON c.id = m.conversation_id
    WHERE c.user_id = :user_id
      AND m.role = 'assistant'
    """
    params = {"user_id": user_id}
    if conversation_id is not None:
        sql += " AND m.conversation_id = :conversation_id"
        params["conversation_id"] = conversation_id
    rows = session.execute(text(sql), params).fetchall()
    totals = _aggregate_usage([row.payload for row in rows])
    return AIUsageResponse(
        user_id=user_id,
        conversation_id=conversation_id,
        prompt_tokens=int(totals.get("prompt_tokens", 0) or 0),
        completion_tokens=int(totals.get("completion_tokens", 0) or 0),
        total_tokens=int(totals.get("total_tokens", 0) or 0),
        estimated_cost_usd=float(totals.get("estimated_cost_usd", 0.0) or 0.0),
        model=totals.get("model"),
    )


@app.post("/ai/conversations/{conversation_id}/messages", response_model=AIMessageResponse)
def ai_conversation_message(
    conversation_id: int,
    payload: AIMessageCreate,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
    current_user_id = _current_user_id(request)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    convo = session.execute(
        text(
            """
            SELECT id, user_id, title, mode, created_at, updated_at
            FROM ai_conversations
            WHERE id = :conversation_id AND user_id = :user_id
            """
        ),
        {"conversation_id": conversation_id, "user_id": current_user_id},
    ).fetchone()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    mode = payload.mode or convo._mapping.get("mode") or "scout"
    if mode != convo._mapping.get("mode"):
        session.execute(
            text(
                "UPDATE ai_conversations SET mode = :mode, updated_at = NOW() WHERE id = :id"
            ),
            {"mode": mode, "id": conversation_id},
        )

    user_row = session.execute(
        text(
            """
            INSERT INTO ai_messages (conversation_id, role, content)
            VALUES (:conversation_id, 'user', :content)
            RETURNING id, conversation_id, role, content, payload, created_at
            """
        ),
        {"conversation_id": conversation_id, "content": payload.prompt},
    ).fetchone()

    title = convo._mapping.get("title")
    if not title:
        session.execute(
            text(
                "UPDATE ai_conversations SET title = :title, updated_at = NOW() WHERE id = :id"
            ),
            {"title": _truncate_title(payload.prompt), "id": conversation_id},
        )

    language = payload.language or "auto"
    if language == "auto":
        language = detect_language(payload.prompt)

    conversation_context = _build_conversation_context(
        session,
        conversation_id,
        exclude_message_id=user_row.id if user_row else None,
        include_roles={"user"},
    )
    prompt_with_context = payload.prompt
    if conversation_context:
        prompt_with_context = f"{conversation_context}\n\nLatest request: {payload.prompt}"

    requested_count = extract_requested_count(payload.prompt)
    if not requested_count and conversation_context:
        requested_count = extract_requested_count(conversation_context)

    if mode == "player":
        if not payload.player_id:
            raise HTTPException(status_code=400, detail="player_id is required for player mode")
        report = player_report(
            payload.player_id,
            player_season_id=payload.player_season_id,
            session=session,
        )
        player_context = _build_player_context(report)
        try:
            llm = get_llm()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"OpenAI client error: {exc}") from exc
        model_name = _resolve_llm_model_name(llm)
        with get_openai_callback() as cb:
            report_text = run_player_agent(
                user_text=prompt_with_context,
                player_context=player_context,
                language=language,
                llm=llm,
            )
        usage_summary = _merge_usage_calls([_usage_from_callback(cb, model_name, "player_report")])
        assistant_payload = {
            "report": report_text,
            "context": player_context,
            "player_id": payload.player_id,
            "usage": usage_summary,
        }
        assistant_content = report_text
    else:
        result = _run_scout_flow(
            session=session,
            prompt=payload.prompt,
            overrides={"season": payload.season} if payload.season else {},
            language_override=language,
            conversation_context=conversation_context,
            requested_count_hint=requested_count,
        )
        assistant_payload = result
        count = len(result.get("candidates", []))
        assistant_content = (
            f"I found {count} profiles matching your brief. Click any card to open the report."
            if count
            else "I could not find profiles matching the brief."
        )

    assistant_row = session.execute(
        text(
            """
            INSERT INTO ai_messages (conversation_id, role, content, payload)
            VALUES (:conversation_id, 'assistant', :content, CAST(:payload AS JSONB))
            RETURNING id, conversation_id, role, content, payload, created_at
            """
        ),
        {
            "conversation_id": conversation_id,
            "content": assistant_content,
            "payload": json.dumps(assistant_payload),
        },
    ).fetchone()

    session.execute(
        text("UPDATE ai_conversations SET updated_at = NOW() WHERE id = :id"),
        {"id": conversation_id},
    )
    session.commit()

    convo_updated = session.execute(
        text(
            """
            SELECT id, user_id, title, mode, created_at, updated_at
            FROM ai_conversations
            WHERE id = :conversation_id
            """
        ),
        {"conversation_id": conversation_id},
    ).fetchone()

    return AIMessageResponse(
        conversation=AIConversation(**_row_to_dict(convo_updated)),
        user_message=AIMessage(**_row_to_dict(user_row)),
        assistant_message=AIMessage(**_row_to_dict(assistant_row)),
    )


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    payload = dict(row._mapping)
    if "payload" in payload and isinstance(payload["payload"], str):
        try:
            payload["payload"] = json.loads(payload["payload"])
        except json.JSONDecodeError:
            pass
    return payload


def _seed_hd_players_if_empty(session: Session) -> None:
    count = session.execute(text("SELECT COUNT(*) AS count FROM hd_players")).fetchone()
    if count and int(count.count or 0) > 0:
        return
    seed_rows = [
        ("Kevin Danso", "Tottenham", "Loan or permanent", "A", None, "Assess future with Tottenham"),
        ("Mario Lemina", "Galatasaray", "Find new club", "B", None, "Find offers to increase gala proposal"),
        ("Simon Banza", "Al Jazira", "Loan or permanent", "A", None, "Discuss TF with Al Jazira"),
        ("Lilian Brassier", "Rennes", "Loan or permanent", "C", 15000000, "Assess future with Rennes"),
        ("Junior Diaz", "Troyes", "Loan or permanent", "A", 8000000, "Send profile to clubs"),
        ("Tom Pouilly", "Pau", "Permanent deal", "B", 500000, "Send profile to clubs"),
        ("Noha Lemina", "Yverdon", "Find new club", "B", 0, "Send profile to clubs"),
        ("Enzo Mongo", "Nantes", "Staying", "C", None, "Standby"),
        ("Massadio Haidara", "Kocaelispor", "Staying", "D", None, "Standby"),
    ]
    for name, club, plan, priority, fee, next_step in seed_rows:
        session.execute(
            text(
                """
                INSERT INTO hd_players (
                  display_name, current_club, plan, priority,
                  demanded_transfer_fee, next_step, assigned_agent,
                  created_at, updated_at
                ) VALUES (
                  :display_name, :current_club, :plan, :priority,
                  :demanded_transfer_fee, :next_step, 'Yannis',
                  NOW(), NOW()
                )
                ON CONFLICT DO NOTHING
                """
            ),
            {
                "display_name": name,
                "current_club": club,
                "plan": plan,
                "priority": priority,
                "demanded_transfer_fee": fee,
                "next_step": next_step,
            },
        )
    session.commit()


def _xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _xlsx_cell(value: Any, row_idx: int, col_idx: int) -> str:
    ref = f"{_xlsx_col_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    escaped = html.escape(str(value), quote=True)
    return f'<c r="{ref}" t="inlineStr"><is><t>{escaped}</t></is></c>'


def _xlsx_sheet_xml(rows: list[list[Any]]) -> str:
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(value, row_idx, col_idx) for col_idx, value in enumerate(row, start=1))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )


def _build_xlsx(sheets: list[tuple[str, list[list[Any]]]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for idx in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(
                f'<sheet name="{html.escape(name[:31], quote=True)}" sheetId="{idx}" r:id="rId{idx}"/>'
                for idx, (name, _) in enumerate(sheets, start=1)
            )
            + "</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
                for idx in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _xlsx_sheet_xml(rows))
    return output.getvalue()


CRM_PROSPECT_STAGES = ["prequalification", "relance1", "relance2", "relance3"]
CRM_CITY_COORDS = {
    ("a coruna", "spain"): (43.3623, -8.4115),
    ("ajaccio", "france"): (41.9192, 8.7386),
    ("albacete", "spain"): (38.9943, -1.8585),
    ("almeria", "spain"): (36.8340, -2.4637),
    ("amadora", "portugal"): (38.7578, -9.2240),
    ("amiens", "france"): (49.8941, 2.2958),
    ("angers", "france"): (47.4784, -0.5632),
    ("andorra", "andorra"): (42.5063, 1.5218),
    ("antwerp", "belgium"): (51.2194, 4.4025),
    ("augsburg", "germany"): (48.3705, 10.8978),
    ("auxerre", "france"): (47.7982, 3.5738),
    ("barcelona", "spain"): (41.3874, 2.1686),
    ("bari", "italy"): (41.1171, 16.8719),
    ("bastia", "france"): (42.6973, 9.4509),
    ("bergamo", "italy"): (45.6983, 9.6773),
    ("berlin", "germany"): (52.5200, 13.4050),
    ("bilbao", "spain"): (43.2630, -2.9350),
    ("birmingham", "england"): (52.4862, -1.8904),
    ("blackburn", "england"): (53.7486, -2.4875),
    ("bochum", "germany"): (51.4818, 7.2162),
    ("bologna", "italy"): (44.4949, 11.3426),
    ("bournemouth", "england"): (50.7192, -1.8808),
    ("braga", "portugal"): (41.5454, -8.4265),
    ("bremen", "germany"): (53.0793, 8.8017),
    ("brescia", "italy"): (45.5416, 10.2118),
    ("brest", "france"): (48.3904, -4.4861),
    ("brighton", "england"): (50.8225, -0.1372),
    ("bristol", "england"): (51.4545, -2.5879),
    ("bruges", "belgium"): (51.2093, 3.2247),
    ("brussels", "belgium"): (50.8503, 4.3517),
    ("burgos", "spain"): (42.3439, -3.6969),
    ("burnley", "england"): (53.7893, -2.2405),
    ("caen", "france"): (49.1829, -0.3707),
    ("cadiz", "spain"): (36.5271, -6.2886),
    ("cagliari", "italy"): (39.2238, 9.1217),
    ("cardiff", "wales"): (51.4816, -3.1791),
    ("cartagena", "spain"): (37.6257, -0.9966),
    ("charleroi", "belgium"): (50.4108, 4.4446),
    ("clermont-ferrand", "france"): (45.7772, 3.0870),
    ("cologne", "germany"): (50.9375, 6.9603),
    ("como", "italy"): (45.8081, 9.0852),
    ("cordoba", "spain"): (37.8882, -4.7794),
    ("coventry", "england"): (52.4068, -1.5197),
    ("cremona", "italy"): (45.1332, 10.0227),
    ("darmstadt", "germany"): (49.8728, 8.6512),
    ("derby", "england"): (52.9225, -1.4746),
    ("dortmund", "germany"): (51.5136, 7.4653),
    ("dunkerque", "france"): (51.0344, 2.3768),
    ("dusseldorf", "germany"): (51.2277, 6.7735),
    ("eibar", "spain"): (43.1849, -2.4716),
    ("elche", "spain"): (38.2699, -0.7126),
    ("estoril", "portugal"): (38.7057, -9.3977),
    ("eupen", "belgium"): (50.6292, 6.0314),
    ("faro", "portugal"): (37.0194, -7.9304),
    ("florence", "italy"): (43.7696, 11.2558),
    ("frankfurt", "germany"): (50.1109, 8.6821),
    ("freiburg", "germany"): (47.9990, 7.8421),
    ("funchal", "portugal"): (32.6669, -16.9241),
    ("gelsenkirchen", "germany"): (51.5177, 7.0857),
    ("genk", "belgium"): (50.9650, 5.5008),
    ("genoa", "italy"): (44.4056, 8.9463),
    ("getafe", "spain"): (40.3083, -3.7320),
    ("ghent", "belgium"): (51.0543, 3.7174),
    ("gijon", "spain"): (43.5322, -5.6611),
    ("girona", "spain"): (41.9794, 2.8214),
    ("granada", "spain"): (37.1773, -3.5986),
    ("grenoble", "france"): (45.1885, 5.7245),
    ("guingamp", "france"): (48.5620, -3.1500),
    ("hamburg", "germany"): (53.5511, 9.9937),
    ("hannover", "germany"): (52.3759, 9.7320),
    ("hull", "england"): (53.7676, -0.3274),
    ("huesca", "spain"): (42.1401, -0.4089),
    ("ipswich", "england"): (52.0567, 1.1482),
    ("karlsruhe", "germany"): (49.0069, 8.4037),
    ("kiel", "germany"): (54.3233, 10.1228),
    ("kortrijk", "belgium"): (50.8276, 3.2659),
    ("la spezia", "italy"): (44.1025, 9.8241),
    ("las palmas", "spain"): (28.1235, -15.4363),
    ("laval", "france"): (48.0785, -0.7669),
    ("lecce", "italy"): (40.3515, 18.1750),
    ("leeds", "england"): (53.8008, -1.5491),
    ("le havre", "france"): (49.4944, 0.1079),
    ("leicester", "england"): (52.6369, -1.1398),
    ("leipzig", "germany"): (51.3397, 12.3731),
    ("lens", "france"): (50.4310, 2.8330),
    ("leverkusen", "germany"): (51.0459, 7.0192),
    ("liege", "belgium"): (50.6326, 5.5797),
    ("lille", "france"): (50.6292, 3.0573),
    ("lisbon", "portugal"): (38.7223, -9.1393),
    ("liverpool", "england"): (53.4084, -2.9916),
    ("london", "england"): (51.5074, -0.1278),
    ("lorient", "france"): (47.7483, -3.3702),
    ("lyon", "france"): (45.7640, 4.8357),
    ("madrid", "spain"): (40.4168, -3.7038),
    ("magdeburg", "germany"): (52.1205, 11.6276),
    ("mainz", "germany"): (49.9929, 8.2473),
    ("malaga", "spain"): (36.7213, -4.4214),
    ("manchester", "england"): (53.4808, -2.2426),
    ("marseille", "france"): (43.2965, 5.3698),
    ("mechelen", "belgium"): (51.0259, 4.4775),
    ("metz", "france"): (49.1193, 6.1757),
    ("middlesbrough", "england"): (54.5742, -1.2351),
    ("milan", "italy"): (45.4642, 9.1900),
    ("modena", "italy"): (44.6471, 10.9252),
    ("monaco", "monaco"): (43.7384, 7.4246),
    ("montpellier", "france"): (43.6108, 3.8767),
    ("monza", "italy"): (45.5845, 9.2744),
    ("munich", "germany"): (48.1351, 11.5820),
    ("nantes", "france"): (47.2184, -1.5536),
    ("naples", "italy"): (40.8518, 14.2681),
    ("newcastle", "england"): (54.9783, -1.6178),
    ("nice", "france"): (43.7102, 7.2620),
    ("norwich", "england"): (52.6309, 1.2974),
    ("nottingham", "england"): (52.9548, -1.1581),
    ("nuremberg", "germany"): (49.4521, 11.0767),
    ("oviedo", "spain"): (43.3619, -5.8494),
    ("oxford", "england"): (51.7520, -1.2577),
    ("paderborn", "germany"): (51.7189, 8.7575),
    ("palermo", "italy"): (38.1157, 13.3615),
    ("pamplona", "spain"): (42.8125, -1.6458),
    ("paris", "france"): (48.8566, 2.3522),
    ("parma", "italy"): (44.8015, 10.3279),
    ("pau", "france"): (43.2951, -0.3708),
    ("pisa", "italy"): (43.7228, 10.4017),
    ("plymouth", "england"): (50.3755, -4.1427),
    ("porto", "portugal"): (41.1579, -8.6291),
    ("portsmouth", "england"): (50.8198, -1.0880),
    ("preston", "england"): (53.7632, -2.7031),
    ("reims", "france"): (49.2583, 4.0317),
    ("rennes", "france"): (48.1173, -1.6778),
    ("rome", "italy"): (41.9028, 12.4964),
    ("salerno", "italy"): (40.6824, 14.7681),
    ("san sebastian", "spain"): (43.3183, -1.9812),
    ("santander", "spain"): (43.4623, -3.8099),
    ("sassuolo", "italy"): (44.5432, 10.7840),
    ("seville", "spain"): (37.3891, -5.9845),
    ("sheffield", "england"): (53.3811, -1.4701),
    ("southampton", "england"): (50.9097, -1.4044),
    ("stoke-on-trent", "england"): (53.0027, -2.1794),
    ("strasbourg", "france"): (48.5734, 7.7521),
    ("stuttgart", "germany"): (48.7758, 9.1829),
    ("sunderland", "england"): (54.9069, -1.3838),
    ("swansea", "wales"): (51.6214, -3.9436),
    ("toulouse", "france"): (43.6047, 1.4442),
    ("troyes", "france"): (48.2973, 4.0744),
    ("turin", "italy"): (45.0703, 7.6869),
    ("udine", "italy"): (46.0711, 13.2346),
    ("ulm", "germany"): (48.4011, 9.9876),
    ("valencia", "spain"): (39.4699, -0.3763),
    ("valladolid", "spain"): (41.6523, -4.7245),
    ("venice", "italy"): (45.4408, 12.3155),
    ("verona", "italy"): (45.4384, 10.9916),
    ("vigo", "spain"): (42.2406, -8.7207),
    ("villarreal", "spain"): (39.9383, -0.1009),
    ("vitoria-gasteiz", "spain"): (42.8467, -2.6727),
    ("watford", "england"): (51.6565, -0.3903),
    ("west bromwich", "england"): (52.5187, -1.9945),
    ("wolfsburg", "germany"): (52.4227, 10.7865),
    ("wolverhampton", "england"): (52.5862, -2.1288),
    ("zaragoza", "spain"): (41.6488, -0.8891),
}

try:
    from crm_location_data import CRM_CITY_COORD_OVERRIDES, CRM_CLUB_LOCATION_FIXES

    CRM_CITY_COORDS.update(CRM_CITY_COORD_OVERRIDES)
except ImportError:
    CRM_CLUB_LOCATION_FIXES = {}



def _crm_new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(12)}"


def _crm_clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _crm_required(value: Optional[str], label: str) -> str:
    cleaned = _crm_clean(value)
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{label} is required")
    return cleaned


def _crm_location_key(city: Optional[str], country: Optional[str]) -> tuple[str, str]:
    def normalize(value: Optional[str]) -> str:
        raw = str(value or "").strip().lower()
        normalized = unicodedata.normalize("NFKD", raw)
        ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", ascii_only)

    return normalize(city), normalize(country)


def _crm_page_response(items: list[dict], total: int, page: int, page_size: int) -> dict:
    total_pages = max(1, (total + page_size - 1) // page_size) if total else 0
    return {"data": items, "total": total, "page": page, "pageSize": page_size, "totalPages": total_pages}


def _crm_pagination(page: int, page_size: int) -> tuple[int, int, int]:
    safe_page = max(1, page)
    safe_size = min(max(1, page_size), 200)
    return safe_page, safe_size, (safe_page - 1) * safe_size


def _crm_contact_select_sql(where_sql: str = "") -> str:
    return f"""
        SELECT
          c.*,
          club.name AS club_name,
          club.city AS club_city,
          club.country AS club_country,
          player.first_name AS player_first_name,
          player.last_name AS player_last_name,
          player.position AS player_position,
          player.age AS player_age,
          player.nationality AS player_nationality
        FROM crm_contacts c
        LEFT JOIN crm_clubs club ON club.id = c.club_id
        LEFT JOIN crm_players player ON player.id = c.player_id
        {where_sql}
    """


def _crm_prospect_select_sql(where_sql: str = "") -> str:
    return f"""
        SELECT
          p.*,
          c.first_name AS contact_first_name,
          c.last_name AS contact_last_name,
          c.role AS contact_role,
          c.email AS contact_email,
          c.phone AS contact_phone,
          c.type AS contact_type,
          club.name AS club_name,
          club.city AS club_city,
          club.country AS club_country,
          player.first_name AS player_first_name,
          player.last_name AS player_last_name,
          player.position AS player_position,
          player.age AS player_age,
          player.nationality AS player_nationality
        FROM crm_prospects p
        JOIN crm_contacts c ON c.id = p.contact_id
        LEFT JOIN crm_clubs club ON club.id = c.club_id
        LEFT JOIN crm_players player ON player.id = c.player_id
        {where_sql}
    """


@app.get("/crm/summary")
def crm_summary(session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    counts = {}
    for key, table_name in {
        "clubs": "crm_clubs",
        "players": "crm_players",
        "contacts": "crm_contacts",
        "prospects": "crm_prospects",
    }.items():
        row = session.execute(text(f"SELECT COUNT(*) AS count FROM {table_name}")).fetchone()
        counts[key] = int(row.count or 0)
    email_row = session.execute(text("SELECT COUNT(*) AS count FROM crm_contacts WHERE email IS NOT NULL AND email <> ''")).fetchone()
    unlinked_row = session.execute(text("SELECT COUNT(*) AS count FROM crm_contacts WHERE club_id IS NULL AND player_id IS NULL")).fetchone()
    counts["contacts_with_email"] = int(email_row.count or 0)
    counts["unlinked_contacts"] = int(unlinked_row.count or 0)
    return counts


@app.get("/crm/options")
def crm_options(session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    clubs = [
        _row_to_dict(row)
        for row in session.execute(
            text("SELECT id, name, city, country, logo FROM crm_clubs ORDER BY LOWER(name), LOWER(city) LIMIT 1500")
        ).fetchall()
    ]
    players = [
        _row_to_dict(row)
        for row in session.execute(
            text(
                """
                SELECT p.id, p.first_name, p.last_name, p.position, p.club_id, c.name AS club_name
                FROM crm_players p
                JOIN crm_clubs c ON c.id = p.club_id
                ORDER BY LOWER(p.last_name), LOWER(p.first_name)
                LIMIT 1500
                """
            )
        ).fetchall()
    ]
    contacts = [
        _row_to_dict(row)
        for row in session.execute(
            text(
                _crm_contact_select_sql("")
                + " ORDER BY LOWER(c.last_name), LOWER(c.first_name) LIMIT 5000"
            )
        ).fetchall()
    ]
    return {"clubs": clubs, "players": players, "contacts": contacts}


@app.get("/crm/clubs")
def crm_clubs(
    page: int = Query(1),
    pageSize: int = Query(25),
    search: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    _ensure_crm_schema(session)
    page, page_size, offset = _crm_pagination(page, pageSize)
    params: dict[str, Any] = {"limit": page_size, "offset": offset}
    where = ""
    if search:
        params["search"] = f"%{search.strip().lower()}%"
        where = "WHERE LOWER(c.name) LIKE :search OR LOWER(c.city) LIKE :search OR LOWER(c.country) LIKE :search"
    total = session.execute(text(f"SELECT COUNT(*) AS count FROM crm_clubs c {where}"), params).fetchone().count
    rows = session.execute(
        text(
            f"""
            SELECT c.*,
              (SELECT COUNT(*) FROM crm_players p WHERE p.club_id = c.id) AS player_count,
              (SELECT COUNT(*) FROM crm_contacts ct WHERE ct.club_id = c.id) AS contact_count
            FROM crm_clubs c
            {where}
            ORDER BY LOWER(c.name), LOWER(c.city)
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    ).fetchall()
    return _crm_page_response([_row_to_dict(row) for row in rows], int(total or 0), page, page_size)


@app.post("/crm/clubs")
def create_crm_club(payload: CrmClubPayload, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    name = _crm_required(payload.name, "Club name")
    city = _crm_required(payload.city, "City")
    country = _crm_required(payload.country, "Country")
    duplicate = session.execute(
        text(
            """
            SELECT 1 FROM crm_clubs
            WHERE LOWER(BTRIM(name)) = LOWER(BTRIM(:name))
              AND LOWER(BTRIM(city)) = LOWER(BTRIM(:city))
              AND LOWER(BTRIM(country)) = LOWER(BTRIM(:country))
            LIMIT 1
            """
        ),
        {"name": name, "city": city, "country": country},
    ).fetchone()
    if duplicate:
        raise HTTPException(status_code=409, detail="A club with the same name, city and country already exists")
    club_id = _crm_new_id("club")
    session.execute(
        text(
            """
            INSERT INTO crm_clubs (id, name, city, country, logo, email, phone, website, created_at, updated_at)
            VALUES (:id, :name, :city, :country, :logo, :email, :phone, :website, NOW(), NOW())
            """
        ),
        {
            "id": club_id,
            "name": name,
            "city": city,
            "country": country,
            "logo": _crm_clean(payload.logo),
            "email": _crm_clean(payload.email),
            "phone": _crm_clean(payload.phone),
            "website": _crm_clean(payload.website),
        },
    )
    session.commit()
    return get_crm_club(club_id, session)


@app.post("/crm/clubs/import")
async def import_crm_clubs(request: Request, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    body = (await request.body()).decode("utf-8-sig")
    imported = 0
    skipped = 0
    for row in csv.DictReader(io.StringIO(body)):
        name = _crm_clean(row.get("name"))
        city = _crm_clean(row.get("city"))
        country = _crm_clean(row.get("country"))
        if not name or not city or not country:
            skipped += 1
            continue
        exists = session.execute(
            text(
                """
                SELECT 1 FROM crm_clubs
                WHERE LOWER(BTRIM(name)) = LOWER(BTRIM(:name))
                  AND LOWER(BTRIM(city)) = LOWER(BTRIM(:city))
                  AND LOWER(BTRIM(country)) = LOWER(BTRIM(:country))
                LIMIT 1
                """
            ),
            {"name": name, "city": city, "country": country},
        ).fetchone()
        if exists:
            skipped += 1
            continue
        session.execute(
            text(
                """
                INSERT INTO crm_clubs (id, name, city, country, logo, email, phone, website, created_at, updated_at)
                VALUES (:id, :name, :city, :country, :logo, :email, :phone, :website, NOW(), NOW())
                """
            ),
            {
                "id": _crm_new_id("club"),
                "name": name,
                "city": city,
                "country": country,
                "logo": _crm_clean(row.get("logo")),
                "email": _crm_clean(row.get("email")),
                "phone": _crm_clean(row.get("phone")),
                "website": _crm_clean(row.get("website")),
            },
        )
        imported += 1
    session.commit()
    return {"imported": imported, "skipped": skipped}


@app.get("/crm/clubs/export.xlsx")
def export_crm_clubs(request: Request, session: Session = Depends(get_session)):
    _require_admin(request)
    _ensure_crm_schema(session)
    rows = session.execute(
        text("SELECT name, city, country, email, phone, website, logo FROM crm_clubs ORDER BY LOWER(name)")
    ).fetchall()
    sheet = [["Name", "City", "Country", "Email", "Phone", "Website", "Logo"]]
    sheet.extend([list(row._mapping.values()) for row in rows])
    return Response(
        _build_xlsx([("CRM Clubs", sheet)]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="crm_clubs.xlsx"'},
    )


@app.get("/crm/clubs/{club_id}")
def get_crm_club(club_id: str, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    row = session.execute(
        text(
            """
            SELECT c.*,
              (SELECT COUNT(*) FROM crm_players p WHERE p.club_id = c.id) AS player_count,
              (SELECT COUNT(*) FROM crm_contacts ct WHERE ct.club_id = c.id) AS contact_count
            FROM crm_clubs c
            WHERE c.id = :id
            """
        ),
        {"id": club_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Club not found")
    payload = _row_to_dict(row)
    payload["players"] = [
        _row_to_dict(player)
        for player in session.execute(
            text("SELECT * FROM crm_players WHERE club_id = :id ORDER BY LOWER(last_name), LOWER(first_name) LIMIT 100"),
            {"id": club_id},
        ).fetchall()
    ]
    payload["contacts"] = [
        _row_to_dict(contact)
        for contact in session.execute(
            text(_crm_contact_select_sql("WHERE c.club_id = :id ORDER BY LOWER(c.last_name), LOWER(c.first_name) LIMIT 100")),
            {"id": club_id},
        ).fetchall()
    ]
    return payload


@app.patch("/crm/clubs/{club_id}")
@app.put("/crm/clubs/{club_id}")
def update_crm_club(club_id: str, payload: CrmClubPayload, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    updated = session.execute(
        text(
            """
            UPDATE crm_clubs
            SET name = :name, city = :city, country = :country, logo = :logo,
                email = :email, phone = :phone, website = :website, updated_at = NOW()
            WHERE id = :id
            RETURNING id
            """
        ),
        {
            "id": club_id,
            "name": _crm_required(payload.name, "Club name"),
            "city": _crm_required(payload.city, "City"),
            "country": _crm_required(payload.country, "Country"),
            "logo": _crm_clean(payload.logo),
            "email": _crm_clean(payload.email),
            "phone": _crm_clean(payload.phone),
            "website": _crm_clean(payload.website),
        },
    ).fetchone()
    if not updated:
        raise HTTPException(status_code=404, detail="Club not found")
    session.commit()
    return get_crm_club(club_id, session)


@app.delete("/crm/clubs/{club_id}")
def delete_crm_club(club_id: str, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    deleted = session.execute(text("DELETE FROM crm_clubs WHERE id = :id RETURNING id"), {"id": club_id}).fetchone()
    if not deleted:
        raise HTTPException(status_code=404, detail="Club not found")
    session.commit()
    return {"deleted": True}


@app.get("/crm/players")
def crm_players(
    page: int = Query(1),
    pageSize: int = Query(25),
    search: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    _ensure_crm_schema(session)
    page, page_size, offset = _crm_pagination(page, pageSize)
    params: dict[str, Any] = {"limit": page_size, "offset": offset}
    where = ""
    if search:
        params["search"] = f"%{search.strip().lower()}%"
        where = """
        WHERE LOWER(p.first_name) LIKE :search OR LOWER(p.last_name) LIKE :search
           OR LOWER(p.position) LIKE :search OR LOWER(p.nationality) LIKE :search
           OR LOWER(c.name) LIKE :search
        """
    total = session.execute(
        text(f"SELECT COUNT(*) AS count FROM crm_players p JOIN crm_clubs c ON c.id = p.club_id {where}"),
        params,
    ).fetchone().count
    rows = session.execute(
        text(
            f"""
            SELECT p.*, c.name AS club_name, c.city AS club_city, c.country AS club_country
            FROM crm_players p
            JOIN crm_clubs c ON c.id = p.club_id
            {where}
            ORDER BY LOWER(p.last_name), LOWER(p.first_name)
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    ).fetchall()
    return _crm_page_response([_row_to_dict(row) for row in rows], int(total or 0), page, page_size)


@app.post("/crm/players")
def create_crm_player(payload: CrmPlayerPayload, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    if not session.execute(text("SELECT 1 FROM crm_clubs WHERE id = :id"), {"id": payload.club_id}).fetchone():
        raise HTTPException(status_code=400, detail="A valid club_id is required")
    first_name = _crm_required(payload.first_name, "First name")
    last_name = _crm_clean(payload.last_name) or ""
    duplicate = session.execute(
        text(
            """
            SELECT 1 FROM crm_players
            WHERE LOWER(BTRIM(first_name)) = LOWER(BTRIM(:first_name))
              AND LOWER(BTRIM(last_name)) = LOWER(BTRIM(:last_name))
              AND club_id = :club_id
            LIMIT 1
            """
        ),
        {"first_name": first_name, "last_name": last_name, "club_id": payload.club_id},
    ).fetchone()
    if duplicate:
        raise HTTPException(status_code=409, detail="A player with the same name and club already exists")
    player_id = _crm_new_id("player")
    session.execute(
        text(
            """
            INSERT INTO crm_players (
              id, first_name, last_name, age, position, nationality, photo, email, phone, club_id, created_at, updated_at
            ) VALUES (
              :id, :first_name, :last_name, :age, :position, :nationality, :photo, :email, :phone, :club_id, NOW(), NOW()
            )
            """
        ),
        {
            "id": player_id,
            "first_name": first_name,
            "last_name": last_name,
            "age": int(payload.age or 0),
            "position": _crm_required(payload.position, "Position"),
            "nationality": _crm_required(payload.nationality, "Nationality"),
            "photo": _crm_clean(payload.photo),
            "email": _crm_clean(payload.email),
            "phone": _crm_clean(payload.phone),
            "club_id": payload.club_id,
        },
    )
    session.commit()
    return get_crm_player(player_id, session)


@app.get("/crm/players/export.xlsx")
def export_crm_players(request: Request, session: Session = Depends(get_session)):
    _require_admin(request)
    _ensure_crm_schema(session)
    rows = session.execute(
        text(
            """
            SELECT p.first_name, p.last_name, p.age, p.position, p.nationality, c.name AS club_name, p.email, p.phone
            FROM crm_players p JOIN crm_clubs c ON c.id = p.club_id
            ORDER BY LOWER(p.last_name), LOWER(p.first_name)
            """
        )
    ).fetchall()
    sheet = [["First name", "Last name", "Age", "Position", "Nationality", "Club", "Email", "Phone"]]
    sheet.extend([list(row._mapping.values()) for row in rows])
    return Response(
        _build_xlsx([("CRM Players", sheet)]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="crm_players.xlsx"'},
    )


@app.get("/crm/players/{player_id}")
def get_crm_player(player_id: str, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    row = session.execute(
        text(
            """
            SELECT p.*, c.name AS club_name, c.city AS club_city, c.country AS club_country
            FROM crm_players p JOIN crm_clubs c ON c.id = p.club_id
            WHERE p.id = :id
            """
        ),
        {"id": player_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Player not found")
    payload = _row_to_dict(row)
    payload["contacts"] = [
        _row_to_dict(contact)
        for contact in session.execute(
            text(_crm_contact_select_sql("WHERE c.player_id = :id ORDER BY LOWER(c.last_name), LOWER(c.first_name) LIMIT 100")),
            {"id": player_id},
        ).fetchall()
    ]
    return payload


@app.patch("/crm/players/{player_id}")
@app.put("/crm/players/{player_id}")
def update_crm_player(player_id: str, payload: CrmPlayerPayload, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    updated = session.execute(
        text(
            """
            UPDATE crm_players
            SET first_name = :first_name, last_name = :last_name, age = :age, position = :position,
                nationality = :nationality, photo = :photo, email = :email, phone = :phone,
                club_id = :club_id, updated_at = NOW()
            WHERE id = :id
            RETURNING id
            """
        ),
        {
            "id": player_id,
            "first_name": _crm_required(payload.first_name, "First name"),
            "last_name": _crm_clean(payload.last_name) or "",
            "age": int(payload.age or 0),
            "position": _crm_required(payload.position, "Position"),
            "nationality": _crm_required(payload.nationality, "Nationality"),
            "photo": _crm_clean(payload.photo),
            "email": _crm_clean(payload.email),
            "phone": _crm_clean(payload.phone),
            "club_id": _crm_required(payload.club_id, "Club"),
        },
    ).fetchone()
    if not updated:
        raise HTTPException(status_code=404, detail="Player not found")
    session.commit()
    return get_crm_player(player_id, session)


@app.delete("/crm/players/{player_id}")
def delete_crm_player(player_id: str, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    deleted = session.execute(text("DELETE FROM crm_players WHERE id = :id RETURNING id"), {"id": player_id}).fetchone()
    if not deleted:
        raise HTTPException(status_code=404, detail="Player not found")
    session.commit()
    return {"deleted": True}


@app.get("/crm/contacts")
def crm_contacts(
    page: int = Query(1),
    pageSize: int = Query(25),
    search: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    _ensure_crm_schema(session)
    page, page_size, offset = _crm_pagination(page, pageSize)
    params: dict[str, Any] = {"limit": page_size, "offset": offset}
    filters = []
    if search:
        params["search"] = f"%{search.strip().lower()}%"
        filters.append(
            """
            (LOWER(c.first_name) LIKE :search OR LOWER(c.last_name) LIKE :search OR LOWER(c.role) LIKE :search
             OR LOWER(COALESCE(club.name, '')) LIKE :search
             OR LOWER(COALESCE(player.first_name, '') || ' ' || COALESCE(player.last_name, '')) LIKE :search)
            """
        )
    if type in ("CLUB", "PLAYER"):
        params["type"] = type
        filters.append("c.type = :type")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    count_sql = f"""
        SELECT COUNT(*) AS count
        FROM crm_contacts c
        LEFT JOIN crm_clubs club ON club.id = c.club_id
        LEFT JOIN crm_players player ON player.id = c.player_id
        {where}
    """
    total = session.execute(text(count_sql), params).fetchone().count
    rows = session.execute(
        text(_crm_contact_select_sql(where) + " ORDER BY LOWER(c.last_name), LOWER(c.first_name) LIMIT :limit OFFSET :offset"),
        params,
    ).fetchall()
    return _crm_page_response([_row_to_dict(row) for row in rows], int(total or 0), page, page_size)


@app.post("/crm/contacts")
def create_crm_contact(payload: CrmContactPayload, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    contact_type = payload.type if payload.type in ("CLUB", "PLAYER") else "CLUB"
    club_id = _crm_clean(payload.club_id)
    player_id = _crm_clean(payload.player_id)
    first_name = _crm_required(payload.first_name, "First name")
    last_name = _crm_required(payload.last_name, "Last name")
    duplicate = session.execute(
        text(
            """
            SELECT 1 FROM crm_contacts
            WHERE LOWER(BTRIM(first_name)) = LOWER(BTRIM(:first_name))
              AND LOWER(BTRIM(last_name)) = LOWER(BTRIM(:last_name))
              AND COALESCE(club_id, '') = COALESCE(:club_id, '')
              AND COALESCE(player_id, '') = COALESCE(:player_id, '')
            LIMIT 1
            """
        ),
        {"first_name": first_name, "last_name": last_name, "club_id": club_id, "player_id": player_id},
    ).fetchone()
    if duplicate:
        raise HTTPException(status_code=409, detail="A contact with the same name and relation already exists")
    contact_id = _crm_new_id("contact")
    session.execute(
        text(
            """
            INSERT INTO crm_contacts (
              id, first_name, last_name, role, email, phone, type, notes, club_id, player_id, created_at, updated_at
            ) VALUES (
              :id, :first_name, :last_name, :role, :email, :phone, :type, :notes, :club_id, :player_id, NOW(), NOW()
            )
            """
        ),
        {
            "id": contact_id,
            "first_name": first_name,
            "last_name": last_name,
            "role": _crm_required(payload.role, "Role"),
            "email": _crm_clean(payload.email),
            "phone": _crm_clean(payload.phone),
            "type": contact_type,
            "notes": _crm_clean(payload.notes),
            "club_id": club_id,
            "player_id": player_id,
        },
    )
    session.commit()
    return get_crm_contact(contact_id, session)


@app.post("/crm/contacts/import")
async def import_crm_contacts(request: Request, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    body = (await request.body()).decode("utf-8-sig")
    imported = 0
    skipped = 0
    for row in csv.DictReader(io.StringIO(body)):
        first_name = _crm_clean(row.get("firstName") or row.get("first_name"))
        last_name = _crm_clean(row.get("lastName") or row.get("last_name"))
        role = _crm_clean(row.get("role"))
        if not first_name or not last_name or not role:
            skipped += 1
            continue
        contact_type = row.get("type") if row.get("type") in ("CLUB", "PLAYER") else "CLUB"
        session.execute(
            text(
                """
                INSERT INTO crm_contacts (
                  id, first_name, last_name, role, email, phone, type, notes, club_id, player_id, created_at, updated_at
                ) VALUES (
                  :id, :first_name, :last_name, :role, :email, :phone, :type, :notes, :club_id, :player_id, NOW(), NOW()
                )
                """
            ),
            {
                "id": _crm_new_id("contact"),
                "first_name": first_name,
                "last_name": last_name,
                "role": role,
                "email": _crm_clean(row.get("email")),
                "phone": _crm_clean(row.get("phone")),
                "type": contact_type,
                "notes": _crm_clean(row.get("notes")),
                "club_id": _crm_clean(row.get("clubId") or row.get("club_id")),
                "player_id": _crm_clean(row.get("playerId") or row.get("player_id")),
            },
        )
        imported += 1
    session.commit()
    return {"imported": imported, "skipped": skipped}


@app.get("/crm/contacts/export.xlsx")
def export_crm_contacts(request: Request, session: Session = Depends(get_session)):
    _require_admin(request)
    _ensure_crm_schema(session)
    rows = session.execute(
        text(
            _crm_contact_select_sql("")
            + " ORDER BY LOWER(c.last_name), LOWER(c.first_name)"
        )
    ).fetchall()
    sheet = [["First name", "Last name", "Role", "Type", "Email", "Phone", "Club", "Player", "Notes"]]
    for row in rows:
        item = _row_to_dict(row)
        player_name = " ".join([item.get("player_first_name") or "", item.get("player_last_name") or ""]).strip()
        sheet.append([
            item.get("first_name"),
            item.get("last_name"),
            item.get("role"),
            item.get("type"),
            item.get("email"),
            item.get("phone"),
            item.get("club_name"),
            player_name,
            item.get("notes"),
        ])
    return Response(
        _build_xlsx([("CRM Contacts", sheet)]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="crm_contacts.xlsx"'},
    )


@app.get("/crm/contacts/{contact_id}")
def get_crm_contact(contact_id: str, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    row = session.execute(text(_crm_contact_select_sql("WHERE c.id = :id")), {"id": contact_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Contact not found")
    payload = _row_to_dict(row)
    prospect = session.execute(text("SELECT * FROM crm_prospects WHERE contact_id = :id"), {"id": contact_id}).fetchone()
    payload["prospect"] = _row_to_dict(prospect) if prospect else None
    return payload


@app.patch("/crm/contacts/{contact_id}")
@app.put("/crm/contacts/{contact_id}")
def update_crm_contact(contact_id: str, payload: CrmContactPayload, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    updated = session.execute(
        text(
            """
            UPDATE crm_contacts
            SET first_name = :first_name, last_name = :last_name, role = :role, email = :email,
                phone = :phone, type = :type, notes = :notes, club_id = :club_id,
                player_id = :player_id, updated_at = NOW()
            WHERE id = :id
            RETURNING id
            """
        ),
        {
            "id": contact_id,
            "first_name": _crm_required(payload.first_name, "First name"),
            "last_name": _crm_required(payload.last_name, "Last name"),
            "role": _crm_required(payload.role, "Role"),
            "email": _crm_clean(payload.email),
            "phone": _crm_clean(payload.phone),
            "type": payload.type if payload.type in ("CLUB", "PLAYER") else "CLUB",
            "notes": _crm_clean(payload.notes),
            "club_id": _crm_clean(payload.club_id),
            "player_id": _crm_clean(payload.player_id),
        },
    ).fetchone()
    if not updated:
        raise HTTPException(status_code=404, detail="Contact not found")
    session.commit()
    return get_crm_contact(contact_id, session)


@app.delete("/crm/contacts/{contact_id}")
def delete_crm_contact(contact_id: str, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    deleted = session.execute(text("DELETE FROM crm_contacts WHERE id = :id RETURNING id"), {"id": contact_id}).fetchone()
    if not deleted:
        raise HTTPException(status_code=404, detail="Contact not found")
    session.commit()
    return {"deleted": True}


@app.get("/crm/prospects")
def crm_prospects(
    stage: Optional[str] = Query(None),
    sort: str = Query("date"),
    session: Session = Depends(get_session),
):
    _ensure_crm_schema(session)
    params: dict[str, Any] = {}
    where = ""
    if stage in CRM_PROSPECT_STAGES:
        where = "WHERE p.stage = :stage"
        params["stage"] = stage
    order_by = {
        "name": "LOWER(c.last_name), LOWER(c.first_name)",
        "entity": "LOWER(COALESCE(club.name, player.last_name, c.last_name))",
        "date": "p.created_at DESC",
    }.get(sort, "p.created_at DESC")
    rows = session.execute(text(_crm_prospect_select_sql(where) + f" ORDER BY {order_by}"), params).fetchall()
    items = [_row_to_dict(row) for row in rows]
    return {"data": items, "stages": CRM_PROSPECT_STAGES}


@app.post("/crm/prospects")
def create_crm_prospect(payload: CrmProspectPayload, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    if payload.stage not in CRM_PROSPECT_STAGES:
        raise HTTPException(status_code=400, detail="Invalid prospect stage")
    if not session.execute(text("SELECT 1 FROM crm_contacts WHERE id = :id"), {"id": payload.contact_id}).fetchone():
        raise HTTPException(status_code=400, detail="A valid contact_id is required")
    existing = session.execute(text("SELECT * FROM crm_prospects WHERE contact_id = :id"), {"id": payload.contact_id}).fetchone()
    if existing:
        return _row_to_dict(existing)
    prospect_id = _crm_new_id("prospect")
    session.execute(
        text(
            """
            INSERT INTO crm_prospects (id, contact_id, stage, notes, created_at, updated_at)
            VALUES (:id, :contact_id, :stage, :notes, NOW(), NOW())
            """
        ),
        {"id": prospect_id, "contact_id": payload.contact_id, "stage": payload.stage, "notes": _crm_clean(payload.notes)},
    )
    session.commit()
    return get_crm_prospect(prospect_id, session)


@app.get("/crm/prospects/{prospect_id}")
def get_crm_prospect(prospect_id: str, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    row = session.execute(text(_crm_prospect_select_sql("WHERE p.id = :id")), {"id": prospect_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Prospect not found")
    return _row_to_dict(row)


@app.patch("/crm/prospects/{prospect_id}")
@app.put("/crm/prospects/{prospect_id}")
def update_crm_prospect(prospect_id: str, payload: CrmProspectPayload, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    if payload.stage not in CRM_PROSPECT_STAGES:
        raise HTTPException(status_code=400, detail="Invalid prospect stage")
    updated = session.execute(
        text(
            """
            UPDATE crm_prospects
            SET stage = :stage, notes = :notes, updated_at = NOW()
            WHERE id = :id
            RETURNING id
            """
        ),
        {"id": prospect_id, "stage": payload.stage, "notes": _crm_clean(payload.notes)},
    ).fetchone()
    if not updated:
        raise HTTPException(status_code=404, detail="Prospect not found")
    session.commit()
    return get_crm_prospect(prospect_id, session)


@app.delete("/crm/prospects/{prospect_id}")
def delete_crm_prospect(prospect_id: str, session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    deleted = session.execute(text("DELETE FROM crm_prospects WHERE id = :id RETURNING id"), {"id": prospect_id}).fetchone()
    if not deleted:
        raise HTTPException(status_code=404, detail="Prospect not found")
    session.commit()
    return {"deleted": True}


@app.get("/crm/cities")
def crm_cities(session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    rows = session.execute(
        text(
            """
            SELECT city, country, COUNT(*) AS club_count,
                   json_agg(json_build_object('id', id, 'name', name, 'logo', logo) ORDER BY name) AS clubs
            FROM crm_clubs
            WHERE city IS NOT NULL AND city <> ''
            GROUP BY city, country
            ORDER BY club_count DESC, city
            LIMIT 500
            """
        )
    ).fetchall()
    return {"data": [_row_to_dict(row) for row in rows]}


@app.get("/crm/map-clusters")
def crm_map_clusters(session: Session = Depends(get_session)):
    _ensure_crm_schema(session)
    rows = session.execute(
        text(
            """
            SELECT city, country, COUNT(*) AS club_count,
                   json_agg(json_build_object('id', id, 'name', name, 'logo', logo) ORDER BY name) AS clubs
            FROM crm_clubs
            WHERE city IS NOT NULL AND city <> ''
            GROUP BY city, country
            ORDER BY club_count DESC, city
            """
        )
    ).fetchall()
    mapped = []
    unmapped_clubs = 0
    unmapped_locations = 0
    for row in rows:
        item = _row_to_dict(row)
        coords = CRM_CITY_COORDS.get(_crm_location_key(item.get("city"), item.get("country")))
        if not coords:
            unmapped_clubs += int(item.get("club_count") or 0)
            unmapped_locations += 1
            continue
        lat, lon = coords
        item["lat"] = lat
        item["lon"] = lon
        mapped.append(item)
    mapped.sort(key=lambda item: int(item.get("club_count") or 0), reverse=True)
    return {
        "data": mapped,
        "mapped_clubs": sum(int(item.get("club_count") or 0) for item in mapped),
        "mapped_locations": len(mapped),
        "unmapped_clubs": unmapped_clubs,
        "unmapped_locations": unmapped_locations,
    }


def _build_player_context(report: Report) -> dict:
    player = report.player.model_dump()
    summary = report.summary
    raw_metrics = report.raw_metrics
    metrics = report.metrics

    key_metrics = {}
    for metric in [
        "goals_per_90",
        "xg_per_90",
        "xa_per_90",
        "assists_per_90",
        "goals",
        "xg",
        "xa",
        "assists",
        "minutes_played",
    ]:
        if metric in metrics:
            key_metrics[metric] = metrics.get(metric)

    return {
        "player": player,
        "summary": summary,
        "role_metrics": raw_metrics,
        "key_metrics": key_metrics,
        "radar_metrics": report.radar_metrics,
        "available_seasons": [item.model_dump() for item in report.available_seasons],
        "score_history": [item.model_dump() for item in report.score_history],
    }


def _truncate_title(text: str, max_len: int = 60) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3] + "..."


def _normalize_name(value: str) -> str:
    cleaned = unicodedata.normalize("NFKD", value)
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", cleaned.lower())


def _normalize_phrase(value: str) -> str:
    cleaned = unicodedata.normalize("NFKD", value)
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned.lower())
    return " ".join(cleaned.split())


_CLUB_LOGO_ALIASES: Optional[dict[str, str]] = None


def _club_name_variants(value: Optional[str]) -> list[str]:
    normalized = _normalize_phrase(value or "")
    if not normalized:
        return []
    variants = {normalized}
    words = normalized.split()
    stop_words = {"fc", "cf", "sc", "afc", "ac", "as", "fk", "sk", "cd", "sd", "ud", "club", "football", "futbol", "soccer"}
    stripped = [word for word in words if word not in stop_words]
    if stripped and stripped != words:
        variants.add(" ".join(stripped))
    expanded = [{"utd": "united", "st": "saint"}.get(word, word) for word in words]
    if expanded != words:
        variants.add(" ".join(expanded))
    return list(variants)


def _load_club_logo_aliases() -> dict[str, str]:
    global _CLUB_LOGO_ALIASES
    if _CLUB_LOGO_ALIASES is not None:
        return _CLUB_LOGO_ALIASES
    path = Path(__file__).resolve().parent / "data" / "club_logos.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _CLUB_LOGO_ALIASES = payload.get("aliases") or {}
    except (OSError, json.JSONDecodeError):
        _CLUB_LOGO_ALIASES = {}
    return _CLUB_LOGO_ALIASES


def _club_logo_url(club_name: Optional[str]) -> str:
    aliases = _load_club_logo_aliases()
    for variant in _club_name_variants(club_name):
        if aliases.get(variant):
            return aliases[variant]
    return ""


def _normalized_transfer_names(*names: Optional[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for name in names:
        normalized = _normalize_phrase(name or "")
        if normalized and normalized not in seen:
            seen.add(normalized)
            values.append(normalized)
    return values


def _team_matches_transfer(row: dict[str, Any], club_name: Optional[str]) -> bool:
    club = _normalize_phrase(club_name or "")
    if not club:
        return False
    teams = [
        row.get("team_name_context"),
        row.get("team_in_name"),
        row.get("team_out_name"),
    ]
    for team in teams:
        normalized_team = _normalize_phrase(team or "")
        if normalized_team and (normalized_team == club or normalized_team in club or club in normalized_team):
            return True
    return False


def _player_transfer_history(session: Session, payload: dict[str, Any]) -> list[dict[str, Any]]:
    _ensure_agency_ops_schema(session)
    names = _normalized_transfer_names(payload.get("linked_player_name"), payload.get("display_name"))
    player_id = payload.get("player_id")
    if not player_id and not names:
        return []
    where_parts = []
    params: dict[str, Any] = {}
    if player_id:
        where_parts.append("linked_player_id = :player_id")
        params["player_id"] = int(player_id)
    if names:
        where_parts.append("normalized_player_name = ANY(CAST(:names AS TEXT[]))")
        params["names"] = names
    rows = session.execute(
        text(
            f"""
            SELECT *
            FROM player_transfer_history
            WHERE {" OR ".join(where_parts)}
            ORDER BY transfer_date DESC NULLS LAST, id DESC
            LIMIT 80
            """
        ),
        params,
    ).fetchall()
    items: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        item = _row_to_dict(row)
        exact_id = bool(player_id and item.get("linked_player_id") == player_id)
        if not exact_id and not _team_matches_transfer(item, payload.get("current_club")):
            continue
        key = (
            item.get("source_player_id"),
            item.get("player_name"),
            item.get("transfer_date"),
            item.get("transfer_type"),
            item.get("team_in_name"),
            item.get("team_out_name"),
            item.get("transfer_fee"),
        )
        if key in seen:
            continue
        seen.add(key)
        item["team_in_logo_url"] = _club_logo_url(item.get("team_in_name"))
        item["team_out_logo_url"] = _club_logo_url(item.get("team_out_name"))
        item["team_context_logo_url"] = _club_logo_url(item.get("team_name_context"))
        item["match_type"] = "player_id" if exact_id else "name_club"
        items.append(item)
    return items[:30]


def _s3_client():
    endpoint = os.getenv("S3_ENDPOINT") or "http://minio:9000"
    access_key = os.getenv("S3_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("S3_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    if not access_key or not secret_key:
        raise HTTPException(status_code=500, detail="Object storage credentials are not configured")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION") or "us-east-1",
    )


def _s3_bucket() -> str:
    bucket = os.getenv("S3_BUCKET") or "nextlegend"
    if not bucket:
        raise HTTPException(status_code=500, detail="Object storage bucket is not configured")
    return bucket


def _ensure_s3_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def _storage_proxy_url(request: Request, file_key: str) -> str:
    quoted = urllib.parse.quote(file_key, safe="")
    public_base = os.getenv("PUBLIC_API_BASE_URL") or os.getenv("API_PUBLIC_BASE_URL")
    if public_base:
        base_url = public_base.rstrip("/")
    else:
        host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        base_url = "http://localhost:8000" if host in {"api", "api:8000"} else f"{proto}://{host}".rstrip("/")
    return f"{base_url}/hd-players/files/{quoted}"


def _manual_transfer_rows(session: Session, hd_player_id: int) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT *
            FROM hd_player_manual_transfers
            WHERE hd_player_id = :hd_player_id
            ORDER BY transfer_date DESC NULLS LAST, id DESC
            """
        ),
        {"hd_player_id": hd_player_id},
    ).fetchall()
    items = []
    for row in rows:
        item = _row_to_dict(row)
        item["source"] = "manual"
        item["player_name"] = ""
        item["team_in_logo_url"] = _club_logo_url(item.get("team_in_name"))
        item["team_out_logo_url"] = _club_logo_url(item.get("team_out_name"))
        item["team_context_logo_url"] = _club_logo_url(item.get("team_in_name") or item.get("team_out_name"))
        item["match_type"] = "manual"
        items.append(item)
    return items


def _resolve_llm_model_name(llm) -> str:
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if value:
            return str(value)
    return os.getenv("OPENAI_MODEL", "gpt-4o")


def _estimate_openai_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    model_lower = str(model or "").lower()
    if "gpt-4o-mini" in model_lower:
        prompt_rate = 0.15
        completion_rate = 0.60
    elif "gpt-4o" in model_lower:
        prompt_rate = 5.0
        completion_rate = 15.0
    else:
        return 0.0
    cost = (prompt_tokens / 1_000_000) * prompt_rate + (completion_tokens / 1_000_000) * completion_rate
    return round(cost, 6)


def _usage_from_callback(cb, model: str, label: str) -> dict[str, object]:
    prompt_tokens = int(getattr(cb, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(cb, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(cb, "total_tokens", prompt_tokens + completion_tokens) or 0)
    cost = _estimate_openai_cost(model, prompt_tokens, completion_tokens)
    return {
        "label": label,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
    }


def _merge_usage_calls(calls: list[dict[str, object]]) -> dict[str, object]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    model = None
    for call in calls:
        totals["prompt_tokens"] += int(call.get("prompt_tokens", 0) or 0)
        totals["completion_tokens"] += int(call.get("completion_tokens", 0) or 0)
        totals["total_tokens"] += int(call.get("total_tokens", 0) or 0)
        totals["estimated_cost_usd"] += float(call.get("estimated_cost_usd", 0.0) or 0.0)
        if not model and call.get("model"):
            model = call.get("model")
    totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 6)
    return {**totals, "model": model, "calls": calls}


def _extract_usage_from_payload(payload: object) -> Optional[dict[str, object]]:
    if payload is None:
        return None
    data = payload
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if usage is None:
        return None
    if isinstance(usage, str):
        try:
            usage = json.loads(usage)
        except json.JSONDecodeError:
            return None
    if not isinstance(usage, dict):
        return None
    return usage


def _aggregate_usage(payloads: list[object]) -> dict[str, object]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    model = None
    for payload in payloads:
        usage = _extract_usage_from_payload(payload)
        if not usage:
            continue
        totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        totals["estimated_cost_usd"] += float(usage.get("estimated_cost_usd", 0.0) or 0.0)
        if not model and usage.get("model"):
            model = usage.get("model")
    totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 6)
    totals["model"] = model
    return totals


def _find_helper_csv(filename: str) -> Optional[Path]:
    candidates = [
        Path("/helpers/csv") / filename,
        Path(__file__).resolve().parent / "helpers" / "csv" / filename,
        Path(__file__).resolve().parent / "helpers" / filename,
        Path(__file__).resolve().parent.parent / "helpers" / "csv" / filename,
        Path(__file__).resolve().parent.parent.parent / "helpers" / "csv" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_league_translation_meta() -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    global _LEAGUE_META, _LEAGUE_META_MTIME, _LEAGUE_ALIAS_MAP
    path = _find_helper_csv("league_translation_meta.csv")
    if not path:
        return {}, {}
    mtime = path.stat().st_mtime
    if _LEAGUE_META is not None and _LEAGUE_META_MTIME == mtime:
        return _LEAGUE_META, _LEAGUE_ALIAS_MAP or {}

    meta: dict[str, dict[str, float]] = {}
    alias_map: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            competition = (row.get("competition") or "").strip()
            if not competition:
                continue
            difficulty = row.get("difficulty")
            intensity = row.get("intensity")
            clubs_count = row.get("clubs_count")
            meta[competition] = {
                "difficulty": float(difficulty) if difficulty else 0.0,
                "intensity": float(intensity) if intensity else 0.0,
                "clubs_count": float(clubs_count) if clubs_count else 0.0,
            }

            normalized_full = _normalize_phrase(competition)
            if normalized_full:
                alias_map.setdefault(normalized_full, []).append(competition)
            if ". " in competition:
                short_name = competition.split(". ", 1)[1]
                normalized_short = _normalize_phrase(short_name)
                if normalized_short:
                    alias_map.setdefault(normalized_short, []).append(competition)

    _LEAGUE_META = meta
    _LEAGUE_META_MTIME = mtime
    _LEAGUE_ALIAS_MAP = alias_map
    return meta, alias_map


def _load_league_aliases() -> dict[str, list[str]]:
    global _LEAGUE_ALIAS_EXTRA, _LEAGUE_ALIAS_EXTRA_MTIME
    path = Path(__file__).resolve().parent / "helpers" / "league_aliases.json"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if _LEAGUE_ALIAS_EXTRA is not None and _LEAGUE_ALIAS_EXTRA_MTIME == mtime:
        return _LEAGUE_ALIAS_EXTRA
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    aliases: dict[str, list[str]] = {}
    if isinstance(data, dict):
        items = data.items()
    else:
        return {}
    for alias, value in items:
        normalized = _normalize_phrase(str(alias))
        if not normalized:
            continue
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
        else:
            values = [str(value).strip()]
        if not values:
            continue
        aliases.setdefault(normalized, [])
        for entry in values:
            if entry and entry not in aliases[normalized]:
                aliases[normalized].append(entry)
    _LEAGUE_ALIAS_EXTRA = aliases
    _LEAGUE_ALIAS_EXTRA_MTIME = mtime
    return aliases


def _resolve_country_from_text(normalized_text: str) -> Optional[str]:
    country_aliases = {
        "spain": "Spain",
        "espagne": "Spain",
        "espana": "Spain",
        "espagnol": "Spain",
        "espagnole": "Spain",
        "spanish": "Spain",
        "france": "France",
        "italy": "Italy",
        "italie": "Italy",
        "england": "England",
        "angleterre": "England",
        "germany": "Germany",
        "allemagne": "Germany",
        "portugal": "Portugal",
        "netherlands": "Netherlands",
        "pays bas": "Netherlands",
        "belgium": "Belgium",
        "belgique": "Belgium",
        "brazil": "Brazil",
        "bresil": "Brazil",
        "argentina": "Argentina",
        "argentine": "Argentina",
    }
    for key, value in country_aliases.items():
        if key in normalized_text:
            return value
    return None


def _resolve_league_from_text(text: str) -> Optional[str]:
    normalized_text = _normalize_phrase(text)
    if not normalized_text:
        return None

    league_meta, alias_map = _load_league_translation_meta()
    if not league_meta:
        return None
    alias_extra = _load_league_aliases()
    if alias_extra:
        merged_map = {**alias_map}
        for key, values in alias_extra.items():
            merged_map.setdefault(key, [])
            for value in values:
                if value not in merged_map[key]:
                    merged_map[key].append(value)
        alias_map = merged_map

    blacklist = {"liga", "league", "division", "serie"}
    matched: list[str] = []
    for alias, competitions in alias_map.items():
        if not alias or alias in blacklist:
            continue
        if alias in normalized_text:
            matched.extend(competitions)

    if matched:
        matched_unique = []
        for comp in matched:
            if comp not in matched_unique:
                matched_unique.append(comp)
        if len(matched_unique) == 1:
            return matched_unique[0]
        country = _resolve_country_from_text(normalized_text)
        if country:
            for comp in matched_unique:
                if comp.lower().startswith(country.lower() + "."):
                    return comp
        return max(
            matched_unique,
            key=lambda comp: league_meta.get(comp, {}).get("difficulty", 0.0),
        )

    country = _resolve_country_from_text(normalized_text)
    if country:
        return f"{country}.%"
    return None


def _load_opta_power_rankings() -> tuple[
    dict[str, dict[str, float | int | str]],
    list[tuple[str, dict[str, float | int | str]]],
]:
    global _OPTA_CLUBS, _OPTA_CLUBS_MTIME, _OPTA_CLUBS_SORTED
    path = _find_helper_csv("opta_power_rankings.csv")
    if not path:
        return {}, []
    mtime = path.stat().st_mtime
    if _OPTA_CLUBS is not None and _OPTA_CLUBS_MTIME == mtime:
        return _OPTA_CLUBS, _OPTA_CLUBS_SORTED or []

    clubs: dict[str, dict[str, float | int | str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            team = (row.get("team") or "").strip()
            if not team:
                continue
            normalized = _normalize_name(team)
            if not normalized:
                continue
            rating = row.get("rating")
            rank = row.get("rank")
            parsed_rating = float(rating) if rating else 0.0
            parsed_rank = int(rank) if rank else 0
            existing = clubs.get(normalized)
            if existing and float(existing.get("rating") or 0.0) >= parsed_rating:
                continue
            clubs[normalized] = {
                "team": team,
                "rating": parsed_rating,
                "rank": parsed_rank,
            }

    sorted_clubs = sorted(
        clubs.items(), key=lambda item: len(item[0]), reverse=True
    )
    _OPTA_CLUBS = clubs
    _OPTA_CLUBS_MTIME = mtime
    _OPTA_CLUBS_SORTED = [(norm, data) for norm, data in sorted_clubs]
    return clubs, _OPTA_CLUBS_SORTED


def _resolve_target_club(text: str) -> Optional[dict[str, float | int | str]]:
    clubs, sorted_clubs = _load_opta_power_rankings()
    if not clubs:
        return None
    normalized_text = _normalize_name(text)
    if not normalized_text:
        return None
    for normalized, data in sorted_clubs:
        if normalized and normalized in normalized_text:
            return data
    return None


def _club_power_for_team(team: Optional[str]) -> dict[str, float | int | str | None]:
    if not team:
        return {"rating": None, "rank": None, "matched_team": None}
    clubs, _ = _load_opta_power_rankings()
    normalized = _normalize_name(team)
    data = clubs.get(normalized) if normalized else None
    if data is None:
        data = _resolve_target_club(team)
    if not data:
        return {"rating": None, "rank": None, "matched_team": None}
    return {
        "rating": float(data.get("rating") or 0.0),
        "rank": int(data.get("rank") or 0) or None,
        "matched_team": data.get("team"),
    }


def _player_search_relevance(item: dict, query: str) -> int:
    query_norm = _normalize_name(query)
    name_norm = _normalize_name(str(item.get("name") or ""))
    if query_norm and name_norm == query_norm:
        return 5
    if query_norm and name_norm.startswith(query_norm):
        return 4
    query_tokens = [_normalize_name(token) for token in re.split(r"\s+", query) if len(token) >= 2]
    if query_tokens and all(token and token in name_norm for token in query_tokens):
        return 3
    if query_tokens:
        words = [_normalize_name(part) for part in re.split(r"\s+", str(item.get("name") or "")) if part]
        if all(any(word.startswith(token) for word in words) for token in query_tokens):
            return 2
    haystack = _normalize_name(" ".join(str(item.get(key) or "") for key in ("team", "competition_name", "calendar")))
    if query_norm and query_norm in haystack:
        return 1
    return 0


def _club_strength_thresholds(target_rating: float) -> tuple[Optional[float], Optional[float]]:
    if target_rating >= 95:
        return 82.0, 8.0
    if target_rating >= 90:
        return 78.0, 7.5
    if target_rating >= 85:
        return 74.0, 7.0
    if target_rating >= 80:
        return 70.0, 6.5
    return None, None


def _apply_strength_filters(
    rows: list[dict],
    *,
    min_club_rating: Optional[float] = None,
    min_league_difficulty: Optional[float] = None,
) -> list[dict]:
    if min_club_rating is None and min_league_difficulty is None:
        return rows

    league_meta, _ = _load_league_translation_meta()
    clubs, _ = _load_opta_power_rankings()
    filtered: list[dict] = []
    for row in rows:
        keep = True
        if min_league_difficulty is not None:
            competition = row.get("competition_name")
            difficulty = league_meta.get(competition, {}).get("difficulty") if competition else None
            if difficulty is not None and difficulty < min_league_difficulty:
                keep = False
        if keep and min_club_rating is not None:
            team = row.get("team") or ""
            rating = clubs.get(_normalize_name(team), {}).get("rating") if team else None
            if rating is not None and rating < min_club_rating:
                keep = False
        if keep:
            filtered.append(row)
    return filtered


def _build_conversation_context(
    session: Session,
    conversation_id: int,
    *,
    exclude_message_id: Optional[int] = None,
    max_messages: int = 6,
    max_chars: int = 1200,
    include_roles: Optional[set[str]] = None,
) -> str:
    rows = session.execute(
        text(
            """
            SELECT id, role, content
            FROM ai_messages
            WHERE conversation_id = :conversation_id
            ORDER BY created_at DESC
            LIMIT :limit
            """
        ),
        {"conversation_id": conversation_id, "limit": max_messages},
    ).fetchall()
    lines = []
    length = 0
    for row in reversed(rows):
        if exclude_message_id and row.id == exclude_message_id:
            continue
        if include_roles is not None and row.role not in include_roles:
            continue
        content = (row.content or "").strip()
        if not content:
            continue
        prefix = "User" if row.role == "user" else "Assistant"
        line = f"{prefix}: {content}"
        length += len(line)
        if length > max_chars:
            break
        lines.append(line)
    if not lines:
        return ""
    return "Previous conversation context:\n" + "\n".join(lines)


def _extract_season_from_text(text: str) -> Optional[str]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return None
    lowered = raw_text.lower()
    if "current season" in lowered or "saison actuelle" in lowered:
        return CURRENT_SEASON_LABEL

    range_match = re.search(r"\b(20\d{2})\s*[/\-_]\s*((?:20)?\d{2,4})\b", raw_text)
    if range_match:
        start = int(range_match.group(1))
        end_raw = range_match.group(2)
        if len(end_raw) == 2:
            end = (start // 100) * 100 + int(end_raw)
            if end < start:
                end += 100
        else:
            end = int(end_raw[-4:])
        return f"{start}/{end}"

    season_year_match = re.search(
        r"(?:season|saison|campaign)[^0-9]{0,12}(20\d{2})",
        lowered,
    )
    if season_year_match:
        return season_year_match.group(1)
    return None


def _detect_trend_mode(text: str) -> Optional[str]:
    normalized = _normalize_phrase(text)
    if not normalized:
        return None
    decline_markers = [
        "plus en difficulte",
        "en difficulte aujourd hui",
        "en baisse",
        "regression",
        "moins performants aujourd hui",
        "decline",
        "drop off",
        "struggling now",
        "worse now",
    ]
    improve_markers = [
        "vice versa",
        "vice versa",
        "a l inverse",
        "inversement",
        "progression",
        "progresses",
        "improved now",
        "better now",
        "now performing better",
        "rebound",
    ]
    decline = any(marker in normalized for marker in decline_markers)
    improve = any(marker in normalized for marker in improve_markers)
    if decline and improve:
        return "both"
    if decline:
        return "decline_now"
    if improve:
        return "improved_now"
    return None


def _extract_prompt_constraints(text: str) -> dict:
    lowered = text.lower()
    normalized_text = _normalize_phrase(text)
    constraints: dict[str, object] = {}

    age_match = re.search(
        r"(?:max(?:imum)?|au maximum|moins de|under|<=)\s*(\d{1,2})",
        lowered,
    )
    if age_match:
        constraints["max_age"] = int(age_match.group(1))
    else:
        age_match = re.search(r"(\d{1,2})\s*(?:ans|yrs|years)", lowered)
        if age_match and "max_age" not in constraints:
            constraints["max_age"] = int(age_match.group(1))

    if "1/2" in lowered or "50%" in lowered or "half" in lowered or "moitié" in lowered:
        constraints["min_minutes_ratio"] = 0.5

    if "top ligue" in lowered or "top league" in lowered:
        constraints["min_league_strength"] = 6
    if re.search(r"\b(big|top)\\s*5\\b", lowered) or "big five" in lowered:
        constraints["min_league_strength"] = 8
    if (
        "club tres fort" in normalized_text
        or "top club" in lowered
        or "elite club" in lowered
    ):
        constraints["min_league_strength"] = max(
            int(constraints.get("min_league_strength") or 0), 7
        )

    league = _resolve_league_from_text(text)
    if league:
        constraints["league"] = league

    if "ailier" in lowered or "winger" in lowered:
        constraints["role"] = "ailier"

    position = resolve_position_from_text(text)
    if position and "position" not in constraints:
        constraints["position"] = position

    season = _extract_season_from_text(text)
    if season:
        constraints["season"] = season

    trend_mode = _detect_trend_mode(text)
    if trend_mode:
        constraints["trend_mode"] = trend_mode

    return constraints


def _trend_candidates(
    session: Session,
    filters: PlayerFilters,
    *,
    trend_mode: str,
    requested_count: Optional[int] = None,
) -> list[dict]:
    reference_labels = (
        _season_filter_values(filters.season) if filters.season else _current_season_labels()
    )
    if not reference_labels:
        reference_labels = [CURRENT_SEASON_LABEL]

    requested_limit = requested_count * 3 if requested_count else 0
    limit_value = max(filters.limit or 30, requested_limit or 0, 20)
    limit_value = min(limit_value, 80)
    params: dict[str, object] = {
        "current_labels": reference_labels,
        "limit": limit_value,
    }
    clauses: list[str] = ["cb.global_score_adjusted IS NOT NULL", "past.past_samples > 0"]

    if filters.league:
        league_value = str(filters.league).strip()
        if "*" in league_value or "%" in league_value:
            clauses.append("cb.competition_name ILIKE :league_pattern")
            params["league_pattern"] = league_value.replace("*", "%")
        else:
            clauses.append("LOWER(cb.competition_name) = LOWER(:league)")
            params["league"] = league_value
    if filters.role:
        clauses.append("cb.assigned_role ILIKE :role")
        params["role"] = f"%{str(filters.role).strip()}%"
    if filters.position:
        clauses.append("(cb.position ILIKE :position OR cb.second_position ILIKE :position)")
        params["position"] = f"%{str(filters.position).strip()}%"
    if filters.max_age is not None:
        clauses.append("cb.age <= :max_age")
        params["max_age"] = filters.max_age
    if filters.min_minutes is not None:
        clauses.append("cb.minutes_played >= :min_minutes")
        params["min_minutes"] = filters.min_minutes
    if filters.min_league_strength is not None:
        clauses.append("cb.league_strength_factor >= :min_league_strength")
        params["min_league_strength"] = filters.min_league_strength

    metric_cols = set(_stats_metric_columns(session))
    for metric, threshold in (filters.min_metrics or {}).items():
        if metric not in metric_cols:
            continue
        key = f"metric_{metric}"
        clauses.append(f'cb."{metric}" >= :{key}')
        params[key] = threshold

    where_clause = " AND ".join(clauses)
    order_dir = "ASC" if trend_mode == "decline_now" else "DESC"
    sql = f"""
    WITH current_base AS (
      SELECT
        ps.id AS player_season_id,
        ps.player_id,
        p.name AS player_name,
        c.name AS competition_name,
        ps.calendar,
        ps.team_in_selected_period AS team,
        ps.position,
        ps.second_position,
        ps.assigned_role,
        ps.minutes_played,
        ps.global_score_adjusted,
        ps.assigned_role_pct_league,
        ps.assigned_role_pct_global,
        ps.league_strength_factor,
        pm.age,
        pm."goals_per_90" AS goals_per_90,
        pm."xg_per_90" AS xg_per_90,
        pm."xa_per_90" AS xa_per_90,
        pm."assists_per_90" AS assists_per_90,
        pm."progressive_runs_per_90" AS progressive_runs_per_90,
        pm."successful_dribbles_percent" AS successful_dribbles_percent
      FROM player_seasons ps
      JOIN players p ON p.id = ps.player_id
      JOIN competitions c ON c.id = ps.competition_id
      LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
      WHERE ps.calendar = ANY(:current_labels)
    ),
    current_best AS (
      SELECT DISTINCT ON (player_id) *
      FROM current_base
      ORDER BY player_id, minutes_played DESC NULLS LAST, global_score_adjusted DESC NULLS LAST
    ),
    past_scores AS (
      SELECT
        ps.player_id,
        AVG(ps.global_score_adjusted) AS past_score_avg,
        MAX(ps.global_score_adjusted) AS past_score_peak,
        COUNT(*) AS past_samples
      FROM player_seasons ps
      WHERE ps.calendar <> ALL(:current_labels)
        AND ps.global_score_adjusted IS NOT NULL
      GROUP BY ps.player_id
    )
    SELECT
      cb.*,
      past.past_score_avg,
      past.past_score_peak,
      past.past_samples,
      (cb.global_score_adjusted - past.past_score_avg) AS trend_delta,
      CASE
        WHEN (cb.global_score_adjusted - past.past_score_avg) >= 0 THEN 'up'
        ELSE 'down'
      END AS trend_direction
    FROM current_best cb
    JOIN past_scores past ON past.player_id = cb.player_id
    WHERE {where_clause}
    ORDER BY trend_delta {order_dir}, cb.minutes_played DESC NULLS LAST
    LIMIT :limit
    """
    rows = session.execute(text(sql), params).fetchall()
    return [_row_to_dict(row) for row in rows]


def _attach_candidate_ids(candidates: list[dict], shortlist: list[dict]) -> list[dict]:
    name_index = {}
    shortlist_rows = []
    for row in shortlist:
        name = row.get("player_name") or row.get("name")
        if not name:
            continue
        norm = _normalize_name(name)
        name_index[norm] = row
        shortlist_rows.append((norm, row))

    for candidate in candidates:
        if candidate.get("player_id") is not None:
            continue
        name = candidate.get("player_name")
        if not name:
            continue
        norm = _normalize_name(name)
        match = name_index.get(norm)
        if match:
            candidate["player_id"] = match.get("player_id")
            if match.get("player_season_id") is not None:
                candidate["player_season_id"] = match.get("player_season_id")
            continue
        for norm_name, row in shortlist_rows:
            if norm in norm_name or norm_name in norm:
                candidate["player_id"] = row.get("player_id")
                if row.get("player_season_id") is not None:
                    candidate["player_season_id"] = row.get("player_season_id")
                break
    return candidates


def _ensure_candidate_count(
    candidates: list[dict],
    shortlist: list[dict],
    requested_count: Optional[int],
) -> list[dict]:
    if not requested_count:
        return candidates

    trimmed = candidates[:requested_count]
    existing_ids = {
        item.get("player_id")
        for item in trimmed
        if item.get("player_id") is not None
    }
    existing_names = {
        (item.get("player_name") or "").strip().lower()
        for item in trimmed
        if item.get("player_name")
    }

    for row in shortlist:
        if len(trimmed) >= requested_count:
            break
        player_id = row.get("player_id")
        name = row.get("player_name") or row.get("name")
        name_key = (name or "").strip().lower()
        if player_id in existing_ids or name_key in existing_names:
            continue
        trimmed.append(
            {
                "player_id": player_id,
                "player_name": name,
                "priority": 2,
                "reason": "High statistical fit from the shortlist.",
                "role_summary": row.get("assigned_role")
                or row.get("position")
                or "Profile fit.",
            }
        )
        existing_ids.add(player_id)
        existing_names.add(name_key)
    return trimmed


def _shortlist_unique_count(rows: list[dict]) -> int:
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    count = 0
    for row in rows:
        player_id = row.get("player_id")
        name = row.get("player_name") or row.get("name")
        name_key = _normalize_name(name) if name else ""
        if player_id is not None:
            if player_id in seen_ids:
                continue
            seen_ids.add(player_id)
        elif name_key:
            if name_key in seen_names:
                continue
        else:
            continue
        if name_key:
            seen_names.add(name_key)
        count += 1
    return count


def _run_scout_flow(
    session: Session,
    prompt: str,
    overrides: Optional[dict] = None,
    language_override: Optional[str] = None,
    conversation_context: Optional[str] = None,
    requested_count_hint: Optional[int] = None,
):
    try:
        llm = get_llm()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OpenAI client error: {exc}") from exc

    overrides = overrides or {}
    requested_count = requested_count_hint or extract_requested_count(prompt) or 3
    if requested_count and not overrides.get("limit"):
        overrides["limit"] = min(50, requested_count * 3)

    column_catalog = build_column_catalog(session)
    prompt_for_llm = (
        f"{conversation_context}\n\nLatest request: {prompt}"
        if conversation_context
        else prompt
    )
    usage_calls: list[dict[str, object]] = []
    model_name = _resolve_llm_model_name(llm)
    with get_openai_callback() as cb:
        filters = run_data_scientist(
            prompt_for_llm,
            column_catalog=column_catalog,
            overrides=overrides,
            llm=llm,
        )
    usage_calls.append(_usage_from_callback(cb, model_name, "data_scientist"))

    merged = {
        **filters.dict(),
        **{k: v for k, v in overrides.items() if v is not None},
    }
    filters = PlayerFilters.parse_obj(merged)
    if requested_count and filters.limit < requested_count * 3:
        filters = PlayerFilters.parse_obj(
            {**filters.dict(), "limit": min(50, requested_count * 3)}
        )

    constraints = _extract_prompt_constraints(prompt)
    if conversation_context:
        constraints = {**_extract_prompt_constraints(conversation_context), **constraints}
    trend_mode = str(constraints.get("trend_mode") or "").strip()
    for key, value in constraints.items():
        if value is None:
            continue
        if key == "trend_mode":
            continue
        if key == "league":
            if filters.league != value:
                filters = PlayerFilters.parse_obj({**filters.dict(), key: value})
            continue
        if getattr(filters, key, None) in (None, ""):
            filters = PlayerFilters.parse_obj({**filters.dict(), key: value})

    if trend_mode:
        if trend_mode == "both":
            decline_rows = _trend_candidates(
                session,
                filters,
                trend_mode="decline_now",
                requested_count=requested_count,
            )
            improve_rows = _trend_candidates(
                session,
                filters,
                trend_mode="improved_now",
                requested_count=requested_count,
            )
            shortlisted: list[dict] = []
            seen: set[int] = set()
            for row in decline_rows + improve_rows:
                pid = row.get("player_id")
                if pid is not None and pid in seen:
                    continue
                if pid is not None:
                    seen.add(pid)
                shortlisted.append(row)
                if len(shortlisted) >= (filters.limit or 30):
                    break
            shortlist_rows = shortlisted
        else:
            shortlist_rows = _trend_candidates(
                session,
                filters,
                trend_mode=trend_mode,
                requested_count=requested_count,
            )
        if not shortlist_rows:
            shortlist_rows = filter_candidates(session, filters)
        unique_count = _shortlist_unique_count(shortlist_rows)
    else:
        shortlist_rows = filter_candidates(session, filters)
        unique_count = _shortlist_unique_count(shortlist_rows)
        if requested_count and unique_count < requested_count and filters.limit < 50:
            expanded = PlayerFilters.parse_obj({**filters.dict(), "limit": 50})
            shortlist_rows = filter_candidates(session, expanded)
            filters = expanded
            unique_count = _shortlist_unique_count(shortlist_rows)

        if not shortlist_rows:
            relaxed = PlayerFilters.parse_obj(
                {
                    **filters.dict(),
                    "min_metrics": {},
                    "max_age": None,
                    "min_minutes": None,
                    "min_minutes_ratio": None,
                }
            )
            shortlist_rows = filter_candidates(session, relaxed)
            filters = relaxed
            unique_count = _shortlist_unique_count(shortlist_rows)

        if (
            requested_count
            and unique_count < requested_count
            and (
                filters.min_metrics
                or filters.min_minutes_ratio
                or filters.min_minutes
            )
        ):
            relaxed = PlayerFilters.parse_obj(
                {
                    **filters.dict(),
                    "min_metrics": {},
                    "min_minutes_ratio": None,
                    "min_minutes": None,
                }
            )
            shortlist_rows = filter_candidates(session, relaxed)
            filters = relaxed
            unique_count = _shortlist_unique_count(shortlist_rows)

        if (
            requested_count
            and unique_count < requested_count
            and filters.min_league_strength is not None
        ):
            relaxed = PlayerFilters.parse_obj(
                {
                    **filters.dict(),
                    "min_league_strength": None,
                }
            )
            shortlist_rows = filter_candidates(session, relaxed)
            filters = relaxed
            unique_count = _shortlist_unique_count(shortlist_rows)

        if (
            (not shortlist_rows or (requested_count and unique_count < requested_count))
            and filters.league
            and "league" not in constraints
        ):
            relaxed = PlayerFilters.parse_obj({**filters.dict(), "league": None})
            shortlist_rows = filter_candidates(session, relaxed)
            filters = relaxed
            unique_count = _shortlist_unique_count(shortlist_rows)

    target_club = _resolve_target_club(f"{conversation_context or ''} {prompt}")
    if target_club:
        target_rating = target_club.get("rating")
        if isinstance(target_rating, (int, float)) and target_rating > 0:
            min_club_rating, min_league_difficulty = _club_strength_thresholds(
                float(target_rating)
            )
            if min_club_rating or min_league_difficulty:
                shortlist_rows = _apply_strength_filters(
                    shortlist_rows,
                    min_club_rating=min_club_rating,
                    min_league_difficulty=min_league_difficulty,
                )
                unique_count = _shortlist_unique_count(shortlist_rows)
                if requested_count and unique_count < requested_count and min_club_rating:
                    shortlist_rows = _apply_strength_filters(
                        shortlist_rows,
                        min_club_rating=None,
                        min_league_difficulty=min_league_difficulty,
                    )
                    unique_count = _shortlist_unique_count(shortlist_rows)
                if (
                    requested_count
                    and unique_count < requested_count
                    and min_league_difficulty
                ):
                    shortlist_rows = _apply_strength_filters(
                        shortlist_rows,
                        min_club_rating=None,
                        min_league_difficulty=None,
                    )
                    unique_count = _shortlist_unique_count(shortlist_rows)


    payload = prepare_scout_payload(shortlist_rows)
    usage_summary = _merge_usage_calls(usage_calls)
    if not payload:
        return {
            "filters": filters.dict(),
            "shortlist": shortlist_rows,
            "candidates": [],
            "usage": usage_summary,
        }

    language = language_override or detect_language(prompt_for_llm)
    with get_openai_callback() as cb:
        scout_response = run_scout_agent(
            user_text=prompt_for_llm,
            players=payload,
            language=language,
            llm=llm,
        )
    usage_calls.append(_usage_from_callback(cb, model_name, "scout"))
    usage_summary = _merge_usage_calls(usage_calls)

    candidates = [item.dict() for item in scout_response.candidates]
    candidates = _attach_candidate_ids(candidates, shortlist_rows)
    candidates = _ensure_candidate_count(candidates, shortlist_rows, requested_count)
    return {
        "filters": filters.dict(),
        "shortlist": shortlist_rows,
        "candidates": candidates,
        "usage": usage_summary,
    }


RAW_METRIC_KEYS = [
  "goals_per_90",
  "xa_per_90",
  "accurate_passes_percent",
    "passes_to_penalty_area_per_90",
    "progressive_passes_per_90",
    "progressive_runs_per_90",
    "successful_dribbles_percent",
    "def_duels_won_percent",
  "interceptions_padj",
  "aerial_duels_won_percent",
]

DEFAULT_COMPETITION_AGGREGATES = [
    {
        "label": "Big 5 Leagues",
        "competitions": [
            "England. Premier League",
            "Spain. La Liga",
            "Italy. Serie A",
            "Germany. Bundesliga",
            "France. Ligue 1",
        ],
    },
    {"label": "Big 10 Competitions", "competitions": []},
    {"label": "First Divisions Only", "competitions": []},
    {"label": "Second Divisions Only", "competitions": []},
]

AGGREGATES_PATH = Path(__file__).resolve().parent / "helpers" / "competition_aggregates.json"
ROLE_METRICS_PATH = Path(__file__).resolve().parent / "helpers" / "role_metrics.json"
LEAGUE_TRANSLATION_PATH = Path(__file__).resolve().parent / "helpers" / "league_translation_matrix.csv"
PLAYER_PROFILES_PATH = Path(__file__).resolve().parent / "helpers" / "player_profiles.json"
MERCATO_LEAGUE_LEVELS_PATH = Path(__file__).resolve().parent / "helpers" / "mercato_league_levels.json"

_COMPETITION_AGGREGATES: Optional[list[dict[str, list[str]]]] = None
_COMPETITION_AGGREGATES_MTIME: Optional[float] = None

_ROLE_METRICS: Optional[dict[str, list[str]]] = None
_ROLE_METRICS_MTIME: Optional[float] = None

_LEAGUE_TRANSLATION: Optional[dict[tuple[str, str], dict[str, float]]] = None
_LEAGUE_TRANSLATION_MTIME: Optional[float] = None
_LEAGUE_TRANSLATION_LEAGUES: Optional[list[str]] = None

_TM_COLUMNS_CACHE: Optional[list[str]] = None
_STATS_METRIC_COLUMNS: Optional[list[str]] = None
_LOWER_IS_BETTER: Optional[set[str]] = None
_LOWER_IS_BETTER_MTIME: Optional[float] = None

_LEAGUE_META: Optional[dict[str, dict[str, float]]] = None
_LEAGUE_META_MTIME: Optional[float] = None
_LEAGUE_ALIAS_MAP: Optional[dict[str, list[str]]] = None
_LEAGUE_ALIAS_EXTRA: Optional[dict[str, list[str]]] = None
_LEAGUE_ALIAS_EXTRA_MTIME: Optional[float] = None

_OPTA_CLUBS: Optional[dict[str, dict[str, float | int | str]]] = None
_OPTA_CLUBS_MTIME: Optional[float] = None
_OPTA_CLUBS_SORTED: Optional[list[tuple[str, dict[str, float | int | str]]]] = None
_MERCATO_LEAGUE_LEVELS: Optional[dict[str, Any]] = None
_MERCATO_LEAGUE_LEVELS_MTIME: Optional[float] = None


def _get_tm_columns(session: Session) -> list[str]:
    global _TM_COLUMNS_CACHE
    rows = session.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'player_seasons' AND column_name LIKE 'tm_%' "
            "ORDER BY column_name"
        )
    ).fetchall()
    _TM_COLUMNS_CACHE = [r[0] for r in rows]
    return _TM_COLUMNS_CACHE


def _tm_select_clause(session: Session, alias: str = "ps") -> str:
    columns = _get_tm_columns(session)
    if not columns:
        return ""
    return ", " + ", ".join([f"{alias}.{col} AS {col}" for col in columns])


def _extract_tm_fields(row_dict: dict) -> dict[str, Optional[float | str]]:
    tm_fields: dict[str, Optional[float | str]] = {}
    for key in list(row_dict.keys()):
        if key.startswith("tm_"):
            tm_fields[key] = row_dict.pop(key)
    return tm_fields


def _load_competition_aggregates() -> list[dict[str, list[str]]]:
    global _COMPETITION_AGGREGATES, _COMPETITION_AGGREGATES_MTIME
    try:
        mtime = AGGREGATES_PATH.stat().st_mtime if AGGREGATES_PATH.exists() else None
    except Exception:
        mtime = None
    if _COMPETITION_AGGREGATES is not None and mtime == _COMPETITION_AGGREGATES_MTIME:
        return _COMPETITION_AGGREGATES

    aggregates = DEFAULT_COMPETITION_AGGREGATES
    if AGGREGATES_PATH.exists():
        try:
            import json

            data = json.loads(AGGREGATES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                aggregates = [{"label": k, "competitions": v} for k, v in data.items()]
            elif isinstance(data, list):
                aggregates = data
        except Exception:
            aggregates = DEFAULT_COMPETITION_AGGREGATES

    normalized: list[dict[str, list[str]]] = []
    for item in aggregates:
        label = str(item.get("label", "")).strip()
        comps = item.get("competitions", []) or []
        if not label:
            continue
        unique = []
        for comp in comps:
            value = str(comp).strip()
            if value and value not in unique:
                unique.append(value)
        normalized.append({"label": label, "competitions": unique})

    _COMPETITION_AGGREGATES = normalized or DEFAULT_COMPETITION_AGGREGATES
    _COMPETITION_AGGREGATES_MTIME = mtime
    return _COMPETITION_AGGREGATES


def _load_role_metrics() -> dict[str, list[str]]:
    global _ROLE_METRICS, _ROLE_METRICS_MTIME
    try:
        mtime = ROLE_METRICS_PATH.stat().st_mtime if ROLE_METRICS_PATH.exists() else None
    except Exception:
        mtime = None
    if _ROLE_METRICS is not None and mtime == _ROLE_METRICS_MTIME:
        return _ROLE_METRICS

    role_metrics: dict[str, list[str]] = {}
    if ROLE_METRICS_PATH.exists():
        try:
            data = json.loads(ROLE_METRICS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for role, metrics in data.items():
                    if not isinstance(metrics, list):
                        continue
                    unique = []
                    seen = set()
                    for item in metrics:
                        value = str(item).strip()
                        if not value or value in seen:
                            continue
                        seen.add(value)
                        unique.append(value)
                    if unique:
                        role_metrics[str(role)] = unique
        except Exception:
            role_metrics = {}

    _ROLE_METRICS = role_metrics
    _ROLE_METRICS_MTIME = mtime
    return _ROLE_METRICS


def _hydrate_report_metric_percentiles(session: Session, player_season_id: int, metrics: dict) -> dict:
    hydrated = dict(metrics or {})
    queries = [
        (
            "global",
            """
            SELECT metric_key, percentile
            FROM player_metric_percentiles_global
            WHERE player_season_id = :player_season_id
            """,
        ),
        (
            "league",
            """
            SELECT metric_key, percentile
            FROM player_metric_percentiles_league
            WHERE player_season_id = :player_season_id
            """,
        ),
    ]
    for scope, sql in queries:
        try:
            rows = session.execute(text(sql), {"player_season_id": player_season_id}).fetchall()
        except Exception:
            rows = []
        suffix = f"_pct_{scope}"
        for row in rows:
            metric_key = row._mapping.get("metric_key")
            if not metric_key:
                continue
            hydrated[f"{metric_key}{suffix}"] = row._mapping.get("percentile")
    return hydrated


def _build_report_average_contexts(session: Session, ps: dict) -> dict[str, Any]:
    season_id = ps.get("season_id")
    competition_id = ps.get("competition_id")
    if season_id is None:
        return {"global": {}, "league": {}}

    contexts: dict[str, Any] = {"global": {}, "league": {}}
    max_matches = 0.0
    if competition_id is not None:
        try:
            max_matches = float(
                session.execute(
                    text(
                        """
                        SELECT COALESCE(MAX(matches_played), 0)
                        FROM player_seasons
                        WHERE season_id = :season_id
                          AND competition_id = :competition_id
                        """
                    ),
                    {"season_id": season_id, "competition_id": competition_id},
                ).scalar()
                or 0
            )
        except Exception:
            max_matches = 0.0
    min_minutes = max(90.0, min(900.0, max_matches * 90.0 * 0.25)) if max_matches > 0 else 90.0

    queries = {
        "global": (
            """
            SELECT
                pmp.position_group,
                pmp.metric_key,
                ARRAY_AGG(pmp.raw_value) FILTER (WHERE pmp.raw_value IS NOT NULL) AS raw_values,
                BOOL_OR(pmp.lower_is_better) AS lower_is_better,
                COUNT(*) AS sample_size
            FROM player_metric_percentiles_global pmp
            JOIN player_seasons ps ON ps.id = pmp.player_season_id
            WHERE pmp.season_id = :season_id
              AND pmp.raw_value IS NOT NULL
              AND COALESCE(ps.minutes_played, 0) >= :min_minutes
            GROUP BY pmp.position_group, pmp.metric_key
            """,
            {"season_id": season_id, "min_minutes": min_minutes},
        ),
        "league": (
            """
            SELECT
                pmp.position_group,
                pmp.metric_key,
                ARRAY_AGG(pmp.raw_value) FILTER (WHERE pmp.raw_value IS NOT NULL) AS raw_values,
                BOOL_OR(pmp.lower_is_better) AS lower_is_better,
                COUNT(*) AS sample_size
            FROM player_metric_percentiles_league pmp
            JOIN player_seasons ps ON ps.id = pmp.player_season_id
            WHERE pmp.season_id = :season_id
              AND pmp.competition_id = :competition_id
              AND pmp.raw_value IS NOT NULL
              AND COALESCE(ps.minutes_played, 0) >= :min_minutes
            GROUP BY pmp.position_group, pmp.metric_key
            """,
            {"season_id": season_id, "competition_id": competition_id, "min_minutes": min_minutes},
        ),
    }

    for context, (sql, params) in queries.items():
        if context == "league" and competition_id is None:
            continue
        try:
            rows = session.execute(text(sql), params).fetchall()
        except Exception:
            continue
        for row in rows:
            item = row._mapping
            position_group = item.get("position_group")
            metric_key = item.get("metric_key")
            if not position_group or not metric_key:
                continue
            group = contexts[context].setdefault(str(position_group), {"metrics": {}, "sample_size": 0})
            sample_size = int(item.get("sample_size") or 0)
            raw_values = [float(value) for value in (item.get("raw_values") or []) if value is not None]
            avg_raw = sum(raw_values) / len(raw_values) if raw_values else None
            lower_is_better = bool(item.get("lower_is_better"))
            avg_percentile = _percentile_rank(raw_values, avg_raw) if avg_raw is not None else None
            if avg_percentile is not None and lower_is_better:
                avg_percentile = 100.0 - avg_percentile
            group["sample_size"] = max(int(group.get("sample_size") or 0), sample_size)
            group["min_minutes"] = min_minutes
            group["metrics"][str(metric_key)] = {
                "raw": avg_raw,
                "percentile": avg_percentile,
                "sample_size": sample_size,
            }
    return contexts


def _load_lower_is_better() -> set[str]:
    global _LOWER_IS_BETTER, _LOWER_IS_BETTER_MTIME
    try:
        mtime = PLAYER_PROFILES_PATH.stat().st_mtime if PLAYER_PROFILES_PATH.exists() else None
    except Exception:
        mtime = None
    if _LOWER_IS_BETTER is not None and mtime == _LOWER_IS_BETTER_MTIME:
        return _LOWER_IS_BETTER

    metrics: set[str] = set()
    if PLAYER_PROFILES_PATH.exists():
        try:
            data = json.loads(PLAYER_PROFILES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for spec in data.values():
                    lower_list = spec.get("lower_is_better", []) if isinstance(spec, dict) else []
                    for item in lower_list or []:
                        value = str(item).strip()
                        if value:
                            metrics.add(value)
        except Exception:
            metrics = set()

    _LOWER_IS_BETTER = metrics
    _LOWER_IS_BETTER_MTIME = mtime
    return _LOWER_IS_BETTER


def _stats_metric_columns(session: Session) -> list[str]:
    global _STATS_METRIC_COLUMNS
    rows = session.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'player_metrics' ORDER BY column_name"
        )
    ).fetchall()
    excluded = {
        "player_season_id",
        "created_at",
        "updated_at",
        "age",
        "minutes_played",
        "matches_played",
    }
    columns = []
    for row in rows:
        name = row[0]
        if name in excluded:
            continue
        if name.startswith("summary_"):
            continue
        if name.endswith("_pct_league") or name.endswith("_pct_global"):
            continue
        if name.endswith("_pct_league_adjusted") or name.endswith("_pct_global_adjusted"):
            continue
        columns.append(name)
    _STATS_METRIC_COLUMNS = columns
    return _STATS_METRIC_COLUMNS


def _percentile_rank(values: list[float], target: float) -> float:
    if not values:
        return 100.0
    sorted_vals = sorted(values)
    position = bisect_right(sorted_vals, target)
    return float(position / len(sorted_vals) * 100.0)


def _load_league_translation() -> dict[tuple[str, str], dict[str, float]]:
    global _LEAGUE_TRANSLATION, _LEAGUE_TRANSLATION_MTIME, _LEAGUE_TRANSLATION_LEAGUES
    try:
        mtime = LEAGUE_TRANSLATION_PATH.stat().st_mtime if LEAGUE_TRANSLATION_PATH.exists() else None
    except Exception:
        mtime = None
    if _LEAGUE_TRANSLATION is not None and mtime == _LEAGUE_TRANSLATION_MTIME:
        return _LEAGUE_TRANSLATION

    translation: dict[tuple[str, str], dict[str, float]] = {}
    leagues: set[str] = set()
    if LEAGUE_TRANSLATION_PATH.exists():
        try:
            with LEAGUE_TRANSLATION_PATH.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    source = str(row.get("source_competition") or "").strip()
                    target = str(row.get("target_competition") or "").strip()
                    if not source or not target:
                        continue
                    leagues.add(source)
                    leagues.add(target)
                    def _to_float(value: str | None) -> float:
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return 1.0
                    translation[(source, target)] = {
                        "difficulty_coeff": _to_float(row.get("difficulty_coeff")),
                        "intensity_coeff": _to_float(row.get("intensity_coeff")),
                        "overall_coeff": _to_float(row.get("overall_coeff")),
                    }
        except Exception:
            translation = {}
            leagues = set()

    _LEAGUE_TRANSLATION = translation
    _LEAGUE_TRANSLATION_MTIME = mtime
    _LEAGUE_TRANSLATION_LEAGUES = sorted(leagues)
    return _LEAGUE_TRANSLATION


def _league_translation_leagues() -> list[str]:
    _load_league_translation()
    return _LEAGUE_TRANSLATION_LEAGUES or []


def _league_translation_coeffs(source: Optional[str], target: Optional[str]) -> dict[str, float]:
    if not source or not target:
        return {"difficulty_coeff": 1.0, "intensity_coeff": 1.0, "overall_coeff": 1.0}
    matrix = _load_league_translation()
    return matrix.get((source, target), {"difficulty_coeff": 1.0, "intensity_coeff": 1.0, "overall_coeff": 1.0})


def _competition_aggregate_map() -> dict[str, list[str]]:
    return {item["label"]: item["competitions"] for item in _load_competition_aggregates()}


def _apply_competition_filter(sql: str, params: dict, competition: Optional[str]) -> tuple[str, dict]:
    if not competition:
        return sql, params
    aggregates = _competition_aggregate_map()
    if competition in aggregates:
        sql += " AND c.name = ANY(:competition_list)"
        params["competition_list"] = aggregates[competition]
    else:
        sql += " AND c.name = :competition"
        params["competition"] = competition
    return sql, params


def _apply_ranking_filters(
    sql: str,
    params: dict,
    role: Optional[str],
    competition: Optional[str],
    season: Optional[str],
    position: Optional[str],
    team: Optional[str],
    age_min: Optional[float],
    age_max: Optional[float],
) -> tuple[str, dict]:
    if role:
        sql += " AND ps.assigned_role = :role"
        params["role"] = role
    sql, params = _apply_competition_filter(sql, params, competition)
    sql, params = _apply_season_filter(sql, params, season)
    if position:
        sql += " AND ps.position = :position"
        params["position"] = position
    if team:
        sql += " AND ps.team_in_selected_period = :team"
        params["team"] = team
    if age_min is not None:
        sql += " AND pm.age >= :age_min"
        params["age_min"] = age_min
    if age_max is not None:
        sql += " AND pm.age <= :age_max"
        params["age_max"] = age_max
    return sql, params


def _ranking_pct_fallback_join() -> str:
    return """
    LEFT JOIN LATERAL (
      SELECT
        ps_alt.assigned_role_pct_league AS fallback_assigned_role_pct_league
      FROM player_seasons ps_alt
      WHERE ps_alt.player_id = ps.player_id
        AND ps_alt.assigned_role_pct_league IS NOT NULL
      ORDER BY
        CASE WHEN ps_alt.assigned_role = ps.assigned_role THEN 0 ELSE 1 END,
        ps_alt.calendar DESC NULLS LAST,
        ps_alt.minutes_played DESC NULLS LAST
      LIMIT 1
    ) pct_league_fallback ON TRUE
    LEFT JOIN LATERAL (
      SELECT
        ps_alt.assigned_role_pct_global AS fallback_assigned_role_pct_global
      FROM player_seasons ps_alt
      WHERE ps_alt.player_id = ps.player_id
        AND ps_alt.assigned_role_pct_global IS NOT NULL
      ORDER BY
        CASE WHEN ps_alt.assigned_role = ps.assigned_role THEN 0 ELSE 1 END,
        ps_alt.calendar DESC NULLS LAST,
        ps_alt.minutes_played DESC NULLS LAST
      LIMIT 1
    ) pct_global_fallback ON TRUE
    """


def _role_pct_fallback_values(
    session: Session,
    *,
    player_id: int,
    assigned_role: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    sql = """
    SELECT
      (
        SELECT ps_alt.assigned_role_pct_league
        FROM player_seasons ps_alt
        WHERE ps_alt.player_id = :player_id
          AND ps_alt.assigned_role_pct_league IS NOT NULL
        ORDER BY
          CASE WHEN ps_alt.assigned_role = :assigned_role THEN 0 ELSE 1 END,
          ps_alt.calendar DESC NULLS LAST,
          ps_alt.minutes_played DESC NULLS LAST
        LIMIT 1
      ) AS league_pct,
      (
        SELECT ps_alt.assigned_role_pct_global
        FROM player_seasons ps_alt
        WHERE ps_alt.player_id = :player_id
          AND ps_alt.assigned_role_pct_global IS NOT NULL
        ORDER BY
          CASE WHEN ps_alt.assigned_role = :assigned_role THEN 0 ELSE 1 END,
          ps_alt.calendar DESC NULLS LAST,
          ps_alt.minutes_played DESC NULLS LAST
        LIMIT 1
      ) AS global_pct
    """
    row = session.execute(
        text(sql),
        {"player_id": player_id, "assigned_role": assigned_role},
    ).fetchone()
    if not row:
        return None, None
    data = _row_to_dict(row)
    return data.get("league_pct"), data.get("global_pct")


DEFAULT_MERCATO_LEAGUE_LEVELS = {
    "bands": [
        {"label": "Premier League", "coefficient": 1.0, "cap": 98, "difficulty_min": 8.9},
        {"label": "Liga / Serie A / Bundesliga", "coefficient": 0.95, "cap": 96, "difficulty_min": 8.3},
        {"label": "Ligue 1", "coefficient": 0.88, "cap": 92, "difficulty_min": 8.0},
        {"label": "Championship / Eredivisie / Liga Portugal", "coefficient": 0.78, "cap": 86, "difficulty_min": 7.4},
        {"label": "Ligue 2 / D2 top pays", "coefficient": 0.68, "cap": 80, "difficulty_min": 6.8},
        {"label": "D1 faible / D2 moyenne", "coefficient": 0.55, "cap": 74, "difficulty_min": 5.7},
        {"label": "D2 Bulgarie / championnat tres faible", "coefficient": 0.45, "cap": 70, "difficulty_min": 0.0},
    ],
    "exact_overrides": [],
}

MERCATO_RECOMMENDATION_SEASONS = ["2025/2026", "2025", "2026"]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    return logic_safe_float(value, default)


def _clamp(value: float, low: float, high: float) -> float:
    return logic_clamp(value, low, high)


def _current_user_id(request: Request) -> Optional[str]:
    user = getattr(request.state, "user", None) or {}
    return user.get("username")


def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _load_mercato_league_levels() -> dict[str, Any]:
    global _MERCATO_LEAGUE_LEVELS, _MERCATO_LEAGUE_LEVELS_MTIME
    try:
        mtime = MERCATO_LEAGUE_LEVELS_PATH.stat().st_mtime if MERCATO_LEAGUE_LEVELS_PATH.exists() else None
    except Exception:
        mtime = None
    if _MERCATO_LEAGUE_LEVELS is not None and mtime == _MERCATO_LEAGUE_LEVELS_MTIME:
        return _MERCATO_LEAGUE_LEVELS
    levels = DEFAULT_MERCATO_LEAGUE_LEVELS
    if MERCATO_LEAGUE_LEVELS_PATH.exists():
        try:
            data = json.loads(MERCATO_LEAGUE_LEVELS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                levels = data
            elif isinstance(data, list):
                levels = {"bands": data, "exact_overrides": []}
        except Exception:
            levels = DEFAULT_MERCATO_LEAGUE_LEVELS
    _MERCATO_LEAGUE_LEVELS = levels
    _MERCATO_LEAGUE_LEVELS_MTIME = mtime
    return levels


def _mercato_league_adjustment(
    competition_name: Optional[str],
    existing_strength_factor: Optional[float] = None,
) -> dict[str, Any]:
    league_meta, _ = _load_league_translation_meta()
    config = _load_mercato_league_levels()
    bands = sorted(
        config.get("bands", []) or [],
        key=lambda item: float(item.get("difficulty_min", 0.0) or 0.0),
        reverse=True,
    )
    overrides = config.get("exact_overrides", []) or []
    difficulty = None
    if competition_name and competition_name in league_meta:
        difficulty = league_meta[competition_name].get("difficulty")
    if difficulty is None and existing_strength_factor is not None and league_meta:
        values = [
            meta.get("difficulty")
            for meta in league_meta.values()
            if meta.get("difficulty") is not None
        ]
        mean_difficulty = sum(values) / len(values) if values else None
        if mean_difficulty:
            difficulty = float(existing_strength_factor) * mean_difficulty
    for override in overrides:
        if competition_name and str(override.get("competition") or "").strip() == competition_name:
            return {
                "label": override.get("label"),
                "coefficient": float(override.get("coefficient", 0.65)),
                "cap": float(override.get("cap", 80)),
                "difficulty": difficulty,
                "existing_strength_factor": existing_strength_factor,
            }
    if difficulty is not None:
        for level in bands:
            minimum = _safe_float(level.get("difficulty_min"), 0.0) or 0.0
            if difficulty >= minimum:
                return {
                    "label": level.get("label"),
                    "coefficient": float(level.get("coefficient", 0.65)),
                    "cap": float(level.get("cap", 80)),
                    "difficulty": difficulty,
                    "existing_strength_factor": existing_strength_factor,
                }
    return {
        "label": "D1 faible / D2 moyenne",
        "coefficient": 0.55,
        "cap": 74.0,
        "difficulty": difficulty,
        "existing_strength_factor": existing_strength_factor,
    }


def calculate_calibrated_player_level(player: dict[str, Any]) -> dict[str, Any]:
    league_meta, _ = _load_league_translation_meta()
    return calculate_calibrated_level(player, league_meta, _load_mercato_league_levels())


def _token_overlap_score(*texts: Optional[str]) -> float:
    source = _normalize_phrase(" ".join([text or "" for text in texts[:1]]))
    target = _normalize_phrase(" ".join([text or "" for text in texts[1:]]))
    if not source or not target:
        return 0.0
    source_tokens = {token for token in source.split() if len(token) >= 3}
    target_tokens = {token for token in target.split() if len(token) >= 3}
    if not source_tokens or not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(source_tokens | target_tokens)


def _position_fit(need: dict[str, Any], player: dict[str, Any]) -> float:
    wanted_position = _normalize_phrase(need.get("position") or "")
    wanted_role = _normalize_phrase(need.get("role") or "")
    position = _normalize_phrase(player.get("position") or "")
    second_position = _normalize_phrase(player.get("second_position") or "")
    assigned_role = _normalize_phrase(player.get("assigned_role") or "")
    score = 0.0
    if wanted_position and (wanted_position in position or wanted_position in second_position):
        score += 0.55 if wanted_position in position else 0.35
    elif wanted_position and position and (position in wanted_position or second_position in wanted_position):
        score += 0.35
    if wanted_role and assigned_role:
        if wanted_role == assigned_role or wanted_role in assigned_role or assigned_role in wanted_role:
            score += 0.45
        else:
            score += 0.25 * _token_overlap_score(wanted_role, assigned_role)
    elif not wanted_role and score > 0:
        score += 0.45
    return _clamp(score, 0.0, 1.0)


def _mercato_match_candidate(need: dict[str, Any], request_row: dict[str, Any], player: dict[str, Any]) -> dict[str, Any]:
    calibration = calculate_calibrated_player_level(player)
    calibrated = float(calibration["calibrated_player_level"])
    required = _safe_float(need.get("required_player_level"), 75.0) or 75.0
    level_fit = _clamp(1.0 - abs(calibrated - required) / 35.0, 0.0, 1.0)
    position_fit = _position_fit(need, player)
    target_level = _normalize_phrase(need.get("target_league_level") or request_row.get("competition_name") or "")
    current_league = _normalize_phrase(player.get("competition_name") or "")
    league_fit = 0.65
    if target_level and current_league:
        league_fit = 1.0 if target_level in current_league or current_league in target_level else 0.65
    league_fit = min(league_fit, _clamp((calibration["league_coefficient"] - 0.4) / 0.6, 0.25, 1.0))
    age = _safe_float(player.get("age"))
    age_fit = 0.75
    if age is not None:
        if need.get("age_min") is not None and age < need["age_min"]:
            age_fit -= min(0.5, (need["age_min"] - age) * 0.12)
        if need.get("age_max") is not None and age > need["age_max"]:
            age_fit -= min(0.6, (age - need["age_max"]) * 0.12)
        if age <= 23:
            age_fit += 0.15
    foot_fit = 1.0
    preferred_foot = _normalize_phrase(need.get("preferred_foot") or "")
    player_foot = _normalize_phrase(player.get("foot") or player.get("tm_foot") or "")
    if preferred_foot and preferred_foot not in {"any", "indifferent"}:
        foot_fit = 1.0 if preferred_foot in player_foot else 0.55
    height_fit = 1.0
    height_min = _safe_float(need.get("height_min"))
    height = _safe_float(player.get("height_cm") or player.get("tm_height"))
    if height_min is not None and height is not None and height < height_min:
        height_fit = 0.65
    availability_fit = 0.75
    market_value = _safe_float(player.get("tm_market_value_numeric"))
    budget_max = _safe_float(request_row.get("budget_max"))
    if budget_max is not None and market_value is not None:
        availability_fit = 1.0 if market_value <= budget_max else max(0.35, 1.0 - (market_value - budget_max) / max(budget_max, 1.0))
    semantic_fit = _token_overlap_score(
        " ".join([str(need.get("notes") or ""), str(request_row.get("extra_info") or "")]),
        " ".join([str(player.get("assigned_role") or ""), str(player.get("position") or ""), str(player.get("competition_name") or "")]),
    )
    minutes = _safe_float(player.get("minutes_played"), 0.0) or 0.0
    reliability = _clamp(minutes / 1800.0, 0.25, 1.0)
    scout_signal = 0.5
    if player.get("tm_profile_url") or player.get("tm_id"):
        scout_signal += 0.15
    breakdown = {
        "position_role_fit": round(position_fit * 25, 2),
        "calibrated_level_fit": round(level_fit * 20, 2),
        "league_fit": round(league_fit * 15, 2),
        "age_potential_fit": round(_clamp(age_fit, 0.0, 1.0) * 10, 2),
        "budget_availability_fit": round(_clamp((availability_fit + foot_fit + height_fit) / 3, 0.0, 1.0) * 10, 2),
        "semantic_fit": round(semantic_fit * 10, 2),
        "data_reliability": round(reliability * 5, 2),
        "scout_signal": round(_clamp(scout_signal, 0.0, 1.0) * 5, 2),
    }
    match_score = round(sum(breakdown.values()), 2)
    strengths = []
    risks = []
    if position_fit >= 0.75:
        strengths.append("Position and role are aligned with the need.")
    if calibrated >= required - 5:
        strengths.append("Calibrated level is close to the requested level.")
    if age is not None and age <= 23:
        strengths.append("Age profile leaves room for progression.")
    if calibration["league_coefficient"] < 0.68:
        risks.append("Strong league-level penalty applied because current competition is below the target context.")
    if reliability < 0.55:
        risks.append("Minutes sample is limited, so confidence is lower.")
    if foot_fit < 1.0:
        risks.append("Preferred-foot criterion is not fully matched.")
    if not strengths:
        strengths.append("Profile remains relevant on aggregate matching score.")
    reason = (
        f"Profile fits the need: {player.get('position') or 'position'} / "
        f"{player.get('assigned_role') or 'undefined role'}, adjusted level {calibrated:.0f} "
        f"for a requested level around {required:.0f}."
    )
    return {
        "match_score": match_score,
        **calibration,
        "explanation_json": {
            "strengths": strengths,
            "risks": risks,
            "score_breakdown": breakdown,
            "league_adjustment": calibration,
            "recommendation_reason": reason,
        },
    }


def _parse_money_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip().lower().replace("€", "").replace(",", ".")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw)
    if not match:
        return None
    numeric = float(match.group(1))
    if "bn" in raw or "b" in raw:
        numeric *= 1_000_000_000
    elif "m" in raw:
        numeric *= 1_000_000
    elif "k" in raw:
        numeric *= 1_000
    return numeric


def _mercato_player_context(
    session: Session,
    player_id: int,
    player_season_id: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    tm_clause = _tm_select_clause(session)
    season_filter = "AND ps.id = :player_season_id" if player_season_id else ""
    row = session.execute(
        text(
            """
            SELECT
              p.id AS player_id,
              p.name,
              p.country,
              p.foot,
              p.height_cm,
              p.tm_id,
              p.tm_profile_url,
              ps.id AS player_season_id,
              ps.calendar,
              ps.team_in_selected_period AS team,
              ps.position,
              ps.second_position,
              ps.assigned_role,
              ps.minutes_played,
              ps.league_strength_factor,
              ps.global_score_adjusted AS raw_player_level,
              ps.assigned_role_pct_league,
              ps.assigned_role_pct_global,
              c.name AS competition_name,
              pm.age AS age""" + tm_clause + """
            FROM players p
            JOIN player_seasons ps ON ps.player_id = p.id
            JOIN competitions c ON c.id = ps.competition_id
            LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
            WHERE p.id = :player_id
            """ + season_filter + """
            ORDER BY ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
            LIMIT 1
            """
        ),
        {"player_id": player_id, "player_season_id": player_season_id},
    ).fetchone()
    if not row:
        return None
    payload = _row_to_dict(row)
    payload["tm_market_value_numeric"] = _parse_money_value(payload.get("tm_market_value"))
    return payload


def _mercato_need_context(session: Session, need_id: int) -> Optional[dict[str, Any]]:
    row = session.execute(
        text(
            """
            SELECT
              mn.*,
              mr.club_id,
              mr.created_by_agent_id,
              mr.assigned_agent_id,
              mr.season,
              mr.title,
              mr.priority,
              mr.status AS request_status,
              mr.budget_min,
              mr.budget_max,
              mr.salary_max,
              mr.deal_type,
              mr.extra_info,
              cl.name AS club_name,
              comp.name AS competition_name
            FROM mercato_needs mn
            JOIN mercato_requests mr ON mr.id = mn.mercato_request_id
            LEFT JOIN clubs cl ON cl.id = mr.club_id
            LEFT JOIN competitions comp ON comp.id = cl.competition_id
            WHERE mn.id = :need_id AND mr.archived_at IS NULL
            """
        ),
        {"need_id": need_id},
    ).fetchone()
    return _row_to_dict(row) if row else None


def _insert_mercato_candidate(
    session: Session,
    *,
    need_id: int,
    player: dict[str, Any],
    source: str,
    status: str,
    agent_note: Optional[str],
    created_by: Optional[str],
    scoring: Optional[dict[str, Any]] = None,
) -> tuple[bool, Optional[int]]:
    need = _mercato_need_context(session, need_id)
    if not need:
        raise HTTPException(status_code=404, detail="Mercato need not found")
    scoring = scoring or _mercato_match_candidate(need, need, player)
    row = session.execute(
        text(
            """
            INSERT INTO mercato_candidates (
	              mercato_need_id,
	              player_id,
	              player_season_id,
	              source,
              status,
              match_score,
              calibrated_player_level,
              raw_player_level,
              league_coefficient,
              explanation_json,
              agent_note,
              created_by_agent_id,
              created_at,
              updated_at
            ) VALUES (
	              :need_id,
	              :player_id,
	              :player_season_id,
	              :source,
              :status,
              :match_score,
              :calibrated_player_level,
              :raw_player_level,
              :league_coefficient,
              CAST(:explanation_json AS JSONB),
              :agent_note,
              :created_by,
              NOW(),
              NOW()
            )
            ON CONFLICT (mercato_need_id, player_id) DO NOTHING
            RETURNING id
            """
        ),
        {
            "need_id": need_id,
            "player_id": player["player_id"],
            "player_season_id": player.get("player_season_id"),
            "source": source,
            "status": status,
            "match_score": scoring.get("match_score"),
            "calibrated_player_level": scoring.get("calibrated_player_level"),
            "raw_player_level": scoring.get("raw_player_level"),
            "league_coefficient": scoring.get("league_coefficient"),
            "explanation_json": json.dumps(scoring.get("explanation_json") or {}),
            "agent_note": agent_note,
            "created_by": created_by,
        },
    ).fetchone()
    return (row is not None), (int(row.id) if row else None)


def _mercato_candidates_for_need(session: Session, need_id: int) -> list[dict[str, Any]]:
    tm_clause = _tm_select_clause(session)
    rows = session.execute(
        text(
            """
            SELECT
              mc.*,
              p.name,
              p.country,
              p.foot,
              p.height_cm,
              p.tm_id,
              p.tm_profile_url,
	              ps.team_in_selected_period AS team,
	              ps.position,
	              ps.second_position,
	              ps.assigned_role,
              ps.global_score_adjusted,
              ps.calendar,
              comp.name AS competition_name,
              pm.age""" + tm_clause + """
            FROM mercato_candidates mc
            JOIN players p ON p.id = mc.player_id
	            LEFT JOIN LATERAL (
	              SELECT ps.*
	              FROM player_seasons ps
	              WHERE ps.id = mc.player_season_id
	                 OR (mc.player_season_id IS NULL AND ps.player_id = p.id)
	              ORDER BY
	                CASE WHEN ps.id = mc.player_season_id THEN 0 ELSE 1 END,
	                ps.calendar DESC NULLS LAST,
	                ps.minutes_played DESC NULLS LAST
	              LIMIT 1
	            ) ps ON true
            LEFT JOIN competitions comp ON comp.id = ps.competition_id
            LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
            WHERE mc.mercato_need_id = :need_id
            ORDER BY mc.match_score DESC NULLS LAST, mc.created_at DESC
            """
        ),
        {"need_id": need_id},
    ).fetchall()
    items = []
    for row in rows:
        payload = _row_to_dict(row)
        payload["tm_fields"] = _extract_tm_fields(payload)
        items.append(payload)
    return items


def _mercato_shortlist_scored_players(
    session: Session,
    need: dict[str, Any],
    payload: Optional[MercatoShortlistGenerate] = None,
) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
    tm_clause = _tm_select_clause(session)
    competition_norm_sql = "TRIM(REGEXP_REPLACE(REGEXP_REPLACE(LOWER(c.name), '[^a-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g'))"
    sql = """
    SELECT DISTINCT ON (p.id)
      p.id AS player_id,
      p.name,
      p.country,
      p.foot,
      p.height_cm,
      p.tm_id,
      p.tm_profile_url,
      ps.id AS player_season_id,
      ps.calendar,
      ps.team_in_selected_period AS team,
      ps.position,
      ps.second_position,
      ps.assigned_role,
      ps.minutes_played,
      ps.league_strength_factor,
      ps.global_score_adjusted AS raw_player_level,
      ps.assigned_role_pct_league,
      ps.assigned_role_pct_global,
      c.name AS competition_name,
      pm.age AS age""" + tm_clause + """
    FROM players p
    JOIN player_seasons ps ON ps.player_id = p.id
    JOIN competitions c ON c.id = ps.competition_id
    LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
    WHERE ps.global_score_adjusted IS NOT NULL
      AND ps.minutes_played >= :min_minutes
      AND ps.calendar = ANY(:recommendation_seasons)
      AND NOT EXISTS (
        SELECT 1 FROM mercato_candidates mc
        WHERE mc.mercato_need_id = :need_id AND mc.player_id = p.id
      )
    """
    selected_competitions = []
    if payload and payload.competitions:
        selected_competitions = [
            _normalize_phrase(str(item))
            for item in payload.competitions
            if _normalize_phrase(str(item))
        ]
    params: dict[str, Any] = {
        "need_id": need["id"],
        "limit": 1200,
        "recommendation_seasons": MERCATO_RECOMMENDATION_SEASONS,
        "min_minutes": payload.min_minutes if payload and payload.min_minutes is not None else 270,
    }
    if selected_competitions:
        sql += f"""
        AND EXISTS (
          SELECT 1
          FROM unnest(CAST(:search_competitions AS TEXT[])) AS selected_competition(name)
          WHERE {competition_norm_sql} = selected_competition.name
             OR {competition_norm_sql} LIKE '%' || selected_competition.name || '%'
             OR selected_competition.name LIKE '%' || {competition_norm_sql} || '%'
        )
        """
        params["search_competitions"] = selected_competitions
    age_min = payload.age_min if payload and payload.age_min is not None else need.get("age_min")
    age_max = payload.age_max if payload and payload.age_max is not None else need.get("age_max")
    if age_min is not None:
        sql += " AND pm.age >= :age_min"
        params["age_min"] = age_min
    if age_max is not None:
        sql += " AND pm.age <= :age_max"
        params["age_max"] = age_max
    if need.get("height_min") is not None:
        sql += " AND (p.height_cm IS NULL OR p.height_cm >= :height_min)"
        params["height_min"] = need["height_min"]
    if need.get("preferred_foot"):
        sql += " AND (p.foot IS NULL OR p.foot ILIKE :preferred_foot)"
        params["preferred_foot"] = f"%{need['preferred_foot']}%"
    if need.get("position"):
        sql += """
        AND (
          ps.position ILIKE :position_like
          OR ps.second_position ILIKE :position_like
          OR ps.assigned_role ILIKE :position_like
        )
        """
        params["position_like"] = f"%{need['position']}%"
    sql += """
    ORDER BY p.id, ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
    LIMIT :limit
    """
    rows = session.execute(text(sql), params).fetchall()
    scored = []
    for row in rows:
        player = _row_to_dict(row)
        player["tm_market_value_numeric"] = _parse_money_value(player.get("tm_market_value"))
        score = _mercato_match_candidate(need, need, player)
        if payload and payload.min_match_score is not None and score["match_score"] < payload.min_match_score:
            continue
        scored.append((score["match_score"], player, score))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _mercato_recommendation_payload(player: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    payload = {
        **player,
        "source": "algorithm_preview",
        "status": "preview",
        "match_score": score.get("match_score"),
        "calibrated_player_level": score.get("calibrated_player_level"),
        "raw_player_level": score.get("raw_player_level"),
        "league_coefficient": score.get("league_coefficient"),
        "explanation_json": score.get("explanation_json") or {},
    }
    payload["tm_fields"] = _extract_tm_fields(payload)
    payload["key_metrics"] = [
        {"label": "Global score", "value": payload.get("raw_player_level")},
        {"label": "Adjusted level", "value": payload.get("calibrated_player_level")},
        {"label": "League role pct", "value": payload.get("assigned_role_pct_league")},
        {"label": "Global role pct", "value": payload.get("assigned_role_pct_global")},
        {"label": "Minutes", "value": payload.get("minutes_played")},
    ]
    return payload


def _mercato_request_payload(session: Session, request_id: int) -> Optional[dict[str, Any]]:
    row = session.execute(
        text(
            """
            SELECT
              mr.*,
              cl.name AS club_name,
              comp.name AS competition_name,
              creator.display_name AS created_by_agent_name,
              assignee.display_name AS assigned_agent_name
            FROM mercato_requests mr
            LEFT JOIN clubs cl ON cl.id = mr.club_id
            LEFT JOIN competitions comp ON comp.id = cl.competition_id
            LEFT JOIN auth_users creator ON creator.username = mr.created_by_agent_id
            LEFT JOIN auth_users assignee ON assignee.username = mr.assigned_agent_id
            WHERE mr.id = :request_id AND mr.archived_at IS NULL
            """
        ),
        {"request_id": request_id},
    ).fetchone()
    if not row:
        return None
    request_payload = _row_to_dict(row)
    need_rows = session.execute(
        text(
            """
            SELECT *
            FROM mercato_needs
            WHERE mercato_request_id = :request_id
            ORDER BY id
            """
        ),
        {"request_id": request_id},
    ).fetchall()
    needs = []
    for need_row in need_rows:
        need = _row_to_dict(need_row)
        need["candidates"] = _mercato_candidates_for_need(session, need["id"])
        needs.append(need)
    request_payload["needs"] = needs
    return request_payload


def _clean_agent_names(values: Optional[List[str]]) -> list[str]:
    cleaned: list[str] = []
    for value in values or []:
        item = _clean_text(value)
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def _validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> None:
    if start_date and end_date and end_date < start_date:
        raise HTTPException(status_code=400, detail="End date cannot be earlier than start date")


def _can_manage_hq_calendar_event(request: Request, row: Any) -> bool:
    user = getattr(request.state, "user", None) or {}
    if user.get("role") == "admin":
        return True
    return bool(user.get("username") and row.created_by_agent_id == user.get("username"))


def _hq_calendar_event_payload(request: Request, row: Any) -> dict[str, Any]:
    item = _row_to_dict(row)
    item["source"] = "event"
    item["can_edit"] = _can_manage_hq_calendar_event(request, row)
    if not isinstance(item.get("agent_names"), list):
        item["agent_names"] = []
    return item


@app.get("/hq/calendar-events")
def hq_calendar_events(
    request: Request,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    _ensure_agency_ops_schema(session)
    start_value = start or "1900-01-01"
    end_value = end or "2999-12-31"
    event_rows = session.execute(
        text(
            """
            SELECT *
            FROM hq_calendar_events
            WHERE start_date <= NULLIF(:end_date, '')::date
              AND COALESCE(end_date, start_date) >= NULLIF(:start_date, '')::date
            ORDER BY start_date ASC, id ASC
            """
        ),
        {"start_date": start_value, "end_date": end_value},
    ).fetchall()
    task_rows = session.execute(
        text(
            """
            SELECT *
            FROM hq_priority_items
            WHERE end_date IS NOT NULL
              AND end_date >= NULLIF(:start_date, '')::date
              AND end_date <= NULLIF(:end_date, '')::date
              AND COALESCE(status, 'todo') NOT IN ('done', 'completed')
            ORDER BY end_date ASC, id ASC
            """
        ),
        {"start_date": start_value, "end_date": end_value},
    ).fetchall()
    items = [_hq_calendar_event_payload(request, row) for row in event_rows]
    for row in task_rows:
        task = _row_to_dict(row)
        items.append(
            {
                "id": f"task-{task['id']}",
                "task_id": task["id"],
                "source": "task",
                "title": task["title"],
                "description": task.get("description"),
                "event_type": "task_due",
                "agent_names": [task.get("agent_name") or "Yannis"],
                "start_date": task.get("end_date"),
                "end_date": task.get("end_date"),
                "location": None,
                "color": task.get("color"),
                "related_page": task.get("related_page") or "/",
                "priority": task.get("priority"),
                "status": task.get("status"),
                "can_edit": False,
            }
        )
    items.sort(key=lambda item: (str(item.get("start_date") or ""), str(item.get("id") or "")))
    return {"items": items}


@app.post("/hq/calendar-events")
def create_hq_calendar_event(payload: HqCalendarEventPayload, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    title = _clean_text(payload.title)
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    if not payload.start_date:
        raise HTTPException(status_code=400, detail="Start date is required")
    _validate_date_range(payload.start_date, payload.end_date)
    row = session.execute(
        text(
            """
            INSERT INTO hq_calendar_events (
              title, description, event_type, agent_names, start_date, end_date,
              location, color, related_page, created_by_agent_id, updated_by_agent_id,
              created_at, updated_at
            ) VALUES (
              :title, :description, :event_type, CAST(:agent_names AS JSONB),
              NULLIF(:start_date, '')::date, NULLIF(:end_date, '')::date,
              :location, :color, :related_page, :user_id, :user_id, NOW(), NOW()
            )
            RETURNING *
            """
        ),
        {
            "title": title,
            "description": _clean_text(payload.description),
            "event_type": _clean_text(payload.event_type) or "team",
            "agent_names": json.dumps(_clean_agent_names(payload.agent_names)),
            "start_date": payload.start_date or "",
            "end_date": payload.end_date or "",
            "location": _clean_text(payload.location),
            "color": _clean_text(payload.color),
            "related_page": _clean_text(payload.related_page),
            "user_id": _current_user_id(request),
        },
    ).fetchone()
    session.commit()
    return _hq_calendar_event_payload(request, row)


@app.patch("/hq/calendar-events/{event_id}")
def update_hq_calendar_event(event_id: int, payload: HqCalendarEventPayload, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    existing = session.execute(
        text("SELECT start_date, end_date, created_by_agent_id FROM hq_calendar_events WHERE id = :event_id"),
        {"event_id": event_id},
    ).fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail="Calendar event not found")
    if not _can_manage_hq_calendar_event(request, existing):
        raise HTTPException(status_code=403, detail="You can only manage events you created")
    allowed = {
        "title",
        "description",
        "event_type",
        "agent_names",
        "start_date",
        "end_date",
        "location",
        "color",
        "related_page",
    }
    data = payload.model_dump(exclude_unset=True)
    next_start_date = data.get("start_date", str(existing.start_date) if existing.start_date else "")
    next_end_date = data.get("end_date", str(existing.end_date) if existing.end_date else "")
    _validate_date_range(next_start_date, next_end_date)
    set_parts = []
    params: dict[str, Any] = {"event_id": event_id, "user_id": _current_user_id(request)}
    for key, value in data.items():
        if key not in allowed:
            continue
        if key == "agent_names":
            params[key] = json.dumps(_clean_agent_names(value))
            set_parts.append("agent_names = CAST(:agent_names AS JSONB)")
        elif key in {"start_date", "end_date"}:
            params[key] = value or ""
            set_parts.append(f"{key} = NULLIF(:{key}, '')::date")
        else:
            params[key] = value
            set_parts.append(f"{key} = :{key}")
    if not set_parts:
        raise HTTPException(status_code=400, detail="No fields to update")
    row = session.execute(
        text(
            f"""
            UPDATE hq_calendar_events
            SET {', '.join(set_parts)}, updated_by_agent_id = :user_id, updated_at = NOW()
            WHERE id = :event_id
            RETURNING *
            """
        ),
        params,
    ).fetchone()
    session.commit()
    return _hq_calendar_event_payload(request, row)


@app.delete("/hq/calendar-events/{event_id}")
def delete_hq_calendar_event(event_id: int, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    existing = session.execute(
        text("SELECT created_by_agent_id FROM hq_calendar_events WHERE id = :event_id"),
        {"event_id": event_id},
    ).fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail="Calendar event not found")
    if not _can_manage_hq_calendar_event(request, existing):
        raise HTTPException(status_code=403, detail="You can only manage events you created")
    session.execute(text("DELETE FROM hq_calendar_events WHERE id = :event_id"), {"event_id": event_id})
    session.commit()
    return {"deleted": True}


@app.get("/hq/priorities")
def hq_priorities(session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    rows = session.execute(
        text(
            """
            SELECT *
            FROM hq_priority_items
            ORDER BY start_date ASC, agent_name ASC, id ASC
            """
        )
    ).fetchall()
    return {"items": [_row_to_dict(row) for row in rows]}


@app.post("/hq/priorities")
def create_hq_priority(payload: HqPriorityItemPayload, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    title = _clean_text(payload.title)
    agent_name = _clean_text(payload.agent_name)
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    if not agent_name:
        raise HTTPException(status_code=400, detail="Agent is required")
    row = session.execute(
        text(
            """
            INSERT INTO hq_priority_items (
              title, description, agent_name, priority, status, start_date, end_date,
              color, related_page, created_by_agent_id, updated_by_agent_id, created_at, updated_at
            ) VALUES (
              :title, :description, :agent_name, :priority, :status,
              COALESCE(NULLIF(:start_date, '')::date, CURRENT_DATE),
              NULLIF(:end_date, '')::date,
              :color, :related_page, :user_id, :user_id, NOW(), NOW()
            )
            RETURNING *
            """
        ),
        {
            "title": title,
            "description": _clean_text(payload.description),
            "agent_name": agent_name,
            "priority": payload.priority or "medium",
            "status": payload.status or "planned",
            "start_date": payload.start_date or "",
            "end_date": payload.end_date or "",
            "color": _clean_text(payload.color),
            "related_page": _clean_text(payload.related_page),
            "user_id": _current_user_id(request),
        },
    ).fetchone()
    session.commit()
    return _row_to_dict(row)


@app.patch("/hq/priorities/{item_id}")
def update_hq_priority(item_id: int, payload: HqPriorityItemPayload, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    allowed = {
        "title",
        "description",
        "agent_name",
        "priority",
        "status",
        "start_date",
        "end_date",
        "color",
        "related_page",
    }
    data = payload.model_dump(exclude_unset=True)
    set_parts = []
    params: dict[str, Any] = {"item_id": item_id, "user_id": _current_user_id(request)}
    for key, value in data.items():
        if key not in allowed:
            continue
        params[key] = value
        if key in {"start_date", "end_date"}:
            set_parts.append(f"{key} = NULLIF(:{key}, '')::date")
        else:
            set_parts.append(f"{key} = :{key}")
    if not set_parts:
        raise HTTPException(status_code=400, detail="No fields to update")
    row = session.execute(
        text(
            f"""
            UPDATE hq_priority_items
            SET {', '.join(set_parts)}, updated_by_agent_id = :user_id, updated_at = NOW()
            WHERE id = :item_id
            RETURNING *
            """
        ),
        params,
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Priority item not found")
    session.commit()
    return _row_to_dict(row)


@app.delete("/hq/priorities/{item_id}")
def delete_hq_priority(item_id: int, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    row = session.execute(
        text("DELETE FROM hq_priority_items WHERE id = :item_id RETURNING id"),
        {"item_id": item_id},
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Priority item not found")
    return {"deleted": True}


@app.get("/hd-players")
def hd_players(q: Optional[str] = Query(None), agent: Optional[str] = Query(None), session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    _seed_hd_players_if_empty(session)
    sql = """
    SELECT hp.*, p.name AS linked_player_name
    FROM hd_players hp
    LEFT JOIN players p ON p.id = hp.player_id
    WHERE COALESCE(hp.status, 'active') <> 'archived'
    """
    params: dict[str, Any] = {}
    if q:
        sql += " AND (hp.display_name ILIKE :q OR hp.current_club ILIKE :q OR hp.plan ILIKE :q)"
        params["q"] = f"%{q}%"
    if agent:
        sql += " AND hp.assigned_agent ILIKE :agent"
        params["agent"] = f"%{agent}%"
    sql += " ORDER BY CASE hp.priority WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 WHEN 'D' THEN 4 ELSE 5 END, hp.display_name"
    rows = session.execute(text(sql), params).fetchall()
    items = [_row_to_dict(row) for row in rows]
    ids = [item["id"] for item in items]
    documents_by_player: dict[int, list[dict[str, Any]]] = {player_id: [] for player_id in ids}
    if ids:
        doc_rows = session.execute(
            text(
                """
                SELECT *
                FROM hd_player_documents
                WHERE hd_player_id = ANY(:ids)
                ORDER BY created_at DESC
                """
            ),
            {"ids": ids},
        ).fetchall()
        for row in doc_rows:
            doc = _row_to_dict(row)
            documents_by_player.setdefault(doc["hd_player_id"], []).append(doc)
    for item in items:
        item["documents"] = documents_by_player.get(item["id"], [])
    return {"items": items}


def _hd_player_payload(session: Session, hd_player_id: int) -> Optional[dict[str, Any]]:
    row = session.execute(
        text(
            """
            SELECT
              hp.*,
              p.name AS linked_player_name
            FROM hd_players hp
            LEFT JOIN players p ON p.id = hp.player_id
            WHERE hp.id = :hd_player_id
              AND COALESCE(hp.status, 'active') <> 'archived'
            """
        ),
        {"hd_player_id": hd_player_id},
    ).fetchone()
    if not row:
        return None
    payload = _row_to_dict(row)
    if payload.get("manual_performance") is None:
        payload["manual_performance"] = {}
    doc_rows = session.execute(
        text(
            """
            SELECT *
            FROM hd_player_documents
            WHERE hd_player_id = :hd_player_id
            ORDER BY created_at DESC
            """
        ),
        {"hd_player_id": hd_player_id},
    ).fetchall()
    payload["documents"] = [_row_to_dict(doc_row) for doc_row in doc_rows]
    manual_prospect_rows = session.execute(
        text(
            """
            SELECT *
            FROM hd_player_prospect_clubs
            WHERE hd_player_id = :hd_player_id
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
            """
        ),
        {"hd_player_id": hd_player_id},
    ).fetchall()
    payload["manual_prospect_clubs"] = [_row_to_dict(prospect_row) for prospect_row in manual_prospect_rows]
    payload["mercato_prospects"] = []
    if payload.get("player_id"):
        prospect_rows = session.execute(
            text(
                """
                SELECT
                  mr.id AS request_id,
                  cl.name AS club_name,
                  comp.name AS competition_name,
                  mn.position,
                  mr.title,
                  mr.priority,
                  mr.status AS request_status,
                  mc.status AS candidate_status,
                  mc.match_score,
                  mc.agent_note,
                  mr.assigned_agent_id,
                  assignee.display_name AS assigned_agent_name
                FROM mercato_candidates mc
                JOIN mercato_needs mn ON mn.id = mc.mercato_need_id
                JOIN mercato_requests mr ON mr.id = mn.mercato_request_id
                LEFT JOIN clubs cl ON cl.id = mr.club_id
                LEFT JOIN competitions comp ON comp.id = cl.competition_id
                LEFT JOIN auth_users assignee ON assignee.username = mr.assigned_agent_id
                WHERE mc.player_id = :player_id AND mr.archived_at IS NULL
                ORDER BY mc.match_score DESC NULLS LAST, mr.updated_at DESC NULLS LAST
                """
            ),
            {"player_id": payload["player_id"]},
        ).fetchall()
        payload["mercato_prospects"] = [_row_to_dict(prospect_row) for prospect_row in prospect_rows]
    transfer_history = [*_manual_transfer_rows(session, hd_player_id), *_player_transfer_history(session, payload)]
    payload["transfer_history"] = sorted(
        transfer_history,
        key=lambda item: (str(item.get("transfer_date") or ""), int(item.get("id") or 0)),
        reverse=True,
    )[:40]
    return payload


@app.get("/hd-players/files/{file_key:path}")
def get_hd_player_file(file_key: str):
    client = _s3_client()
    bucket = _s3_bucket()
    key = urllib.parse.unquote(file_key)
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except ClientError:
        raise HTTPException(status_code=404, detail="File not found")
    content = obj["Body"].read()
    content_type = obj.get("ContentType") or mimetypes.guess_type(key)[0] or "application/octet-stream"
    file_name = key.rsplit("/", 1)[-1] or "file"
    disposition = "inline" if content_type.startswith("image/") or "pdf" in content_type else "attachment"
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f'{disposition}; filename="{file_name}"'},
    )


@app.post("/hd-players/{hd_player_id}/upload")
async def upload_hd_player_file(
    hd_player_id: int,
    request: Request,
    purpose: str = Query("document"),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    _ensure_agency_ops_schema(session)
    player_row = session.execute(
        text("SELECT id FROM hd_players WHERE id = :hd_player_id"),
        {"hd_player_id": hd_player_id},
    ).fetchone()
    if not player_row:
        raise HTTPException(status_code=404, detail="HD player not found")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    max_bytes = 20 * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")
    clean_purpose = re.sub(r"[^a-z0-9_-]+", "-", purpose.lower()).strip("-") or "document"
    clean_name = re.sub(r"[^A-Za-z0-9._-]+", "-", file.filename or "upload").strip("-") or "upload"
    key = f"{os.getenv('S3_PREFIX', 'new_nextlegend').strip('/')}/hd-players/{hd_player_id}/{clean_purpose}/{uuid.uuid4().hex}-{clean_name}"
    content_type = file.content_type or mimetypes.guess_type(clean_name)[0] or "application/octet-stream"
    client = _s3_client()
    bucket = _s3_bucket()
    _ensure_s3_bucket(client, bucket)
    client.put_object(Bucket=bucket, Key=key, Body=raw, ContentType=content_type)
    return {
        "file_key": key,
        "storage_url": _storage_proxy_url(request, key),
        "file_name": file.filename,
        "content_type": content_type,
        "size_bytes": len(raw),
    }


@app.get("/hd-players/{hd_player_id}")
def get_hd_player(hd_player_id: int, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    _ensure_mercato_schema(session)
    payload = _hd_player_payload(session, hd_player_id)
    if not payload:
        raise HTTPException(status_code=404, detail="HD player not found")
    return payload


@app.post("/hd-players")
def create_hd_player(payload: HdPlayerPayload, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    display_name = _clean_text(payload.display_name)
    if not display_name:
        raise HTTPException(status_code=400, detail="Display name is required")
    row = session.execute(
        text(
            """
            INSERT INTO hd_players (
              player_id, display_name, position, current_club, contract_expiry,
              current_club_situation, plan, priority, demanded_transfer_fee, next_step,
              assigned_agent, photo_url, birth_date, player_phone, player_email, entourage_phone, entourage_email,
              season_objectives, eyeball_url, transfermarkt_url, is_young_player, manual_performance,
              contract_status, mandate_status, medical_status,
              market_notes, scouting_notes, status, created_by_agent_id, updated_by_agent_id,
              created_at, updated_at
            ) VALUES (
              :player_id, :display_name, :position, :current_club, NULLIF(:contract_expiry, '')::date,
              :current_club_situation, :plan, :priority, :demanded_transfer_fee, :next_step,
              :assigned_agent, :photo_url, NULLIF(:birth_date, '')::date, :player_phone, :player_email, :entourage_phone, :entourage_email,
              :season_objectives, :eyeball_url, :transfermarkt_url, :is_young_player, CAST(:manual_performance AS JSONB),
              :contract_status, :mandate_status, :medical_status,
              :market_notes, :scouting_notes, :status, :user_id, :user_id, NOW(), NOW()
            )
            RETURNING *
            """
        ),
        {
            **payload.model_dump(),
            "display_name": display_name,
            "contract_expiry": payload.contract_expiry or "",
            "birth_date": payload.birth_date or "",
            "priority": payload.priority or "B",
            "is_young_player": bool(payload.is_young_player),
            "manual_performance": json.dumps(payload.manual_performance or {}),
            "status": payload.status or "active",
            "user_id": _current_user_id(request),
        },
    ).fetchone()
    session.commit()
    return _hd_player_payload(session, int(row.id)) or _row_to_dict(row)


@app.patch("/hd-players/{hd_player_id}")
def update_hd_player(hd_player_id: int, payload: HdPlayerPayload, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    allowed = set(HdPlayerPayload.model_fields.keys())
    data = payload.model_dump(exclude_unset=True)
    set_parts = []
    params: dict[str, Any] = {"hd_player_id": hd_player_id, "user_id": _current_user_id(request)}
    for key, value in data.items():
        if key not in allowed:
            continue
        if key == "status":
            _require_admin(request)
        params[key] = json.dumps(value or {}) if key == "manual_performance" else value if value is not None else None
        if key in {"contract_expiry", "birth_date"}:
            set_parts.append(f"{key} = NULLIF(:{key}, '')::date")
        elif key == "manual_performance":
            set_parts.append("manual_performance = CAST(:manual_performance AS JSONB)")
        else:
            set_parts.append(f"{key} = :{key}")
    if not set_parts:
        raise HTTPException(status_code=400, detail="No fields to update")
    row = session.execute(
        text(
            f"""
            UPDATE hd_players
            SET {', '.join(set_parts)}, updated_by_agent_id = :user_id, updated_at = NOW()
            WHERE id = :hd_player_id
            RETURNING *
            """
        ),
        params,
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="HD player not found")
    session.commit()
    return _hd_player_payload(session, hd_player_id) or _row_to_dict(row)


@app.post("/hd-players/{hd_player_id}/documents")
def create_hd_player_document(hd_player_id: int, payload: HdPlayerDocumentPayload, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    player_row = session.execute(
        text("SELECT player_id FROM hd_players WHERE id = :hd_player_id"),
        {"hd_player_id": hd_player_id},
    ).fetchone()
    if not player_row:
        raise HTTPException(status_code=404, detail="HD player not found")
    row = session.execute(
        text(
            """
            INSERT INTO hd_player_documents (
              hd_player_id, player_id, document_type, title, file_name, file_key,
              storage_url, content_type, size_bytes, notes, created_by_agent_id, created_at
            ) VALUES (
              :hd_player_id, :player_id, :document_type, :title, :file_name, :file_key,
              :storage_url, :content_type, :size_bytes, :notes, :user_id, NOW()
            )
            RETURNING *
            """
        ),
        {
            **payload.model_dump(),
            "hd_player_id": hd_player_id,
            "player_id": player_row.player_id,
            "user_id": _current_user_id(request),
        },
    ).fetchone()
    session.commit()
    return _row_to_dict(row)


@app.patch("/hd-players/documents/{document_id}")
def update_hd_player_document(document_id: int, payload: HdPlayerDocumentPayload, request: Request, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    data = payload.model_dump(exclude_unset=True)
    allowed = set(HdPlayerDocumentPayload.model_fields.keys())
    set_parts = []
    params: dict[str, Any] = {"document_id": document_id, "user_id": _current_user_id(request)}
    for key, value in data.items():
        if key not in allowed:
            continue
        params[key] = value
        set_parts.append(f"{key} = :{key}")
    if not set_parts:
        raise HTTPException(status_code=400, detail="No fields to update")
    row = session.execute(
        text(
            f"""
            UPDATE hd_player_documents
            SET {', '.join(set_parts)}
            WHERE id = :document_id
            RETURNING *
            """
        ),
        params,
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    session.commit()
    return _row_to_dict(row)


@app.delete("/hd-players/documents/{document_id}")
def delete_hd_player_document(document_id: int, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    row = session.execute(
        text("DELETE FROM hd_player_documents WHERE id = :document_id RETURNING id"),
        {"document_id": document_id},
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


@app.post("/hd-players/{hd_player_id}/prospect-clubs")
def create_hd_player_prospect_club(
    hd_player_id: int,
    payload: HdPlayerProspectClubPayload,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_agency_ops_schema(session)
    club_name = _clean_text(payload.club_name)
    if not club_name:
        raise HTTPException(status_code=400, detail="Club name is required")
    row = session.execute(
        text(
            """
            INSERT INTO hd_player_prospect_clubs (
              hd_player_id, club_id, club_name, competition_name, status, offer,
              contact, notes, created_by_agent_id, created_at, updated_at
            ) VALUES (
              :hd_player_id, :club_id, :club_name, :competition_name, :status, :offer,
              :contact, :notes, :user_id, NOW(), NOW()
            )
            RETURNING *
            """
        ),
        {
            **payload.model_dump(),
            "hd_player_id": hd_player_id,
            "club_name": club_name,
            "user_id": _current_user_id(request),
        },
    ).fetchone()
    session.commit()
    return _row_to_dict(row)


@app.delete("/hd-players/prospect-clubs/{prospect_id}")
def delete_hd_player_prospect_club(prospect_id: int, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    row = session.execute(
        text("DELETE FROM hd_player_prospect_clubs WHERE id = :prospect_id RETURNING id"),
        {"prospect_id": prospect_id},
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Prospect club not found")
    return {"deleted": True}


@app.post("/hd-players/{hd_player_id}/transfers")
def create_hd_player_manual_transfer(
    hd_player_id: int,
    payload: HdPlayerManualTransferPayload,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_agency_ops_schema(session)
    row = session.execute(
        text(
            """
            INSERT INTO hd_player_manual_transfers (
              hd_player_id, transfer_date, transfer_type, transfer_fee, team_in_name,
              team_out_name, league_name, notes, created_by_agent_id, created_at, updated_at
            ) VALUES (
              :hd_player_id, NULLIF(:transfer_date, '')::date, :transfer_type, :transfer_fee,
              :team_in_name, :team_out_name, :league_name, :notes, :user_id, NOW(), NOW()
            )
            RETURNING *
            """
        ),
        {
            **payload.model_dump(),
            "transfer_date": payload.transfer_date or "",
            "hd_player_id": hd_player_id,
            "user_id": _current_user_id(request),
        },
    ).fetchone()
    session.commit()
    return _row_to_dict(row)


@app.delete("/hd-players/transfers/{transfer_id}")
def delete_hd_player_manual_transfer(transfer_id: int, session: Session = Depends(get_session)):
    _ensure_agency_ops_schema(session)
    row = session.execute(
        text("DELETE FROM hd_player_manual_transfers WHERE id = :transfer_id RETURNING id"),
        {"transfer_id": transfer_id},
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Manual transfer not found")
    return {"deleted": True}


@app.delete("/hd-players/{hd_player_id}")
def archive_hd_player(hd_player_id: int, request: Request, session: Session = Depends(get_session)):
    _require_admin(request)
    _ensure_agency_ops_schema(session)
    row = session.execute(
        text(
            """
            UPDATE hd_players
            SET status = 'archived', updated_by_agent_id = :user_id, updated_at = NOW()
            WHERE id = :hd_player_id
              AND COALESCE(status, 'active') <> 'archived'
            RETURNING id
            """
        ),
        {"hd_player_id": hd_player_id, "user_id": _current_user_id(request)},
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="HD player not found")
    return {"archived": True, "id": hd_player_id}


@app.get("/mercato/requests/export.xlsx")
def export_mercato_requests(session: Session = Depends(get_session)):
    _ensure_mercato_schema(session)
    _ensure_agency_ops_schema(session)
    _seed_hd_players_if_empty(session)
    request_rows = session.execute(
        text(
            """
            SELECT
              mr.id,
              cl.name AS club,
              comp.name AS league,
              mn.position,
              mn.notes AS profile,
              mr.budget_max AS fee,
              mr.salary_max AS net_wages,
              mr.assigned_agent_id AS agent,
              mr.priority,
              mr.status,
              mr.title,
              mr.updated_at
            FROM mercato_requests mr
            LEFT JOIN clubs cl ON cl.id = mr.club_id
            LEFT JOIN competitions comp ON comp.id = cl.competition_id
            LEFT JOIN mercato_needs mn ON mn.mercato_request_id = mr.id
            WHERE mr.archived_at IS NULL
            ORDER BY cl.name, mn.position, mr.id
            """
        )
    ).fetchall()
    candidate_rows = session.execute(
        text(
            """
            SELECT
              mr.id AS request_id,
              cl.name AS club,
              mn.position,
              p.name AS player,
              mc.status,
              mc.match_score,
              mc.calibrated_player_level,
              mc.agent_note,
              mr.assigned_agent_id AS agent
            FROM mercato_candidates mc
            JOIN mercato_needs mn ON mn.id = mc.mercato_need_id
            JOIN mercato_requests mr ON mr.id = mn.mercato_request_id
            LEFT JOIN clubs cl ON cl.id = mr.club_id
            LEFT JOIN players p ON p.id = mc.player_id
            WHERE mr.archived_at IS NULL
            ORDER BY mr.id, mc.match_score DESC NULLS LAST
            """
        )
    ).fetchall()
    hd_rows = session.execute(
        text(
            """
            SELECT display_name, current_club, contract_expiry, current_club_situation,
                   plan, priority, demanded_transfer_fee, next_step, assigned_agent
            FROM hd_players
            ORDER BY CASE priority WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 WHEN 'D' THEN 4 ELSE 5 END, display_name
            """
        )
    ).fetchall()
    overview = [["Name", "Current club", "Club logo URL", "Contract expiry", "Current club situation", "Plan", "Priority", "Demanded TF", "Next step", "Agent"]]
    overview.extend([[row.display_name, row.current_club, _club_logo_url(row.current_club), row.contract_expiry, row.current_club_situation, row.plan, row.priority, row.demanded_transfer_fee, row.next_step, row.assigned_agent] for row in hd_rows])
    requirements = [["Clubs", "Club logo URL", "Category", "Profile", "Fee", "Net wages", "Suggestion", "Status", "Agent", "Priority", "Request ID"]]
    requirements.extend([[row.club, _club_logo_url(row.club), row.position, row.profile, row.fee, row.net_wages, row.title, row.status, row.agent, row.priority, row.id] for row in request_rows])
    candidates = [["Request ID", "Club", "Club logo URL", "Category", "Player", "Status", "Match score", "Calibrated level", "Agent note", "Agent"]]
    candidates.extend([[row.request_id, row.club, _club_logo_url(row.club), row.position, row.player, row.status, row.match_score, row.calibrated_player_level, row.agent_note, row.agent] for row in candidate_rows])
    player_sheets = [
        ("Kevin", [["Club", "Club logo URL", "Status", "Offer", "Contact", "Notes"], *[[club, _club_logo_url(club), status, offer, contact, notes] for club, status, offer, contact, notes in [["Monaco", "Interest", "", "", ""], ["Marseille", "Pending", "", "", ""], ["Everton", "Pending", "", "", ""], ["Nottingham", "Pending", "", "", ""], ["Brentford", "No interest", "", "", ""], ["Newcaslte", "No interest", "", "", ""], ["Atletico Madrid", "No interest", "", "", ""]]]]),
        ("Lilian", [["Club", "Club logo URL", "Status", "Offer", "Contact", "Meet", "Notes"], ["Bologna", _club_logo_url("Bologna"), "Interest", "", "", "", ""]]),
        ("Mario", [["Club", "Club logo URL", "Status", "Offer", "Contact", "Meet", "Notes"], ["Al Shabab", _club_logo_url("Al Shabab"), "Offer", "3.5 net", "", "", ""]]),
        ("Simon", [["Club", "Club logo URL", "Status", "Offer", "Contact", "Meet", "Notes"], ["Cruzeiro Esporte Clube", _club_logo_url("Cruzeiro Esporte Clube"), "No interest", "", "Damon intermediary", "", ""]]),
    ]
    workbook = _build_xlsx([
        ("Overview", overview),
        ("Clubs requirements", requirements),
        *player_sheets,
        ("Matching shortlist", candidates),
    ])
    return Response(
        workbook,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="nextlegend_mercato_2026.xlsx"'},
    )


@app.get("/mercato/requests")
def mercato_requests(
    club: Optional[str] = Query(None),
    agent: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    competition: Optional[str] = Query(None),
    deal_type: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    _ensure_mercato_schema(session)
    sql = """
    SELECT DISTINCT mr.id
    FROM mercato_requests mr
    LEFT JOIN clubs cl ON cl.id = mr.club_id
    LEFT JOIN competitions comp ON comp.id = cl.competition_id
    LEFT JOIN mercato_needs mn ON mn.mercato_request_id = mr.id
    WHERE mr.archived_at IS NULL
    """
    params: dict[str, Any] = {}
    if club:
        sql += " AND cl.name ILIKE :club"
        params["club"] = f"%{club}%"
    if agent:
        sql += " AND (mr.assigned_agent_id ILIKE :agent OR mr.created_by_agent_id ILIKE :agent)"
        params["agent"] = f"%{agent}%"
    if position:
        sql += " AND mn.position ILIKE :position"
        params["position"] = f"%{position}%"
    if priority:
        sql += " AND mr.priority = :priority"
        params["priority"] = priority
    if status:
        sql += " AND mr.status = :status"
        params["status"] = status
    if competition:
        sql += " AND comp.name ILIKE :competition"
        params["competition"] = f"%{competition}%"
    if deal_type:
        sql += " AND mr.deal_type = :deal_type"
        params["deal_type"] = deal_type
    sql += " ORDER BY mr.id DESC"
    ids = [row.id for row in session.execute(text(sql), params).fetchall()]
    items = []
    for request_id in ids:
        payload = _mercato_request_payload(session, request_id)
        if payload:
            items.append(payload)
    active = [item for item in items if item.get("status") != "closed"]
    club_ids = {item.get("club_id") for item in active if item.get("club_id")}
    candidates_count = sum(len(need.get("candidates") or []) for item in items for need in item.get("needs") or [])
    urgent_count = sum(1 for item in active if item.get("priority") == "urgent")
    return {
        "items": items,
        "kpis": {
            "active_requests": len(active),
            "clubs_count": len(club_ids),
            "shortlisted_players": candidates_count,
            "urgent_requests": urgent_count,
        },
    }


@app.post("/mercato/requests")
def create_mercato_request(payload: MercatoRequestCreate, request: Request, session: Session = Depends(get_session)):
    _ensure_mercato_schema(session)
    if not payload.club_id:
        raise HTTPException(status_code=400, detail="Club is required")
    if not _clean_text(payload.need.position):
        raise HTTPException(status_code=400, detail="Position is required")
    created_by = _current_user_id(request)
    title = _clean_text(payload.title)
    if not title:
        title = payload.need.position or "Mercato need"
    row = session.execute(
        text(
            """
            INSERT INTO mercato_requests (
              club_id,
              created_by_agent_id,
              assigned_agent_id,
              season,
              title,
              priority,
              status,
              budget_min,
              budget_max,
              salary_max,
              deal_type,
              extra_info,
              created_at,
              updated_at
            ) VALUES (
              :club_id,
              :created_by,
              :assigned_agent_id,
              :season,
              :title,
              :priority,
              :status,
              :budget_min,
              :budget_max,
              :salary_max,
              :deal_type,
              :extra_info,
              NOW(),
              NOW()
            )
            RETURNING id
            """
        ),
        {
            "club_id": payload.club_id,
            "created_by": created_by,
            "assigned_agent_id": payload.assigned_agent_id or created_by,
            "season": payload.season or "2026",
            "title": title,
            "priority": payload.priority or "medium",
            "status": payload.status or "new",
            "budget_min": payload.budget_min,
            "budget_max": payload.budget_max,
            "salary_max": payload.salary_max,
            "deal_type": payload.deal_type or "any",
            "extra_info": _clean_text(payload.extra_info),
        },
    ).fetchone()
    request_id = int(row.id)
    session.execute(
        text(
            """
            INSERT INTO mercato_needs (
              mercato_request_id,
              position,
              role,
              age_min,
              age_max,
              preferred_foot,
              height_min,
              target_league_level,
              required_player_level,
              nationality_preferences,
              contract_preferences,
              notes,
              created_at,
              updated_at
            ) VALUES (
              :request_id,
              :position,
              :role,
              :age_min,
              :age_max,
              :preferred_foot,
              :height_min,
              :target_league_level,
              :required_player_level,
              :nationality_preferences,
              :contract_preferences,
              :notes,
              NOW(),
              NOW()
            )
            """
        ),
        {
            "request_id": request_id,
            "position": _clean_text(payload.need.position),
            "role": None,
            "age_min": payload.need.age_min,
            "age_max": payload.need.age_max,
            "preferred_foot": _clean_text(payload.need.preferred_foot),
            "height_min": payload.need.height_min,
            "target_league_level": _clean_text(payload.need.target_league_level),
            "required_player_level": payload.need.required_player_level,
            "nationality_preferences": _clean_text(payload.need.nationality_preferences),
            "contract_preferences": None,
            "notes": _clean_text(payload.need.notes),
        },
    )
    session.commit()
    return _mercato_request_payload(session, request_id)


@app.get("/mercato/requests/{request_id}")
def get_mercato_request(request_id: int, session: Session = Depends(get_session)):
    _ensure_mercato_schema(session)
    payload = _mercato_request_payload(session, request_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Mercato request not found")
    return payload


@app.patch("/mercato/requests/{request_id}")
def update_mercato_request(
    request_id: int,
    payload: MercatoRequestUpdate,
    session: Session = Depends(get_session),
):
    _ensure_mercato_schema(session)
    existing = _mercato_request_payload(session, request_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Mercato request not found")
    data = payload.model_dump(exclude_unset=True)
    if "club_id" in data and not data.get("club_id"):
        raise HTTPException(status_code=400, detail="Club is required")
    request_fields = {
        "club_id",
        "assigned_agent_id",
        "season",
        "title",
        "priority",
        "status",
        "budget_min",
        "budget_max",
        "salary_max",
        "deal_type",
        "extra_info",
    }
    updates = {key: data[key] for key in request_fields if key in data}
    if updates:
        set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
        session.execute(
            text(f"UPDATE mercato_requests SET {set_clause}, updated_at = NOW() WHERE id = :request_id"),
            {**updates, "request_id": request_id},
        )
    need_payload = data.get("need")
    if need_payload:
        if "position" in need_payload and not _clean_text(need_payload.get("position")):
            raise HTTPException(status_code=400, detail="Position is required")
        need_id = need_payload.get("id") or (existing.get("needs") or [{}])[0].get("id")
        if need_id:
            need_fields = {
                "position",
                "role",
                "age_min",
                "age_max",
                "preferred_foot",
                "height_min",
                "target_league_level",
                "required_player_level",
                "nationality_preferences",
                "contract_preferences",
                "notes",
            }
            need_updates = {key: need_payload[key] for key in need_fields if key in need_payload}
            if need_updates:
                set_clause = ", ".join([f"{key} = :{key}" for key in need_updates.keys()])
                session.execute(
                    text(f"UPDATE mercato_needs SET {set_clause}, updated_at = NOW() WHERE id = :need_id"),
                    {**need_updates, "need_id": need_id},
                )
    session.commit()
    return _mercato_request_payload(session, request_id)


@app.delete("/mercato/requests/{request_id}")
def archive_mercato_request(request_id: int, session: Session = Depends(get_session)):
    _ensure_mercato_schema(session)
    result = session.execute(
        text(
            """
            UPDATE mercato_requests
            SET status = 'closed',
                archived_at = NOW(),
                updated_at = NOW()
            WHERE id = :request_id AND archived_at IS NULL
            """
        ),
        {"request_id": request_id},
    )
    session.commit()
    return {"archived": result.rowcount > 0}


@app.post("/mercato/needs/{need_id}/candidates")
def add_mercato_candidate(
    need_id: int,
    payload: MercatoCandidateCreate,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_mercato_schema(session)
    player = _mercato_player_context(session, payload.player_id, payload.player_season_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    added, candidate_id = _insert_mercato_candidate(
        session,
        need_id=need_id,
        player=player,
        source=payload.source or "manual",
        status=payload.status or "suggested",
        agent_note=_clean_text(payload.agent_note),
        created_by=_current_user_id(request),
    )
    session.commit()
    return {"added": added, "candidate_id": candidate_id}


@app.post("/mercato/needs/{need_id}/generate-shortlist")
def generate_mercato_shortlist(
    need_id: int,
    payload: Optional[MercatoShortlistGenerate] = None,
    request: Request = None,
    session: Session = Depends(get_session),
):
    _ensure_mercato_schema(session)
    need = _mercato_need_context(session, need_id)
    if not need:
        raise HTTPException(status_code=404, detail="Mercato need not found")
    if need.get("request_status") == "closed":
        raise HTTPException(status_code=400, detail="Cannot generate shortlist for a closed need")
    session.execute(
        text(
            """
            DELETE FROM mercato_candidates
            WHERE mercato_need_id = :need_id
              AND source = 'algorithm'
              AND status = 'suggested'
            """
        ),
        {"need_id": need_id},
    )
    scored = _mercato_shortlist_scored_players(session, need, payload)
    inserted = []
    for _, player, score in scored[:5]:
        added, candidate_id = _insert_mercato_candidate(
            session,
            need_id=need_id,
            player=player,
            source="algorithm",
            status="suggested",
            agent_note=None,
            created_by=_current_user_id(request) if request is not None else None,
            scoring=score,
        )
        if added:
            inserted.append({"candidate_id": candidate_id, "player_id": player["player_id"], "match_score": score["match_score"]})
    if inserted:
        session.execute(
            text("UPDATE mercato_requests SET status = 'shortlist_ready', updated_at = NOW() WHERE id = :request_id"),
            {"request_id": need["mercato_request_id"]},
        )
    session.commit()
    return {"generated": len(inserted), "candidates": _mercato_candidates_for_need(session, need_id)}


@app.post("/mercato/needs/{need_id}/preview-shortlist")
def preview_mercato_shortlist(
    need_id: int,
    payload: Optional[MercatoShortlistGenerate] = None,
    session: Session = Depends(get_session),
):
    _ensure_mercato_schema(session)
    need = _mercato_need_context(session, need_id)
    if not need:
        raise HTTPException(status_code=404, detail="Mercato need not found")
    if need.get("request_status") == "closed":
        raise HTTPException(status_code=400, detail="Cannot run matching for a closed need")
    scored = _mercato_shortlist_scored_players(session, need, payload)
    return {
        "need_id": need_id,
        "generated": min(len(scored), 5),
        "candidates": [_mercato_recommendation_payload(player, score) for _, player, score in scored[:5]],
    }


@app.patch("/mercato/candidates/{candidate_id}")
def update_mercato_candidate(
    candidate_id: int,
    payload: MercatoCandidateUpdate,
    request: Request,
    session: Session = Depends(get_session),
):
    _ensure_mercato_schema(session)
    existing = session.execute(
        text("SELECT id, status FROM mercato_candidates WHERE id = :candidate_id"),
        {"candidate_id": candidate_id},
    ).fetchone()
    if not existing:
        raise HTTPException(status_code=404, detail="Mercato candidate not found")
    data = payload.model_dump(exclude_unset=True)
    updates = {key: data[key] for key in ("status", "agent_note") if key in data}
    if not updates:
        return {"updated": False}
    set_clause = ", ".join([f"{key} = :{key}" for key in updates])
    session.execute(
        text(f"UPDATE mercato_candidates SET {set_clause}, updated_at = NOW() WHERE id = :candidate_id"),
        {**updates, "candidate_id": candidate_id},
    )
    if "status" in updates and updates["status"] != existing.status:
        session.execute(
            text(
                """
                INSERT INTO mercato_candidate_events (
                  mercato_candidate_id,
                  event_type,
                  old_status,
                  new_status,
                  note,
                  created_by_agent_id,
                  created_at
                ) VALUES (
                  :candidate_id,
                  'status_changed',
                  :old_status,
                  :new_status,
                  :note,
                  :created_by,
                  NOW()
                )
                """
            ),
            {
                "candidate_id": candidate_id,
                "old_status": existing.status,
                "new_status": updates["status"],
                "note": data.get("agent_note"),
                "created_by": _current_user_id(request),
            },
        )
    session.commit()
    return {"updated": True}


@app.post("/mercato/candidates/{candidate_id}/status")
def update_mercato_candidate_status(
    candidate_id: int,
    payload: MercatoCandidateStatus,
    request: Request,
    session: Session = Depends(get_session),
):
    return update_mercato_candidate(
        candidate_id,
        MercatoCandidateUpdate(status=payload.status, agent_note=payload.note),
        request,
        session,
    )


@app.get("/ranking", response_model=List[RankingRow])
def ranking(
    role: Optional[str] = Query(None),
    competition: Optional[str] = Query(None),
    min_minutes: Optional[float] = Query(270),
    season: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    age_min: Optional[float] = Query(None, ge=0),
    age_max: Optional[float] = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    limit: int = Query(30),
    session: Session = Depends(get_session),
):
    tm_clause = _tm_select_clause(session)
    sql = """
    SELECT
      ps.id AS player_season_id,
      p.id AS player_id,
      p.name,
      p.tm_id,
      p.tm_profile_url,
      c.name AS competition_name,
      ps.calendar,
      ps.team_in_selected_period AS team,
      ps.position,
      ps.assigned_role,
      ps.minutes_played,
      ps.global_score_adjusted,
      COALESCE(ps.assigned_role_pct_league, pct_league_fallback.fallback_assigned_role_pct_league) AS assigned_role_pct_league,
      COALESCE(ps.assigned_role_pct_global, pct_global_fallback.fallback_assigned_role_pct_global) AS assigned_role_pct_global,
      pm.age AS age""" + tm_clause + """
    FROM player_seasons ps
    JOIN players p ON p.id = ps.player_id
    JOIN competitions c ON c.id = ps.competition_id
    LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
    """ + _ranking_pct_fallback_join() + """
    WHERE ps.global_score_adjusted IS NOT NULL
      AND ps.minutes_played >= :min_minutes
    """
    params = {"min_minutes": min_minutes or 0, "limit": limit}
    sql, params = _apply_ranking_filters(sql, params, role, competition, season, position, team, age_min, age_max)
    sql += " ORDER BY ps.global_score_adjusted DESC NULLS LAST OFFSET :offset LIMIT :limit"
    params["offset"] = offset

    rows = session.execute(text(sql), params).fetchall()
    items = []
    for row in rows:
        payload = _row_to_dict(row)
        payload["tm_fields"] = _extract_tm_fields(payload)
        items.append(RankingRow(**payload))
    return items


@app.get("/ranking/page", response_model=RankingPage)
def ranking_page(
    role: Optional[str] = Query(None),
    competition: Optional[str] = Query(None),
    min_minutes: Optional[float] = Query(270),
    season: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    age_min: Optional[float] = Query(None, ge=0),
    age_max: Optional[float] = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=200),
    session: Session = Depends(get_session),
):
    tm_clause = _tm_select_clause(session)
    base_sql = """
    FROM player_seasons ps
    JOIN players p ON p.id = ps.player_id
    JOIN competitions c ON c.id = ps.competition_id
    LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
    """ + _ranking_pct_fallback_join() + """
    WHERE ps.global_score_adjusted IS NOT NULL
      AND ps.minutes_played >= :min_minutes
    """
    params = {"min_minutes": min_minutes or 0}
    base_sql, params = _apply_ranking_filters(base_sql, params, role, competition, season, position, team, age_min, age_max)

    count_sql = "SELECT COUNT(*) " + base_sql
    total = session.execute(text(count_sql), params).scalar() or 0

    data_sql = """
    SELECT
      ps.id AS player_season_id,
      p.id AS player_id,
      p.name,
      p.tm_id,
      p.tm_profile_url,
      c.name AS competition_name,
      ps.calendar,
      ps.team_in_selected_period AS team,
      ps.position,
      ps.assigned_role,
      ps.minutes_played,
      ps.global_score_adjusted,
      COALESCE(ps.assigned_role_pct_league, pct_league_fallback.fallback_assigned_role_pct_league) AS assigned_role_pct_league,
      COALESCE(ps.assigned_role_pct_global, pct_global_fallback.fallback_assigned_role_pct_global) AS assigned_role_pct_global,
      pm.age AS age""" + tm_clause + """
    """ + base_sql + " ORDER BY ps.global_score_adjusted DESC NULLS LAST OFFSET :offset LIMIT :limit"
    params = {**params, "offset": offset, "limit": limit}
    rows = session.execute(text(data_sql), params).fetchall()
    items = []
    for row in rows:
        payload = _row_to_dict(row)
        payload["tm_fields"] = _extract_tm_fields(payload)
        items.append(RankingRow(**payload))
    return RankingPage(items=items, total=total, offset=offset, limit=limit)


@app.get("/players/{player_id}", response_model=RankingRow)
def player_card(player_id: int, session: Session = Depends(get_session)):
    tm_clause = _tm_select_clause(session)
    sql = """
    SELECT
      ps.id AS player_season_id,
      p.id AS player_id,
      p.name,
      p.tm_id,
      p.tm_profile_url,
      c.name AS competition_name,
      ps.calendar,
      ps.team_in_selected_period AS team,
      ps.position,
      ps.assigned_role,
      ps.minutes_played,
      ps.matches_played,
      ps.global_score_adjusted,
      ps.assigned_role_pct_league,
      ps.assigned_role_pct_global,
      pm.age AS age""" + tm_clause + """
    FROM player_seasons ps
    JOIN players p ON p.id = ps.player_id
    JOIN competitions c ON c.id = ps.competition_id
    LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
    WHERE p.id = :player_id
    ORDER BY ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
    LIMIT 1
    """
    row = session.execute(text(sql), {"player_id": player_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Player not found")
    payload = _row_to_dict(row)
    payload["tm_fields"] = _extract_tm_fields(payload)
    return RankingRow(**payload)


@app.get("/players/{player_id}/seasons", response_model=List[ReportSeasonOption])
def player_seasons(player_id: int, session: Session = Depends(get_session)):
    items = _load_player_season_context(session, player_id)
    if not items:
        raise HTTPException(status_code=404, detail="Player not found")
    return [
        ReportSeasonOption(
            player_season_id=int(item["player_season_id"]),
            calendar=item.get("calendar"),
            competition_name=item.get("competition_name"),
            team=item.get("team"),
            minutes_played=item.get("minutes_played"),
            global_score_adjusted=item.get("global_score_adjusted"),
        )
        for item in items
    ]


@app.get("/players/{player_id}/report", response_model=Report)
def player_report(
    player_id: int,
    player_season_id: Optional[int] = Query(None, ge=1),
    session: Session = Depends(get_session),
):
    # Load the requested season when valid; otherwise fall back to the latest season.
    row = None
    if player_season_id is not None:
        sql = """
        SELECT ps.id AS player_season_id, ps.*, p.name, p.tm_id, p.tm_profile_url, c.name AS competition_name
        FROM player_seasons ps
        JOIN players p ON p.id = ps.player_id
        JOIN competitions c ON c.id = ps.competition_id
        WHERE p.id = :player_id AND ps.id = :player_season_id
        LIMIT 1
        """
        row = session.execute(
            text(sql),
            {"player_id": player_id, "player_season_id": player_season_id},
        ).fetchone()
    if row is None:
        sql = """
        SELECT ps.id AS player_season_id, ps.*, p.name, p.tm_id, p.tm_profile_url, c.name AS competition_name
        FROM player_seasons ps
        JOIN players p ON p.id = ps.player_id
        JOIN competitions c ON c.id = ps.competition_id
        WHERE p.id = :player_id
        ORDER BY ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
        LIMIT 1
        """
        row = session.execute(text(sql), {"player_id": player_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Player not found")
    ps = _row_to_dict(row)
    season_items = _load_player_season_context(session, player_id)
    available_seasons = [
        ReportSeasonOption(
            player_season_id=int(item["player_season_id"]),
            calendar=item.get("calendar"),
            competition_name=item.get("competition_name"),
            team=item.get("team"),
            minutes_played=item.get("minutes_played"),
            global_score_adjusted=item.get("global_score_adjusted"),
        )
        for item in season_items
    ]
    score_history = _build_score_history(season_items)
    similarities_enabled = True

    if (
        ps.get("assigned_role_pct_league") is None
        or ps.get("assigned_role_pct_global") is None
    ):
        fallback_league, fallback_global = _role_pct_fallback_values(
            session,
            player_id=player_id,
            assigned_role=ps.get("assigned_role"),
        )
        if ps.get("assigned_role_pct_league") is None:
            ps["assigned_role_pct_league"] = fallback_league
        if ps.get("assigned_role_pct_global") is None:
            ps["assigned_role_pct_global"] = fallback_global

    # Metrics (wide)
    metrics_row = session.execute(
        text("SELECT * FROM player_metrics WHERE player_season_id = :psid"),
        {"psid": ps["id"]},
    ).fetchone()
    metrics = _row_to_dict(metrics_row) if metrics_row else {}
    metrics.pop("player_season_id", None)
    metrics.pop("created_at", None)
    metrics.pop("updated_at", None)
    metrics = _hydrate_report_metric_percentiles(session, int(ps["id"]), metrics)
    role_metrics_map = _load_role_metrics()
    assigned_role = ps.get("assigned_role") or ""
    role_metrics = role_metrics_map.get(assigned_role) or []
    if not role_metrics:
        role_metrics = RAW_METRIC_KEYS
    raw_metrics = {key: metrics.get(key) for key in role_metrics}

    # Role scores
    def _load_role_rows(psid: int):
        return session.execute(
            text(
                "SELECT profile, raw_score, pct_league, pct_global, pct_global_adjusted "
                "FROM role_scores WHERE player_season_id = :psid"
            ),
            {"psid": psid},
        ).fetchall()

    role_rows = _load_role_rows(ps["id"])
    role_fit_metrics = metrics
    has_role_percentiles = any(
        (r._mapping.get("pct_league") is not None) or (r._mapping.get("pct_global") is not None)
        for r in role_rows
    )
    if not has_role_percentiles:
        fallback_role_ps = session.execute(
            text(
                """
                SELECT ps.id
                FROM player_seasons ps
                JOIN role_scores rs ON rs.player_season_id = ps.id
                WHERE ps.player_id = :player_id
                  AND (rs.pct_league IS NOT NULL OR rs.pct_global IS NOT NULL)
                GROUP BY ps.id, ps.assigned_role, ps.calendar, ps.minutes_played
                ORDER BY
                  CASE WHEN ps.assigned_role = :assigned_role THEN 0 ELSE 1 END,
                  ps.calendar DESC NULLS LAST,
                  ps.minutes_played DESC NULLS LAST
                LIMIT 1
                """
            ),
            {"player_id": player_id, "assigned_role": ps.get("assigned_role")},
        ).fetchone()
        if fallback_role_ps:
            fallback_psid = int(fallback_role_ps.id)
            role_rows = _load_role_rows(fallback_psid)
            fallback_metrics_row = session.execute(
                text("SELECT * FROM player_metrics WHERE player_season_id = :psid"),
                {"psid": fallback_psid},
            ).fetchone()
            if fallback_metrics_row:
                role_fit_metrics = _row_to_dict(fallback_metrics_row)
                role_fit_metrics.pop("player_season_id", None)
                role_fit_metrics.pop("created_at", None)
                role_fit_metrics.pop("updated_at", None)

    role_scores: list[RoleScore] = []
    for r in role_rows:
        profile_name = r._mapping.get("profile")
        if not profile_name:
            continue
        pct_league = r._mapping.get("pct_league")
        pct_global = r._mapping.get("pct_global")
        raw_score = r._mapping.get("raw_score")
        pct_global_adjusted = r._mapping.get("pct_global_adjusted")

        if pct_league is None:
            pct_league = role_fit_metrics.get(f"{profile_name}_pct_league")
        if pct_global is None:
            pct_global = role_fit_metrics.get(f"{profile_name}_pct_global")
        if pct_global is None:
            pct_global = role_fit_metrics.get(profile_name)
        if raw_score is None:
            raw_score = role_fit_metrics.get(profile_name)
        if pct_global_adjusted is None and profile_name == ps.get("assigned_role"):
            pct_global_adjusted = ps.get("global_score_adjusted")

        role_scores.append(
            RoleScore(
                profile=profile_name,
                raw_score=raw_score,
                pct_league=pct_league,
                pct_global=pct_global,
                pct_global_adjusted=pct_global_adjusted,
            )
        )
    assigned_role_name = ps.get("assigned_role")
    role_scores.sort(
        key=lambda r: (
            -(r.pct_global if r.pct_global is not None else -1.0),
            -(r.pct_league if r.pct_league is not None else -1.0),
            0 if assigned_role_name and r.profile == assigned_role_name else 1,
        )
    )

    # Summary (garde seulement summary_* colonnes)
    summary = {k: v for k, v in metrics.items() if k.startswith("summary_")}
    tm_fields = {k: v for k, v in ps.items() if k.startswith("tm_")}
    try:
        _ensure_agency_ops_schema(session)
        hd_photo_row = session.execute(
            text(
                """
                SELECT photo_url
                FROM hd_players
                WHERE player_id = :player_id
                  AND photo_url IS NOT NULL
                  AND photo_url <> ''
                ORDER BY updated_at DESC NULLS LAST, id DESC
                LIMIT 1
                """
            ),
            {"player_id": player_id},
        ).fetchone()
        if hd_photo_row and hd_photo_row.photo_url:
            tm_fields["app_photo_url"] = hd_photo_row.photo_url
    except Exception:
        pass

    player = RankingRow(
        player_season_id=ps["id"],
        player_id=ps["player_id"],
        name=ps["name"],
        competition_name=ps["competition_name"],
        calendar=ps.get("calendar"),
        team=ps.get("team_in_selected_period"),
        position=ps.get("position"),
        assigned_role=ps.get("assigned_role"),
        minutes_played=ps.get("minutes_played"),
        matches_played=ps.get("matches_played"),
        global_score_adjusted=ps.get("global_score_adjusted"),
        age=metrics.get("age") if metrics else None,
        assigned_role_pct_league=ps.get("assigned_role_pct_league"),
        assigned_role_pct_global=ps.get("assigned_role_pct_global"),
        tm_id=ps.get("tm_id"),
        tm_profile_url=ps.get("tm_profile_url"),
    )
    transfer_history = _player_transfer_history(
        session,
        {
            "player_id": player_id,
            "linked_player_name": ps.get("name"),
            "display_name": ps.get("name"),
            "current_club": ps.get("team_in_selected_period"),
        },
    )

    return Report(
        player=player,
        metrics=metrics,
        raw_metrics=raw_metrics,
        radar_metrics=role_metrics,
        tm_fields=tm_fields,
        role_scores=role_scores,
        summary=summary,
        available_seasons=available_seasons,
        score_history=score_history,
        score_snapshots=_load_score_snapshots(session, player_id=player_id, player_season_id=int(ps["id"])),
        season_metric_history=_build_season_metric_history(session, score_history),
        transfer_history=[TransferHistoryItem(**item) for item in transfer_history],
        similarities_enabled=similarities_enabled,
        current_season_label=CURRENT_SEASON_LABEL,
        average_contexts=_build_report_average_contexts(session, ps),
    )


@app.get("/players/{player_id}/score-snapshots", response_model=List[ScoreSnapshotPoint])
def player_score_snapshots(
    player_id: int,
    player_season_id: Optional[int] = Query(None, ge=1),
    session: Session = Depends(get_session),
):
    return _load_score_snapshots(session, player_id=player_id, player_season_id=player_season_id)


@app.get("/players/{player_id}/similarities", response_model=List[SimilarityRow])
def player_similarities(
    player_id: int,
    player_season_id: Optional[int] = Query(None, ge=1),
    profile: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=10),
    offset: int = Query(0, ge=0),
    age_min: Optional[float] = Query(None, ge=0),
    age_max: Optional[float] = Query(None, ge=0),
    big5_only: bool = Query(False),
    current_season_only: bool = Query(False),
    session: Session = Depends(get_session),
):
    tm_clause = _tm_select_clause(session, alias="psb")
    big5 = _competition_aggregate_map().get("Big 5 Leagues", [])
    if player_season_id is not None:
        seed_sql = """
        SELECT ps.id AS player_season_id, ps.assigned_role, ps.calendar
        FROM player_seasons ps
        WHERE ps.player_id = :player_id AND ps.id = :player_season_id
        LIMIT 1
        """
        seed = session.execute(
            text(seed_sql),
            {"player_id": player_id, "player_season_id": player_season_id},
        ).fetchone()
    else:
        seed_sql = """
        SELECT ps.id AS player_season_id, ps.assigned_role, ps.calendar
        FROM player_seasons ps
        WHERE ps.player_id = :player_id
        ORDER BY ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
        LIMIT 1
        """
        seed = session.execute(text(seed_sql), {"player_id": player_id}).fetchone()
    if not seed:
        raise HTTPException(status_code=404, detail="Player not found")
    profile_value = profile or seed.assigned_role
    if not profile_value:
        return []

    sql = """
    SELECT
      sim.player_b_id,
      pb.name AS player_b_name,
      psb.team_in_selected_period AS team,
      cb.name AS competition_name,
      psb.calendar,
      sim.profile,
      sim.similarity,
      psb.global_score_adjusted,
      psb.assigned_role_pct_league,
      psb.assigned_role_pct_global,
      pm.age AS age""" + tm_clause + """
    FROM player_similarity sim
    JOIN players pb ON pb.id = sim.player_b_id
    LEFT JOIN player_seasons psb ON psb.id = sim.player_b_season_id
    LEFT JOIN competitions cb ON cb.id = psb.competition_id
    LEFT JOIN player_metrics pm ON pm.player_season_id = psb.id
    WHERE sim.player_a_season_id = :player_season_id
      AND sim.profile = :profile
    """
    params = {
        "player_season_id": seed.player_season_id,
        "profile": profile_value,
        "limit": limit,
        "offset": offset,
    }
    if age_min is not None:
        sql += " AND pm.age >= :age_min"
        params["age_min"] = age_min
    if age_max is not None:
        sql += " AND pm.age <= :age_max"
        params["age_max"] = age_max
    if big5_only:
        sql += " AND cb.name = ANY(:big5)"
        params["big5"] = big5
    if current_season_only:
        sql += " AND psb.calendar = ANY(:current_season_labels)"
        params["current_season_labels"] = _current_season_labels()
    sql += """
    ORDER BY sim.similarity DESC NULLS LAST
    OFFSET :offset LIMIT :limit
    """
    rows = session.execute(
        text(sql),
        params,
    ).fetchall()
    if not rows:
        fallback_sql = """
        SELECT
          sim.player_b_id,
          pb.name AS player_b_name,
          psb.team_in_selected_period AS team,
          cb.name AS competition_name,
          psb.calendar,
          sim.profile,
          sim.similarity,
          psb.global_score_adjusted,
          psb.assigned_role_pct_league,
          psb.assigned_role_pct_global,
          pm.age AS age""" + tm_clause + """
        FROM player_similarity sim
        JOIN players pb ON pb.id = sim.player_b_id
        LEFT JOIN player_seasons psb ON psb.id = sim.player_b_season_id
        LEFT JOIN competitions cb ON cb.id = psb.competition_id
        LEFT JOIN player_metrics pm ON pm.player_season_id = psb.id
        WHERE sim.player_a_id = :player_id
          AND sim.profile = :profile
        """
        fallback_params = {
            "player_id": player_id,
            "profile": profile_value,
            "limit": limit,
            "offset": offset,
        }
        if age_min is not None:
            fallback_sql += " AND pm.age >= :age_min"
            fallback_params["age_min"] = age_min
        if age_max is not None:
            fallback_sql += " AND pm.age <= :age_max"
            fallback_params["age_max"] = age_max
        if big5_only:
            fallback_sql += " AND cb.name = ANY(:big5)"
            fallback_params["big5"] = big5
        if current_season_only:
            fallback_sql += " AND psb.calendar = ANY(:current_season_labels)"
            fallback_params["current_season_labels"] = _current_season_labels()
        fallback_sql += """
        ORDER BY sim.similarity DESC NULLS LAST
        OFFSET :offset LIMIT :limit
        """
        rows = session.execute(
            text(fallback_sql),
            fallback_params,
        ).fetchall()
    items = []
    for row in rows:
        payload = _row_to_dict(row)
        payload["tm_fields"] = _extract_tm_fields(payload)
        items.append(SimilarityRow(**payload))
    return items


@app.get("/meta/competitions")
def meta_competitions(session: Session = Depends(get_session)):
    sql = """
    SELECT c.name, array_agg(DISTINCT ps.calendar) AS seasons
    FROM competitions c
    JOIN player_seasons ps ON ps.competition_id = c.id
    GROUP BY c.name
    ORDER BY c.name
    """
    rows = session.execute(text(sql)).fetchall()
    items = [{"name": r.name, "seasons": r.seasons} for r in rows]
    season_map = {item["name"]: item.get("seasons") or [] for item in items}
    aggregates = _load_competition_aggregates()
    aggregate_items = []
    for aggregate in aggregates:
        seasons = sorted(
            {
                season
                for comp in aggregate.get("competitions", [])
                for season in season_map.get(comp, [])
                if season
            }
        )
        aggregate_items.append({"name": aggregate["label"], "seasons": seasons})
    return aggregate_items + items


@app.get("/meta/positions")
def meta_positions(session: Session = Depends(get_session)):
    sql = """
    SELECT DISTINCT position, second_position
    FROM player_seasons
    WHERE position IS NOT NULL OR second_position IS NOT NULL
    """
    rows = session.execute(text(sql)).fetchall()
    codes: set[str] = set()
    for row in rows:
        for value in (row.position, row.second_position):
            if not value:
                continue
            for token in str(value).split(","):
                cleaned = token.strip()
                if cleaned:
                    codes.add(cleaned)
    return sorted(codes)


@app.get("/meta/stats-research/metrics")
def meta_stats_research_metrics(session: Session = Depends(get_session)):
    return {
        "metrics": _stats_metric_columns(session),
        "lower_is_better": sorted(_load_lower_is_better()),
    }


YOUTH_STATS_EXCLUDED_METRICS = {
    "club_strength_modifier",
    "competition_cap",
    "competition_modifier",
    "metric_score",
    "minutes_confidence",
    "minutes_regularity_modifier",
    "production_bonus",
    "scoring_model_version",
    "scoring_position_group",
    "team_strength_z",
}

YOUTH_STATS_LOWER_IS_BETTER = {
    "goals_conceded_per_90",
}


def _youth_stats_metric_keys(session: Session) -> list[str]:
    _ensure_youth_schema(session)
    rows = session.execute(
        text(
            """
            SELECT DISTINCT jsonb_object_keys(metrics) AS metric
            FROM youth_player_rankings
            WHERE metrics IS NOT NULL
              AND metrics <> '{}'::jsonb
            """
        )
    ).fetchall()
    metrics = []
    for row in rows:
        metric = str(row.metric or "").strip()
        if not metric or metric in YOUTH_STATS_EXCLUDED_METRICS:
            continue
        metrics.append(metric)
    return sorted(set(metrics))


@app.get("/youth/meta/stats-research/metrics")
def youth_meta_stats_research_metrics(session: Session = Depends(get_session)):
    return {
        "metrics": _youth_stats_metric_keys(session),
        "lower_is_better": sorted(YOUTH_STATS_LOWER_IS_BETTER),
    }


@app.get("/meta/league-translation/leagues")
def meta_league_translation_leagues():
    return _league_translation_leagues()


@app.get("/meta/league-translation")
def meta_league_translation(
    source: str = Query(..., min_length=1),
    target: str = Query(..., min_length=1),
):
    coeffs = _league_translation_coeffs(source, target)
    return {"source": source, "target": target, **coeffs}


@app.get("/meta/seasons")
def meta_seasons(session: Session = Depends(get_session)):
    sql = """
    SELECT DISTINCT label
    FROM seasons
    WHERE label IS NOT NULL AND label <> ''
    ORDER BY label
    """
    rows = session.execute(text(sql)).fetchall()
    labels = [str(r.label).strip() for r in rows if r.label]
    unique = []
    seen = set()
    for label in labels:
        if not label or label in seen:
            continue
        seen.add(label)
        unique.append(label)
    expanded = set(unique)
    filtered = []
    for label in unique:
        year_match = re.fullmatch(r"\d{4}", label)
        if year_match:
            year = int(label)
            full = f"{year}/{year + 1}"
            short = f"{year}/{str(year + 1)[-2:]}"
            if full in expanded or short in expanded:
                continue
        filtered.append(label)
    return filtered


@app.get("/meta/roles")
def meta_roles(session: Session = Depends(get_session)):
    sql = """
    SELECT DISTINCT assigned_role as role FROM player_seasons WHERE assigned_role IS NOT NULL
    UNION
    SELECT DISTINCT profile FROM role_scores WHERE profile IS NOT NULL
    """
    rows = session.execute(text(sql)).fetchall()
    return [r.role for r in rows if r.role]


@app.get("/ops/metrics")
def ops_metrics(session: Session = Depends(get_session)):
    counts_sql = """
    SELECT
      (SELECT COUNT(*) FROM players) AS players,
      (SELECT COUNT(*) FROM player_seasons) AS player_seasons,
      (SELECT COUNT(*) FROM player_metrics) AS player_metrics,
      (SELECT COUNT(*) FROM role_scores) AS role_scores,
      (SELECT COUNT(*) FROM player_similarity) AS player_similarity
    """
    counts = session.execute(text(counts_sql)).fetchone()
    last_run = session.execute(
        text(
            """
            SELECT run_id, status, started_at, ended_at, rows_processed
            FROM pipeline_runs
            ORDER BY started_at DESC NULLS LAST
            LIMIT 1
            """
        )
    ).fetchone()
    return {
        "status": "ok",
        "counts": _row_to_dict(counts),
        "last_pipeline_run": _row_to_dict(last_run) if last_run else None,
    }


@app.get("/meta/positions")
def meta_positions(
    competition: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    sql = """
    SELECT DISTINCT ps.position
    FROM player_seasons ps
    JOIN competitions c ON c.id = ps.competition_id
    WHERE ps.position IS NOT NULL AND ps.position <> ''
    """
    params = {}
    sql, params = _apply_competition_filter(sql, params, competition)
    sql, params = _apply_season_filter(sql, params, season)
    sql += " ORDER BY ps.position"
    rows = session.execute(text(sql), params).fetchall()
    return [r.position for r in rows if r.position]


@app.get("/meta/teams")
def meta_teams(
    competition: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    sql = """
    SELECT DISTINCT ps.team_in_selected_period AS team
    FROM player_seasons ps
    JOIN competitions c ON c.id = ps.competition_id
    WHERE ps.team_in_selected_period IS NOT NULL AND ps.team_in_selected_period <> ''
    """
    params = {}
    sql, params = _apply_competition_filter(sql, params, competition)
    sql, params = _apply_season_filter(sql, params, season)
    sql += " ORDER BY ps.team_in_selected_period"
    rows = session.execute(text(sql), params).fetchall()
    return [r.team for r in rows if r.team]


@app.get("/meta/clubs")
def meta_clubs(session: Session = Depends(get_session)):
    sql = """
    SELECT cl.id, cl.name, c.name AS competition_name
    FROM clubs cl
    LEFT JOIN competitions c ON c.id = cl.competition_id
    ORDER BY cl.name
    """
    rows = session.execute(text(sql)).fetchall()
    return [
        {
            "id": r.id,
            "name": r.name,
            "competition_name": r.competition_name,
        }
        for r in rows
    ]


@app.get("/meta/players")
def meta_players(
    competition: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    limit: int = Query(300, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    sql = """
    SELECT DISTINCT p.id AS player_id, p.name
    FROM players p
    JOIN player_seasons ps ON ps.player_id = p.id
    JOIN competitions c ON c.id = ps.competition_id
    WHERE p.name IS NOT NULL AND p.name <> ''
    """
    params = {"limit": limit}
    sql, params = _apply_competition_filter(sql, params, competition)
    sql, params = _apply_season_filter(sql, params, season)
    if team:
        sql += " AND ps.team_in_selected_period = :team"
        params["team"] = team
    sql += " ORDER BY p.name LIMIT :limit"
    rows = session.execute(text(sql), params).fetchall()
    return [{"id": r.player_id, "name": r.name} for r in rows]


@app.get("/players")
def search_players(
    q: str = Query(..., min_length=1),
    season: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    session: Session = Depends(get_session),
):
    cleaned = q.strip()
    if len(cleaned) < 2:
        return []

    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    params = {"limit": limit}
    token_clauses = []
    for idx, token in enumerate(tokens):
        if len(token) < 2:
            continue
        prefix_key = f"t{idx}_prefix"
        space_key = f"t{idx}_space"
        dot_key = f"t{idx}_dot"
        dash_key = f"t{idx}_dash"
        team_key = f"t{idx}_team"
        comp_key = f"t{idx}_comp"
        params[prefix_key] = f"{token}%"
        params[space_key] = f"% {token}%"
        params[dot_key] = f"%.{token}%"
        params[dash_key] = f"%-{token}%"
        params[team_key] = f"%{token}%"
        params[comp_key] = f"%{token}%"
        token_clauses.append(
            "("
            f"p.name ILIKE :{prefix_key} "
            f"OR p.name ILIKE :{space_key} "
            f"OR p.name ILIKE :{dot_key} "
            f"OR p.name ILIKE :{dash_key} "
            f"OR ps.team_in_selected_period ILIKE :{team_key} "
            f"OR c.name ILIKE :{comp_key}"
            ")"
        )
    if not token_clauses:
        return []
    where_clause = " AND ".join(token_clauses)

    fetch_limit = min(2500, max(limit * 60, 500))
    params["fetch_limit"] = fetch_limit
    sql = f"""
    SELECT DISTINCT ON (p.id, ps.calendar, c.name, ps.team_in_selected_period)
      p.id,
      ps.id AS player_season_id,
      p.name,
      c.name AS competition_name,
      ps.calendar,
      ps.team_in_selected_period AS team,
      ps.minutes_played
    FROM players p
    JOIN player_seasons ps ON ps.player_id = p.id
    JOIN competitions c ON c.id = ps.competition_id
    WHERE {where_clause}
    """
    if season:
        season_values = _season_filter_values(season)
        if len(season_values) <= 1:
            sql += " AND ps.calendar = :season"
            params["season"] = season_values[0] if season_values else season
        else:
            sql += " AND ps.calendar = ANY(:season_values)"
            params["season_values"] = season_values
    sql += """
    ORDER BY p.id, ps.calendar, c.name, ps.team_in_selected_period, ps.minutes_played DESC NULLS LAST
    LIMIT :fetch_limit
    """
    rows = session.execute(text(sql), params).fetchall()
    items = [_row_to_dict(row) for row in rows]
    for item in items:
        club_power = _club_power_for_team(item.get("team"))
        item["club_power_rating"] = club_power["rating"]
        item["club_power_rank"] = club_power["rank"]
        item["club_power_matched_team"] = club_power["matched_team"]
        item["_search_relevance"] = _player_search_relevance(item, cleaned)
    items.sort(
        key=lambda item: (
            float(item.get("club_power_rating") or -1),
            int(item.get("_search_relevance") or 0),
            _season_sort_key_desc(item.get("calendar")),
            float(item.get("minutes_played") or -1),
            str(item.get("name") or ""),
            str(item.get("competition_name") or ""),
            str(item.get("team") or ""),
        ),
        reverse=True,
    )
    trimmed = items[:limit]
    for item in trimmed:
        item.pop("minutes_played", None)
        item.pop("_search_relevance", None)
    return trimmed


def _apply_youth_filters(
    sql: str,
    params: dict[str, Any],
    *,
    season: Optional[int],
    championship: Optional[str],
    age_category: Optional[str],
    birth_year: Optional[int],
    position_group: Optional[str],
    position: Optional[str],
    club: Optional[str],
    min_minutes: Optional[float],
) -> tuple[str, dict[str, Any]]:
    if season:
        sql += " AND season = :season"
        params["season"] = season
    if championship and championship.lower() != "all":
        sql += " AND championship = :championship"
        params["championship"] = championship
    if age_category and age_category.lower() != "all":
        sql += " AND age_category = :age_category"
        params["age_category"] = age_category
    if birth_year:
        sql += " AND birth_year = :birth_year"
        params["birth_year"] = birth_year
    if position_group and position_group.lower() != "all":
        sql += " AND position_group = :position_group"
        params["position_group"] = position_group
    if position and position.lower() != "all":
        sql += " AND position ILIKE :position_like"
        params["position_like"] = f"%{position}%"
    if club:
        sql += " AND club_name ILIKE :club"
        params["club"] = f"%{club.strip()}%"
    if min_minutes is not None:
        sql += " AND COALESCE(minutes_played, 0) >= :min_minutes"
        params["min_minutes"] = min_minutes
    return sql, params


def _youth_row_payload(row: Any) -> dict[str, Any]:
    payload = _row_to_dict(row)
    payload["metrics"] = payload.get("metrics") or {}
    payload["metric_percentiles"] = payload.get("metric_percentiles") or {}
    raw_payload = payload.get("raw_payload") or {}
    payload["birth_date"] = (
        payload.get("birth_date")
        or raw_payload.get("birthDate")
        or raw_payload.get("birthdate")
        or raw_payload.get("dateOfBirth")
        or raw_payload.get("birthday")
    )
    payload["nationality_code"] = (
        raw_payload.get("nationality_countryCode")
        or raw_payload.get("nationality2_countryCode")
        or payload.get("country_code")
    )
    payload["nationality_label"] = (
        raw_payload.get("nationality_countryLabel")
        or raw_payload.get("nationality2_countryLabel")
        or payload.get("country_code")
    )
    return payload


def _average_youth_contexts(session: Session, player: dict[str, Any]) -> dict[str, Any]:
    contexts = {
        "global": "season = :season AND position_group = :position_group",
        "league": "season = :season AND position_group = :position_group AND championship = :championship",
    }
    output: dict[str, Any] = {}
    metric_keys = set((player.get("metrics") or {}).keys())
    for context, where_sql in contexts.items():
        rows = session.execute(
            text(
                f"""
                SELECT metrics, metric_percentiles, minutes_played
                FROM youth_player_rankings
                WHERE {where_sql}
                """
            ),
            {
                "season": player.get("season"),
                "position_group": player.get("position_group"),
                "championship": player.get("championship"),
            },
        ).fetchall()
        sums: dict[str, float] = {}
        pct_sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        pct_counts: dict[str, int] = {}
        min_minutes_values = []
        for row in rows:
            row_dict = _row_to_dict(row)
            row_metrics = row_dict.get("metrics") or {}
            row_percentiles = row_dict.get("metric_percentiles") or {}
            minutes = row_dict.get("minutes_played")
            if minutes is not None:
                min_minutes_values.append(float(minutes))
            metric_keys.update(row_metrics.keys())
            for key, value in row_metrics.items():
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                sums[key] = sums.get(key, 0.0) + numeric
                counts[key] = counts.get(key, 0) + 1
                pct_value = (row_percentiles.get(key) or {}).get("global_position" if context == "global" else "championship")
                if pct_value is not None:
                    pct_sums[key] = pct_sums.get(key, 0.0) + float(pct_value)
                    pct_counts[key] = pct_counts.get(key, 0) + 1
        metrics = {
            key: {
                "raw": round(sums[key] / counts[key], 4) if counts.get(key) else None,
                "percentile": round(pct_sums[key] / pct_counts[key], 4) if pct_counts.get(key) else None,
            }
            for key in sorted(metric_keys)
        }
        output[context] = {
            player.get("position_group") or "Unknown": {
                "sample_size": len(rows),
                "min_minutes": min(min_minutes_values) if min_minutes_values else None,
                "metrics": metrics,
            }
        }
    return output


@app.get("/youth/meta")
def youth_meta(session: Session = Depends(get_session)):
    _ensure_youth_schema(session)
    row = session.execute(
        text(
            """
            SELECT
              array_agg(DISTINCT season ORDER BY season DESC) FILTER (WHERE season IS NOT NULL) AS seasons,
              array_agg(DISTINCT championship ORDER BY championship) FILTER (WHERE championship IS NOT NULL AND championship <> '') AS championships,
              array_agg(DISTINCT age_category ORDER BY age_category) FILTER (WHERE age_category IS NOT NULL AND age_category <> '') AS age_categories,
              array_agg(DISTINCT birth_year ORDER BY birth_year DESC) FILTER (WHERE birth_year IS NOT NULL) AS birth_years,
              array_agg(DISTINCT position_group ORDER BY position_group) FILTER (WHERE position_group IS NOT NULL AND position_group <> '') AS position_groups,
              array_agg(DISTINCT primary_position ORDER BY primary_position) FILTER (WHERE primary_position IS NOT NULL AND primary_position <> '') AS positions
            FROM youth_player_rankings
            """
        )
    ).fetchone()
    counts = session.execute(
        text(
            """
            SELECT
              COUNT(*) AS rows,
              COUNT(DISTINCT provider_player_id) AS players,
              COUNT(DISTINCT club_name) AS clubs,
              COUNT(DISTINCT championship) AS championships
            FROM youth_player_rankings
            """
        )
    ).fetchone()
    return {
        **_row_to_dict(row),
        "counts": _row_to_dict(counts),
    }


@app.get("/youth/ranking/page")
def youth_ranking_page(
    season: Optional[int] = Query(2027),
    championship: Optional[str] = Query(None),
    age_category: Optional[str] = Query(None),
    birth_year: Optional[int] = Query(None),
    position_group: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    club: Optional[str] = Query(None),
    min_minutes: Optional[float] = Query(90, ge=0),
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=200),
    session: Session = Depends(get_session),
):
    _ensure_youth_schema(session)
    base_sql = """
    FROM youth_player_rankings
    WHERE score IS NOT NULL
    """
    params: dict[str, Any] = {}
    base_sql, params = _apply_youth_filters(
        base_sql,
        params,
        season=season,
        championship=championship,
        age_category=age_category,
        birth_year=birth_year,
        position_group=position_group,
        position=position,
        club=club,
        min_minutes=min_minutes,
    )
    total = session.execute(text("SELECT COUNT(*) " + base_sql), params).scalar() or 0
    rows = session.execute(
        text(
            """
            SELECT
              id, provider, provider_player_id, provider_player_url, season, calendar, is_current_season,
              display_name, birth_year, birth_date, age, age_category, championship,
              club_name, team_name, team_level, position, primary_position,
              position_group, strong_foot, height_cm, weight_kg, games_count,
              minutes_played, rating, score, score_raw,
              score_percentile_global, score_percentile_age_category,
              score_percentile_birth_year, score_percentile_championship,
              COALESCE(raw_payload->>'nationality_countryCode', raw_payload->>'nationality2_countryCode', country_code) AS nationality_code,
              COALESCE(raw_payload->>'nationality_countryLabel', raw_payload->>'nationality2_countryLabel', country_code) AS nationality_label
            """
            + base_sql
            + """
            ORDER BY score DESC NULLS LAST, minutes_played DESC NULLS LAST, rating DESC NULLS LAST
            OFFSET :offset LIMIT :limit
            """
        ),
        {**params, "offset": offset, "limit": limit},
    ).fetchall()
    return {
        "items": [_row_to_dict(row) for row in rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/youth/players")
def youth_search_players(
    q: str = Query(..., min_length=1),
    season: Optional[int] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    session: Session = Depends(get_session),
):
    _ensure_youth_schema(session)
    cleaned = q.strip()
    if len(cleaned) < 2:
        return []
    params: dict[str, Any] = {"q": f"%{cleaned}%", "limit": limit}
    sql = """
    SELECT
      id, provider_player_id, display_name, club_name, championship, calendar,
      age_category, birth_year, birth_date, position, position_group, minutes_played, score,
      COALESCE(raw_payload->>'nationality_countryCode', raw_payload->>'nationality2_countryCode', country_code) AS nationality_code,
      COALESCE(raw_payload->>'nationality_countryLabel', raw_payload->>'nationality2_countryLabel', country_code) AS nationality_label
    FROM youth_player_rankings
    WHERE (display_name ILIKE :q OR club_name ILIKE :q OR championship ILIKE :q)
    """
    if season:
        sql += " AND season = :season"
        params["season"] = season
    sql += """
    ORDER BY score DESC NULLS LAST, minutes_played DESC NULLS LAST, display_name
    LIMIT :limit
    """
    rows = session.execute(text(sql), params).fetchall()
    return [_row_to_dict(row) for row in rows]


@app.get("/youth/players/{youth_id}/report")
def youth_player_report(youth_id: int, session: Session = Depends(get_session)):
    _ensure_youth_schema(session)
    row = session.execute(
        text("SELECT * FROM youth_player_rankings WHERE id = :id"),
        {"id": youth_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Youth player not found")
    player = _youth_row_payload(row)
    season_rows = session.execute(
        text(
            """
            SELECT *
            FROM youth_player_rankings
            WHERE provider = :provider
              AND provider_player_id = :provider_player_id
            ORDER BY season DESC, minutes_played DESC NULLS LAST, score DESC NULLS LAST, id DESC
            """
        ),
        {
            "provider": player.get("provider") or "eyeball",
            "provider_player_id": player.get("provider_player_id"),
        },
    ).fetchall()
    season_payloads = [_youth_row_payload(item) for item in season_rows]
    score_rows_by_season: dict[int, dict[str, Any]] = {}
    for item in season_payloads:
        season_value = item.get("season")
        if season_value is None:
            continue
        current = score_rows_by_season.get(int(season_value))
        if not current:
            score_rows_by_season[int(season_value)] = item
            continue
        current_minutes = current.get("minutes_played") or 0
        item_minutes = item.get("minutes_played") or 0
        current_score = current.get("score") or 0
        item_score = item.get("score") or 0
        if (item_minutes, item_score) > (current_minutes, current_score):
            score_rows_by_season[int(season_value)] = item
    score_history = [
        score_rows_by_season[season_value]
        for season_value in sorted(score_rows_by_season)
    ]
    available_seasons = [
        score_rows_by_season[season_value]
        for season_value in sorted(score_rows_by_season, reverse=True)
    ]
    similar_rows = session.execute(
        text(
            """
            SELECT
              id, display_name, club_name, championship, calendar, age_category, birth_year,
              position, position_group, minutes_played, rating, score,
              score_percentile_age_category, score_percentile_championship,
              COALESCE(raw_payload->>'nationality_countryCode', raw_payload->>'nationality2_countryCode', country_code) AS nationality_code,
              COALESCE(raw_payload->>'nationality_countryLabel', raw_payload->>'nationality2_countryLabel', country_code) AS nationality_label
            FROM youth_player_rankings
            WHERE id <> :id
              AND season = :season
              AND position_group = :position_group
              AND (
                championship = :championship
                OR age_category = :age_category
                OR birth_year = :birth_year
              )
            ORDER BY
              CASE WHEN championship = :championship THEN 0 ELSE 1 END,
              ABS(COALESCE(score, 0) - COALESCE(:score, 0)),
              score DESC NULLS LAST
            LIMIT 12
            """
        ),
        {
            "id": youth_id,
            "season": player.get("season"),
            "position_group": player.get("position_group"),
            "championship": player.get("championship"),
            "age_category": player.get("age_category"),
            "birth_year": player.get("birth_year"),
            "score": player.get("score"),
        },
    ).fetchall()
    metrics = player.get("metrics") or {}
    metric_percentiles = player.get("metric_percentiles") or {}
    key_metrics = sorted(
        [
            {
                "key": key,
                "value": metrics.get(key),
                **(metric_percentiles.get(key) or {}),
            }
            for key in metrics
            if metrics.get(key) is not None and metric_percentiles.get(key)
        ],
        key=lambda item: item.get("global_position") or 0,
        reverse=True,
    )[:12]
    return {
        "player": player,
        "metrics": metrics,
        "metric_percentiles": metric_percentiles,
        "key_metrics": key_metrics,
        "similar_players": [_row_to_dict(row) for row in similar_rows],
        "available_seasons": available_seasons,
        "score_history": score_history,
        "season_metric_history": score_history,
        "average_contexts": _average_youth_contexts(session, player),
        "summary": {
            "score": player.get("score"),
            "global": player.get("score_percentile_global"),
            "age_category": player.get("score_percentile_age_category"),
            "birth_year": player.get("score_percentile_birth_year"),
            "championship": player.get("score_percentile_championship"),
        },
    }


def _youth_prospect_identity(session: Session, youth_id: int) -> dict[str, Any]:
    _ensure_youth_schema(session)
    row = session.execute(
        text(
            """
            SELECT id, provider, season, source_row_hash
            FROM youth_player_rankings
            WHERE id = :youth_id
            """
        ),
        {"youth_id": youth_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Youth player not found")
    return _row_to_dict(row)


@app.get("/youth/prospects/ids")
def youth_prospect_ids(season: Optional[int] = Query(None), session: Session = Depends(get_session)):
    _ensure_youth_schema(session)
    sql = """
    SELECT y.id
    FROM youth_prospects yp
    JOIN youth_player_rankings y
      ON y.provider = yp.provider
     AND y.season = yp.season
     AND y.source_row_hash = yp.source_row_hash
    WHERE 1=1
    """
    params: dict[str, Any] = {}
    if season:
        sql += " AND y.season = :season"
        params["season"] = season
    rows = session.execute(text(sql), params).fetchall()
    return {"youth_ids": [int(row.id) for row in rows]}


@app.post("/youth/prospects")
def add_youth_prospect(payload: YouthProspectToggle, session: Session = Depends(get_session)):
    identity = _youth_prospect_identity(session, payload.youth_id)
    result = session.execute(
        text(
            """
            INSERT INTO youth_prospects (provider, season, source_row_hash)
            VALUES (:provider, :season, :source_row_hash)
            ON CONFLICT (provider, season, source_row_hash) DO NOTHING
            RETURNING id
            """
        ),
        {
            "provider": identity["provider"],
            "season": identity["season"],
            "source_row_hash": identity["source_row_hash"],
        },
    ).fetchone()
    session.commit()
    return {"youth_id": payload.youth_id, "is_prospect": True, "added": bool(result)}


@app.get("/youth/prospects/page")
def youth_prospects_page(
    season: Optional[int] = Query(2027),
    championship: Optional[str] = Query(None),
    age_category: Optional[str] = Query(None),
    birth_year: Optional[int] = Query(None),
    position_group: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    club: Optional[str] = Query(None),
    min_minutes: Optional[float] = Query(0, ge=0),
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=200),
    session: Session = Depends(get_session),
):
    _ensure_youth_schema(session)
    base_sql = """
    FROM (
      SELECT y.*, yp.created_at AS prospect_created_at
      FROM youth_prospects yp
      JOIN youth_player_rankings y
        ON y.provider = yp.provider
       AND y.season = yp.season
       AND y.source_row_hash = yp.source_row_hash
    ) youth_player_rankings
    WHERE score IS NOT NULL
    """
    params: dict[str, Any] = {}
    base_sql, params = _apply_youth_filters(
        base_sql,
        params,
        season=season,
        championship=championship,
        age_category=age_category,
        birth_year=birth_year,
        position_group=position_group,
        position=position,
        club=club,
        min_minutes=min_minutes,
    )
    total = session.execute(text("SELECT COUNT(*) " + base_sql), params).scalar() or 0
    rows = session.execute(
        text(
            """
            SELECT
              id, provider, provider_player_id, provider_player_url, season, calendar, is_current_season,
              display_name, birth_year, birth_date, age, age_category, championship,
              club_name, team_name, team_level, position, primary_position,
              position_group, strong_foot, height_cm, weight_kg, games_count,
              minutes_played, rating, score, score_raw,
              score_percentile_global, score_percentile_age_category,
              score_percentile_birth_year, score_percentile_championship,
              prospect_created_at,
              COALESCE(raw_payload->>'nationality_countryCode', raw_payload->>'nationality2_countryCode', country_code) AS nationality_code,
              COALESCE(raw_payload->>'nationality_countryLabel', raw_payload->>'nationality2_countryLabel', country_code) AS nationality_label
            """
            + base_sql
            + """
            ORDER BY prospect_created_at DESC NULLS LAST, score DESC NULLS LAST, minutes_played DESC NULLS LAST
            OFFSET :offset LIMIT :limit
            """
        ),
        {**params, "offset": offset, "limit": limit},
    ).fetchall()
    return {
        "items": [_row_to_dict(row) for row in rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/youth/prospects/{youth_id}")
def youth_prospect_status(youth_id: int, session: Session = Depends(get_session)):
    identity = _youth_prospect_identity(session, youth_id)
    exists = session.execute(
        text(
            """
            SELECT 1
            FROM youth_prospects
            WHERE provider = :provider
              AND season = :season
              AND source_row_hash = :source_row_hash
            """
        ),
        {
            "provider": identity["provider"],
            "season": identity["season"],
            "source_row_hash": identity["source_row_hash"],
        },
    ).fetchone()
    return {"youth_id": youth_id, "is_prospect": bool(exists)}


@app.delete("/youth/prospects/{youth_id}")
def remove_youth_prospect(youth_id: int, session: Session = Depends(get_session)):
    identity = _youth_prospect_identity(session, youth_id)
    result = session.execute(
        text(
            """
            DELETE FROM youth_prospects
            WHERE provider = :provider
              AND season = :season
              AND source_row_hash = :source_row_hash
            """
        ),
        {
            "provider": identity["provider"],
            "season": identity["season"],
            "source_row_hash": identity["source_row_hash"],
        },
    )
    session.commit()
    return {"youth_id": youth_id, "removed": result.rowcount > 0}


@app.get("/prospects/ids")
def prospect_ids(session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    rows = session.execute(text("SELECT player_id FROM prospects")).fetchall()
    return {"player_ids": [r.player_id for r in rows]}


@app.post("/prospects")
def add_prospect(payload: ProspectToggle, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    if payload.player_season_id is not None:
        season_row = session.execute(
            text("SELECT player_id FROM player_seasons WHERE id = :player_season_id"),
            {"player_season_id": payload.player_season_id},
        ).fetchone()
        if not season_row:
            raise HTTPException(status_code=404, detail="Player season not found")
        if int(season_row.player_id) != int(payload.player_id):
            raise HTTPException(status_code=400, detail="Player season does not belong to this player")
    result = session.execute(
        text(
            """
            INSERT INTO prospects (player_id, player_season_id)
            VALUES (:player_id, :player_season_id)
            ON CONFLICT (player_id) DO UPDATE
            SET player_season_id = COALESCE(EXCLUDED.player_season_id, prospects.player_season_id)
            RETURNING (xmax = 0) AS inserted
            """
        ),
        {"player_id": payload.player_id, "player_season_id": payload.player_season_id},
    ).fetchone()
    session.commit()
    return {
        "player_id": payload.player_id,
        "player_season_id": payload.player_season_id,
        "added": bool(result.inserted) if result else False,
    }


@app.get("/prospects/page", response_model=RankingPage)
def prospects_page(
    role: Optional[str] = Query(None),
    competition: Optional[str] = Query(None),
    min_minutes: Optional[float] = Query(0),
    season: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    age_min: Optional[float] = Query(None, ge=0),
    age_max: Optional[float] = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=200),
    session: Session = Depends(get_session),
):
    _ensure_prospect_schema(session)
    tm_clause = _tm_select_clause(session)
    base_sql = """
    FROM prospects pr
    JOIN players p ON p.id = pr.player_id
    JOIN player_seasons ps ON ps.player_id = p.id
      AND (pr.player_season_id IS NULL OR ps.id = pr.player_season_id)
    JOIN competitions c ON c.id = ps.competition_id
    LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
    """ + _ranking_pct_fallback_join() + """
    WHERE 1=1
    """
    params = {}
    if min_minutes is not None:
        base_sql += " AND ps.minutes_played >= :min_minutes"
        params["min_minutes"] = min_minutes
    base_sql, params = _apply_ranking_filters(base_sql, params, role, competition, season, position, team, age_min, age_max)

    count_sql = "SELECT COUNT(DISTINCT p.id) " + base_sql
    total = session.execute(text(count_sql), params).scalar() or 0

    data_sql = """
    SELECT DISTINCT ON (p.id)
      ps.id AS player_season_id,
      p.id AS player_id,
      p.name,
      p.tm_id,
      p.tm_profile_url,
      c.name AS competition_name,
      ps.calendar,
      ps.team_in_selected_period AS team,
      ps.position,
      ps.assigned_role,
      ps.minutes_played,
      ps.global_score_adjusted,
      COALESCE(ps.assigned_role_pct_league, pct_league_fallback.fallback_assigned_role_pct_league) AS assigned_role_pct_league,
      COALESCE(ps.assigned_role_pct_global, pct_global_fallback.fallback_assigned_role_pct_global) AS assigned_role_pct_global,
      pm.age AS age""" + tm_clause + """
    """ + base_sql + """
    ORDER BY p.id, ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
    OFFSET :offset LIMIT :limit
    """
    params = {**params, "offset": offset, "limit": limit}
    rows = session.execute(text(data_sql), params).fetchall()
    items = []
    for row in rows:
        payload = _row_to_dict(row)
        payload["tm_fields"] = _extract_tm_fields(payload)
        items.append(RankingRow(**payload))
    return RankingPage(items=items, total=total, offset=offset, limit=limit)


@app.get("/prospects/{player_id}")
def prospect_status(player_id: int, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    exists = session.execute(
        text("SELECT 1 FROM prospects WHERE player_id = :player_id"),
        {"player_id": player_id},
    ).fetchone()
    return {"player_id": player_id, "is_prospect": bool(exists)}


@app.delete("/prospects/{player_id}")
def remove_prospect(player_id: int, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    result = session.execute(
        text("DELETE FROM prospects WHERE player_id = :player_id"),
        {"player_id": player_id},
    )
    session.commit()
    return {"player_id": player_id, "removed": result.rowcount > 0}


@app.get("/prospect/club-needs")
def prospect_club_needs(session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    needs_sql = """
    SELECT
      cn.id,
      cn.club_id,
      cn.need_label,
      cn.contact_name,
      cn.contact_phone,
      cn.assigned_user,
      cn.priority_stage,
      cn.sort_order,
      cl.name AS club_name,
      c.name AS competition_name
    FROM club_needs cn
    LEFT JOIN clubs cl ON cl.id = cn.club_id
    LEFT JOIN competitions c ON c.id = cl.competition_id
    ORDER BY cn.priority_stage, cn.sort_order, cn.id
    """
    needs_rows = session.execute(text(needs_sql)).fetchall()
    needs = []
    for row in needs_rows:
        needs.append({**_row_to_dict(row), "players": []})

    players_sql = """
    SELECT
      cnp.club_need_id,
      cnp.sort_order,
      p.id AS player_id,
      p.name,
      ps.team_in_selected_period AS team,
      comp.name AS competition_name,
      ps.calendar
    FROM club_need_players cnp
    JOIN players p ON p.id = cnp.player_id
    LEFT JOIN LATERAL (
      SELECT ps.*
      FROM player_seasons ps
      WHERE ps.player_id = p.id
      ORDER BY ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
      LIMIT 1
    ) ps ON true
    LEFT JOIN competitions comp ON comp.id = ps.competition_id
    ORDER BY cnp.club_need_id, cnp.sort_order
    """
    player_rows = session.execute(text(players_sql)).fetchall()
    needs_map = {need["id"]: need for need in needs}
    for row in player_rows:
        payload = _row_to_dict(row)
        need = needs_map.get(payload["club_need_id"])
        if need is not None:
            need["players"].append(payload)
    return {"needs": needs}


@app.post("/prospect/club-needs")
def create_club_need(payload: ClubNeedCreate, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    stage = payload.priority_stage or "Priority 1"
    max_sort = session.execute(
        text("SELECT COALESCE(MAX(sort_order), -1) FROM club_needs WHERE priority_stage = :stage"),
        {"stage": stage},
    ).scalar()
    sort_order = (max_sort or -1) + 1
    assigned = payload.assigned_user or "admin"
    row = session.execute(
        text(
            """
            INSERT INTO club_needs (
              club_id,
              need_label,
              contact_name,
              contact_phone,
              assigned_user,
              priority_stage,
              sort_order,
              created_at,
              updated_at
            ) VALUES (
              :club_id,
              :need_label,
              :contact_name,
              :contact_phone,
              :assigned_user,
              :priority_stage,
              :sort_order,
              NOW(),
              NOW()
            )
            RETURNING id
            """
        ),
        {
            "club_id": payload.club_id,
            "need_label": payload.need_label,
            "contact_name": payload.contact_name,
            "contact_phone": payload.contact_phone,
            "assigned_user": assigned,
            "priority_stage": stage,
            "sort_order": sort_order,
        },
    ).fetchone()
    session.commit()
    return {"id": row.id, "priority_stage": stage, "sort_order": sort_order}


@app.patch("/prospect/club-needs/reorder")
def reorder_club_needs(payload: ClubNeedReorder, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    for need in payload.needs:
        session.execute(
            text(
                """
                UPDATE club_needs
                SET priority_stage = :priority_stage,
                    sort_order = :sort_order,
                    updated_at = NOW()
                WHERE id = :id
                """
            ),
            {
                "id": need.id,
                "priority_stage": need.priority_stage,
                "sort_order": need.sort_order,
            },
        )
    session.commit()
    return {"updated": len(payload.needs)}


@app.post("/prospect/club-needs/{need_id}/players")
def add_need_player(need_id: int, payload: ClubNeedPlayerAdd, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    max_sort = session.execute(
        text("SELECT COALESCE(MAX(sort_order), -1) FROM club_need_players WHERE club_need_id = :need_id"),
        {"need_id": need_id},
    ).scalar()
    sort_order = (max_sort or -1) + 1
    result = session.execute(
        text(
            """
            INSERT INTO club_need_players (club_need_id, player_id, sort_order, created_at)
            VALUES (:need_id, :player_id, :sort_order, NOW())
            ON CONFLICT (club_need_id, player_id) DO NOTHING
            """
        ),
        {"need_id": need_id, "player_id": payload.player_id, "sort_order": sort_order},
    )
    session.commit()
    return {"added": result.rowcount > 0}


@app.delete("/prospect/club-needs/{need_id}/players/{player_id}")
def remove_need_player(need_id: int, player_id: int, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    result = session.execute(
        text(
            """
            DELETE FROM club_need_players
            WHERE club_need_id = :need_id AND player_id = :player_id
            """
        ),
        {"need_id": need_id, "player_id": player_id},
    )
    session.commit()
    return {"removed": result.rowcount > 0}


@app.patch("/prospect/club-needs/{need_id}/players/reorder")
def reorder_need_players(need_id: int, payload: ClubNeedPlayerOrder, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    for order, player_id in enumerate(payload.player_ids):
        session.execute(
            text(
                """
                UPDATE club_need_players
                SET sort_order = :sort_order
                WHERE club_need_id = :need_id AND player_id = :player_id
                """
            ),
            {"need_id": need_id, "player_id": player_id, "sort_order": order},
        )
    session.commit()
    return {"updated": len(payload.player_ids)}


class VizPercentileRequest(BaseModel):
    player_id: int
    player_season_id: Optional[int] = None
    metrics: list[str]
    context: str = "League"
    positions: list[str] = []
    min_minutes: float = 270


@app.post("/viz/percentiles")
def viz_percentiles(payload: VizPercentileRequest, session: Session = Depends(get_session)):
    available = set(_stats_metric_columns(session))
    metrics = [metric for metric in payload.metrics if metric in available]
    if not metrics:
        raise HTTPException(status_code=400, detail="No valid metrics provided")

    if payload.player_season_id:
        player_sql = """
        SELECT ps.id AS player_season_id,
          p.id AS player_id,
          p.name,
          ps.team_in_selected_period AS team,
          c.name AS competition_name,
          ps.position,
          ps.second_position,
          ps.minutes_played,
          ps.calendar
        FROM player_seasons ps
        JOIN players p ON p.id = ps.player_id
        JOIN competitions c ON c.id = ps.competition_id
        WHERE p.id = :player_id
          AND ps.id = :player_season_id
        LIMIT 1
        """
        player_row = session.execute(
            text(player_sql),
            {"player_id": payload.player_id, "player_season_id": payload.player_season_id},
        ).fetchone()
    else:
        player_sql = """
        SELECT ps.id AS player_season_id,
          p.id AS player_id,
          p.name,
          ps.team_in_selected_period AS team,
          c.name AS competition_name,
          ps.position,
          ps.second_position,
          ps.minutes_played,
          ps.calendar
        FROM player_seasons ps
        JOIN players p ON p.id = ps.player_id
        JOIN competitions c ON c.id = ps.competition_id
        WHERE p.id = :player_id
        ORDER BY ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
        LIMIT 1
        """
        player_row = session.execute(text(player_sql), {"player_id": payload.player_id}).fetchone()
    if not player_row:
        raise HTTPException(status_code=404, detail="Player not found")
    player = _row_to_dict(player_row)

    metric_cols = ", ".join([f'pm."{metric}" AS "{metric}"' for metric in metrics])
    metrics_sql = f"""
    SELECT {metric_cols}
    FROM player_metrics pm
    WHERE pm.player_season_id = :player_season_id
    """
    metrics_row = session.execute(text(metrics_sql), {"player_season_id": player["player_season_id"]}).fetchone()
    player_metrics = _row_to_dict(metrics_row) if metrics_row else {}

    cohort_sql = f"""
    SELECT {metric_cols}
    FROM player_seasons ps
    JOIN competitions c ON c.id = ps.competition_id
    JOIN player_metrics pm ON pm.player_season_id = ps.id
    WHERE ps.minutes_played IS NOT NULL
    """
    params = {"min_minutes": payload.min_minutes}
    if payload.min_minutes is not None:
        cohort_sql += " AND ps.minutes_played >= :min_minutes"
    context = payload.context.strip().lower()
    if context == "league":
        cohort_sql += " AND c.name = :competition"
        params["competition"] = player.get("competition_name")
    if player.get("calendar"):
        cohort_sql += " AND ps.calendar = :season"
        params["season"] = player.get("calendar")

    positions = [p.strip() for p in payload.positions or [] if p.strip()]
    if positions:
        params["positions"] = positions
        params["pos_patterns"] = [f"%{p}%" for p in positions]
        cohort_sql += """
        AND (
          ps.position = ANY(:positions)
          OR ps.second_position = ANY(:positions)
          OR ps.position ILIKE ANY(:pos_patterns)
          OR ps.second_position ILIKE ANY(:pos_patterns)
        )
        """

    rows = session.execute(text(cohort_sql), params).fetchall()
    values_by_metric: dict[str, list[float]] = {metric: [] for metric in metrics}
    for row in rows:
        data = _row_to_dict(row)
        for metric in metrics:
            value = data.get(metric)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            values_by_metric[metric].append(numeric)

    percentiles: dict[str, float] = {}
    for metric in metrics:
        value = player_metrics.get(metric)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        cohort_values = list(values_by_metric.get(metric, []))
        if numeric not in cohort_values:
            cohort_values.append(numeric)
        percentiles[metric] = _percentile_rank(cohort_values, numeric)

    return {
        "player": player,
        "values": player_metrics,
        "percentiles": percentiles,
        "cohort_count": len(rows),
    }


@app.get("/stats-research")
def stats_research(
    metric_x: str = Query(..., min_length=1),
    metric_y: str = Query(..., min_length=1),
    league: Optional[str] = Query(None),
    season: Optional[str] = Query(None),
    positions: Optional[str] = Query(None),
    min_minutes: float = Query(270, ge=0),
    limit: int = Query(5000, ge=1, le=20000),
    session: Session = Depends(get_session),
):
    available = _stats_metric_columns(session)
    if metric_x not in available or metric_y not in available:
        raise HTTPException(status_code=400, detail="Invalid metric selection")

    sql = f"""
    WITH base AS (
      SELECT DISTINCT ON (p.id, c.name, ps.team_in_selected_period)
        p.id AS player_id,
        p.name,
        ps.team_in_selected_period AS team,
        c.name AS competition_name,
        ps.position,
        ps.second_position,
        ps.minutes_played,
        pm.age,
        pm.\"{metric_x}\" AS metric_x,
        pm.\"{metric_y}\" AS metric_y
      FROM player_seasons ps
      JOIN players p ON p.id = ps.player_id
      JOIN competitions c ON c.id = ps.competition_id
      JOIN player_metrics pm ON pm.player_season_id = ps.id
      WHERE pm.\"{metric_x}\" IS NOT NULL
        AND pm.\"{metric_y}\" IS NOT NULL
    """
    params: dict = {"limit": limit}
    if league and league not in {"All leagues", "All"}:
        sql, params = _apply_competition_filter(sql, params, league)
    if min_minutes is not None:
        sql += " AND ps.minutes_played >= :min_minutes"
        params["min_minutes"] = min_minutes
    if season:
        season_values = _season_filter_values(season)
        if len(season_values) <= 1:
            sql += " AND ps.calendar = :season"
            params["season"] = season_values[0] if season_values else season
        else:
            sql += " AND ps.calendar = ANY(:season_values)"
            params["season_values"] = season_values
    if positions:
        pos_list = [p.strip() for p in positions.split(",") if p.strip()]
        if pos_list:
            params["positions"] = pos_list
            params["pos_patterns"] = [f"%{p}%" for p in pos_list]
            sql += """
            AND (
              ps.position = ANY(:positions)
              OR ps.second_position = ANY(:positions)
              OR ps.position ILIKE ANY(:pos_patterns)
              OR ps.second_position ILIKE ANY(:pos_patterns)
            )
            """
    sql += """
      ORDER BY p.id, c.name, ps.team_in_selected_period,
        ps.calendar DESC NULLS LAST, ps.minutes_played DESC NULLS LAST
    )
    SELECT *
    FROM base
    """
    sql += f' ORDER BY metric_x DESC NULLS LAST, metric_y DESC NULLS LAST LIMIT :limit'

    rows = session.execute(text(sql), params).fetchall()
    return [_row_to_dict(row) for row in rows]


@app.get("/youth/stats-research")
def youth_stats_research(
    metric_x: str = Query(..., min_length=1),
    metric_y: str = Query(..., min_length=1),
    league: Optional[str] = Query(None),
    season: Optional[int] = Query(None),
    positions: Optional[str] = Query(None),
    min_minutes: float = Query(90, ge=0),
    limit: int = Query(5000, ge=1, le=20000),
    session: Session = Depends(get_session),
):
    available = _youth_stats_metric_keys(session)
    if metric_x not in available or metric_y not in available:
        raise HTTPException(status_code=400, detail="Invalid metric selection")

    numeric_pattern = r"^-?[0-9]+(\.[0-9]+)?$"
    sql = """
    WITH base AS (
      SELECT DISTINCT ON (provider_player_id, season, championship, club_name)
        id AS player_id,
        display_name AS name,
        club_name AS team,
        championship AS competition_name,
        calendar,
        position,
        primary_position,
        position_group,
        minutes_played,
        age,
        birth_year,
        age_category,
        CASE
          WHEN metrics ->> :metric_x_key ~ :numeric_pattern
          THEN (metrics ->> :metric_x_key)::double precision
          ELSE NULL
        END AS metric_x,
        CASE
          WHEN metrics ->> :metric_y_key ~ :numeric_pattern
          THEN (metrics ->> :metric_y_key)::double precision
          ELSE NULL
        END AS metric_y
      FROM youth_player_rankings
      WHERE metrics ->> :metric_x_key ~ :numeric_pattern
        AND metrics ->> :metric_y_key ~ :numeric_pattern
    """
    params: dict[str, Any] = {
        "metric_x_key": metric_x,
        "metric_y_key": metric_y,
        "numeric_pattern": numeric_pattern,
        "limit": limit,
    }
    if league and league not in {"All leagues", "All"}:
        sql += " AND championship = :league"
        params["league"] = league
    if min_minutes is not None:
        sql += " AND COALESCE(minutes_played, 0) >= :min_minutes"
        params["min_minutes"] = min_minutes
    if season:
        sql += " AND season = :season"
        params["season"] = season
    if positions:
        pos_list = [position.strip() for position in positions.split(",") if position.strip()]
        if pos_list:
            params["positions"] = pos_list
            params["pos_patterns"] = [f"%{position}%" for position in pos_list]
            sql += """
            AND (
              primary_position = ANY(:positions)
              OR position_group = ANY(:positions)
              OR position ILIKE ANY(:pos_patterns)
            )
            """
    sql += """
      ORDER BY provider_player_id, season, championship, club_name,
        minutes_played DESC NULLS LAST, score DESC NULLS LAST, id DESC
    )
    SELECT *
    FROM base
    WHERE metric_x IS NOT NULL
      AND metric_y IS NOT NULL
    ORDER BY metric_x DESC NULLS LAST, metric_y DESC NULLS LAST
    LIMIT :limit
    """
    rows = session.execute(text(sql), params).fetchall()
    return [_row_to_dict(row) for row in rows]
