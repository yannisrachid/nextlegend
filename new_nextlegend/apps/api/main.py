from fastapi import FastAPI, Depends, Query, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional, List
from pathlib import Path
import os
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext

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
from langchain.callbacks import get_openai_callback
import toml
import re
import unicodedata
import json
import csv
from bisect import bisect_right

app = FastAPI(title="NextLegend v2 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    public_paths = {"/", "/health", "/auth/login", "/auth/logout", "/auth/me"}
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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
"""

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
    if _table_exists(session, "auth_sessions"):
        session.execute(
            text("CREATE INDEX IF NOT EXISTS auth_sessions_user_id_idx ON auth_sessions(user_id)")
        )
    if _table_exists(session, "auth_users"):
        session.execute(
            text("CREATE INDEX IF NOT EXISTS auth_users_email_idx ON auth_users(email)")
        )
    session.commit()
    _seed_auth_users(session)


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


class AuthLoginRequest(BaseModel):
    username: str
    password: str
    legacy_user_id: Optional[str] = None


class AuthUserResponse(BaseModel):
    username: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None


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


def _require_admin(request: Request) -> None:
    user = getattr(request.state, "user", None) or {}
    if user.get("role") != "admin":
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
def ai_conversations(user_id: str = Query(...), session: Session = Depends(get_session)):
    _ensure_ai_schema(session)
    rows = session.execute(
        text(
            """
            SELECT id, user_id, title, mode, created_at, updated_at
            FROM ai_conversations
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            """
        ),
        {"user_id": user_id},
    ).fetchall()
    items = [AIConversation(**_row_to_dict(row)) for row in rows]
    return AIConversationList(items=items)


@app.post("/ai/conversations", response_model=AIConversation)
def ai_conversation_create(
    payload: AIConversationCreate,
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
    row = session.execute(
        text(
            """
            INSERT INTO ai_conversations (user_id, title, mode)
            VALUES (:user_id, :title, :mode)
            RETURNING id, user_id, title, mode, created_at, updated_at
            """
        ),
        {
            "user_id": payload.user_id,
            "title": payload.title,
            "mode": payload.mode or "scout",
        },
    ).fetchone()
    session.commit()
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
    return AIConversation(**_row_to_dict(row))


@app.get("/ai/users")
def ai_users(session: Session = Depends(get_session)):
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
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
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
            "user_id": payload.user_id,
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
    user_id: str = Query(...),
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
    convo = session.execute(
        text(
            """
            SELECT id, user_id, title, mode, created_at, updated_at
            FROM ai_conversations
            WHERE id = :conversation_id AND user_id = :user_id
            """
        ),
        {"conversation_id": conversation_id, "user_id": user_id},
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
    user_id: str = Query(...),
    conversation_id: Optional[int] = Query(None),
    session: Session = Depends(get_session),
):
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
    session: Session = Depends(get_session),
):
    _ensure_ai_schema(session)
    convo = session.execute(
        text(
            """
            SELECT id, user_id, title, mode, created_at, updated_at
            FROM ai_conversations
            WHERE id = :conversation_id AND user_id = :user_id
            """
        ),
        {"conversation_id": conversation_id, "user_id": payload.user_id},
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
    cleaned = re.sub(r"[^a-z0-9\\s]", " ", cleaned.lower())
    return " ".join(cleaned.split())


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
            clubs[normalized] = {
                "team": team,
                "rating": float(rating) if rating else 0.0,
                "rank": int(rank) if rank else 0,
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
    if season:
        sql += " AND ps.calendar = :season"
        params["season"] = season
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
    # Récupère la saison explicitement demandée, sinon la plus récente.
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
    else:
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
        similarities_enabled=similarities_enabled,
        current_season_label=CURRENT_SEASON_LABEL,
    )


@app.get("/players/{player_id}/similarities", response_model=List[SimilarityRow])
def player_similarities(
    player_id: int,
    player_season_id: Optional[int] = Query(None, ge=1),
    profile: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=100),
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
    if season:
        sql += " AND ps.calendar = :season"
        params["season"] = season
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
    if season:
        sql += " AND ps.calendar = :season"
        params["season"] = season
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
    if season:
        sql += " AND ps.calendar = :season"
        params["season"] = season
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

    fetch_limit = min(500, max(limit * 20, limit))
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
    items.sort(
        key=lambda item: (
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
    return trimmed


@app.get("/prospects/ids")
def prospect_ids(session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    rows = session.execute(text("SELECT player_id FROM prospects")).fetchall()
    return {"player_ids": [r.player_id for r in rows]}


@app.post("/prospects")
def add_prospect(payload: ProspectToggle, session: Session = Depends(get_session)):
    _ensure_prospect_schema(session)
    result = session.execute(
        text("INSERT INTO prospects (player_id) VALUES (:player_id) ON CONFLICT DO NOTHING"),
        {"player_id": payload.player_id},
    )
    session.commit()
    return {"player_id": payload.player_id, "added": result.rowcount > 0}


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
