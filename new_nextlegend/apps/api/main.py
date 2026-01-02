from fastapi import FastAPI, Depends, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Optional, List
from pathlib import Path

from settings import settings
from db import get_session
from models import RankingRow, Report, SimilarityRow, RankingPage, RoleScore
import re
import json

app = FastAPI(title="NextLegend v2 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _row_to_dict(row) -> dict:
    return dict(row._mapping) if row is not None else {}


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

_COMPETITION_AGGREGATES: Optional[list[dict[str, list[str]]]] = None
_COMPETITION_AGGREGATES_MTIME: Optional[float] = None

_ROLE_METRICS: Optional[dict[str, list[str]]] = None
_ROLE_METRICS_MTIME: Optional[float] = None


_TM_COLUMNS_CACHE: Optional[list[str]] = None


def _get_tm_columns(session: Session) -> list[str]:
    global _TM_COLUMNS_CACHE
    if _TM_COLUMNS_CACHE is None:
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
      ps.assigned_role_pct_league,
      ps.assigned_role_pct_global,
      pm.age AS age""" + tm_clause + """
    FROM player_seasons ps
    JOIN players p ON p.id = ps.player_id
    JOIN competitions c ON c.id = ps.competition_id
    LEFT JOIN player_metrics pm ON pm.player_season_id = ps.id
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
      ps.assigned_role_pct_league,
      ps.assigned_role_pct_global,
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


@app.get("/players/{player_id}/report", response_model=Report)
def player_report(player_id: int, session: Session = Depends(get_session)):
    # Récup player_season le plus récent
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
    role_rows = session.execute(
        text("SELECT profile, raw_score, pct_league, pct_global, pct_global_adjusted FROM role_scores WHERE player_season_id = :psid"),
        {"psid": ps["id"]},
    ).fetchall()
    role_scores = [
        RoleScore(
            profile=r._mapping["profile"],
            raw_score=r._mapping["raw_score"],
            pct_league=r._mapping["pct_league"],
            pct_global=r._mapping["pct_global"],
            pct_global_adjusted=r._mapping["pct_global_adjusted"],
        )
        for r in role_rows
        if r._mapping.get("profile")
    ]
    role_scores.sort(key=lambda r: r.pct_global if r.pct_global is not None else -1, reverse=True)

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
    )


@app.get("/players/{player_id}/similarities", response_model=List[SimilarityRow])
def player_similarities(
    player_id: int,
    profile: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    age_min: Optional[float] = Query(None, ge=0),
    age_max: Optional[float] = Query(None, ge=0),
    big5_only: bool = Query(False),
    session: Session = Depends(get_session),
):
    tm_clause = _tm_select_clause(session, alias="psb")
    big5 = _competition_aggregate_map().get("Big 5 Leagues", [])
    seed_sql = """
    SELECT ps.id AS player_season_id, ps.assigned_role
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


@app.get("/meta/seasons")
def meta_seasons(session: Session = Depends(get_session)):
    sql = """
    SELECT DISTINCT label
    FROM seasons
    WHERE label IS NOT NULL AND label <> ''
    ORDER BY label
    """
    rows = session.execute(text(sql)).fetchall()
    return [r.label for r in rows if r.label]


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
        prefix_key = f"t{idx}_prefix"
        space_key = f"t{idx}_space"
        dot_key = f"t{idx}_dot"
        dash_key = f"t{idx}_dash"
        params[prefix_key] = f"{token}%"
        params[space_key] = f"% {token}%"
        params[dot_key] = f"%.{token}%"
        params[dash_key] = f"%-{token}%"
        token_clauses.append(
            f"(p.name ILIKE :{prefix_key} OR p.name ILIKE :{space_key} OR p.name ILIKE :{dot_key} OR p.name ILIKE :{dash_key})"
        )
    where_clause = " AND ".join(token_clauses) if token_clauses else "TRUE"

    sql = f"""
    SELECT DISTINCT ON (p.name, ps.team_in_selected_period, c.name, ps.calendar)
      p.id,
      p.name,
      c.name AS competition_name,
      ps.calendar,
      ps.team_in_selected_period AS team
    FROM players p
    JOIN player_seasons ps ON ps.player_id = p.id
    JOIN competitions c ON c.id = ps.competition_id
    WHERE {where_clause}
    ORDER BY p.name, ps.team_in_selected_period, c.name, ps.calendar, ps.minutes_played DESC NULLS LAST
    LIMIT :limit
    """
    rows = session.execute(text(sql), params).fetchall()
    return [_row_to_dict(row) for row in rows]
