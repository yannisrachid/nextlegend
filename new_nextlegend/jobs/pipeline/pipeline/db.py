"""
DB utilities: ensure schema and perform bulk upserts aligned with DATA_MODEL.md.
This is a minimal implementation to unblock ingestion; it handles core tables and
uses psycopg2 execute_values for ON CONFLICT upserts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from io import StringIO

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS competitions (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    country TEXT,
    type TEXT,
    tier INT,
    strength_factor DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seasons (
    id SERIAL PRIMARY KEY,
    label TEXT UNIQUE NOT NULL,
    start_year INT,
    end_year INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clubs (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    competition_id INT REFERENCES competitions(id),
    country TEXT,
    external_wyscout_id TEXT,
    external_tm_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, competition_id)
);

CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    wyscout_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    country TEXT,
    birth_date DATE,
    age INT,
    height_cm DOUBLE PRECISION,
    weight_kg DOUBLE PRECISION,
    foot TEXT,
    tm_id TEXT,
    tm_profile_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS player_seasons (
    id SERIAL PRIMARY KEY,
    player_id INT REFERENCES players(id),
    competition_id INT REFERENCES competitions(id),
    season_id INT REFERENCES seasons(id),
    club_id INT REFERENCES clubs(id),
    calendar TEXT,
    team_in_selected_period TEXT,
    position TEXT,
    second_position TEXT,
    minutes_played DOUBLE PRECISION,
    matches_played DOUBLE PRECISION,
    assigned_role TEXT,
    assigned_role_pct_league DOUBLE PRECISION,
    assigned_role_pct_global DOUBLE PRECISION,
    global_score_adjusted DOUBLE PRECISION,
    league_strength_factor DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(player_id, competition_id, season_id, club_id)
);

CREATE TABLE IF NOT EXISTS player_metrics (
    player_season_id INT PRIMARY KEY REFERENCES player_seasons(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS role_scores (
    id SERIAL PRIMARY KEY,
    player_season_id INT REFERENCES player_seasons(id),
    profile TEXT,
    raw_score DOUBLE PRECISION,
    pct_league DOUBLE PRECISION,
    pct_global DOUBLE PRECISION,
    pct_global_adjusted DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS role_scores_unique ON role_scores (player_season_id, profile);

CREATE TABLE IF NOT EXISTS player_similarity (
    id SERIAL PRIMARY KEY,
    profile TEXT,
    player_a_id INT REFERENCES players(id),
    player_b_id INT REFERENCES players(id),
    player_a_season_id INT REFERENCES player_seasons(id),
    player_b_season_id INT REFERENCES player_seasons(id),
    similarity DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id TEXT UNIQUE,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    status TEXT,
    source_uri TEXT,
    enriched_uri TEXT,
    rows_processed INT,
    message TEXT,
    git_sha TEXT
);

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


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(SCHEMA_SQL))


def _execute_upsert(
    conn,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence],
    conflict_cols: Sequence[str],
    update_cols: Sequence[str],
    page_size: int = 1000,
):
    col_list = ", ".join(columns)
    conflict = ", ".join(conflict_cols)
    update = ", ".join([f"{c}=EXCLUDED.{c}" for c in update_cols]) if update_cols else ""
    placeholders = ", ".join([f":{c}" for c in columns])
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
    if conflict_cols:
        sql += f" ON CONFLICT ({conflict})"
        if update:
            sql += f" DO UPDATE SET {update}"
        else:
            sql += " DO NOTHING"

    def _clean_value(value):
        if value is None:
            return None
        try:
            return None if pd.isna(value) else value
        except Exception:
            return value

    batch = []
    for row in rows:
        cleaned = {col: _clean_value(val) for col, val in zip(columns, row)}
        batch.append(cleaned)
        if len(batch) >= page_size:
            conn.execute(text(sql), batch)
            batch = []
    if batch:
        conn.execute(text(sql), batch)


def upsert_dimensions(engine: Engine, competitions: pd.DataFrame, seasons: pd.DataFrame, players: pd.DataFrame, clubs: pd.DataFrame):
    with engine.begin() as conn:
        _execute_upsert(
            conn,
            "competitions",
            ["name"],
            competitions[["name"]].itertuples(index=False, name=None),
            ["name"],
            [],
        )
        _execute_upsert(
            conn,
            "seasons",
            ["label"],
            seasons[["label"]].itertuples(index=False, name=None),
            ["label"],
            [],
        )
        player_cols = ["wyscout_id", "name"]
        update_cols = ["name"]
        if "tm_id" in players.columns:
            player_cols.append("tm_id")
            update_cols.append("tm_id")
        if "tm_profile_url" in players.columns:
            player_cols.append("tm_profile_url")
            update_cols.append("tm_profile_url")
        _execute_upsert(
            conn,
            "players",
            player_cols,
            players[player_cols].itertuples(index=False, name=None),
            ["wyscout_id"],
            update_cols,
        )
    # clubs dépend des IDs competitions -> map après insertion
    if not clubs.empty:
        comp_map = _id_map(engine, "competitions", "name")
        clubs = clubs.copy()
        clubs["competition_id"] = clubs["competition_name"].map(comp_map)
        clubs = clubs.dropna(subset=["competition_id"])
        with engine.begin() as conn:
            _execute_upsert(
                conn,
                "clubs",
                ["name", "competition_id"],
                clubs[["name", "competition_id"]].itertuples(index=False, name=None),
                ["name", "competition_id"],
                [],
            )


def _id_map(engine: Engine, table: str, key_col: str) -> dict[str, int]:
    with engine.begin() as conn:
        rows = conn.execute(text(f"SELECT id, {key_col} FROM {table}")).fetchall()
    return {str(r[1]): r[0] for r in rows}


def resolve_ids(engine: Engine):
    return {
        "competitions": _id_map(engine, "competitions", "name"),
        "seasons": _id_map(engine, "seasons", "label"),
        "players": _id_map(engine, "players", "wyscout_id"),
        "clubs": _id_map(engine, "clubs", "name"),
    }


def upsert_player_seasons(engine: Engine, fact: pd.DataFrame, ids: dict):
    if fact.empty:
        return pd.DataFrame()
    fact = fact.copy()
    fact["competition_id"] = fact["competition_name"].map(ids["competitions"])
    fact["season_id"] = fact["calendar"].map(ids["seasons"])
    fact["player_id"] = fact["wyscout_id"].astype(str).map(ids["players"])
    fact["club_id"] = fact["team_in_selected_period"].map(ids["clubs"]) if "team_in_selected_period" in fact.columns else None
    fact = fact.dropna(subset=["competition_id", "season_id", "player_id"])

    tm_cols = [c for c in fact.columns if c.startswith("tm_")]
    if tm_cols:
        with engine.begin() as conn:
            existing = conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_seasons'")
            ).fetchall()
            existing_cols = {r[0] for r in existing}
            for col in tm_cols:
                if col in existing_cols:
                    continue
                if pd.api.types.is_numeric_dtype(fact[col]):
                    col_type = "DOUBLE PRECISION"
                else:
                    col_type = "TEXT"
                conn.execute(text(f'ALTER TABLE player_seasons ADD COLUMN "{col}" {col_type}'))

    cols = [
        "player_id",
        "competition_id",
        "season_id",
        "club_id",
        "calendar",
        "team_in_selected_period",
        "position",
        "second_position",
        "minutes_played",
        "matches_played",
        "assigned_role",
        "assigned_role_pct_league",
        "assigned_role_pct_global",
        "global_score_adjusted",
        "league_strength_factor",
    ]
    cols.extend(tm_cols)
    rows = fact[cols].itertuples(index=False, name=None)

    update_cols = [
        "position",
        "second_position",
        "minutes_played",
        "matches_played",
        "assigned_role",
        "assigned_role_pct_league",
        "assigned_role_pct_global",
        "global_score_adjusted",
        "league_strength_factor",
        "team_in_selected_period",
    ] + tm_cols

    with engine.begin() as conn:
        _execute_upsert(
            conn,
            "player_seasons",
            cols,
            rows,
            ["player_id", "competition_id", "season_id", "club_id"],
            update_cols,
        )
        # Return id map for player_seasons
        map_rows = conn.execute(
            text("SELECT id, player_id, competition_id, season_id, COALESCE(club_id, -1) AS club_id FROM player_seasons")
        ).fetchall()
    index = {(r.player_id, r.competition_id, r.season_id, r.club_id): r.id for r in map_rows}
    return index


def _copy_dataframe(conn, table: str, df: pd.DataFrame, columns: Sequence[str], chunk_size: int = 5000):
    col_list = ", ".join([f'"{c}"' for c in columns])
    copy_sql = f"COPY {table} ({col_list}) FROM STDIN WITH CSV"
    raw = conn.connection
    with raw.cursor() as cur:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            buffer = StringIO()
            chunk.to_csv(buffer, index=False, header=False, na_rep="")
            buffer.seek(0)
            with cur.copy(copy_sql) as copy:
                copy.write(buffer.getvalue())


def upsert_player_metrics(
    engine: Engine,
    metrics: pd.DataFrame,
    season_index: dict,
    ids: dict,
    replace: bool = False,
    use_copy: bool = False,
):
    if metrics.empty:
        print("[DB] player_metrics empty; skip.")
        return
    metrics = metrics.copy()
    print(f"[DB] player_metrics input rows={len(metrics)} cols={len(metrics.columns)}")
    metrics["player_id"] = metrics["wyscout_id"].astype(str).map(ids["players"])
    metrics["competition_id"] = metrics["competition_name"].map(ids["competitions"])
    metrics["season_id"] = metrics["calendar"].map(ids["seasons"])
    metrics["club_id"] = metrics.get("team_in_selected_period", pd.Series([-1] * len(metrics))).map(ids["clubs"]).fillna(-1)
    metrics["player_season_id"] = list(
        zip(
            metrics["player_id"],
            metrics["competition_id"],
            metrics["season_id"],
            metrics["club_id"].fillna(-1),
        )
    )
    metrics["player_season_id"] = metrics["player_season_id"].map(season_index)
    before_rows = len(metrics)
    metrics = metrics.dropna(subset=["player_season_id"])
    dropped = before_rows - len(metrics)
    print(f"[DB] player_metrics mapped rows={len(metrics)} dropped={dropped}")
    # Remove helper columns
    metrics = metrics.drop(columns=["club_id", "player_id", "competition_id", "season_id"], errors="ignore")
    metrics = metrics.drop(columns=["wyscout_id", "competition_name", "calendar", "team_in_selected_period", "team"], errors="ignore")

    metric_cols = [c for c in metrics.columns if c != "player_season_id"]
    print(f"[DB] player_metrics columns={len(metric_cols)}")
    if not metric_cols:
        print("[WARN] player_metrics has no numeric columns after cleanup")
    # Ensure columns exist
    if metric_cols:
        with engine.begin() as conn:
            existing = conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_metrics'")
            ).fetchall()
            existing_cols = {r[0] for r in existing}
            for col in metric_cols:
                if col not in existing_cols:
                    conn.execute(text(f'ALTER TABLE player_metrics ADD COLUMN "{col}" DOUBLE PRECISION'))

    metrics = metrics.where(pd.notna(metrics), None)

    with engine.begin() as conn:
        if replace:
            conn.execute(text("TRUNCATE player_metrics"))
        else:
            unique_ids = [int(val) for val in metrics["player_season_id"].unique() if pd.notna(val)]
            conn.execute(
                text("DELETE FROM player_metrics WHERE player_season_id = ANY(:ids)"),
                {"ids": unique_ids},
            )

        if use_copy:
            _copy_dataframe(conn, "player_metrics", metrics, metrics.columns.tolist())
        else:
            metrics.to_sql(
                "player_metrics",
                conn,
                if_exists="append",
                index=False,
                chunksize=2000,
                method="multi",
            )


def upsert_role_scores(engine: Engine, role_scores: pd.DataFrame, season_index: dict, ids: dict):
    if role_scores.empty:
        return
    role_scores = role_scores.copy()
    role_scores["player_season_id"] = list(
        zip(
            role_scores["wyscout_id"].astype(str).map(ids["players"]),
            role_scores["competition_name"].map(ids["competitions"]),
            role_scores["calendar"].map(ids["seasons"]),
            role_scores.get("team_in_selected_period", pd.Series([-1] * len(role_scores))).map(ids["clubs"]).fillna(-1),
        )
    )
    role_scores["player_season_id"] = role_scores["player_season_id"].map(season_index)
    role_scores = role_scores.dropna(subset=["player_season_id"])
    cols = ["player_season_id", "profile", "raw_score", "pct_league", "pct_global", "pct_global_adjusted"]
    rows = role_scores[cols].itertuples(index=False, name=None)
    with engine.begin() as conn:
        _execute_upsert(
            conn,
            "role_scores",
            cols,
            rows,
            ["player_season_id", "profile"],
            ["raw_score", "pct_league", "pct_global", "pct_global_adjusted"],
        )


def upsert_similarity(
    engine: Engine,
    similarity: pd.DataFrame,
    ids: dict,
    season_index: dict,
    replace: bool = False,
    use_copy: bool = False,
):
    if similarity.empty:
        return
    similarity = similarity.copy()
    if "player_a_id" in similarity.columns:
        similarity["player_a_id"] = similarity["player_a_id"].astype(str).map(ids["players"])
    else:
        similarity["player_a_id"] = similarity["player_a"].astype(str).map(ids["players"])
    if "player_b_id" in similarity.columns:
        similarity["player_b_id"] = similarity["player_b_id"].astype(str).map(ids["players"])
    else:
        similarity["player_b_id"] = similarity["player_b"].astype(str).map(ids["players"])
    similarity["competition_a_id"] = similarity.get("competition_a", pd.Series()).map(ids["competitions"])
    similarity["competition_b_id"] = similarity.get("competition_b", pd.Series()).map(ids["competitions"])
    similarity["season_a_id"] = similarity.get("calendar_a", pd.Series()).map(ids["seasons"])
    similarity["season_b_id"] = similarity.get("calendar_b", pd.Series()).map(ids["seasons"])
    similarity["club_a_id"] = similarity.get("team_a", pd.Series([-1] * len(similarity))).map(ids.get("clubs", {})).fillna(-1)
    similarity["club_b_id"] = similarity.get("team_b", pd.Series([-1] * len(similarity))).map(ids.get("clubs", {})).fillna(-1)

    similarity["player_a_season_id"] = list(
        zip(
            similarity["player_a_id"],
            similarity["competition_a_id"],
            similarity["season_a_id"],
            similarity["club_a_id"],
        )
    )
    similarity["player_b_season_id"] = list(
        zip(
            similarity["player_b_id"],
            similarity["competition_b_id"],
            similarity["season_b_id"],
            similarity["club_b_id"],
        )
    )
    similarity["player_a_season_id"] = similarity["player_a_season_id"].map(season_index)
    similarity["player_b_season_id"] = similarity["player_b_season_id"].map(season_index)
    similarity = similarity.dropna(subset=["player_a_id", "player_b_id"])
    for col in ("player_a_id", "player_b_id", "player_a_season_id", "player_b_season_id"):
        if col in similarity.columns:
            similarity[col] = similarity[col].apply(
                lambda val: int(val) if pd.notna(val) else None
            )
    cols = ["profile", "player_a_id", "player_b_id", "player_a_season_id", "player_b_season_id", "similarity"]
    similarity = similarity[cols].where(pd.notna(similarity[cols]), None)
    with engine.begin() as conn:
        if replace:
            conn.execute(text("TRUNCATE player_similarity"))
        if use_copy:
            _copy_dataframe(conn, "player_similarity", similarity, cols)
        else:
            rows = similarity[cols].itertuples(index=False, name=None)
            _execute_upsert(
                conn,
                "player_similarity",
                cols,
                rows,
                [],
                [],
            )


def insert_pipeline_run(engine: Engine, run_id: str, status: str, source_uri: str, rows_processed: int, message: str = ""):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pipeline_runs (run_id, started_at, ended_at, status, source_uri, rows_processed, message)
                VALUES (:run_id, NOW(), NOW(), :status, :source_uri, :rows_processed, :message)
                ON CONFLICT (run_id) DO UPDATE SET
                    ended_at = NOW(),
                    status = EXCLUDED.status,
                    rows_processed = EXCLUDED.rows_processed,
                    message = EXCLUDED.message
                """
            ),
            {"run_id": run_id, "status": status, "source_uri": source_uri, "rows_processed": rows_processed, "message": message},
        )
