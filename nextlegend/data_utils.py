"""Utility helpers for loading and preparing the NextLegend dataset."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

from s3_utils import object_exists, read_csv_from_s3

try:  # Python 3.11+ ships tomllib natively
    import tomllib  # type: ignore[attr-defined]
    _TOML_BINARY = True
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    try:
        import tomli as tomllib  # type: ignore
        _TOML_BINARY = True
    except ModuleNotFoundError:
        import toml as tomllib  # type: ignore
        _TOML_BINARY = False

try:
    from rapidfuzz import fuzz, process
except ImportError as exc:  # pragma: no cover - make missing dependency obvious
    raise RuntimeError(
        "The package 'rapidfuzz' is required for schema reconciliation. "
        "Install it with `pip install rapidfuzz`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SOURCE = "data/wyscout_players_cleaned.csv"
SCHEMA_PATH = PROJECT_ROOT / "config" / "schema_map.yaml"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.toml"

PLAYER_ID_PATTERN = re.compile(r"\(([-\\d]+)\)")

# Metrics that should receive an automatic per90 computation when available.
PER90_SOURCE_METRICS = [
    "goals",
    "assists",
    "xG",
    "xA",
    "npxG",
    "shots",
    "key_passes",
    "smart_passes",
    "progressive_passes",
    "deep_completions",
    "progressive_runs",
    "progressive_carries",
    "dribbles_attempted",
    "dribbles_succeeded",
    "def_duels_total",
    "def_duels_won",
    "interceptions",
    "tackles_sliding",
    "blocks",
    "aerial_duels_total",
    "aerial_duels_won",
]

logger = logging.getLogger("nextlegend.data")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _use_local_path(path: Union[str, Path]) -> bool:
    if isinstance(path, Path):
        return True
    text = str(path)
    return os.path.isabs(text) or text.startswith("./") or text.startswith("../")


def _load_raw_dataset(source: Union[str, Path, None], *, n_rows: Optional[int] = None) -> pd.DataFrame:
    target = source or DATA_SOURCE
    if isinstance(target, Path) or _use_local_path(target):
        path_obj = Path(target)
        if not path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {path_obj}")
        logger.info("Loading dataset from local path %s", path_obj)
        return pd.read_csv(path_obj, nrows=n_rows)

    key = str(target)
    logger.info("Loading dataset from S3 key %s", key)
    try:
        return read_csv_from_s3(key, nrows=n_rows)
    except FileNotFoundError as exc:
        available = object_exists(key)
        hint = "" if available else " (object missing from S3)"
        raise FileNotFoundError(f"S3 data file not found: {key}{hint}") from exc


@dataclass(frozen=True)
class SchemaMappingResult:
    """Expose how internal feature names map to CSV columns."""

    resolved: Dict[str, str]
    suggestions: Dict[str, tuple[str, float]]
    missing: tuple[str, ...]

    @property
    def has_missing(self) -> bool:
        return bool(self.missing)


def _normalise_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_toml(path: Path) -> dict:
    if _TOML_BINARY:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def load_settings() -> dict:
    return _load_toml(SETTINGS_PATH)


@lru_cache(maxsize=1)
def load_schema_map() -> dict:
    return _load_yaml(SCHEMA_PATH)


def load_index_weights() -> dict[str, float]:
    settings = load_settings()
    return {k: float(v) for k, v in settings.get("index_weights", {}).items()}


def _match_column(
    internal_name: str,
    candidates: Iterable[str],
    available_columns: Iterable[str],
    score_threshold: int = 78,
) -> Optional[tuple[str, float]]:
    """Find the best matching column for an internal label."""

    available = list(available_columns)
    lowered = {col.lower(): col for col in available}
    normalised = {_normalise_label(col): col for col in available}

    for candidate in candidates:
        if candidate in available:
            return candidate, 100.0
        lowered_candidate = candidate.lower()
        if lowered_candidate in lowered:
            return lowered[lowered_candidate], 100.0
        normalised_candidate = _normalise_label(candidate)
        if normalised_candidate in normalised:
            return normalised[normalised_candidate], 95.0

    # Nothing found in direct synonyms → fuzzy match against available columns.
    search_terms = list(dict.fromkeys([*candidates, internal_name]))
    best_match: Optional[tuple[str, float]] = None

    for term in search_terms:
        result = process.extractOne(
            term,
            available,
            scorer=fuzz.WRatio,
        )
        if not result:
            continue
        column, score, _ = result
        if score < score_threshold:
            continue
        if best_match is None or score > best_match[1]:
            best_match = (column, float(score))

    return best_match


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("M", "e6", regex=False)
        .str.replace("k", "e3", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"-": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    )
    converted = pd.to_numeric(cleaned, errors="coerce")
    return converted


def _extract_player_tokens(value: object) -> tuple[str, Optional[str]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "", None

    text = str(value)
    token_candidate = text.split(";")[0].strip()
    display_name = token_candidate or text.strip()
    match = PLAYER_ID_PATTERN.search(text)
    player_id = match.group(1) if match else None
    return display_name, player_id


def _parse_age(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    match = re.search(r"\\((\\d{1,2})\\)", text)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\\d+", text)
    if digits:
        try:
            return int(digits[-1])
        except ValueError:
            return None
    return None


def align_schema(df: pd.DataFrame, schema_map: Optional[dict] = None) -> SchemaMappingResult:
    schema = schema_map or load_schema_map()
    resolved: Dict[str, str] = {}
    suggestions: Dict[str, tuple[str, float]] = {}

    for internal_name, candidates in schema.items():
        if not isinstance(candidates, (list, tuple, set)):
            candidates = [candidates]
        match = _match_column(internal_name, candidates, df.columns)
        if match:
            column, score = match
            resolved[internal_name] = column
            if score < 100:
                suggestions[internal_name] = (column, score)
        else:
            hint = process.extractOne(
                internal_name,
                df.columns,
                scorer=fuzz.WRatio,
            )
            if hint:
                column, score, _ = hint
                suggestions[internal_name] = (column, float(score))

    missing = tuple(sorted(set(schema.keys()) - set(resolved.keys())))
    return SchemaMappingResult(resolved=resolved, suggestions=suggestions, missing=missing)


def apply_mapping(df: pd.DataFrame, mapping: SchemaMappingResult) -> pd.DataFrame:
    """Create internal columns based on the resolved mapping."""

    normalized = df.copy()
    for internal_name, source_column in mapping.resolved.items():
        if internal_name in normalized.columns:
            continue
        if source_column not in normalized.columns:
            continue
        normalized[internal_name] = normalized[source_column]
    return normalized


def _post_process_identity(df: pd.DataFrame, mapping: SchemaMappingResult) -> pd.DataFrame:
    working = df.copy()
    name_source = mapping.resolved.get("player_name")
    id_source = mapping.resolved.get("player_id", name_source)

    if name_source and name_source in working.columns:
        parsed = working[name_source].apply(_extract_player_tokens)
        working["player_name"] = parsed.str[0].str.strip()
        if id_source:
            working["player_id"] = parsed.str[1]
    elif id_source and id_source in working.columns:
        parsed = working[id_source].apply(_extract_player_tokens)
        working.setdefault("player_name", parsed.str[0])
        working["player_id"] = parsed.str[1]

    if "player_id" in working.columns:
        working["player_id"] = working["player_id"].fillna("").astype(str).replace({"": np.nan})

    if "age" in working.columns:
        working["age"] = working["age"].apply(_parse_age)

    if "strong_foot" in working.columns:
        working["strong_foot"] = (
            working["strong_foot"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"Droite": "Right", "Gauche": "Left"})
        )

    if "nationality" in working.columns:
        working["nationality"] = working["nationality"].astype(str).str.strip()

    for col in ("team", "league", "season", "position"):
        if col in working.columns:
            working[col] = working[col].astype(str).str.strip()

    return working


def _ensure_numeric_metrics(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    candidate_columns = [
        col
        for col in working.columns
        if col not in {"player_name", "player_id", "team", "league", "season", "position", "role_primary", "role_secondary"}
    ]

    for col in candidate_columns:
        if working[col].dtype == object:
            coerced = _coerce_numeric(working[col])
            numeric_ratio = np.isfinite(coerced).mean()
            if numeric_ratio >= 0.5:
                working[col] = coerced
    return working


def add_per90_metrics(df: pd.DataFrame, minutes_column: str = "minutes") -> pd.DataFrame:
    if minutes_column not in df.columns:
        return df

    working = df.copy()
    minutes = working[minutes_column].replace({0: np.nan})

    for metric in PER90_SOURCE_METRICS:
        if metric not in working.columns:
            continue
        target_column = f"{metric}_per90"
        if target_column in working.columns:
            continue
        alternate = f"{metric}_per_90"
        if alternate in working.columns:
            continue
        working[target_column] = (working[metric] * 90) / minutes

    return working


def add_rate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if {"def_duels_won", "def_duels_total"}.issubset(working.columns):
        working["def_duels_win_rate"] = working["def_duels_won"] / working["def_duels_total"].replace({0: np.nan})

    if {"aerial_duels_won", "aerial_duels_total"}.issubset(working.columns):
        working["aerial_win_rate"] = working["aerial_duels_won"] / working["aerial_duels_total"].replace({0: np.nan})

    if {"dribbles_succeeded", "dribbles_attempted"}.issubset(working.columns):
        working["dribbles_success_rate"] = working["dribbles_succeeded"] / working["dribbles_attempted"].replace({0: np.nan})

    return working


def load_player_dataset(
    csv_path: Optional[Union[str, Path]] = None,
    *,
    competitions: Optional[Sequence[str]] = None,
    n_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, SchemaMappingResult]:
    """Load, map and enrich the players dataset.

    Parameters
    ----------
    competitions:
        Optional list of competition names to keep at load time. Filtering happens
        before schema alignment for performance reasons.
    n_rows:
        Optional maximum number of rows to read from disk (sampling utility for tests).
    """

    df = _load_raw_dataset(csv_path, n_rows=n_rows)

    if competitions:
        competition_col = "competition_name" if "competition_name" in df.columns else None
        if competition_col:
            wanted = {str(item) for item in competitions if item is not None}
            df = df[df[competition_col].astype(str).isin(wanted)]
            logger.info(
                "Competition filtering applied (%s selected) → %s rows",
                len(wanted),
                len(df),
            )
        else:
            logger.warning(
                "Competition filter requested but raw column 'competition_name' is absent."
            )

    mapping = align_schema(df)
    normalized = apply_mapping(df, mapping)
    normalized = _post_process_identity(normalized, mapping)
    normalized = _ensure_numeric_metrics(normalized)
    normalized = add_per90_metrics(normalized)
    normalized = add_rate_metrics(normalized)

    logger.info("Dataset ready: %s rows × %s columns", len(normalized), len(normalized.columns))
    if mapping.missing:
        logger.warning("Missing mapped fields: %s", ", ".join(mapping.missing))
    return normalized, mapping


def filter_by_global_context(
    df: pd.DataFrame,
    *,
    season: Optional[str] = None,
    league: Optional[str | Iterable[str]] = None,
    min_minutes: Optional[int | float] = None,
) -> pd.DataFrame:
    working = df.copy()
    initial_count = len(working)
    if season:
        working = working[working["season"].astype(str) == str(season)]
    if league:
        if isinstance(league, str):
            working = working[working["league"].astype(str) == str(league)]
        else:
            leagues = {str(item) for item in league if item is not None}
            if leagues:
                working = working[working["league"].astype(str).isin(leagues)]
    if min_minutes is not None and "minutes" in working.columns:
        working = working[working["minutes"] >= float(min_minutes)]
    logger.info(
        "Global filters applied → %s rows (initial=%s, season=%s, leagues=%s, min_minutes=%s)",
        len(working),
        initial_count,
        season or "all",
        league if league else "all",
        min_minutes,
    )
    return working


def available_filter_options(df: pd.DataFrame) -> dict:
    options = {}
    if "season" in df.columns:
        options["season"] = sorted({val for val in df["season"].dropna().astype(str)})
    if "league" in df.columns:
        options["league"] = sorted({val for val in df["league"].dropna().astype(str)})
    return options
