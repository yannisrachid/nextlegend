from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

import pandas as pd

try:
    from fuzzywuzzy import fuzz
except Exception:  # pragma: no cover - fallback for minimal environments
    fuzz = None


NULL_STRINGS = {"", "nan", "none", "<na>", "null", "nat"}
MONTH_CODE = {
    "01": "A",
    "02": "B",
    "03": "C",
    "04": "D",
    "05": "E",
    "06": "F",
    "07": "G",
    "08": "H",
    "09": "I",
    "10": "J",
    "11": "K",
    "12": "L",
}


@dataclass(frozen=True)
class MatchConfig:
    min_candidate_score: float = 0.72
    auto_accept_score: float = 0.9
    auto_accept_margin: float = 0.035
    max_candidates_per_player: int = 5


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in NULL_STRINGS


def clean_text(value: Any) -> str:
    if _is_blank(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text).lower()
    return " ".join(text.split())


def normalize_name(value: Any) -> str:
    tokens = [tok for tok in clean_text(value).split() if tok not in {"jr", "junior", "ii", "iii"}]
    return " ".join(tokens)


def normalize_club(value: Any) -> str:
    text = clean_text(value)
    stop = {"fc", "cf", "sc", "ac", "afc", "club", "de", "the"}
    return " ".join(tok for tok in text.split() if tok not in stop)


def parse_date(value: Any) -> Optional[dt.date]:
    if _is_blank(value):
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})", text)
    if match:
        day, month, year = match.groups()
        try:
            return dt.date(int(year), int(month), int(day))
        except ValueError:
            return None
    return None


def parse_birth_year(value: Any) -> Optional[int]:
    if _is_blank(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        year = int(value)
        if 1900 <= year <= 2050:
            return year
    text = str(value).strip()
    match = re.search(r"'(\d{2})\b", text)
    if match:
        year = int(match.group(1))
        return 2000 + year if year <= 35 else 1900 + year
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if match:
        return int(match.group(1))
    birth = parse_date(text)
    return birth.year if birth else None


def infer_birth_year_candidates(age: Any, calendar: Any = None, reference_year: Optional[int] = None) -> set[int]:
    if _is_blank(age):
        return set()
    birth_year = parse_birth_year(age)
    if birth_year:
        return {birth_year}
    try:
        age_int = int(float(str(age).strip()))
    except ValueError:
        return set()
    if age_int < 12 or age_int > 50:
        return set()
    if reference_year is None:
        calendar_text = "" if _is_blank(calendar) else str(calendar)
        years = [int(match) for match in re.findall(r"\b(20\d{2})\b", calendar_text)]
        reference_year = min(years) if years else dt.datetime.now(dt.timezone.utc).year
    estimated_birth_year = reference_year - age_int
    return {estimated_birth_year - 1, estimated_birth_year, estimated_birth_year + 1}


def encoded_birth_date(value: Any) -> str:
    birth = parse_date(value)
    if not birth:
        return "00000"
    return f"{birth.day:02d}{MONTH_CODE[f'{birth.month:02d}']}{str(birth.year)[-2:]}"


def parse_market_value(value: Any) -> Optional[float]:
    if _is_blank(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value) if isinstance(value, float) else False:
            return None
        return float(value)
    text = str(value).strip().replace(",", ".")
    numeric = re.sub(r"[^0-9.]", "", text)
    if not numeric:
        return None
    try:
        amount = float(numeric)
    except ValueError:
        return None
    lower = text.lower()
    if "bn" in lower or "billion" in lower:
        amount *= 1_000_000_000
    elif "m" in lower or "mill" in lower:
        amount *= 1_000_000
    elif "k" in lower or "th" in lower:
        amount *= 1_000
    return float(amount)


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is not None:
        return max(fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b)) / 100.0
    return SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()


def _name_similarity(wyscout_name: str, tm_name: str) -> tuple[float, str]:
    base = _similarity(wyscout_name, tm_name)
    w_tokens = wyscout_name.split()
    tm_tokens = tm_name.split()
    if not w_tokens or not tm_tokens:
        return base, "empty"

    short_tokens = [tok for tok in w_tokens if len(tok) == 1]
    long_tokens = [tok for tok in w_tokens if len(tok) > 1]
    if not short_tokens:
        return base, "full"

    long_matches = 0
    for token in long_tokens:
        if token in tm_tokens or max((_similarity(token, tm_token) for tm_token in tm_tokens), default=0.0) >= 0.88:
            long_matches += 1
    initial_matches = 0
    for token in short_tokens:
        if any(tm_token.startswith(token) for tm_token in tm_tokens):
            initial_matches += 1

    long_ok = bool(long_tokens) and long_matches == len(long_tokens)
    initials_ok = initial_matches == len(short_tokens)
    same_last_name = bool(long_tokens) and bool(tm_tokens) and _similarity(long_tokens[-1], tm_tokens[-1]) >= 0.9

    if long_ok and initials_ok:
        return max(base, 0.96 if len(long_tokens) > 1 else 0.94), "abbreviated"
    if same_last_name and initial_matches:
        return max(base, 0.92), "abbreviated_partial"
    return base, "abbreviated_weak"


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_lookup = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_lookup:
            return lower_lookup[candidate]
    return None


def _extract_birth_from_description(value: Any) -> Optional[str]:
    if _is_blank(value):
        return None
    match = re.search(r"\*\s*(\d{2}/\d{2}/\d{4})", str(value))
    birth = parse_date(match.group(1)) if match else None
    return birth.isoformat() if birth else None


def _extract_age_from_description(value: Any) -> Optional[float]:
    if _is_blank(value):
        return None
    match = re.search(r",\s*(\d{1,2})\s*,", str(value))
    if not match:
        return None
    return float(match.group(1))


def prepare_transfermarkt_profiles(raw: pd.DataFrame) -> pd.DataFrame:
    tm = raw.copy()
    tm.columns = [str(col).strip().lower() for col in tm.columns]

    id_col = _first_existing_column(tm, ["tm_player_id", "player_id", "id"])
    name_col = _first_existing_column(tm, ["tm_player_name", "player_name", "profile_name", "name"])
    if not id_col or not name_col:
        raise ValueError("Transfermarkt input must include a player id and player name column.")

    profile_url_col = _first_existing_column(tm, ["tm_profile_url", "profile_url", "url"])
    profile_image_col = _first_existing_column(tm, ["tm_profile_image_url", "profile_image_url", "imageurl", "image_url"])
    birth_col = _first_existing_column(tm, ["tm_birth_date", "birth_date", "date_of_birth", "dateofbirth"])
    age_col = _first_existing_column(tm, ["tm_age", "age"])
    club_id_col = _first_existing_column(tm, ["tm_club_id", "club_id"])
    club_name_col = _first_existing_column(tm, ["tm_club_name", "club_name"])
    position_col = _first_existing_column(tm, ["tm_position_main", "position_main", "position"])
    citizenship_col = _first_existing_column(tm, ["tm_citizenship", "citizenship", "nationality", "nationalities"])
    market_value_col = _first_existing_column(tm, ["tm_market_value", "market_value", "marketvalue"])
    agent_col = _first_existing_column(tm, ["tm_agent_name", "agent_name"])
    description_col = _first_existing_column(tm, ["profile_description", "description"])
    fetched_at_col = _first_existing_column(tm, ["tm_fetched_at", "fetched_at", "profile_updated_at"])

    out = pd.DataFrame(index=tm.index)
    out["tm_player_id"] = tm[id_col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    out["tm_player_name"] = tm[name_col]
    out["tm_profile_url"] = tm[profile_url_col] if profile_url_col else pd.NA
    out["tm_profile_image_url"] = tm[profile_image_col] if profile_image_col else pd.NA
    if birth_col:
        out["tm_birth_date"] = tm[birth_col].apply(lambda value: parse_date(value).isoformat() if parse_date(value) else None)
    elif description_col:
        out["tm_birth_date"] = tm[description_col].apply(_extract_birth_from_description)
    else:
        out["tm_birth_date"] = pd.NA
    if age_col:
        out["tm_age"] = pd.to_numeric(tm[age_col], errors="coerce")
    elif description_col:
        out["tm_age"] = tm[description_col].apply(_extract_age_from_description)
    else:
        out["tm_age"] = pd.NA
    out["tm_club_id"] = tm[club_id_col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True) if club_id_col else pd.NA
    out["tm_club_name"] = tm[club_name_col] if club_name_col else pd.NA
    out["tm_position_main"] = tm[position_col] if position_col else pd.NA
    out["tm_citizenship"] = tm[citizenship_col] if citizenship_col else pd.NA
    out["tm_market_value"] = tm[market_value_col] if market_value_col else pd.NA
    out["tm_market_value_eur"] = out["tm_market_value"].apply(parse_market_value)
    out["tm_agent_name"] = tm[agent_col] if agent_col else pd.NA
    out["tm_fetched_at"] = tm[fetched_at_col] if fetched_at_col else pd.NA
    out["tm_name_norm"] = out["tm_player_name"].apply(normalize_name)
    out["tm_birth_key"] = out["tm_birth_date"].apply(encoded_birth_date)
    out["tm_birth_year"] = out["tm_birth_date"].apply(parse_birth_year)
    out["tm_club_norm"] = out["tm_club_name"].apply(normalize_club)
    out = out[out["tm_player_id"].notna() & ~out["tm_player_id"].astype(str).str.lower().isin(NULL_STRINGS)]
    return out.drop_duplicates(subset=["tm_player_id"], keep="last").reset_index(drop=True)


def prepare_wyscout_players(raw: pd.DataFrame) -> pd.DataFrame:
    wyscout = raw.copy()
    wyscout["name_norm"] = wyscout["name"].apply(normalize_name)
    wyscout["birth_key"] = wyscout["birth_date"].apply(encoded_birth_date) if "birth_date" in wyscout.columns else "00000"
    if "birth_year" in wyscout.columns:
        wyscout["birth_year"] = wyscout["birth_year"].apply(parse_birth_year)
    elif "birth_date" in wyscout.columns:
        wyscout["birth_year"] = wyscout["birth_date"].apply(parse_birth_year)
    else:
        wyscout["birth_year"] = pd.NA
    birth_year_candidates = []
    for _, row in wyscout.iterrows():
        candidates = infer_birth_year_candidates(row.get("age"), row.get("calendar"))
        birth_year = parse_birth_year(row.get("birth_year"))
        if birth_year:
            candidates.add(birth_year)
        birth_year_candidates.append(sorted(candidates))
    wyscout["birth_year_candidates"] = birth_year_candidates
    wyscout["club_norm"] = wyscout["club_name"].apply(normalize_club) if "club_name" in wyscout.columns else ""
    if "age" not in wyscout.columns and "season_age" in wyscout.columns:
        wyscout["age"] = wyscout["season_age"]
    return wyscout


def _birth_score(w_row: pd.Series, tm_row: pd.Series) -> tuple[float, Optional[float]]:
    w_birth = parse_date(w_row.get("birth_date"))
    tm_birth = parse_date(tm_row.get("tm_birth_date"))
    if w_birth and tm_birth:
        if w_birth == tm_birth:
            return 1.0, 0.0
        if w_birth.year == tm_birth.year:
            return 0.82, 0.0
        return 0.0, float(abs(w_birth.year - tm_birth.year))

    tm_year = parse_birth_year(tm_row.get("tm_birth_year")) or (tm_birth.year if tm_birth else None)
    w_year = parse_birth_year(w_row.get("birth_year"))
    candidates = set(w_row.get("birth_year_candidates") or [])
    if w_year:
        candidates.add(w_year)
    if tm_year and candidates:
        if tm_year in candidates:
            return 0.9, 0.0
        min_diff = min(abs(int(candidate) - int(tm_year)) for candidate in candidates)
        if min_diff == 1:
            return 0.68, 1.0
        return 0.0, float(min_diff)

    w_age = w_row.get("age")
    tm_age = tm_row.get("tm_age")
    try:
        if pd.notna(w_age) and pd.notna(tm_age):
            diff = abs(float(w_age) - float(tm_age))
            if diff <= 1:
                return 0.82, diff
            if diff <= 2:
                return 0.45, diff
            return 0.0, diff
    except Exception:
        pass
    return 0.55, None


def _country_score(w_row: pd.Series, tm_row: pd.Series) -> float:
    country = clean_text(w_row.get("country"))
    citizenship = clean_text(tm_row.get("tm_citizenship"))
    if not country or not citizenship:
        return 0.55
    return 1.0 if country in citizenship or citizenship in country else 0.0


def _position_score(w_row: pd.Series, tm_row: pd.Series) -> float:
    position = clean_text(w_row.get("position"))
    tm_position = clean_text(tm_row.get("tm_position_main"))
    if not position or not tm_position:
        return 0.55
    if position in tm_position or tm_position in position:
        return 1.0
    return _similarity(position, tm_position)


def score_candidate(w_row: pd.Series, tm_row: pd.Series) -> dict[str, Any]:
    name_score, name_pattern = _name_similarity(str(w_row.get("name_norm", "")), str(tm_row.get("tm_name_norm", "")))
    club_score = _similarity(str(w_row.get("club_norm", "")), str(tm_row.get("tm_club_norm", ""))) or 0.55
    birth_score, age_diff = _birth_score(w_row, tm_row)
    country_score = _country_score(w_row, tm_row)
    position_score = _position_score(w_row, tm_row)

    if name_score < 0.7:
        total = name_score * 0.55
    elif birth_score == 0.0:
        total = (0.74 * name_score) + (0.12 * club_score) + (0.08 * country_score) + (0.06 * position_score)
        total = min(total, 0.84)
    else:
        total = (
            (0.62 * name_score)
            + (0.18 * birth_score)
            + (0.11 * club_score)
            + (0.05 * country_score)
            + (0.04 * position_score)
        )
    if name_pattern in {"abbreviated", "abbreviated_partial"} and club_score >= 0.92 and birth_score >= 0.55:
        total = max(total, 0.91 if name_pattern == "abbreviated" else 0.88)

    return {
        "confidence_score": round(float(total), 4),
        "name_score": round(float(name_score), 4),
        "name_pattern": name_pattern,
        "birth_score": round(float(birth_score), 4),
        "club_score": round(float(club_score), 4),
        "country_score": round(float(country_score), 4),
        "position_score": round(float(position_score), 4),
        "age_diff": age_diff,
    }


def _block_indexes(tm: pd.DataFrame) -> dict[str, dict[str, set[int]]]:
    blocks = {"last": {}, "club": {}, "birth": {}}
    for idx, row in tm.iterrows():
        name_tokens = str(row.get("tm_name_norm") or "").split()
        if name_tokens:
            blocks["last"].setdefault(name_tokens[-1], set()).add(idx)
        club = str(row.get("tm_club_norm") or "")
        if club:
            blocks["club"].setdefault(club, set()).add(idx)
        birth = str(row.get("tm_birth_key") or "")
        if birth and birth != "00000":
            blocks["birth"].setdefault(birth, set()).add(idx)
    return blocks


def _candidate_indexes(w_row: pd.Series, blocks: dict[str, dict[str, set[int]]], tm_count: int) -> set[int]:
    indexes: set[int] = set()
    tokens = str(w_row.get("name_norm") or "").split()
    if tokens:
        indexes.update(blocks["last"].get(tokens[-1], set()))
    club = str(w_row.get("club_norm") or "")
    if club:
        indexes.update(blocks["club"].get(club, set()))
    birth = str(w_row.get("birth_key") or "")
    if birth and birth != "00000":
        indexes.update(blocks["birth"].get(birth, set()))
    if not indexes and tm_count <= 5000:
        indexes.update(range(tm_count))
    return indexes


def build_match_candidates(
    wyscout_raw: pd.DataFrame,
    tm_raw: pd.DataFrame,
    *,
    config: MatchConfig | None = None,
) -> pd.DataFrame:
    config = config or MatchConfig()
    wyscout = prepare_wyscout_players(wyscout_raw)
    tm = prepare_transfermarkt_profiles(tm_raw) if "tm_name_norm" not in tm_raw.columns else tm_raw.copy()
    blocks = _block_indexes(tm)
    rows: list[dict[str, Any]] = []

    tm_by_id = tm.set_index("tm_player_id", drop=False)
    for _, w_row in wyscout.iterrows():
        existing_tm_id = str(w_row.get("existing_tm_id") or "").strip().replace(".0", "")
        if existing_tm_id and existing_tm_id.lower() not in NULL_STRINGS and existing_tm_id in tm_by_id.index:
            tm_row = tm_by_id.loc[existing_tm_id]
            if isinstance(tm_row, pd.DataFrame):
                tm_row = tm_row.iloc[0]
            evidence = score_candidate(w_row, tm_row)
            evidence["existing_tm_id"] = existing_tm_id
            evidence["previous_link_preserved"] = True
            evidence["observed_confidence_score"] = evidence["confidence_score"]
            evidence["confidence_score"] = max(float(evidence["confidence_score"]), 0.95)
            rows.append(_candidate_row(w_row, tm_row, evidence, "existing_link"))
            continue

        candidates = []
        for tm_idx in _candidate_indexes(w_row, blocks, len(tm)):
            tm_row = tm.iloc[tm_idx]
            evidence = score_candidate(w_row, tm_row)
            if evidence["confidence_score"] >= config.min_candidate_score:
                candidates.append((evidence["confidence_score"], tm_idx, evidence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, tm_idx, evidence in candidates[: config.max_candidates_per_player]:
            rows.append(_candidate_row(w_row, tm.iloc[tm_idx], evidence, "bidirectional_fuzzy"))

    if not rows:
        return pd.DataFrame()
    matches = pd.DataFrame(rows)
    matches = _assign_match_status(matches, config)
    return matches.sort_values(["player_id", "confidence_score"], ascending=[True, False], kind="stable").reset_index(drop=True)


def _candidate_row(w_row: pd.Series, tm_row: pd.Series, evidence: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "player_id": int(w_row["player_id"]),
        "wyscout_id": str(w_row.get("wyscout_id") or ""),
        "wyscout_name": w_row.get("name"),
        "wyscout_club": w_row.get("club_name"),
        "tm_player_id": str(tm_row.get("tm_player_id")),
        "tm_player_name": tm_row.get("tm_player_name"),
        "tm_club_name": tm_row.get("tm_club_name"),
        "confidence_score": evidence["confidence_score"],
        "method": method,
        "evidence": evidence,
    }


def _assign_match_status(matches: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    matches = matches.copy()
    matches["rank_for_player"] = matches.groupby("player_id")["confidence_score"].rank(method="first", ascending=False)
    matches["rank_for_tm"] = matches.groupby("tm_player_id")["confidence_score"].rank(method="first", ascending=False)
    margins = []
    for _, group in matches.groupby("player_id", sort=False):
        ordered = group.sort_values("confidence_score", ascending=False)
        best = float(ordered.iloc[0]["confidence_score"])
        second = float(ordered.iloc[1]["confidence_score"]) if len(ordered) > 1 else 0.0
        margins.extend([(idx, best - second) for idx in ordered.index])
    margin_lookup = dict(margins)
    matches["score_margin"] = matches.index.map(lambda idx: round(float(margin_lookup.get(idx, 0.0)), 4))
    matches["is_primary"] = False
    matches["status"] = "review"
    auto_mask = (
        (matches["rank_for_player"] == 1)
        & (matches["rank_for_tm"] == 1)
        & (matches["confidence_score"] >= config.auto_accept_score)
        & (
            (matches["score_margin"] >= config.auto_accept_margin)
            | (matches["method"] == "existing_link")
        )
    )
    matches.loc[auto_mask, "status"] = "accepted"
    matches.loc[auto_mask, "is_primary"] = True
    low_mask = matches["confidence_score"] < config.min_candidate_score
    matches.loc[low_mask, "status"] = "rejected"
    return matches.drop(columns=["rank_for_player", "rank_for_tm"])
