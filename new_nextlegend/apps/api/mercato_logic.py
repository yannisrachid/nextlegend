from __future__ import annotations

from typing import Any, Optional


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


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def league_adjustment(
    competition_name: Optional[str],
    existing_strength_factor: Optional[float],
    league_meta: dict[str, dict[str, float]],
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    config = config or DEFAULT_MERCATO_LEAGUE_LEVELS
    bands = sorted(
        config.get("bands", []) or [],
        key=lambda item: float(item.get("difficulty_min", 0.0) or 0.0),
        reverse=True,
    )
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
    for override in config.get("exact_overrides", []) or []:
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
            if difficulty >= (safe_float(level.get("difficulty_min"), 0.0) or 0.0):
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


def calculate_calibrated_level(
    player: dict[str, Any],
    league_meta: dict[str, dict[str, float]],
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    raw_level = safe_float(player.get("raw_player_level"), 0.0) or 0.0
    adjustment = league_adjustment(
        player.get("competition_name"),
        safe_float(player.get("league_strength_factor")),
        league_meta,
        config,
    )
    age = safe_float(player.get("age"))
    minutes = safe_float(player.get("minutes_played"), 0.0) or 0.0
    context_bonus = 0.0
    bonus_reasons = []
    if age is not None and age <= 21 and raw_level >= 72:
        context_bonus += 2.0
        bonus_reasons.append("young high-performing profile")
    if minutes >= 1800:
        context_bonus += 2.0
        bonus_reasons.append("strong minutes sample")
    elif minutes >= 900:
        context_bonus += 1.0
        bonus_reasons.append("usable minutes sample")
    competition = str(player.get("competition_name") or "").lower()
    if "uefa champions league" in competition:
        context_bonus += 3.0
        bonus_reasons.append("Champions League context")
    elif "uefa europa" in competition:
        context_bonus += 2.0
        bonus_reasons.append("European context")
    context_bonus = clamp(context_bonus, 0.0, 8.0)
    coefficient = float(adjustment["coefficient"])
    cap = float(adjustment["cap"])
    calibrated = min(raw_level * coefficient + context_bonus, cap)
    return {
        "raw_player_level": raw_level,
        "calibrated_player_level": round(calibrated, 2),
        "league_coefficient": coefficient,
        "league_cap": cap,
        "league_level": adjustment.get("label"),
        "difficulty": adjustment.get("difficulty"),
        "context_bonus": round(context_bonus, 2),
        "bonus_reasons": bonus_reasons,
    }
