from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any


_JSON_BLOCK = re.compile(r"\b[A-Za-z_$][\w$]*=JSON\.parse\(`(.*?)`\)", re.S)
_UPDATED = re.compile(
    r'([A-Za-z_$][\w$]*)="([A-Z][a-z]{2} \d{1,2}, \d{4})",\s*'
    r'[A-Za-z_$][\w$]*=\{lastUpdated:\1\}'
)


def parse_rankings(bundle: str, gender: str) -> list[dict[str, Any]]:
    datasets: list[list[dict[str, Any]]] = []
    for match in _JSON_BLOCK.finditer(bundle):
        raw = match.group(1)
        if '"contestantId"' not in raw[:1000] or '"currentRating"' not in raw[:2000]:
            continue
        for candidate in (raw, raw.replace('\\\\"', '\\"')):
            try:
                rows = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(rows, list)
                and rows
                and isinstance(rows[0], dict)
                and {"contestantId", "contestantName", "currentGlobalRank", "currentRating"}.issubset(rows[0])
            ):
                datasets.append(rows)
                break

    datasets.sort(key=len, reverse=True)
    if len(datasets) < 2 or len(datasets[0]) < len(datasets[1]) * 1.25:
        raise RuntimeError("Cannot distinguish men's and women's Opta ranking datasets.")
    if gender == "mens":
        return datasets[0]
    if gender == "womens":
        return datasets[1]
    raise ValueError(f"Unsupported gender: {gender}")


def parse_last_updated(bundle: str) -> tuple[str | None, str | None]:
    match = _UPDATED.search(bundle)
    if not match:
        return None, None
    label = match.group(2)
    try:
        parsed = datetime.strptime(label, "%b %d, %Y").date().isoformat()
    except ValueError:
        return label, None
    return label, parsed
