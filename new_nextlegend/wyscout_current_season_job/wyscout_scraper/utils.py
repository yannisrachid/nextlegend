from __future__ import annotations

import re


_SLUG_RE = re.compile(r"[^\w\d]+")


def slugify(value: str, max_length: int = 100) -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    if not slug:
        slug = "export"
    return slug[:max_length]
