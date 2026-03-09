from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from playwright.sync_api import Download, Request, Response


@dataclass
class NetworkRecorder:
    """Capture selected Playwright requests for offline inspection."""

    output_path: Path
    max_entries: int = 500
    entries: List[Dict[str, object]] = field(default_factory=list)

    def record(self, request: Request) -> None:
        url = request.url
        if not any(token in url for token in ("wyscout", "statsbomb")):
            return
        entry = {
            "method": request.method,
            "url": url,
            "headers": dict(request.headers),
        }
        with contextlib.suppress(Exception):
            if request.post_data:
                entry["post_data"] = request.post_data
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def flush(self) -> None:
        if not self.entries:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in self.entries)
        self.output_path.write_text(payload, encoding="utf-8")

    def reset(self) -> None:
        self.entries.clear()
        with contextlib.suppress(OSError):
            if self.output_path.exists():
                self.output_path.unlink()

    def record_response(self, response: Response) -> None:
        url = response.url
        if not any(token in url for token in ("wyscout", "statsbomb")):
            return
        info: Dict[str, object] = {
            "event": "response",
            "url": url,
            "status": response.status,
            "headers": dict(response.headers),
        }
        content_disposition = response.headers.get("content-disposition", "")
        if "attachment" in content_disposition or "excel" in response.headers.get("content-type", ""):
            with contextlib.suppress(Exception):
                body = response.body()
                info["body_sample"] = body[:512].decode("latin-1", errors="ignore")
        self.entries.append(info)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def record_download(self, download: Download) -> None:
        info: Dict[str, object] = {
            "event": "download",
            "suggested_filename": download.suggested_filename,
        }
        with contextlib.suppress(Exception):
            info["url"] = download.url
        with contextlib.suppress(Exception):
            info["content_type"] = download.headers.get("content-type")
        self.entries.append(info)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
