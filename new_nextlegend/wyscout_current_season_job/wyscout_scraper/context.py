from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from .config import ScraperConfig

if TYPE_CHECKING:
    from .driver import PlaywrightDriver
    from .workflow import WyscoutScraper

@dataclass
class ScraperContext:
    """Runtime context partagé entre les agents."""

    config: ScraperConfig
    driver: "PlaywrightDriver"
    logger: Callable[[str], None]
    options: Dict[str, object] = field(default_factory=dict)
    competitions: List[str] = field(default_factory=list)
    filtered_competitions: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.logger(message)

    def option(self, key: str, default: Optional[object] = None) -> Optional[object]:
        return self.options.get(key, default)

    @property
    def scraper(self) -> "WyscoutScraper":
        value = self.options.get("scraper")
        if value is None:
            raise RuntimeError("Contexte sans référence vers le scraper")
        return value  # type: ignore[return-value]
