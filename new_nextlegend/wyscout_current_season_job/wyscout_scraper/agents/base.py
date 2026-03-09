from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

from ..context import ScraperContext


class BaseAgent(ABC):
    """Interface minimale pour les agents du scraper."""

    name: Final[str]

    def __init__(self, name: str) -> None:
        self.name = name

    def log(self, ctx: ScraperContext, message: str) -> None:
        ctx.log(f"[{self.name}] {message}")

    @abstractmethod
    def run(self, ctx: ScraperContext) -> None:
        """Execute l'agent en utilisant le contexte partagé."""
