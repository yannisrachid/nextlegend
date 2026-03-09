from __future__ import annotations

from ..context import ScraperContext
from .base import BaseAgent


class CompetitionExportAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("Export")

    def run(self, ctx: ScraperContext) -> None:
        if ctx.option("list_only", False):
            self.log(ctx, "Mode liste uniquement, aucun export lancé.")
            return
        scraper = ctx.scraper
        consecutive_failures = 0
        for competition in ctx.filtered_competitions:
            self.log(ctx, f"Traitement de la compétition : {competition}")
            try:
                scraper._process_competition(competition)
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                scraper.mark_competition_status(competition, "error", exc.__class__.__name__)
                self.log(ctx, f"Erreur lors du traitement de '{competition}': {exc!r}")
                if scraper.is_access_interrupted():
                    hint = getattr(scraper, "_last_access_interruption_hint", None)
                    detail = f"{exc.__class__.__name__}:{hint}" if hint else exc.__class__.__name__
                    scraper.mark_competition_status(competition, "blocked", detail)
                    scraper._run_aborted_competition = competition
                    raise RuntimeError(
                        f"Interruption d'accès détectée pendant '{competition}'. Arrêt du run pour reprise."
                    ) from exc
                if consecutive_failures >= 3:
                    scraper._run_aborted_competition = competition
                    raise RuntimeError(
                        f"Arrêt de sécurité après {consecutive_failures} échecs consécutifs "
                        f"(dernier: '{competition}')."
                    ) from exc
                continue
