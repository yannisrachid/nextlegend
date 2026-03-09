from __future__ import annotations

from ..context import ScraperContext
from .base import BaseAgent


class CompetitionCollectionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("Collect")

    def run(self, ctx: ScraperContext) -> None:
        scraper = ctx.scraper
        selected_from_file = ctx.option("selected_from_file") or []
        only = ctx.option("only")
        skip = ctx.option("skip")

        manual_competitions = None
        if only:
            manual_competitions = [name.strip() for name in only if name and name.strip()]
        elif selected_from_file:
            manual_competitions = [name.strip() for name in selected_from_file if name and name.strip()]

        if manual_competitions is not None:
            competitions = list(dict.fromkeys(manual_competitions))
            ctx.competitions = competitions
            self.log(ctx, f"{len(competitions)} compétitions fournies par configuration.")
            filtered = scraper._filter_competitions(competitions, None, skip)
        else:
            competitions = scraper._collect_competitions()
            ctx.competitions = competitions
            self.log(ctx, f"{len(competitions)} compétitions détectées avant filtrage.")
            filtered = scraper._filter_competitions(competitions, only, skip)
        ctx.filtered_competitions = filtered
        self.log(ctx, f"{len(filtered)} compétitions à traiter après filtrage.")

        if selected_from_file and manual_competitions is None:
            detected = {name.strip().lower() for name in filtered}
            missing = [
                name for name in selected_from_file if name.strip().lower() not in detected
            ]
            if missing:
                self.log(
                    ctx,
                    "Compétitions présentes dans le fichier mais non détectées : "
                    + ", ".join(missing),
                )

        list_only = bool(ctx.option("list_only", False))
        if list_only:
            scraper._output_competitions(filtered, ctx.option("list_output"))
