from __future__ import annotations

from ..context import ScraperContext
from .base import BaseAgent


class PreparationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("Preparation")

    def run(self, ctx: ScraperContext) -> None:
        scraper = ctx.scraper
        self.log(ctx, "Ouverture de l'application Wyscout…")
        scraper._open_application()
        if ctx.option("auto_login", False):
            self.log(ctx, "Tentative de connexion automatique via variables d'environnement…")
            scraper._try_auto_login_from_env()
        scraper._wait_for_manual_login()
        if ctx.config.auto_open_advanced_search:
            self.log(ctx, "Ouverture automatique d'Advanced Search…")
            scraper._open_advanced_search()
        else:
            scraper._prompt_user_preparation()

        if ctx.config.auto_set_current_season:
            self.log(ctx, "Sélection automatique de la saison en cours…")
            scraper._ensure_current_season()

        if ctx.config.auto_select_all_columns:
            self.log(ctx, "Activation automatique de toutes les colonnes…")
            scraper._enable_all_columns()
        else:
            print(
                "Assurez-vous d'avoir sélectionné toutes les colonnes de statistiques et la saison souhaitée."
            )
