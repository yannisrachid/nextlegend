from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_SEASON_OPTIONS = ("Saison en cours", "Current season", "Temporada actual")
DEFAULT_COLUMN_TOGGLE_LABEL = ("Toutes les colonnes", "All columns", "Todas las columnas")
DEFAULT_APPLY_COLUMNS_LABEL = ("Appliquer", "Apply", "Aplicar")
DEFAULT_CALENDAR_PREFERENCES = ("2025/2026", "2026")
DEFAULT_CHUNK_TERMS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)
DEFAULT_COMPETITION_BLACKLIST = (
    "Toutes les compétitions seniors",
    "Toutes les compétitions",
    "Tous les championnats",
    "All senior competitions",
    "Todas las competiciones senior",
)
DEFAULT_COMPETITION_PREFIX_BLACKLIST = ("top",)
DEFAULT_COMPETITION_TOKENS_TO_CLEAR = (
    "toutes les compétitions seniors",
    "toutes les compétitions",
    "tous les championnats",
)
DEFAULT_ADVANCED_SEARCH_TITLES = (
    "advanced search",
    "recherche avancée",
    "busqueda avanzada",
    "búsqueda avanzada",
    "ricerca avanzata",
)


@dataclass
class SelectorConfig:
    """Centralised storage for the CSS/XPath selectors used in the workflow."""

    app_switcher_button: str = "a.gears-button-inner, button.gears-button-inner, [data*='switcher_apps']"
    advanced_search_tile: str = "div.drawer-title"
    season_dropdown_container: str = "div[id^='react-select'][id*='season'][id$='--value']"
    season_dropdown_input: str = "div[id^='react-select'][id*='season'] input"
    column_settings_button: str = "div[class*='custom-btn']"
    column_toggle_checkbox: str = "span[class*='checkbox-label'], span[class*='checkbox-icon']"
    column_apply_button: str = "div[class*='confirm'], button[class*='confirm']"
    competitions_dropdown_container: str = "div[id^='react-select'][id*='competition'][id$='--value']"
    competitions_dropdown_input: str = "div[id^='react-select'][id*='competition'] input"
    competitions_dropdown_arrow: str = "div[id^='react-select'][id*='competition'][id$='--value'] ~ span.Select-arrow-zone"
    competitions_options_container: str = (
        "div.Select-menu, div.Select-menu-outer, div[class*='menu'], ul[role='listbox']"
    )
    competition_selected_tokens: str = (
        "div[id^='react-select'][id*='competition'][id$='--value'] span.Select-value-label, "
        "span.Select-value-label, span[class*='multi-value__label']"
    )
    competition_option_elements: str = (
        "div[id^='react-select'][id*='competition'][id^='react-select-4--option'],"
        " div.Select-option, li[role='option'], span[role='option']"
    )
    page_size_container: str = (
        "div#react-select-3--value, "
        "div[id^='react-select'][id*='page'][id$='--value'], "
        "div[id^='react-select'][id*='size'][id$='--value'], "
        "div[class*='page-size'] div[class*='Select']"
    )
    page_size_input: str = (
        "div#react-select-3--value input, "
        "div[id^='react-select'][id*='page'][id$='--value'] input, "
        "div[id^='react-select'][id*='size'][id$='--value'] input"
    )
    page_size_option_elements: str = "div.Select-option, li[role='option'], span[role='option']"
    select_all_checkbox: str = "table thead input[type='checkbox']"
    pagination_next_button: str = "button.next--FcQn9"
    pagination_next_disabled: str = "button.next--FcQn9[disabled], button.next--FcQn9[aria-disabled='true']"
    total_results_counter: str = (
        "div.current-page--2mMXn, div[class*='total-results'], span[class*='total-results'],"
        " div[class*='results-count']"
    )
    quick_search_input: str = "input[placeholder*='Recherche'], input[placeholder*='Search']"
    export_button: str = "a.export--O6aG-, a[class*='export'], button[class*='export']"
    calendar_filter_container: str = "div.chosen-filter--2Jv8u, div[class*='chosen-filter']"
    calendar_filter_label: str = "label.filter-label--2Djgi, label"
    calendar_select_control: str = "div.Select-control"
    calendar_value_label: str = "div.Select-value-label, span.Select-value-label"
    calendar_arrow: str = "span.Select-arrow-zone, div.Select-arrow-zone"
    calendar_menu_container: str = "div.Select-menu-outer, div.Select-menu"
    calendar_option_elements: str = "div.Select-option, li[role='option'], span[role='option']"


@dataclass
class ScraperConfig:
    """Configuration object consumed by the scraper."""

    download_dir: Path = Path("data")
    output_csv_name: str = "wyscout_players.csv"
    headless: bool = False
    wait_timeout: int = 45
    wait_for_login_timeout: int = 600  # allow ample time for manual login
    chunk_search_terms: Iterable[str] = field(default_factory=lambda: list(DEFAULT_CHUNK_TERMS))
    competition_blacklist: Iterable[str] = field(
        default_factory=lambda: tuple(DEFAULT_COMPETITION_BLACKLIST)
    )
    competition_prefix_blacklist: Iterable[str] = field(
        default_factory=lambda: tuple(DEFAULT_COMPETITION_PREFIX_BLACKLIST)
    )
    competition_tokens_to_clear: Iterable[str] = field(
        default_factory=lambda: tuple(DEFAULT_COMPETITION_TOKENS_TO_CLEAR)
    )
    selectors: SelectorConfig = field(default_factory=SelectorConfig)
    season_option_labels: Iterable[str] = field(default_factory=lambda: DEFAULT_SEASON_OPTIONS)
    column_toggle_labels: Iterable[str] = field(default_factory=lambda: DEFAULT_COLUMN_TOGGLE_LABEL)
    apply_columns_labels: Iterable[str] = field(default_factory=lambda: DEFAULT_APPLY_COLUMNS_LABEL)
    calendar_preferences: Iterable[str] = field(
        default_factory=lambda: DEFAULT_CALENDAR_PREFERENCES
    )
    advanced_search_titles: Iterable[str] = field(
        default_factory=lambda: DEFAULT_ADVANCED_SEARCH_TITLES
    )
    auto_open_advanced_search: bool = False
    auto_select_all_columns: bool = False
    auto_set_current_season: bool = False
    selected_competitions_file: Optional[Path] = Path("data/competitions_selected.txt")

    def ensure_directories(self) -> None:
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        self.chunk_search_terms = list(self.chunk_search_terms)
        self.competition_blacklist = tuple(self.competition_blacklist)
        self.competition_prefix_blacklist = tuple(
            prefix.strip().lower() for prefix in self.competition_prefix_blacklist
        )
        self.competition_tokens_to_clear = tuple(
            token.strip().lower() for token in self.competition_tokens_to_clear
        )
        self.season_option_labels = tuple(self.season_option_labels)
        self.column_toggle_labels = tuple(self.column_toggle_labels)
        self.apply_columns_labels = tuple(self.apply_columns_labels)
        self.calendar_preferences = tuple(
            str(value).strip() for value in self.calendar_preferences if str(value).strip()
        )
        if not self.calendar_preferences:
            self.calendar_preferences = tuple(DEFAULT_CALENDAR_PREFERENCES)
        self.advanced_search_titles = tuple(title.lower() for title in self.advanced_search_titles)
        self.selectors.competition_option_elements = (
            self.selectors.competition_option_elements
            or "div.Select-option, div[role='option'], li[role='option']"
        )
        self.selectors.calendar_option_elements = (
            self.selectors.calendar_option_elements
            or "div.Select-option, li[role='option'], span[role='option']"
        )
        self.auto_open_advanced_search = bool(self.auto_open_advanced_search)
        self.auto_select_all_columns = bool(self.auto_select_all_columns)
        self.auto_set_current_season = bool(self.auto_set_current_season)
        if self.selected_competitions_file:
            self.selected_competitions_file = Path(self.selected_competitions_file)
            if not self.selected_competitions_file.exists():
                self.selected_competitions_file = None

    def to_json(self) -> str:
        payload: Dict[str, object] = {
            "download_dir": str(self.download_dir),
            "output_csv_name": self.output_csv_name,
            "headless": self.headless,
            "wait_timeout": self.wait_timeout,
            "wait_for_login_timeout": self.wait_for_login_timeout,
            "chunk_search_terms": list(self.chunk_search_terms),
            "competition_blacklist": list(self.competition_blacklist),
            "competition_prefix_blacklist": list(self.competition_prefix_blacklist),
            "competition_tokens_to_clear": list(self.competition_tokens_to_clear),
            "season_option_labels": list(self.season_option_labels),
            "column_toggle_labels": list(self.column_toggle_labels),
            "apply_columns_labels": list(self.apply_columns_labels),
            "calendar_preferences": list(self.calendar_preferences),
            "advanced_search_titles": list(self.advanced_search_titles),
            "auto_open_advanced_search": self.auto_open_advanced_search,
            "auto_select_all_columns": self.auto_select_all_columns,
            "auto_set_current_season": self.auto_set_current_season,
            "selected_competitions_file": str(self.selected_competitions_file)
            if self.selected_competitions_file
            else None,
        }
        payload["selectors"] = json.loads(json.dumps(self.selectors, default=lambda obj: obj.__dict__))
        return json.dumps(payload, indent=2, ensure_ascii=True)

    @classmethod
    def from_json(cls, raw: str) -> "ScraperConfig":
        data = json.loads(raw)
        selectors = SelectorConfig(**data.get("selectors", {}))
        return cls(
            download_dir=Path(data.get("download_dir", "data")),
            output_csv_name=str(data.get("output_csv_name", "wyscout_players.csv")),
            headless=bool(data.get("headless", False)),
            wait_timeout=int(data.get("wait_timeout", 45)),
            wait_for_login_timeout=int(data.get("wait_for_login_timeout", 600)),
            chunk_search_terms=data.get("chunk_search_terms") or list(DEFAULT_CHUNK_TERMS),
            competition_blacklist=data.get("competition_blacklist") or tuple(DEFAULT_COMPETITION_BLACKLIST),
            competition_prefix_blacklist=
            tuple(prefix.strip().lower() for prefix in data.get("competition_prefix_blacklist", []))
            or tuple(DEFAULT_COMPETITION_PREFIX_BLACKLIST),
            competition_tokens_to_clear=
            tuple(token.strip().lower() for token in data.get("competition_tokens_to_clear", []))
            or tuple(DEFAULT_COMPETITION_TOKENS_TO_CLEAR),
            selectors=selectors,
            season_option_labels=data.get("season_option_labels") or DEFAULT_SEASON_OPTIONS,
            column_toggle_labels=data.get("column_toggle_labels") or DEFAULT_COLUMN_TOGGLE_LABEL,
            apply_columns_labels=data.get("apply_columns_labels") or DEFAULT_APPLY_COLUMNS_LABEL,
            calendar_preferences=data.get("calendar_preferences") or DEFAULT_CALENDAR_PREFERENCES,
            advanced_search_titles=
            [title.lower() for title in data.get("advanced_search_titles", [])]
            or DEFAULT_ADVANCED_SEARCH_TITLES,
            auto_open_advanced_search=bool(data.get("auto_open_advanced_search", False)),
            auto_select_all_columns=bool(data.get("auto_select_all_columns", False)),
            auto_set_current_season=bool(data.get("auto_set_current_season", False)),
            selected_competitions_file=
            Path(data["selected_competitions_file"]).expanduser()
            if data.get("selected_competitions_file")
            else None,
        )
