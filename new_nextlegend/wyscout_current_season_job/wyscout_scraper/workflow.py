from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from playwright.sync_api import Frame, Locator, Page, TimeoutError as PlaywrightTimeoutError

from .agents import (
    CompetitionCollectionAgent,
    CompetitionExportAgent,
    PreparationAgent,
)
from .config import DEFAULT_CALENDAR_PREFERENCES, ScraperConfig
from .context import ScraperContext
from .driver import PlaywrightDriver, create_driver
from .network import NetworkRecorder
from .utils import slugify


Scope = Union[Page, Frame]

_EXPORT_BASE_URL = "https://wyscout.hudl.com/app/"

_LOGIN_EMAIL_SELECTORS: Tuple[str, ...] = (
    "input[type='email']",
    "input[name='email']",
    "input[name='username']",
    "input[autocomplete='username']",
    "input[id*='email' i]",
    "input[name*='email' i]",
)
_LOGIN_PASSWORD_SELECTORS: Tuple[str, ...] = (
    "input[type='password']",
    "input[name='password']",
    "input[autocomplete='current-password']",
    "input[id*='password' i]",
    "input[name*='password' i]",
)
_LOGIN_SUBMIT_SELECTORS: Tuple[str, ...] = (
    "button[type='submit']",
    "input[type='submit']",
    "button:has-text('Log in')",
    "button:has-text('Login')",
    "button:has-text('Sign in')",
    "button:has-text('Connexion')",
    "button:has-text('Se connecter')",
    "button:has-text('Continue')",
    "button:has-text('Next')",
)
_POST_LOGIN_ACCESS_LABELS: Tuple[str, ...] = (
    "forcer l'accès",
    "forcer l’acces",
    "force access",
)
_COLUMN_PANEL_LABELS: Tuple[str, ...] = (
    "toutes les colonnes",
    "all columns",
    "todas las columnas",
)
_STRICT_AUTOCOMPLETE_COMPETITIONS: Tuple[str, ...] = (
    "Europe. UEFA Champions League",
    "Europe. UEFA Europa Conference League",
    "Europe. UEFA Europa League",
)
_ADVANCED_SEARCH_BLOCK_PATTERNS: Tuple[str, ...] = (
    "too many requests",
    "too many attempts",
    "try again later",
    "temporarily unavailable",
    "access denied",
    "forbidden",
    "rate limit",
    "limit exceeded",
    "blocked",
    "temporarily blocked",
    "réessayez plus tard",
    "reessayez plus tard",
    "temporairement indisponible",
    "accès refusé",
    "acces refuse",
    "limite",
)


class CsvBatchWriter:
    """Persists rows to CSV and appends safely when resuming an interrupted run."""

    def __init__(self, target: Path, fieldnames: Sequence[str]) -> None:
        """Open the target CSV in append-or-create mode and validate the existing header."""
        self.target = target
        self.fieldnames = list(dict.fromkeys(fieldnames))
        self.target.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
        if self.target.exists():
            with self.target.open("r", newline="", encoding="utf-8") as existing:
                reader = csv.reader(existing)
                existing_header = next(reader, None)
            if existing_header:
                if list(existing_header) != self.fieldnames:
                    raise RuntimeError(
                        "Schema CSV incompatible avec le fichier existant. "
                        "Supprimez le CSV precedent ou redemarrez avec --fresh-start."
                    )
                write_header = False
        mode = "a" if self.target.exists() and not write_header else "w"
        self.file = self.target.open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if write_header:
            self.writer.writeheader()
            self._fsync()

    def write_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        """Write one batch of rows and force a flush to disk."""
        for row in rows:
            sanitized = {field: row.get(field, "") for field in self.fieldnames}
            self.writer.writerow(sanitized)
        self._fsync()

    def close(self, success: bool = True) -> Path:
        """Close the underlying file handle and return the target path."""
        if not self.file.closed:
            self.file.flush()
            self.file.close()
        return self.target

    def _fsync(self) -> None:
        """Flush buffered writes and best-effort fsync the file descriptor."""
        self.file.flush()
        with contextlib.suppress(OSError):
            os.fsync(self.file.fileno())


class WyscoutScraper:
    """Automates Advanced Search exports on Wyscout via Playwright."""

    def __init__(self, config: Optional[ScraperConfig] = None) -> None:
        self.config = config or ScraperConfig()
        self.config.ensure_directories()
        self.driver: PlaywrightDriver = create_driver(self.config)
        self.page: Page = self.driver.page
        self._results_cache: Optional[int] = None
        self._login_timeout_ms = max(1000, int(self.config.wait_for_login_timeout * 1000))
        self._wait_timeout_ms = max(1000, int(self.config.wait_timeout * 1000))
        self._recorder = NetworkRecorder(self.config.download_dir / "network_log.jsonl")
        self._recorder.reset()
        self.page.on("request", self._recorder.record)
        self.driver.context.on("request", self._recorder.record)
        self.page.on("response", self._recorder.record_response)
        self.driver.context.on("response", self._recorder.record_response)
        self.page.on("download", self._recorder.record_download)
        self.driver.context.on("download", self._recorder.record_download)
        self._combined_csv_path = self.config.download_dir / self.config.output_csv_name
        self._csv_writer: Optional[CsvBatchWriter] = None
        self._csv_column_keys: Optional[List[str]] = None
        self._total_rows_written = 0
        self._skipped_competitions: List[str] = []
        self._page_source_dumped_once = False
        self._competition_status: Dict[str, Dict[str, str]] = {}
        self._run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._run_aborted_reason: Optional[str] = None
        self._run_aborted_competition: Optional[str] = None
        self._last_access_interruption_hint: Optional[str] = None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        prefix = time.strftime("[%H:%M:%S]")
        print(f"{prefix} [DEBUG] {message}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        only: Optional[Iterable[str]] = None,
        skip: Optional[Iterable[str]] = None,
        list_only: bool = False,
        list_output: Optional[str] = None,
        auto_login: bool = False,
    ) -> None:
        selected_from_file = self._load_selected_competitions()
        combined_only: Optional[List[str]]
        if only and selected_from_file:
            only_set = {name.strip() for name in only}
            combined_only = [name for name in selected_from_file if name in only_set]
        elif only:
            combined_only = [name.strip() for name in only]
        else:
            combined_only = selected_from_file

        options = {
            "scraper": self,
            "only": combined_only,
            "skip": [s.strip() for s in skip] if skip else None,
            "list_only": list_only,
            "list_output": list_output,
            "selected_from_file": selected_from_file,
            "auto_login": bool(auto_login),
        }

        ctx = ScraperContext(
            config=self.config,
            driver=self.driver,
            logger=self._log,
            options=options,
        )

        agents = [PreparationAgent(), CompetitionCollectionAgent()]

        try:
            for agent in agents:
                agent.run(ctx)
            if not list_only:
                CompetitionExportAgent().run(ctx)
            self._summarise_skipped_competitions()
        except Exception as exc:
            self._run_aborted_reason = repr(exc)
            raise
        finally:
            with contextlib.suppress(Exception):
                self._recorder.flush()
            with contextlib.suppress(Exception):
                self._write_run_report(ctx)
            with contextlib.suppress(Exception):
                self._close_csv_writer()
            self.shutdown()

    def shutdown(self) -> None:
        self.driver.close()

    def mark_competition_status(self, name: str, status: str, detail: Optional[str] = None) -> None:
        record: Dict[str, str] = {"status": status}
        if detail:
            record["detail"] = detail
        self._competition_status[name] = record

    def _write_run_report(self, ctx: ScraperContext) -> None:
        report_path = self.config.download_dir / "run_report.json"
        selected = ctx.option("only") or ctx.filtered_competitions or []
        payload = {
            "started_at": self._run_started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "output_csv": str(self._combined_csv_path),
            "total_rows_written": self._total_rows_written,
            "list_only": bool(ctx.option("list_only", False)),
            "aborted_reason": self._run_aborted_reason,
            "aborted_competition": self._run_aborted_competition,
            "requested_competitions": list(selected) if isinstance(selected, list) else [],
            "competition_status": self._competition_status,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Browser navigation helpers
    # ------------------------------------------------------------------

    def _open_application(self) -> None:
        self.page.goto(_EXPORT_BASE_URL, wait_until="domcontentloaded")

    def _wait_for_manual_login(self) -> None:
        if self._is_logged_in():
            self._log("Connexion déjà détectée, poursuite du script.")
            return
        if self._click_post_login_access_button():
            time.sleep(1.0)
            if self._is_logged_in():
                self._log("Connexion détectée après validation de l'accès.")
                return
        print("Connectez-vous à Wyscout dans la fenêtre Chromium qui s'est ouverte.")
        print("Le script continuera automatiquement après la connexion.")
        selectors = self._selector_list(self.config.selectors.app_switcher_button)
        deadline = time.time() + max(5, self.config.wait_for_login_timeout)
        while time.time() < deadline:
            if self._click_post_login_access_button():
                time.sleep(1.0)
                continue
            for scope in self._candidate_scopes():
                for selector in selectors:
                    locator = scope.locator(selector).first
                    try:
                        handle = locator.element_handle(timeout=500)
                    except PlaywrightTimeoutError:
                        continue
                    if handle is None:
                        continue
                    with contextlib.suppress(PlaywrightTimeoutError):
                        if locator.is_visible(timeout=200):
                            self._log("Connexion détectée, poursuite du script.")
                            return
            time.sleep(1)
        raise RuntimeError(
            "La connexion n'a pas été détectée dans le délai imparti. Relancez le script après vous être connecté."
        )

    def _try_auto_login_from_env(self) -> bool:
        email = (
            os.getenv("WYSCOUT_EMAIL")
            or os.getenv("WYSCOUT_USERNAME")
            or os.getenv("HUDL_EMAIL")
            or os.getenv("HUDL_USERNAME")
        )
        password = os.getenv("WYSCOUT_PASSWORD") or os.getenv("HUDL_PASSWORD")
        if not email or not password:
            self._log(
                "Auto-login demandé mais variables manquantes "
                "(attendu: WYSCOUT_EMAIL/WYSCOUT_PASSWORD)."
            )
            return False

        self._log("Tentative d'auto-login via variables d'environnement.")
        email_submitted = False
        password_submitted = False
        last_email_submit_at = 0.0
        last_password_submit_at = 0.0
        last_force_access_click_at = 0.0
        deadline = time.time() + min(max(10, self.config.wait_timeout), 40)

        while time.time() < deadline:
            if self._is_logged_in(timeout_ms=300):
                self._log("Auto-login: session déjà authentifiée.")
                return True

            email_info = self._find_locator(_LOGIN_EMAIL_SELECTORS, visible=True, timeout_ms=250)
            password_info = self._find_locator(_LOGIN_PASSWORD_SELECTORS, visible=True, timeout_ms=250)

            # Step 1: email -> Enter
            if not email_submitted:
                if email_info:
                    _, email_input = email_info
                    if self._fill_login_field(email_input, email):
                        email_submitted = True
                        last_email_submit_at = time.time()
                        self._press_enter(email_input)
                        self._log("Auto-login: identifiant saisi puis validé (Entrée).")
                        time.sleep(0.5)
                        continue

                time.sleep(0.25)
                continue

            # Step 2: wait for password field, then password -> Enter
            if email_submitted and not password_submitted:
                if password_info:
                    _, password_input = password_info
                    if self._fill_login_field(password_input, password):
                        password_submitted = True
                        last_password_submit_at = time.time()
                        self._press_enter(password_input)
                        self._log("Auto-login: mot de passe saisi puis validé (Entrée).")
                        time.sleep(0.8)
                        continue

                # Some flows need a second Enter on the email step to reveal password.
                if email_info and (time.time() - last_email_submit_at) > 2.0:
                    _, email_input = email_info
                    self._press_enter(email_input)
                    last_email_submit_at = time.time()
                    self._log("Auto-login: relance validation identifiant (Entrée).")
                time.sleep(0.25)
                continue

            # Step 3: post-password transitions (MFA / force access / redirect)
            if password_info and (time.time() - last_password_submit_at) > 2.0:
                _, password_input = password_info
                self._press_enter(password_input)
                last_password_submit_at = time.time()
                self._log("Auto-login: relance validation mot de passe (Entrée).")
                time.sleep(0.5)
                continue

            if (time.time() - last_force_access_click_at) > 1.5 and self._click_post_login_access_button():
                last_force_access_click_at = time.time()
                time.sleep(0.8)
                continue

            time.sleep(0.3)

        if self._find_locator(self.config.selectors.app_switcher_button, visible=True, timeout_ms=300):
            self._log("Auto-login: connexion détectée.")
            return True
        self._log("Auto-login non finalisé (MFA/SSO/captcha ou sélecteurs non reconnus).")
        return False

    def _press_enter(self, prefer_on: Optional[Locator] = None) -> bool:
        if prefer_on is not None:
            with contextlib.suppress(Exception):
                prefer_on.press("Enter")
                return True
        with contextlib.suppress(Exception):
            self.page.keyboard.press("Enter")
            return True
        return False

    def _is_logged_in(self, timeout_ms: int = 500) -> bool:
        return bool(
            self._find_locator(
                self.config.selectors.app_switcher_button,
                visible=True,
                timeout_ms=timeout_ms,
            )
        )

    def _click_post_login_access_button(self) -> bool:
        label_regex = re.compile(
            "|".join(re.escape(label) for label in _POST_LOGIN_ACCESS_LABELS),
            re.IGNORECASE,
        )
        for scope in self._candidate_scopes():
            for role in ("button", "link"):
                with contextlib.suppress(Exception):
                    locator = scope.get_by_role(role, name=label_regex)
                    if self._click_matching_locator(locator):
                        self._log("Bouton d'accès post-login détecté et cliqué.")
                        return True
            with contextlib.suppress(Exception):
                locator = scope.locator("button, a").filter(has_text=label_regex)
                if self._click_matching_locator(locator):
                    self._log("Accès post-login validé via fallback texte.")
                    return True
        return False

    def _has_post_login_access_button(self) -> bool:
        label_regex = re.compile(
            "|".join(re.escape(label) for label in _POST_LOGIN_ACCESS_LABELS),
            re.IGNORECASE,
        )
        for scope in self._candidate_scopes():
            for role in ("button", "link"):
                with contextlib.suppress(Exception):
                    locator = scope.get_by_role(role, name=label_regex)
                    if self._locator_has_visible_match(locator):
                        return True
            with contextlib.suppress(Exception):
                locator = scope.locator("button, a").filter(has_text=label_regex)
                if self._locator_has_visible_match(locator):
                    return True
        return False

    def _locator_has_visible_match(self, locator: Locator, max_count: int = 10) -> bool:
        try:
            count = locator.count()
        except Exception:
            return False
        for index in range(min(count, max_count)):
            candidate = locator.nth(index)
            try:
                if candidate.is_visible(timeout=150):
                    return True
            except Exception:
                continue
        return False

    def _find_advanced_search_block_message(self) -> Optional[str]:
        """Detect visible blocking/error text displayed in Advanced Search while rows disappear."""
        pattern_re = re.compile(
            "|".join(re.escape(p) for p in _ADVANCED_SEARCH_BLOCK_PATTERNS),
            re.IGNORECASE,
        )
        candidate_selectors = [
            "[role='alert']",
            "[role='status']",
            "div[class*='error']",
            "div[class*='warning']",
            "div[class*='alert']",
            "div[class*='empty']",
            "div[class*='message']",
            "span[class*='error']",
            "span[class*='warning']",
        ]
        for scope in self._candidate_scopes():
            for selector in candidate_selectors:
                locator = scope.locator(selector)
                try:
                    count = locator.count()
                except Exception:
                    continue
                for idx in range(min(count, 30)):
                    item = locator.nth(idx)
                    try:
                        if not item.is_visible(timeout=120):
                            continue
                    except Exception:
                        continue
                    with contextlib.suppress(Exception):
                        raw = item.inner_text(timeout=150) or item.text_content(timeout=150)
                        text = self._sanitize_cell_text(raw)
                        if text and pattern_re.search(text):
                            return text[:400]
            # Fallback: visible body text scan (slower, but catches unknown containers).
            with contextlib.suppress(Exception):
                raw_body = scope.locator("body").inner_text(timeout=1500)
                body_text = self._sanitize_cell_text(raw_body)
                if body_text and pattern_re.search(body_text):
                    # Keep only a small excerpt around the first match.
                    match = pattern_re.search(body_text)
                    if match:
                        start = max(0, match.start() - 80)
                        end = min(len(body_text), match.end() + 160)
                        return body_text[start:end]
        return None

    def _fill_login_field(self, locator: Locator, value: str) -> bool:
        if not value:
            return False
        try:
            self._focus_locator(locator)
            locator.fill("")
            locator.fill(value)
            return True
        except PlaywrightTimeoutError:
            return False
        except Exception:
            with contextlib.suppress(Exception):
                locator.click()
                locator.type(value, delay=20)
                return True
        return False

    def _submit_login_step(self, prefer_enter_on: Optional[Locator] = None) -> bool:
        for scope in self._candidate_scopes():
            for selector in _LOGIN_SUBMIT_SELECTORS:
                locator = scope.locator(selector).first
                try:
                    handle = locator.element_handle(timeout=200)
                except PlaywrightTimeoutError:
                    continue
                if handle is None:
                    continue
                try:
                    if not locator.is_visible(timeout=200):
                        continue
                except PlaywrightTimeoutError:
                    continue
                with contextlib.suppress(Exception):
                    locator.click(timeout=500)
                    return True
                with contextlib.suppress(Exception):
                    locator.click()
                    return True
        target = prefer_enter_on
        if target is not None:
            with contextlib.suppress(Exception):
                target.press("Enter")
                return True
        with contextlib.suppress(Exception):
            self.page.keyboard.press("Enter")
            return True
        return False

    def _open_advanced_search(self) -> None:
        # Already on Advanced Search (e.g. rerun with persisted session)
        if self._advanced_search_ready_quick():
            return

        button_info = self._find_locator(
            self.config.selectors.app_switcher_button,
            visible=True,
            timeout_ms=self._login_timeout_ms,
        )
        if not button_info:
            raise RuntimeError("Impossible de trouver le bouton d'ouverture du sélecteur d'applications.")
        _, button = button_info
        button.click()
        if not self._is_dialog_open():
            with contextlib.suppress(Exception):
                self.page.evaluate("el => el.click()", button)
        if self._click_advanced_search_tile():
            return
        if self._open_advanced_search_via_js():
            return
        raise RuntimeError(
            "Impossible d'ouvrir 'Advanced Search' (tuile introuvable et fallback JS indisponible)."
        )

    def _click_advanced_search_tile(self) -> bool:
        """Try multiple runtime DOM strategies for the App Switcher tile (React-safe)."""
        labels = tuple(dict.fromkeys(self.config.advanced_search_titles))
        label_regex = re.compile("|".join(re.escape(label) for label in labels if label), re.IGNORECASE)

        # Wait a bit for the app switcher content to mount/hydrate.
        # Keep this short: fallback JS is much faster/reliable than waiting ~45s on some UIs.
        deadline = time.time() + min(max(2, self.config.wait_timeout), 5)
        while time.time() < deadline:
            for scope in self._candidate_scopes():
                # 1) Most stable when translations are rendered as <span key="ADVANCED_SEARCH">
                for selector in (
                    "[key='ADVANCED_SEARCH']",
                    "[data-app='advanced_search']",
                    "[data-app-name='advanced_search']",
                    "[data-testid*='advanced' i]",
                    "[class*='advanced-search' i]",
                ):
                    locator = scope.locator(selector)
                    if self._click_matching_locator(locator):
                        if self._wait_for_advanced_search_ready_short():
                            self._log(f"Advanced Search ouvert via sélecteur: {selector}")
                            return True

                # 2) Role-based/text-based selectors against live React DOM.
                for role in ("button", "link"):
                    with contextlib.suppress(Exception):
                        locator = scope.get_by_role(role, name=label_regex)
                        if self._click_matching_locator(locator):
                            if self._wait_for_advanced_search_ready_short():
                                self._log(f"Advanced Search ouvert via rôle: {role}")
                                return True
                with contextlib.suppress(Exception):
                    locator = scope.get_by_text(label_regex)
                    if self._click_matching_locator(locator):
                        if self._wait_for_advanced_search_ready_short():
                            self._log("Advanced Search ouvert via texte visible.")
                            return True

                # 3) Existing configured tile selector as fallback.
                with contextlib.suppress(Exception):
                    tiles_locator = scope.locator(self.config.selectors.advanced_search_tile)
                    if self._click_matching_locator(tiles_locator.filter(has_text=label_regex)):
                        if self._wait_for_advanced_search_ready_short():
                            self._log("Advanced Search ouvert via selector config.advanced_search_tile.")
                            return True

            time.sleep(0.2)
        return False

    def _click_matching_locator(self, locator: Locator) -> bool:
        """Click first visible candidate, trying ancestor clickable if needed."""
        try:
            count = locator.count()
        except Exception:
            count = 0
        for index in range(min(count, 10)):
            candidate = locator.nth(index)
            try:
                handle = candidate.element_handle(timeout=200)
            except PlaywrightTimeoutError:
                continue
            if handle is None:
                continue
            visible = True
            with contextlib.suppress(PlaywrightTimeoutError):
                visible = candidate.is_visible(timeout=200)
            if not visible:
                continue
            with contextlib.suppress(Exception):
                candidate.click(timeout=500)
                return True
            with contextlib.suppress(Exception):
                candidate.evaluate(
                    """
                    (node) => {
                      const clickable = node.closest('a,button,[role="button"]') || node;
                      clickable.click();
                    }
                    """
                )
                return True
        return False

    def is_access_interrupted(self) -> bool:
        """Heuristic used to stop a run when Wyscout redirects/blocks access mid-process."""
        self._last_access_interruption_hint = None
        if self._is_logged_in(timeout_ms=150):
            # Still authenticated: could be an in-app throttle/block message.
            message = self._find_advanced_search_block_message()
            if message:
                self._last_access_interruption_hint = message
                return True
            return False
        if self._find_locator(_LOGIN_EMAIL_SELECTORS, visible=True, timeout_ms=150):
            self._last_access_interruption_hint = "login_email_visible"
            return True
        if self._find_locator(_LOGIN_PASSWORD_SELECTORS, visible=True, timeout_ms=150):
            self._last_access_interruption_hint = "login_password_visible"
            return True
        if self._has_post_login_access_button():
            self._last_access_interruption_hint = "post_login_access_button_visible"
            return True
        return False

    def _advanced_search_ready_quick(self) -> bool:
        selectors = [
            self.config.selectors.competitions_dropdown_input,
            self.config.selectors.export_button,
        ]
        for selector in selectors:
            if self._find_locator(selector, visible=True, timeout_ms=250):
                return True
        return False

    def _wait_for_advanced_search_ready_short(self, timeout_seconds: float = 4.0) -> bool:
        deadline = time.time() + max(0.5, timeout_seconds)
        while time.time() < deadline:
            if self._advanced_search_ready_quick():
                return True
            time.sleep(0.15)
        return False

    def _open_advanced_search_via_js(self) -> bool:
        """Fallback: use Wyscout internal JS API if exposed in the page runtime."""
        with contextlib.suppress(Exception):
            result = self.page.evaluate(
                """
                () => {
                  try {
                    const w = window;
                    const ae = w.ae;
                    if (ae && typeof ae.getCmp === 'function') {
                      const app = ae.getCmp('app');
                      if (app && typeof app.showAdvancedSearchPopUp === 'function') {
                        app.showAdvancedSearchPopUp({});
                        return 'showAdvancedSearchPopUp';
                      }
                    }
                    if (ae && typeof ae.cmd === 'function') {
                      ae.cmd({
                        command: 'loadApp',
                        owner: 'app',
                        type: 'component',
                        params: { appName: 'advanced_search', track: true }
                      });
                      return 'ae.cmd(loadApp)';
                    }
                    return null;
                  } catch (err) {
                    return `error:${String(err)}`;
                  }
                }
                """
            )
            if result:
                self._log(f"Fallback JS app switcher utilisé: {result}")
                return self._wait_for_advanced_search_ready_short(timeout_seconds=6.0)
        return False

    def _is_dialog_open(self) -> bool:
        selectors = [
            "div.app-switcher-dialog",
            "div.drawer-title",
            "div[class*='drawer-']",
        ]
        for selector in selectors:
            try:
                locator = self.page.locator(selector).first
                handle = locator.element_handle(timeout=300)
                if handle is None:
                    continue
                if locator.is_visible(timeout=200):
                    return True
            except PlaywrightTimeoutError:
                continue
        return False

    def _prompt_user_preparation(self) -> None:
        instructions = (
            "Merci de naviguer manuellement vers l'outil 'Advanced Search', "
            "de choisir la saison désirée et d'activer toutes les colonnes de statistiques.\n"
            "Assurez-vous que le filtre 'Compétition' est visible (cliquez dedans une fois si besoin).\n"
            "Quand tout est prêt, appuyez sur Entrée pour que l'automatisation reprenne."
        )
        print(instructions)
        try:
            input(": ")
        except EOFError:
            pass
        self._wait_until_advanced_search_ready()

    def _wait_until_advanced_search_ready(self) -> None:
        deadline = time.time() + max(5, self.config.wait_timeout)
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                if self._find_locator(self.config.selectors.competitions_dropdown_input, visible=True):
                    return
                if self._find_locator(self.config.selectors.export_button, visible=True):
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(0.5)
        message = (
            "Impossible de détecter les sélecteurs de l'Advanced Search. Vérifiez l'interface puis relancez."
        )
        if last_error:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _ensure_current_season(self) -> None:
        info = self._find_locator(self.config.selectors.season_dropdown_input, visible=True)
        if not info:
            print("Aucun contrôle de saison détecté : la sélection existante sera conservée.")
            return
        _, input_locator = info
        input_locator.click()
        for label in self.config.season_option_labels:
            input_locator.fill(label)
            input_locator.press("Enter")
            updated = self._wait_for_results_update()
            if updated is not None:
                return
        print("Saison en cours introuvable, la sélection actuelle est conservée.")
        with contextlib.suppress(Exception):
            input_locator.press("Escape")

    def _enable_all_columns(self) -> None:
        info = self._find_locator(self.config.selectors.column_settings_button, visible=True)
        if not info:
            self._log("Impossible de trouver le menu de configuration des colonnes.")
            return
        _, button = info
        button.click()
        time.sleep(0.1)
        if not self._click_all_columns_toggle():
            print("Impossible de trouver l'option 'Toutes les colonnes'.")
        if not self._click_apply_columns():
            self._log("Bouton 'Appliquer' introuvable dans le panneau des colonnes.")
        self._wait_for_results_update()

    def _click_all_columns_toggle(self) -> bool:
        label_values = tuple(dict.fromkeys((*self.config.column_toggle_labels, *_COLUMN_PANEL_LABELS)))
        label_regex = re.compile("|".join(re.escape(v) for v in label_values if v), re.IGNORECASE)

        for scope in self._candidate_scopes():
            # 1) Direct label click when label is the clickable target.
            for selector in (
                "span[class*='checkbox-label']",
                "label",
                "div[class*='checkbox']",
                "div",
            ):
                with contextlib.suppress(Exception):
                    locator = scope.locator(selector).filter(has_text=label_regex)
                    if self._click_matching_locator(locator):
                        self._log("Option 'Toutes les colonnes' activée via label.")
                        return True

            # 2) Click checkbox icon inside the labeled container.
            containers = [
                scope.locator("div[class*='checkbox']").filter(has_text=label_regex),
                scope.locator("label").filter(has_text=label_regex),
                scope.locator("div").filter(has_text=label_regex),
            ]
            for container_locator in containers:
                try:
                    count = container_locator.count()
                except Exception:
                    count = 0
                for idx in range(min(count, 10)):
                    container = container_locator.nth(idx)
                    for icon_selector in (
                        "span.checkbox-icon--FU2FX",
                        "span[class*='checkbox-icon']",
                        "input[type='checkbox']",
                    ):
                        with contextlib.suppress(Exception):
                            if self._click_matching_locator(container.locator(icon_selector)):
                                self._log("Option 'Toutes les colonnes' activée via icône checkbox.")
                                return True

            # 3) Last resort: single visible checkbox icon in the currently open panel.
            with contextlib.suppress(Exception):
                fallback_icons = scope.locator("span.checkbox-icon--FU2FX, span[class*='checkbox-icon']")
                if self._click_matching_locator(fallback_icons):
                    self._log("Fallback: clic sur une icône checkbox visible du panneau colonnes.")
                    return True
        return False

    def _click_apply_columns(self) -> bool:
        label_regex = re.compile(
            "|".join(re.escape(v) for v in self.config.apply_columns_labels if v),
            re.IGNORECASE,
        )
        for scope in self._candidate_scopes():
            for role in ("button", "link"):
                with contextlib.suppress(Exception):
                    locator = scope.get_by_role(role, name=label_regex)
                    if self._click_matching_locator(locator):
                        self._log("Panneau colonnes appliqué via rôle.")
                        return True
            for selector in (
                "div.confirm--3Lkgo",
                self.config.selectors.column_apply_button,
                "div[class*='confirm']",
                "button[class*='confirm']",
                "div",
                "button",
            ):
                with contextlib.suppress(Exception):
                    locator = scope.locator(selector).filter(has_text=label_regex)
                    if self._click_matching_locator(locator):
                        self._log("Panneau colonnes appliqué via texte/selector.")
                        return True
        return False

    def _collect_competitions(self) -> List[str]:
        input_info = self._competition_input()
        if not input_info:
            raise RuntimeError("Impossible de trouver le sélecteur de compétition.")
        _, input_locator = input_info
        self._clear_selected_competitions()
        input_locator.click()
        with contextlib.suppress(Exception):
            input_locator.press("ArrowDown")
        competitions: List[str] = []
        deadline = time.time() + self.config.wait_timeout
        options_selectors = self._selector_list(self.config.selectors.competition_option_elements)
        while time.time() < deadline:
            captured = False
            for selector in options_selectors:
                locator = self.page.locator(selector)
                try:
                    locator.wait_for(state="visible", timeout=500)
                except PlaywrightTimeoutError:
                    continue
                texts = locator.all_inner_texts()
                for text in texts:
                    label = text.strip()
                    if label and label not in competitions:
                        competitions.append(label)
                        captured = True
            if competitions:
                break
            time.sleep(0.2)
        with contextlib.suppress(Exception):
            input_locator.press("Escape")
        self._log(f"{len(competitions)} compétitions détectées.")
        return competitions

    def _competition_input(self) -> Optional[Tuple[Scope, Locator]]:
        wrapper_selectors = [
            "div#react-select-4--value",
            "div[id^='react-select'][id*='competition'][id$='--value']",
            "div[id^='react-select'][id*='competition']",
        ]
        input_selectors = [
            "input[role='combobox']",
            "input",
        ]
        for scope in self._candidate_scopes():
            for wrapper_selector in wrapper_selectors:
                wrapper = scope.locator(wrapper_selector).first
                try:
                    handle = wrapper.element_handle(timeout=200)
                except PlaywrightTimeoutError:
                    continue
                if handle is None:
                    continue
                for input_selector in input_selectors:
                    candidate = wrapper.locator(input_selector).first
                    try:
                        if candidate.element_handle(timeout=200):
                            return scope, candidate
                    except PlaywrightTimeoutError:
                        continue
        candidates: List[str] = self._selector_list(self.config.selectors.competitions_dropdown_input)
        candidates.extend(
            [
                "div[id^='react-select'][id*='competition'] input",
                "input[placeholder*='Comp']",
                "input[placeholder*='Champ']",
                "input[placeholder*='League']",
                "input[aria-label*='Comp']",
                "input[aria-label*='Champ']",
                "input[name*='competition']",
                "input[id*='competition']",
                "input[data-test*='competition']",
                "input[data-testid*='competition']",
                "input[role='combobox']",
            ]
        )
        seen = []
        for selector in candidates:
            if selector not in seen:
                seen.append(selector)
        for visible in (True, False):
            info = self._find_locator(seen, visible=visible)
            if info:
                return info
        keywords = ("compet", "champ", "league", "liga", "tournament", "tournoi")
        for scope in self._candidate_scopes():
            input_locator = scope.locator("input")
            try:
                count = input_locator.count()
            except PlaywrightTimeoutError:
                continue
            for idx in range(count):
                candidate = input_locator.nth(idx)
                try:
                    handle = candidate.element_handle(timeout=200)
                except PlaywrightTimeoutError:
                    continue
                if handle is None:
                    continue
                snapshot_parts = []
                for attr in ("placeholder", "aria-label", "name", "id", "data-testid", "data-test"):
                    value = candidate.get_attribute(attr) or ""
                    if value:
                        snapshot_parts.append(value.lower())
                snapshot = " ".join(snapshot_parts)
                if any(keyword in snapshot for keyword in keywords):
                    return scope, candidate
        return None

    def _filter_competitions(
        self,
        competitions: Iterable[str],
        only: Optional[Iterable[str]],
        skip: Optional[Iterable[str]],
    ) -> List[str]:
        comp_list = [c for c in competitions if c]
        blacklist = {name.strip().lower() for name in self.config.competition_blacklist}
        comp_list = [c for c in comp_list if c.strip().lower() not in blacklist]
        if only:
            desired = {name.strip().lower() for name in only}
            comp_list = [c for c in comp_list if c.strip().lower() in desired]
        if skip:
            blocked = {name.strip().lower() for name in skip}
            comp_list = [c for c in comp_list if c.strip().lower() not in blocked]
        if self.config.competition_prefix_blacklist:
            prefixes = tuple(self.config.competition_prefix_blacklist)
            comp_list = [c for c in comp_list if not c.strip().lower().startswith(prefixes)]
        return comp_list

    def _read_current_competitions_from_tokens(self) -> List[str]:
        selectors = self._selector_list(self.config.selectors.competition_selected_tokens)
        tokens: List[str] = []
        for scope in self._candidate_scopes():
            for selector in selectors:
                locator = scope.locator(selector)
                texts: List[str] = []
                with contextlib.suppress(PlaywrightTimeoutError):
                    texts = locator.all_inner_texts()
                for text in texts:
                    label = text.strip()
                    if label:
                        tokens.append(label)
        return tokens

    def _clear_selected_competitions(self) -> None:
        info = self._competition_input()
        if not info:
            return
        scope, input_locator = info
        deadline = time.time() + 6
        tokens_selector = self._selector_list(self.config.selectors.competition_selected_tokens)
        remove_selectors = [
            "span.Select-value-icon",
            "span[class*='multi-value__remove']",
            "button[aria-label*='Remove']",
            "button[aria-label*='Supprimer']",
        ]
        clear_all_selectors = [
            "span.Select-clear-zone",
            "div.Select-clear-zone",
            "button[aria-label*='Clear']",
            "button[aria-label*='Tout']",
        ]
        for css in clear_all_selectors:
            button = scope.locator(css).first
            with contextlib.suppress(PlaywrightTimeoutError):
                handle = button.element_handle(timeout=200)
                if handle is not None:
                    button.click(timeout=200)
                    time.sleep(0.05)
        while time.time() < deadline:
            removed_any = False
            labels: List[Locator] = []
            for selector in tokens_selector:
                locator = scope.locator(selector)
                try:
                    count = locator.count()
                except PlaywrightTimeoutError:
                    continue
                for idx in range(count):
                    labels.append(locator.nth(idx))
            if not labels:
                break
            for label in labels:
                try:
                    text = (label.inner_text(timeout=200) or "").strip()
                except PlaywrightTimeoutError:
                    continue
                if not text:
                    continue
                container = label.locator(
                    "xpath=ancestor::div[contains(@class,'Select-value') or contains(@class,'multi-value')]"
                )
                remover: Optional[Locator] = None
                for css in remove_selectors:
                    candidate = container.locator(css).first
                    try:
                        handle = candidate.element_handle(timeout=100)
                    except PlaywrightTimeoutError:
                        continue
                    if handle is not None:
                        remover = candidate
                        break
                if remover is not None:
                    try:
                        remover.click(timeout=200)
                        removed_any = True
                        time.sleep(0.05)
                        break
                    except PlaywrightTimeoutError:
                        pass
            if removed_any:
                continue
            # Fallback: clear via clavier
            self._focus_locator(input_locator)
            for combo in ("Control+A", "Meta+A"):
                with contextlib.suppress(PlaywrightTimeoutError):
                    self.page.keyboard.press(combo)
            for key in ("Delete", "Backspace"):
                with contextlib.suppress(PlaywrightTimeoutError):
                    self.page.keyboard.press(key)
            time.sleep(0.1)
            if not self._read_current_competitions_from_tokens():
                break
            for _ in range(3):
                self._focus_locator(input_locator)
                with contextlib.suppress(PlaywrightTimeoutError):
                    self.page.keyboard.press("Backspace")
                time.sleep(0.05)
            if not self._read_current_competitions_from_tokens():
                break

    def _focus_locator(self, locator: Locator) -> None:
        try:
            locator.focus(timeout=200)
            return
        except PlaywrightTimeoutError:
            pass
        handle = None
        with contextlib.suppress(PlaywrightTimeoutError):
            handle = locator.element_handle(timeout=200)
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.focus()

    def _set_page_size(self, size: int = 100) -> bool:
        container_info = self._find_locator(self.config.selectors.page_size_container, visible=True)
        if not container_info:
            self._log("Impossible de localiser le sélecteur de pagination.")
            return False
        _, container = container_info
        label = container.inner_text(timeout=500).strip()
        current_size = self._parse_first_int(label)
        if current_size is not None and current_size >= size:
            return True
        before_rows = self._read_visible_table_row_count()
        container.click()
        options = self._selector_list(self.config.selectors.page_size_option_elements)
        best_option: Optional[Tuple[int, Locator]] = None
        for scope in self._candidate_scopes():
            for selector in options:
                locator = scope.locator(selector)
                try:
                    count = locator.count()
                except Exception:
                    count = 0
                for idx in range(min(count, 30)):
                    option = locator.nth(idx)
                    try:
                        if not option.is_visible(timeout=200):
                            continue
                    except PlaywrightTimeoutError:
                        continue
                    with contextlib.suppress(Exception):
                        raw = option.inner_text(timeout=200) or option.text_content(timeout=200)
                        option_size = self._parse_first_int(raw)
                        if option_size is None:
                            continue
                        if best_option is None or option_size > best_option[0]:
                            best_option = (option_size, option)
                        if option_size == size:
                            best_option = (option_size, option)
        if best_option is not None:
            selected_size, option = best_option
            with contextlib.suppress(Exception):
                option.click(timeout=500)
                if self._wait_for_page_size_applied(container, selected_size, before_rows):
                    self._log(f"Pagination: {selected_size} lignes/page.")
                    return True
        # Fallback react-select input path (type then Enter).
        page_size_input_info = self._find_locator(self.config.selectors.page_size_input, visible=True, timeout_ms=300)
        if page_size_input_info:
            _, page_size_input = page_size_input_info
            with contextlib.suppress(Exception):
                self._focus_locator(page_size_input)
                page_size_input.fill(str(size))
                page_size_input.press("Enter")
                if self._wait_for_page_size_applied(container, size, before_rows):
                    self._log(f"Pagination: {size} lignes/page (fallback input).")
                    return True
        with contextlib.suppress(Exception):
            container.press("Escape")
        return False

    def _wait_for_page_size_applied(
        self,
        container: Locator,
        selected_size: int,
        before_rows: Optional[int],
        timeout_seconds: float = 6.0,
    ) -> bool:
        deadline = time.time() + max(0.5, timeout_seconds)
        while time.time() < deadline:
            with contextlib.suppress(Exception):
                updated = container.inner_text(timeout=200).strip()
                updated_size = self._parse_first_int(updated)
                if updated_size is not None and updated_size >= selected_size:
                    return True
            current_rows = self._read_visible_table_row_count()
            if current_rows is not None:
                if current_rows >= selected_size:
                    return True
                if before_rows is not None and current_rows != before_rows:
                    return True
            time.sleep(0.12)
        return False

    def _go_to_next_page(self, competition_name: str) -> bool:
        disabled_selectors = self._selector_list(self.config.selectors.pagination_next_disabled)
        button_locator: Optional[Locator] = None
        rows_locator: Optional[Locator] = None
        for scope in self._candidate_scopes():
            disabled = False
            for selector in disabled_selectors:
                locator = scope.locator(selector).first
                try:
                    handle = locator.element_handle(timeout=200)
                except PlaywrightTimeoutError:
                    continue
                if handle is None:
                    continue
                try:
                    if locator.is_visible(timeout=200):
                        disabled = True
                        break
                except PlaywrightTimeoutError:
                    continue
            if disabled:
                self._log(f"{competition_name}: bouton 'Next' désactivé.")
                return False
            for selector in self._selector_list(self.config.selectors.pagination_next_button):
                locator = scope.locator(selector).first
                try:
                    handle = locator.element_handle(timeout=200)
                except PlaywrightTimeoutError:
                    continue
                if handle is None:
                    continue
                try:
                    if not locator.is_visible(timeout=200):
                        continue
                except PlaywrightTimeoutError:
                    continue
                button_locator = locator
                rows_locator = scope.locator("table tbody tr")
                break
            if button_locator:
                break
        if button_locator is None or rows_locator is None:
            self._log(f"{competition_name}: bouton 'Next' introuvable.")
            return False
        before_snapshot: List[str] = []
        with contextlib.suppress(Exception):
            handles = rows_locator.element_handles()
            before_snapshot = [handle.inner_text() for handle in handles[:1]]
        button_locator.click()
        deadline = time.time() + self.config.wait_timeout
        while time.time() < deadline:
            handles = rows_locator.element_handles()
            if not handles:
                time.sleep(0.2)
                continue
            current_first = handles[0].inner_text()
            if not before_snapshot or current_first != before_snapshot[0]:
                self._log(f"{competition_name}: page suivante chargée.")
                return True
            time.sleep(0.2)
        self._log(f"{competition_name}: pagination sans changement détecté.")
        return False

    # ------------------------------------------------------------------
    # Competition processing
    # ------------------------------------------------------------------

    def _process_competition(self, name: str) -> None:
        print(f"Traitement de la compétition : {name}")
        self._select_competition(name)
        calendar_value = self._ensure_calendar_value(name)
        if not calendar_value:
            self._log(f"{name}: calendrier non conforme, compétition ignorée.")
            self._skipped_competitions.append(name)
            self.mark_competition_status(name, "skipped_calendar")
            return
        total = self._wait_for_results_update()
        if total is None or total == 0:
            print(f"Aucun joueur trouvé pour {name}.")
            self.mark_competition_status(name, "empty")
            return
        if not self._set_page_size(100):
            self._log(f"{name}: impossible de passer à 100 lignes/page, poursuite avec la valeur actuelle.")
        if total and not self._wait_for_table_rows(timeout_seconds=8.0):
            self._log(f"{name}: tableau non prêt après changement de pagination.")
            if self.is_access_interrupted():
                hint = self._last_access_interruption_hint or "unknown_block_message"
                raise RuntimeError(f"Access interrupted in Advanced Search after page-size change: {hint}")
        if total is None:
            total = self._read_total_results() or 0
        slug = slugify(name)
        column_keys: List[str] = []
        scraped = 0
        page_number = 1
        if total:
            self._log(f"{name}: {total} lignes annoncées par Wyscout.")
        while True:
            header_cells, raw_rows = self._collect_table_snapshot()
            if not raw_rows:
                # First page can still be re-rendering just after page-size / calendar updates.
                if scraped == 0 and total and self._wait_for_table_rows(timeout_seconds=6.0):
                    header_cells, raw_rows = self._collect_table_snapshot()
                if raw_rows:
                    pass
                else:
                    if total and self.is_access_interrupted():
                        hint = self._last_access_interruption_hint or "unknown_block_message"
                        raise RuntimeError(f"Access interrupted in Advanced Search with empty table: {hint}")
                    self._log("Aucune donnée détectée sur la page courante.")
                    break
            if not column_keys:
                column_keys = self._derive_column_keys(header_cells, raw_rows)
                writer = self._ensure_csv_writer(column_keys)
                self._log(
                    f"{name}: schéma détecté ({len(column_keys)} colonnes) -> "
                    + ", ".join(column_keys)
                )
            else:
                writer = self._ensure_csv_writer(column_keys)
            rows = self._build_rows_for_csv(
                column_keys,
                raw_rows,
                competition=name,
                competition_slug=slug,
                calendar_value=calendar_value,
                page_number=page_number,
                start_index=scraped,
            )
            if not rows:
                self._log("Lignes vides détectées, arrêt du traitement.")
                break
            writer.write_rows(rows)
            scraped += len(rows)
            self._total_rows_written += len(rows)
            self._log(
                f"{name}: page {page_number} -> {len(rows)} lignes sauvegardées "
                f"(cumul {scraped}" + (f"/{total}" if total else "") + ")"
            )
            if total and scraped >= total:
                break
            self._log(f"{name}: appui sur 'Next' vers la page {page_number + 1}.")
            if not self._go_to_next_page(name):
                self._log(f"{name}: pagination interrompue (bouton indisponible).")
                break
            page_number += 1
        if scraped == 0:
            self._log(f"Aucune donnée persistée pour {name}.")
            self.mark_competition_status(name, "error", "no_rows_persisted")
        elif total and scraped != total:
            self._log(f"Attention : {name} incomplet ({scraped}/{total} lignes).")
            self.mark_competition_status(name, "partial", f"{scraped}/{total}")
        else:
            self._log(f"Données enregistrées pour {name} ({scraped} lignes).")
            self.mark_competition_status(name, "success", str(scraped))

    def _select_competition(self, name: str) -> None:
        input_info = self._competition_input()
        if not input_info:
            raise RuntimeError("Champ 'Compétition' introuvable.")
        _, input_locator = input_info
        self._clear_selected_competitions()
        self._focus_locator(input_locator)
        for combo in ("Control+A", "Meta+A"):
            with contextlib.suppress(PlaywrightTimeoutError):
                input_locator.press(combo)
        with contextlib.suppress(PlaywrightTimeoutError):
            input_locator.press("Delete")
        try:
            input_locator.fill(name)
        except PlaywrightTimeoutError:
            for char in name:
                input_locator.type(char)
        time.sleep(0.08)
        strict_exact = name in _STRICT_AUTOCOMPLETE_COMPETITIONS
        if strict_exact:
            if not self._click_exact_competition_option(name, timeout_seconds=3.0):
                raise RuntimeError(
                    f"Option exacte introuvable pour la compétition '{name}' "
                    "(pour éviter une sélection erronée du type Qualification)."
                )
        else:
            input_locator.press("Enter")
        deadline = time.time() + min(max(2, self.config.wait_timeout), 6)
        while time.time() < deadline:
            tokens = self._read_current_competitions_from_tokens()
            if any(token.strip() == name for token in tokens):
                break
            time.sleep(0.2)
        # HTML dump is useful for debugging selectors, but dumping the whole page for each competition is costly.
        if not self._page_source_dumped_once:
            self._dump_page_source(name)
            self._page_source_dumped_once = True

    def _click_exact_competition_option(self, target_name: str, timeout_seconds: float = 3.0) -> bool:
        normalized_target = self._sanitize_cell_text(target_name)
        deadline = time.time() + max(0.5, timeout_seconds)
        option_selectors = self._selector_list(self.config.selectors.competition_option_elements)
        while time.time() < deadline:
            for scope in self._candidate_scopes():
                for selector in option_selectors:
                    locator = scope.locator(selector)
                    try:
                        count = locator.count()
                    except Exception:
                        count = 0
                    for idx in range(min(count, 30)):
                        option = locator.nth(idx)
                        try:
                            if not option.is_visible(timeout=150):
                                continue
                        except PlaywrightTimeoutError:
                            continue
                        with contextlib.suppress(Exception):
                            raw = option.inner_text(timeout=200) or option.text_content(timeout=200)
                            label = self._sanitize_cell_text(raw)
                            if label != normalized_target:
                                continue
                            option.click(timeout=500)
                            self._log(f"Sélection exacte de la compétition via menu: {target_name}")
                            return True
            time.sleep(0.08)
        return False

    # ------------------------------------------------------------------
    # Scraping helpers
    # ------------------------------------------------------------------

    def _ensure_csv_writer(self, column_keys: Sequence[str]) -> CsvBatchWriter:
        """Create or reuse the CSV writer while enforcing a stable schema across competitions."""
        if self._csv_column_keys is None:
            self._csv_column_keys = list(column_keys)
        elif list(column_keys) != self._csv_column_keys:
            raise RuntimeError(
                "Schéma de colonnes incohérent entre les compétitions détectées."
            )
        if self._csv_writer is None:
            fieldnames = [
                "competition_name",
                "competition_slug",
                "calendar",
                "page_number",
                "row_number",
                *column_keys,
            ]
            self._csv_writer = CsvBatchWriter(self._combined_csv_path, fieldnames)
            self._log(f"Fichier CSV cible : {self._combined_csv_path}")
        return self._csv_writer

    def _locate_calendar_control(self) -> Optional[Tuple[Scope, Locator]]:
        containers = self._selector_list(self.config.selectors.calendar_filter_container)
        label_selectors = self._selector_list(self.config.selectors.calendar_filter_label)
        control_selector = self.config.selectors.calendar_select_control
        keywords = ("calend", "calendar")
        for scope in self._candidate_scopes():
            for container_selector in containers:
                container_locator = scope.locator(container_selector)
                try:
                    count = container_locator.count()
                except PlaywrightTimeoutError:
                    continue
                for idx in range(count):
                    container = container_locator.nth(idx)
                    label_text = ""
                    for label_selector in label_selectors:
                        label = container.locator(label_selector).first
                        try:
                            label_text = (label.inner_text(timeout=200) or "").strip()
                        except PlaywrightTimeoutError:
                            continue
                        if label_text:
                            break
                    if not label_text or not any(keyword in label_text.lower() for keyword in keywords):
                        continue
                    control = container.locator(control_selector).first
                    try:
                        if control.element_handle(timeout=200):
                            return scope, control
                    except PlaywrightTimeoutError:
                        pass
                    try:
                        if container.element_handle(timeout=200):
                            return scope, container
                    except PlaywrightTimeoutError:
                        continue
        return None

    def _read_calendar_value(self, control: Locator) -> Optional[str]:
        value_selectors = self._selector_list(self.config.selectors.calendar_value_label)
        for selector in value_selectors:
            value_locator = control.locator(selector).first
            with contextlib.suppress(PlaywrightTimeoutError):
                raw = value_locator.inner_text(timeout=200) or value_locator.text_content(timeout=200)
                text = self._sanitize_cell_text(raw)
                if text:
                    return text
        fallback_locator = control.locator("div.Select-value, span.Select-value").first
        with contextlib.suppress(PlaywrightTimeoutError):
            raw = fallback_locator.inner_text(timeout=200) or fallback_locator.text_content(timeout=200)
            text = self._sanitize_cell_text(raw)
            if text:
                return text
        with contextlib.suppress(PlaywrightTimeoutError):
            raw = control.inner_text(timeout=200) or control.text_content(timeout=200)
            text = self._sanitize_cell_text(raw)
            if text:
                return text
        return None

    def _collect_calendar_options(self) -> List[Tuple[str, Locator]]:
        option_selectors = self._selector_list(self.config.selectors.calendar_option_elements)
        options: List[Tuple[str, Locator]] = []
        for scope in self._candidate_scopes():
            for selector in option_selectors:
                locator = scope.locator(selector)
                try:
                    count = locator.count()
                except PlaywrightTimeoutError:
                    continue
                for idx in range(count):
                    option = locator.nth(idx)
                    try:
                        if not option.is_visible(timeout=200):
                            continue
                    except PlaywrightTimeoutError:
                        continue
                    with contextlib.suppress(PlaywrightTimeoutError):
                        raw = option.inner_text(timeout=200) or option.text_content(timeout=200)
                    text = self._sanitize_cell_text(raw)
                    if text:
                        options.append((text, option))
        return options

    def _ensure_calendar_value(self, competition_name: str) -> Optional[str]:
        info = self._locate_calendar_control()
        if not info:
            self._log(f"{competition_name}: champ 'Calendrier' introuvable.")
            return None
        _, control = info

        preferred_calendars = tuple(self.config.calendar_preferences) or tuple(DEFAULT_CALENDAR_PREFERENCES)

        def normalize(value: str) -> str:
            return re.sub(r"\s+", "", value or "").lower()

        preferred_lookup = {normalize(value): value for value in preferred_calendars}
        current = self._read_calendar_value(control)
        if current and normalize(current) in preferred_lookup:
            resolved = preferred_lookup[normalize(current)]
            self._log(f"{competition_name}: calendrier déjà positionné sur {resolved}.")
            return resolved
        arrow_selectors = self._selector_list(self.config.selectors.calendar_arrow)
        opened = False
        for selector in arrow_selectors:
            arrow = control.locator(selector).first
            with contextlib.suppress(PlaywrightTimeoutError):
                arrow.click(timeout=300)
                opened = True
                break
            with contextlib.suppress(Exception):
                arrow.click()
                opened = True
                break
        if not opened:
            with contextlib.suppress(Exception):
                control.click()
            time.sleep(0.08)
        time.sleep(0.08)
        options = self._collect_calendar_options()
        if not options:
            self._log(f"{competition_name}: aucune option de calendrier détectée.")
            with contextlib.suppress(Exception):
                self.page.keyboard.press("Escape")
            return None
        available = [label for label, _ in options]
        selected_value: Optional[str] = None
        for preferred in preferred_calendars:
            preferred_key = normalize(preferred)
            target_locator: Optional[Locator] = None
            for label, option_locator in options:
                if normalize(label) == preferred_key:
                    target_locator = option_locator
                    break
            if target_locator:
                try:
                    target_locator.click(timeout=500)
                    time.sleep(0.08)
                    selected_value = preferred_lookup.get(preferred_key, preferred)
                    break
                except PlaywrightTimeoutError:
                    continue
        if not selected_value:
            self._log(
                f"{competition_name}: calendrier requis introuvable "
                f"(attendu: {', '.join(preferred_calendars)} ; options: {', '.join(available)})."
            )
            with contextlib.suppress(Exception):
                self.page.keyboard.press("Escape")
            return None
        updated = self._read_calendar_value(control) or selected_value
        self._log(f"{competition_name}: calendrier sélectionné -> {updated}.")
        return updated.strip()

    def _collect_table_snapshot(self) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        script = """
        () => {
            const result = { header: [], rows: [] };
            const tables = Array.from(document.querySelectorAll("table"));
            let target = null;
            for (const table of tables) {
                if (table.querySelector("td[class*='playerCell']") || table.querySelector("td[class*='search-results-table']")) {
                    target = table;
                    break;
                }
            }
            if (!target) {
                return result;
            }
            const headRows = Array.from(target.querySelectorAll("thead tr")).filter(Boolean);
            if (headRows.length) {
                const lastRow = headRows[headRows.length - 1];
                result.header = Array.from(lastRow.children).map((cell, index) => ({
                    index,
                    className: cell.getAttribute("class") || "",
                    text: (cell.innerText || "").replace(/\\s+/g, " ").trim(),
                    dataColumn: cell.getAttribute("data-column") || (cell.dataset ? cell.dataset.column : "") || "",
                    dataField: cell.getAttribute("data-field") || (cell.dataset ? cell.dataset.field : "") || "",
                }));
            }
            const bodies = Array.from(target.querySelectorAll("tbody"));
            for (const body of bodies) {
                const row = body.querySelector("tr");
                if (!row) {
                    continue;
                }
                const cells = Array.from(row.children);
                if (!cells.length) {
                    continue;
                }
                const rowData = cells.map((cell, index) => {
                    const dataset = {};
                    if (cell && cell.dataset) {
                        for (const [key, value] of Object.entries(cell.dataset)) {
                            dataset[key] = value;
                        }
                    }
                    const images = Array.from(cell.querySelectorAll("img")).map((img) => ({
                        alt: img.getAttribute("alt") || "",
                        title: img.getAttribute("title") || "",
                    }));
                    return {
                        index,
                        className: cell.getAttribute("class") || "",
                        text: (cell.innerText || "").replace(/\\s+/g, " ").trim(),
                        dataset,
                        images,
                        title: cell.getAttribute("title") || "",
                    };
                });
                if (rowData.some((cell) => (cell.text && cell.text.trim()) || (cell.images && cell.images.length))) {
                    result.rows.push(rowData);
                }
            }
            return result;
        }
        """
        for scope in self._candidate_scopes():
            try:
                snapshot = scope.evaluate(script)
            except Exception:
                continue
            if not isinstance(snapshot, dict):
                continue
            header = snapshot.get("header")
            rows = snapshot.get("rows")
            header_list = header if isinstance(header, list) else []
            rows_list = rows if isinstance(rows, list) else []
            if rows_list:
                return header_list, rows_list
        return [], []

    def _derive_column_keys(
        self,
        header_cells: Sequence[Dict[str, Any]],
        rows: Sequence[Sequence[Dict[str, Any]]],
    ) -> List[str]:
        sample_row: Sequence[Dict[str, Any]] = []
        for row in rows:
            if row:
                sample_row = row
                break
        length = max(len(header_cells), len(sample_row))
        seen: Dict[str, int] = {}
        keys: List[str] = []
        for idx in range(length):
            header_info: Dict[str, Any] = header_cells[idx] if idx < len(header_cells) else {}
            cell_info: Dict[str, Any] = sample_row[idx] if idx < len(sample_row) else {}
            candidates = [
                header_info.get("dataColumn", ""),
                header_info.get("dataField", ""),
                self._extract_key_from_class(header_info.get("className", "")),
                self._extract_key_from_class(cell_info.get("className", "")),
                header_info.get("text", ""),
            ]
            dataset = cell_info.get("dataset") or {}
            if isinstance(dataset, dict):
                candidates.extend(dataset.values())
            key = ""
            for candidate in candidates:
                key = self._normalize_identifier(candidate)
                if key:
                    break
            if not key:
                key = f"column_{idx+1:02d}"
            if key in seen:
                seen[key] += 1
                key = f"{key}_{seen[key]}"
            else:
                seen[key] = 1
            keys.append(key)
        return keys

    def _extract_key_from_class(self, class_name: str) -> str:
        if not class_name:
            return ""
        classes = [cls for cls in class_name.split() if cls]
        for cls in classes:
            if "search-results-table-column-" in cls:
                return cls.split("search-results-table-column-", 1)[-1]
        for cls in classes:
            if cls.startswith("search-results-table-"):
                suffix = cls.split("search-results-table-", 1)[-1]
                if suffix and suffix not in {"video", "column"}:
                    return suffix
        for cls in classes:
            if "playerCell" in cls:
                return "player"
            if "birthCountryCell" in cls:
                return "birth_country"
            if "passportCountryCell" in cls:
                return "passport_country"
            if "Cell--" in cls:
                return cls.split("Cell--", 1)[0]
        return ""

    def _normalize_identifier(self, raw: Any) -> str:
        if raw is None:
            return ""
        text = str(raw).strip()
        if not text:
            return ""
        text = text.replace("\u202f", " ").replace("\xa0", " ")
        text = re.sub(r"search[-_]?results[-_]?table[-_]?column-", "", text, flags=re.IGNORECASE)
        text = re.sub(r"search[-_]?results[-_]?table-", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[\\s/]+", "_", text)
        text = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", text)
        text = text.replace("-", "_")
        text = re.sub(r"__+", "_", text)
        text = text.strip("_").lower()
        adjustments = {
            "playercell": "player",
            "player_cell": "player",
            "playername": "player",
            "birthcountrycell": "birth_country",
            "birth_country_cell": "birth_country",
            "passportcountrycell": "passport_country",
            "passport_country_cell": "passport_country",
        }
        if text in adjustments:
            text = adjustments[text]
        return text

    def _compose_cell_value(self, cell: Dict[str, Any]) -> str:
        candidates: List[str] = []
        for key in ("text", "title"):
            value = self._sanitize_cell_text(cell.get(key))
            if value:
                candidates.append(value)
        images = cell.get("images") or []
        for image in images:
            if not isinstance(image, dict):
                continue
            for key in ("alt", "title"):
                value = self._sanitize_cell_text(image.get(key))
                if value:
                    candidates.append(value)
        dataset = cell.get("dataset")
        if isinstance(dataset, dict):
            for value in dataset.values():
                normalized = self._sanitize_cell_text(value)
                if normalized:
                    candidates.append(normalized)
        deduped: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return "; ".join(deduped)

    def _sanitize_cell_text(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).replace("\u202f", " ").replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _parse_first_int(self, value: Any) -> Optional[int]:
        text = self._sanitize_cell_text(value)
        if not text:
            return None
        match = re.search(r"\d+", text)
        if not match:
            return None
        with contextlib.suppress(ValueError):
            return int(match.group(0))
        return None

    def _build_rows_for_csv(
        self,
        column_keys: Sequence[str],
        raw_rows: Sequence[Sequence[Dict[str, Any]]],
        *,
        competition: str,
        competition_slug: str,
        calendar_value: str,
        page_number: int,
        start_index: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for offset, cells in enumerate(raw_rows, start=1):
            if not cells:
                continue
            if not any(self._sanitize_cell_text(cell.get("text")) for cell in cells):
                continue
            row_data: Dict[str, Any] = {
                "competition_name": competition,
                "competition_slug": competition_slug,
                "calendar": calendar_value,
                "page_number": page_number,
                "row_number": start_index + offset,
            }
            for idx, key in enumerate(column_keys):
                cell_value = ""
                if idx < len(cells):
                    cell_value = self._compose_cell_value(cells[idx])
                row_data[key] = cell_value
            rows.append(row_data)
        return rows

    # ------------------------------------------------------------------
    # Results helpers
    # ------------------------------------------------------------------

    def _wait_for_results_update(self, previous: Optional[int] = None) -> Optional[int]:
        timeout = time.time() + max(5, self.config.wait_timeout)
        while time.time() < timeout:
            current = self._read_total_results()
            if current is None:
                time.sleep(0.2)
                continue
            if previous is None or current != previous:
                self._results_cache = current
                return current
            time.sleep(0.3)
        return previous

    def _read_total_results(self) -> Optional[int]:
        selectors = self._selector_list(self.config.selectors.total_results_counter)
        for scope in self._candidate_scopes():
            for selector in selectors:
                locator = scope.locator(selector)
                text = ""
                with contextlib.suppress(PlaywrightTimeoutError):
                    text = locator.inner_text(timeout=300).strip()
                if not text:
                    continue
                digits = re.findall(r"\d+", text.replace("\u202f", "").replace(" ", ""))
                if digits:
                    return int(digits[-1])
        rows_locator = self.page.locator("table tbody tr")
        try:
            handles = rows_locator.element_handles()
        except PlaywrightTimeoutError:
            return None
        if handles:
            return len(handles)
        return None

    def _read_visible_table_row_count(self) -> Optional[int]:
        best: Optional[int] = None
        for scope in self._candidate_scopes():
            locator = scope.locator("table tbody tr")
            try:
                count = locator.count()
            except PlaywrightTimeoutError:
                continue
            if count <= 0:
                continue
            if best is None or count > best:
                best = count
        return best

    def _wait_for_table_rows(self, timeout_seconds: float = 6.0, min_rows: int = 1) -> bool:
        deadline = time.time() + max(0.5, timeout_seconds)
        while time.time() < deadline:
            row_count = self._read_visible_table_row_count()
            if row_count is not None and row_count >= min_rows:
                return True
            # Fallback via DOM snapshot in case row counting misses the right frame/table structure.
            with contextlib.suppress(Exception):
                _, raw_rows = self._collect_table_snapshot()
                if raw_rows and len(raw_rows) >= min_rows:
                    return True
            time.sleep(0.12)
        return False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_selected_competitions(self) -> Optional[List[str]]:
        path = self.config.selected_competitions_file
        if not path:
            return None
        with contextlib.suppress(FileNotFoundError, OSError):
            raw = path.read_text(encoding="utf-8")
            selections = [line.strip() for line in raw.splitlines() if line.strip()]
            if selections:
                self._log(f"{len(selections)} compétitions chargées depuis {path}.")
                return selections
        self._log(f"Aucune compétition trouvée dans {path}, poursuite sans filtre.")
        return None

    def _output_competitions(self, competitions: Iterable[str], output_path: Optional[str]) -> None:
        competitions = list(dict.fromkeys(competitions))
        print("\nListe complète des compétitions détectées :")
        for competition in competitions:
            print(f"- {competition}")
        if output_path:
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(competitions), encoding="utf-8")
            print(f"\nListe enregistrée dans {path}")

    def _close_csv_writer(self) -> None:
        if self._csv_writer is None:
            return
        try:
            success = self._total_rows_written > 0
            target = self._csv_writer.close(success=success)
            if success:
                self._log(
                    f"CSV final disponible : {target} ({self._total_rows_written} lignes agrégées)."
                )
        finally:
            self._csv_writer = None
            self._csv_column_keys = None
            self._total_rows_written = 0

    def _summarise_skipped_competitions(self) -> None:
        if not self._skipped_competitions:
            return
        unique = list(dict.fromkeys(self._skipped_competitions))
        print("\nCompétitions ignorées :")
        for competition in unique:
            print(f"- {competition}")
        error_path = self.config.download_dir / "error_leagues.txt"
        error_path.write_text("\n".join(unique), encoding="utf-8")
        self._log(f"Liste des compétitions ignorées enregistrée dans {error_path}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _selector_list(self, raw: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(raw, str):
            parts = raw.split(",")
        else:
            parts = []
            for entry in raw:
                parts.extend(entry.split(","))
        return [part.strip() for part in parts if part and part.strip()]

    def _candidate_scopes(self) -> List[Scope]:
        scopes: List[Scope] = []
        for page in self.page.context.pages:
            scopes.append(page)
            main_frame = page.main_frame
            for frame in page.frames:
                if frame is main_frame:
                    continue
                if frame.is_detached():
                    continue
                scopes.append(frame)
        return scopes

    def _find_locator(
        self,
        selectors: Union[str, Sequence[str]],
        *,
        visible: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> Optional[Tuple[Scope, Locator]]:
        timeout = timeout_ms or 500
        for scope in self._candidate_scopes():
            for selector in self._selector_list(selectors):
                locator = scope.locator(selector)
                try:
                    handle = locator.first.element_handle(timeout=timeout)
                except PlaywrightTimeoutError:
                    continue
                if handle is None:
                    continue
                if visible:
                    try:
                        if not locator.first.is_visible(timeout=timeout):
                            continue
                    except PlaywrightTimeoutError:
                        continue
                return scope, locator.first
        return None

    def _dump_page_source(self, name: str) -> None:
        try:
            content = self.page.content()
        except PlaywrightTimeoutError:
            return
        path = self.config.download_dir / "source_competition.html"
        header = f"<!-- {name} -->\n"
        with contextlib.suppress(OSError):
            path.write_text(header + content, encoding="utf-8")
