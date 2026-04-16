from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from .config import ScraperConfig


@dataclass
class PlaywrightDriver:
    """Wrapper around Playwright objects (browser/context/page)."""

    config: ScraperConfig
    browser: Browser
    context: BrowserContext
    page: Page
    _manager: object

    @classmethod
    def launch(cls, config: ScraperConfig) -> "PlaywrightDriver":
        manager = sync_playwright().start()
        launch_args = [
            "--window-size=1600,1000",
            "--disable-notifications",
            "--disable-popup-blocking",
            "--no-default-browser-check",
            "--no-first-run",
            "--force-device-scale-factor=1",
        ]
        browser_channel = (os.getenv("PLAYWRIGHT_BROWSER_CHANNEL") or "").strip() or None
        browser_executable = (os.getenv("PLAYWRIGHT_BROWSER_EXECUTABLE_PATH") or "").strip() or None
        launch_kwargs = {"headless": config.headless, "args": launch_args}
        if browser_channel:
            launch_kwargs["channel"] = browser_channel
        if browser_executable:
            launch_kwargs["executable_path"] = browser_executable
        browser = manager.chromium.launch(**launch_kwargs)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            accept_downloads=True,
        )
        timeout_ms = max(1000, int(config.wait_timeout * 1000))
        context.set_default_timeout(timeout_ms)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        return cls(config=config, browser=browser, context=context, page=page, _manager=manager)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.page.close()
        with contextlib.suppress(Exception):
            self.context.close()
        with contextlib.suppress(Exception):
            self.browser.close()
        with contextlib.suppress(Exception):
            self._manager.stop()


def create_driver(config: ScraperConfig) -> PlaywrightDriver:
    return PlaywrightDriver.launch(config)
