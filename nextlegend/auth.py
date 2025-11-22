"""Simple credential-based authentication for NextLegend."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import streamlit as st
import toml
import base64

from components.sidebar import LOGO_PATH

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
CREDENTIALS_PATH = ROOT_DIR / "config" / "credentials.toml"

LOGIN_TITLE = "NextLegend by Your Legend"
LOGIN_SUBTITLE = "Scout with Intelligence with NextLegend"
LOGIN_COPY = "A tool designed by Your Legend for scouting"

AUTH_SESSION_KEY = "auth_user"
AUTH_ERROR_KEY = "auth_error"

# Environment/secret handling
APP_ENV = os.environ.get("NEXTLEGEND_ENV", "dev").strip().lower()
PRODUCTION_ENV_NAMES = {"prod", "production"}
SECRETS_CANDIDATES = [
    Path.home() / ".streamlit" / "secrets.toml",
    PROJECT_ROOT / ".streamlit" / "secrets.toml",
    ROOT_DIR / ".streamlit" / "secrets.toml",
]


def _secrets_config_available() -> bool:
    return any(path.exists() for path in SECRETS_CANDIDATES)


def _should_use_streamlit_secrets() -> bool:
    return APP_ENV in PRODUCTION_ENV_NAMES or _secrets_config_available()


def hash_password(password: str) -> str:
    """Return a SHA-256 hex digest for the provided password."""

    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _normalize_users(entries: Optional[Iterable]) -> list[dict]:
    if not entries:
        return []
    normalized: list[dict] = []
    for entry in entries:
        if isinstance(entry, dict):
            normalized.append(entry)
        else:
            try:
                normalized.append(dict(entry))
            except TypeError:
                continue
    return normalized


def _load_users_from_secrets() -> list[dict]:
    if not _should_use_streamlit_secrets():
        return []
    try:
        secrets_obj = st.secrets
    except FileNotFoundError:
        if APP_ENV in PRODUCTION_ENV_NAMES:
            st.error(
                "Streamlit secrets are required in production, but no `secrets.toml` file was found. "
                "Falling back to local credentials for now."
            )
        return []
    except Exception:
        return []

    def _get_value(container: Any, key: str) -> Any:
        if container is None:
            return None
        if isinstance(container, dict):
            return container.get(key)
        try:
            return container[key]
        except Exception:
            return None

    # Preferred structure: [credentials] with nested users
    section = _get_value(secrets_obj, "credentials")
    if section is not None:
        users = _get_value(section, "users")
        if users:
            return _normalize_users(users)

    # Fallback: allow [[users]] at the root
    users = _get_value(secrets_obj, "users")
    return _normalize_users(users)


def _load_users_from_file() -> list[dict]:
    if not CREDENTIALS_PATH.exists():
        return []
    try:
        data = toml.loads(CREDENTIALS_PATH.read_text(encoding="utf-8")) or {}
    except toml.TomlDecodeError:
        return []
    return _normalize_users(data.get("users"))


@st.cache_data(show_spinner=False)
def load_credentials(active_env: str = APP_ENV) -> Dict[str, Dict[str, str]]:
    """Load credential entries from secrets or config/credentials.toml."""

    users = _load_users_from_secrets()
    if not users:
        users = _load_users_from_file()

    result: Dict[str, Dict[str, str]] = {}
    for entry in users:
        username = str(entry.get("username", "")).strip()
        if not username:
            continue
        stored_hash = entry.get("password_hash")
        plain_password = entry.get("password")
        if not stored_hash and plain_password:
            stored_hash = hash_password(str(plain_password))
        if not stored_hash:
            continue
        result[username.lower()] = {
            "username": username,
            "password_hash": str(stored_hash),
            "display_name": entry.get("display_name") or username,
            "email": entry.get("email"),
        }
    return result


def _authenticate(username: str, password: str) -> Optional[Dict[str, str]]:
    credentials = load_credentials()
    if not credentials:
        return None
    user_entry = credentials.get(username.strip().lower())
    if not user_entry:
        return None
    provided_hash = hash_password(password)
    if provided_hash != user_entry["password_hash"]:
        return None
    return {
        "username": user_entry["username"],
        "display_name": user_entry.get("display_name", user_entry["username"]),
        "email": user_entry.get("email"),
    }


def _render_branding() -> None:
    if LOGO_PATH.exists():
        image_bytes = LOGO_PATH.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        st.markdown(
            f"""
            <div style="text-align:center; margin-bottom:0.5rem;">
                <img src="data:image/png;base64,{encoded}" style="width:170px;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='text-align:center; font-size:1.5rem; color:#e2e8f0; font-weight:600;'>Your Legend FC</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div style="text-align:center; padding: 1.5rem 0;">
            <h1 style="font-size:3rem; margin-bottom:0.2rem;">{LOGIN_TITLE}</h1>
            <h3 style="margin-bottom:0.4rem; color:#7BD389;">{LOGIN_SUBTITLE}</h3>
            <p style="font-size:1.05rem; color:#cbd5f5;">{LOGIN_COPY}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_login_form() -> None:
    _render_branding()
    credentials = load_credentials()
    if not credentials:
        st.warning(
            "No credentials detected. "
            "Add your users to Streamlit secrets under `[credentials]` (with a `users` array) "
            "or leave a local `nextlegend/config/credentials.toml`. The secrets option is recommended for deployment."
        )

    container = st.container()
    with container.form("login-form", clear_on_submit=False):
        username = st.text_input("Login", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            profile = _authenticate(username, password)
            if profile:
                st.session_state[AUTH_SESSION_KEY] = profile
                st.session_state.pop(AUTH_ERROR_KEY, None)
                st.success("Authentication successful. Redirecting…")
                st.rerun()
            else:
                st.session_state[AUTH_ERROR_KEY] = True

    if st.session_state.get(AUTH_ERROR_KEY):
        st.error("Invalid login or password. Please contact the administrator if you forgot your password.")

    st.markdown(
        "<div style='text-align:center; margin-top:2rem; font-size:0.9rem; color:#94a3b8;'>© 2025 YOUR LEGEND — All Rights Reserved.</div>",
        unsafe_allow_html=True,
    )


def require_authentication() -> Dict[str, str]:
    """Enforce authentication. Render login form when necessary."""

    user = st.session_state.get(AUTH_SESSION_KEY)
    if user:
        return user
    _render_login_form()
    st.stop()


def render_account_controls() -> None:
    """Render logout control in the sidebar."""

    user = st.session_state.get(AUTH_SESSION_KEY)
    if not user:
        return
    with st.sidebar.expander("Account", expanded=True):
        st.markdown(f"**Signed in as** {user.get('display_name', user.get('username'))}")
        if st.button("Log out", key="logout-button"):
            logout()


def logout() -> None:
    """Clear authentication state and reload the app."""

    for key in (AUTH_SESSION_KEY, AUTH_ERROR_KEY):
        if key in st.session_state:
            st.session_state.pop(key)
    st.rerun()


def current_user() -> Optional[Dict[str, str]]:
    return st.session_state.get(AUTH_SESSION_KEY)
