from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# Streamlit runs files inside `nextlegend/`, so modules can be imported relatively.
from s3_utils import (
    S3ConfigurationError,
    read_csv_from_s3,
    write_csv_to_s3,
)

logger = logging.getLogger(__name__)

PROSPECTS_FILE = Path("prospects_data.csv")
PROSPECTS_S3_KEY = "data/prospects_data.csv"

PROSPECT_COLUMNS = [
    "player",
    "team",
    "competition_name",
    "position",
    "role",
]


def _normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for column in PROSPECT_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA
    return working[PROSPECT_COLUMNS]


def load_prospects_csv() -> pd.DataFrame:
    try:
        df = read_csv_from_s3(PROSPECTS_S3_KEY)
        return _normalise_dataframe(df)
    except FileNotFoundError:
        pass
    except S3ConfigurationError as exc:
        logger.warning("S3 configuration missing for prospects file: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to read prospects from S3 (%s): %s", PROSPECTS_S3_KEY, exc)

    if PROSPECTS_FILE.exists():
        df = pd.read_csv(PROSPECTS_FILE)
    else:
        df = pd.DataFrame(columns=PROSPECT_COLUMNS)
    return _normalise_dataframe(df)


def save_prospects_csv(df: pd.DataFrame) -> None:
    cleaned = _normalise_dataframe(df)

    PROSPECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROSPECTS_FILE, index=False)

    try:
        write_csv_to_s3(cleaned, PROSPECTS_S3_KEY, index=False)
    except S3ConfigurationError as exc:
        logger.warning("S3 configuration missing for prospects file: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to write prospects to S3 (%s): %s", PROSPECTS_S3_KEY, exc)
