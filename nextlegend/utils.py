from pathlib import Path

import pandas as pd

PROSPECTS_FILE = Path("prospects_data.csv")

PROSPECT_COLUMNS = [
    "player",
    "team",
    "competition_name",
    "position",
    "role",
]


def load_prospects_csv() -> pd.DataFrame:
    if PROSPECTS_FILE.exists():
        df = pd.read_csv(PROSPECTS_FILE)
    else:
        df = pd.DataFrame(columns=PROSPECT_COLUMNS)
    for column in PROSPECT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[PROSPECT_COLUMNS]


def save_prospects_csv(df: pd.DataFrame) -> None:
    df = df[PROSPECT_COLUMNS]
    df.to_csv(PROSPECTS_FILE, index=False)
