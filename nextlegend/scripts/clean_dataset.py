"""Clean the raw Wyscout dataset by splitting the `player` column.

This script now supports reading/writing directly from/to S3 via
``--input-s3`` / ``--output-s3`` (or any S3 URI/key). When both S3
arguments are omitted, it behaves exactly like before using local files.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from nextlegend.s3_utils import read_csv_from_s3, write_csv_to_s3

NAME_PATTERN = re.compile(r"\(([-\d]+)\)")
AGE_PATTERN = re.compile(r"'?(\d{2})(?:\s*\((\d{1,2})\))?")


def _extract_tokens(cell: object) -> Tuple[str, Optional[str]]:
    if pd.isna(cell):
        return "", None

    text = str(cell)
    # The display name precedes the semicolon.
    base_name = text.split(";", 1)[0].strip()

    match = NAME_PATTERN.search(text)
    identifier = match.group(1) if match else None
    return base_name, identifier


def _normalise_age(cell: object) -> Tuple[Optional[int], Optional[int]]:
    if pd.isna(cell):
        return None, None

    text = str(cell).strip()
    match = AGE_PATTERN.search(text)
    if not match:
        return None, None

    two_digit_year = match.group(1)
    age_value = match.group(2)

    # Determine birth year.
    year_int = int(two_digit_year)
    birth_year = 2000 + year_int if year_int <= 24 else 1900 + year_int

    age = int(age_value) if age_value else None
    return birth_year, age


def _load_dataframe(input_path: str, *, use_s3: bool) -> pd.DataFrame:
    if use_s3:
        return read_csv_from_s3(input_path)
    local_path = Path(input_path).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {local_path}")
    return pd.read_csv(local_path)


def _save_dataframe(df: pd.DataFrame, output_path: str, *, use_s3: bool) -> None:
    if use_s3:
        write_csv_to_s3(df, output_path, index=False)
        return
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def clean_dataset(input_path: str, output_path: str, *, input_from_s3: bool = False, output_to_s3: bool = False) -> None:
    df = _load_dataframe(input_path, use_s3=input_from_s3)
    if "player" not in df.columns:
        raise KeyError("Expected column 'player' in the dataset.")

    names, identifiers = zip(*df["player"].apply(_extract_tokens))
    df["player"] = names
    df.insert(df.columns.get_loc("player") + 1, "player_id", identifiers)

    if "age" in df.columns:
        birth_years = []
        ages = []
        for value in df["age"]:
            birth, age = _normalise_age(value)
            birth_years.append(birth)
            ages.append(age)
        df["age"] = ages
        df.insert(df.columns.get_loc("age"), "birth_year", birth_years)

    _save_dataframe(df, output_path, use_s3=output_to_s3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Wyscout player column and extract IDs.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent.parent / "data" / "wyscout_players_final.csv"),
        help="Path to the raw CSV file (local).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "data" / "wyscout_players_cleaned.csv"),
        help="Path where the cleaned CSV should be saved (local).",
    )
    parser.add_argument(
        "--input-s3",
        dest="input_s3",
        default=None,
        help="Optional S3 key/URI to read the raw CSV from (overrides --input).",
    )
    parser.add_argument(
        "--output-s3",
        dest="output_s3",
        default=None,
        help="Optional S3 key/URI to write the cleaned CSV to (overrides --output).",
    )
    args = parser.parse_args()

    input_path = args.input_s3 or args.input
    output_path = args.output_s3 or args.output
    clean_dataset(
        input_path,
        output_path,
        input_from_s3=bool(args.input_s3),
        output_to_s3=bool(args.output_s3),
    )
    destination = args.output_s3 or args.output
    print(f"Cleaned dataset saved to: {destination}")


if __name__ == "__main__":
    main()
