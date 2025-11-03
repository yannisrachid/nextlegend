"""Clean the raw Wyscout dataset by splitting the `player` column."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

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


def clean_dataset(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Wyscout player column and extract IDs.")
    parser.add_argument(
        "--input",
        default=Path(__file__).resolve().parent.parent / "data" / "wyscout_players_final.csv",
        type=Path,
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).resolve().parent.parent / "data" / "wyscout_players_cleaned.csv",
        type=Path,
        help="Path where the cleaned CSV should be saved.",
    )
    args = parser.parse_args()

    clean_dataset(args.input, args.output)
    print(f"Cleaned dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
