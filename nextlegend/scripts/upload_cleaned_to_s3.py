#!/usr/bin/env python3
"""Upload the cleaned Wyscout dataset to S3 using project utilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from nextlegend.s3_utils import object_exists, write_csv_to_s3

DEFAULT_LOCAL = PROJECT_ROOT / "data" / "wyscout_players_cleaned.csv"
DEFAULT_KEY = "data/wyscout_players_cleaned.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the cleaned Wyscout CSV to S3."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_LOCAL),
        help="Local path to the cleaned CSV (default: nextlegend/data/wyscout_players_cleaned.csv).",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help="Destination S3 key (default: data/wyscout_players_cleaned.csv).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the existence check after upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_path = Path(args.input).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(f"Local CSV not found: {local_path}")

    df = pd.read_csv(local_path)
    write_csv_to_s3(df, args.key, index=False)
    print(f"Uploaded {local_path} -> s3://<bucket>/{args.key}")

    if not args.skip_verify:
        exists = object_exists(args.key)
        status = "OK" if exists else "MISSING"
        print(f"Verification ({args.key}): {status}")


if __name__ == "__main__":
    main()
