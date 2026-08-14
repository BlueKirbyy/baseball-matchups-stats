"""Import timestamped prop markets from CSV or manual CLI arguments."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from analytics_store import initialize
from prediction_store import add_market_snapshot

REQUIRED_COLUMNS = {"captured_at", "provider", "platform_type", "player_name", "prop_type", "line"}


def _integer(value):
    return int(value) if str(value or "").strip() else None


def _boolean(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def validate_row(row):
    missing = [column for column in REQUIRED_COLUMNS if not str(row.get(column, "")).strip()]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")
    if row["platform_type"] not in {"sportsbook", "pickem"}:
        raise ValueError("platform_type must be sportsbook or pickem")
    captured_at = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    float(row["line"])
    if row["platform_type"] == "sportsbook":
        for field in ("over_price", "under_price"):
            if str(row.get(field, "")).strip():
                price = int(row[field])
                if price == 0:
                    raise ValueError(f"{field} cannot be zero")


def normalize_row(row, source):
    validate_row(row)
    return {
        "captured_at": row["captured_at"],
        "provider": row["provider"].strip(),
        "platform_type": row["platform_type"].strip(),
        "game_pk": _integer(row.get("game_pk")),
        "player_id": _integer(row.get("player_id")),
        "player_name": row["player_name"].strip(),
        "prop_type": row["prop_type"].strip(),
        "line": float(row["line"]),
        "over_price": _integer(row.get("over_price")),
        "under_price": _integer(row.get("under_price")),
        "payout_json": row.get("payout_json") or None,
        "is_closing": _boolean(row.get("is_closing")),
        "source": source,
    }


def import_csv(path, db_path=None):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        absent = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(absent))}")
        identifiers = []
        for line_number, row in enumerate(reader, start=2):
            try:
                identifiers.append(add_market_snapshot(normalize_row(row, f"csv:{path.name}"), db_path))
            except Exception as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return identifiers


def main():
    parser = argparse.ArgumentParser(description="Import immutable player-prop market snapshots")
    subparsers = parser.add_subparsers(dest="command", required=True)
    csv_parser = subparsers.add_parser("import", help="Import a CSV file")
    csv_parser.add_argument("path")
    add = subparsers.add_parser("add", help="Add one manual market snapshot")
    add.add_argument("--captured-at", required=True, help="ISO-8601 timestamp with timezone")
    add.add_argument("--provider", required=True)
    add.add_argument("--platform-type", choices=("sportsbook", "pickem"), default="sportsbook")
    add.add_argument("--game-pk", type=int)
    add.add_argument("--player-id", type=int)
    add.add_argument("--player-name", required=True)
    add.add_argument("--prop-type", default="pitcher_strikeouts")
    add.add_argument("--line", required=True, type=float)
    add.add_argument("--over-price", type=int)
    add.add_argument("--under-price", type=int)
    add.add_argument("--payout-json")
    add.add_argument("--closing", action="store_true")
    args = parser.parse_args()
    initialize()
    if args.command == "import":
        identifiers = import_csv(args.path)
        print(f"Imported {len(identifiers)} immutable market snapshot(s).")
        return
    row = vars(args)
    row["is_closing"] = row.pop("closing")
    row["captured_at"] = row.pop("captured_at")
    identifier = add_market_snapshot(normalize_row(row, "manual-cli"))
    print(f"Saved market snapshot {identifier}.")


if __name__ == "__main__":
    main()
