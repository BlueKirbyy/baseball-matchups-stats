"""Persistence helpers for immutable market and model records."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from analytics_store import connect


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def is_before_start(captured_at, scheduled_start):
    if not scheduled_start:
        return False
    try:
        return parse_timestamp(captured_at) < parse_timestamp(scheduled_start)
    except (TypeError, ValueError):
        return False


def record_game(game, observed_at=None, db_path=None):
    observed_at = observed_at or utc_now()
    with connect(db_path) as db:
        db.execute(
            """INSERT INTO games(
                 game_pk, scheduled_start, official_date, away_team_id,
                 away_team_name, home_team_id, home_team_name, venue_name,
                 first_seen_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(game_pk) DO UPDATE SET
                 scheduled_start=excluded.scheduled_start,
                 official_date=excluded.official_date,
                 away_team_id=excluded.away_team_id,
                 away_team_name=excluded.away_team_name,
                 home_team_id=excluded.home_team_id,
                 home_team_name=excluded.home_team_name,
                 venue_name=excluded.venue_name,
                 updated_at=excluded.updated_at""",
            (
                game["game_pk"], game.get("scheduled_start"), game.get("official_date"),
                game.get("away_team_id"), game.get("away_team_name"),
                game.get("home_team_id"), game.get("home_team_name"),
                game.get("venue_name"), observed_at, observed_at,
            ),
        )


def record_pregame_snapshot(game_pk, captured_at, scheduled_start, probable, lineups, db_path=None):
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT INTO pregame_snapshots(
                 game_pk, captured_at, scheduled_start, away_probable_pitcher_id,
                 home_probable_pitcher_id, away_lineup_json, home_lineup_json,
                 away_lineup_confirmed, home_lineup_confirmed
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_pk, captured_at, scheduled_start,
                (probable.get("away") or {}).get("id"), (probable.get("home") or {}).get("id"),
                json.dumps(lineups.get("away") or []), json.dumps(lineups.get("home") or []),
                int(bool(lineups.get("away"))), int(bool(lineups.get("home"))),
            ),
        )
        return cursor.lastrowid


def add_market_snapshot(row, db_path=None):
    now = utc_now()
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT INTO market_snapshots(
                 captured_at, provider, platform_type, game_pk, player_id,
                 player_name, prop_type, line, over_price, under_price,
                 payout_json, is_closing, source, imported_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.get("captured_at") or now, row["provider"], row.get("platform_type", "sportsbook"),
                row.get("game_pk"), row.get("player_id"), row["player_name"], row["prop_type"],
                float(row["line"]), row.get("over_price"), row.get("under_price"),
                json.dumps(row.get("payout")) if row.get("payout") is not None else row.get("payout_json"),
                int(bool(row.get("is_closing"))), row.get("source", "manual"), now,
            ),
        )
        return cursor.lastrowid


def latest_markets(game_pk, prop_type="pitcher_strikeouts", as_of=None, player_id=None, player_name=None, db_path=None):
    as_of = as_of or utc_now()
    player_clause = "1=1"
    player_params = []
    if player_id is not None:
        player_clause = "(candidate.player_id=? OR (candidate.player_id IS NULL AND lower(candidate.player_name)=lower(?)))"
        player_params = [player_id, player_name or ""]
    query = f"""
      SELECT candidate.* FROM market_snapshots candidate
      WHERE candidate.game_pk=? AND candidate.prop_type=? AND candidate.captured_at<=?
        AND {player_clause}
        AND candidate.market_snapshot_id=(
          SELECT latest.market_snapshot_id FROM market_snapshots latest
          WHERE latest.game_pk=candidate.game_pk AND latest.prop_type=candidate.prop_type
            AND latest.provider=candidate.provider AND latest.platform_type=candidate.platform_type
            AND (latest.player_id=candidate.player_id OR (
              latest.player_id IS NULL AND candidate.player_id IS NULL
              AND lower(latest.player_name)=lower(candidate.player_name)
            ))
            AND latest.captured_at<=?
          ORDER BY latest.captured_at DESC, latest.market_snapshot_id DESC LIMIT 1
        )
      ORDER BY candidate.provider, candidate.platform_type
    """
    params = [game_pk, prop_type, as_of, *player_params, as_of]
    with connect(db_path) as db:
        rows = [dict(row) for row in db.execute(query, params)]
    chosen = {}
    for row in rows:
        key = (row["provider"], row["platform_type"])
        current = chosen.get(key)
        row_exact = player_id is not None and row.get("player_id") == player_id
        current_exact = current and player_id is not None and current.get("player_id") == player_id
        if current is None or (row_exact and not current_exact) or (row_exact == current_exact and row["captured_at"] > current["captured_at"]):
            chosen[key] = row
    return list(chosen.values())


def save_prediction(game_pk, prediction, market=None, data_freshness_seconds=None, db_path=None):
    """Append a prediction; intentionally no update path exists."""
    market = market or prediction.get("market") or {}
    if data_freshness_seconds is None:
        data_freshness_seconds = prediction.get("data_freshness_seconds")
    if not is_before_start(prediction["as_of"], prediction.get("scheduled_start")):
        raise ValueError("prediction as_of must be earlier than scheduled_start")
    with connect(db_path) as db:
        cursor = db.execute(
            """INSERT INTO model_predictions(
                 created_at, as_of, game_pk, scheduled_start, player_id, player_name,
                 prop_type, model_version, feature_version, projection, median,
                 interval_low, interval_high, probability_over, probability_under,
                 fair_over_price, fair_under_price, market_snapshot_id, no_vig_over,
                 expected_value_over, expected_value_under, confidence,
                 arsenal_coverage, effective_sample_size, lineup_confirmed,
                 data_freshness_seconds, decision, factors_json, inputs_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                utc_now(), prediction["as_of"], game_pk, prediction.get("scheduled_start"),
                prediction["player_id"], prediction["player_name"], prediction["prop_type"],
                prediction["model_version"], prediction["feature_version"], prediction["projection"],
                prediction["median"], prediction["interval_low"], prediction["interval_high"],
                prediction.get("probability_over"), prediction.get("probability_under"),
                prediction.get("fair_over_price"), prediction.get("fair_under_price"),
                market.get("market_snapshot_id"), market.get("no_vig_over"),
                market.get("expected_value_over"), market.get("expected_value_under"),
                prediction["confidence"], prediction["arsenal_coverage"],
                prediction["effective_sample_size"], int(prediction["lineup_confirmed"]),
                data_freshness_seconds, prediction["decision"],
                json.dumps(prediction.get("factors", [])), json.dumps(prediction, sort_keys=True),
            ),
        )
        return cursor.lastrowid


def settle_prediction(prediction_id, actual_value, source="manual", settled_at=None, db_path=None):
    """Settle once; replacing a result would undermine the audit trail."""
    with connect(db_path) as db:
        db.execute(
            "INSERT INTO prediction_results(prediction_id, settled_at, actual_value, result_source) VALUES (?, ?, ?, ?)",
            (prediction_id, settled_at or utc_now(), float(actual_value), source),
        )
