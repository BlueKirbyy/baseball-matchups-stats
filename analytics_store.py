"""SQLite storage and non-destructive migrations for Diamond Intel."""
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).with_name("diamond_intel.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sync_runs (
  season INTEGER PRIMARY KEY,
  synced_at TEXT NOT NULL,
  pitcher_count INTEGER NOT NULL,
  batter_count INTEGER NOT NULL,
  failed_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pitcher_arsenal (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pitch_code TEXT NOT NULL,
  pitch_name TEXT NOT NULL,
  pitches INTEGER NOT NULL,
  usage REAL NOT NULL,
  velo REAL,
  whiff_rate REAL,
  PRIMARY KEY (season, player_id, pitch_code)
);
CREATE TABLE IF NOT EXISTS batter_season_stats (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pa INTEGER NOT NULL,
  avg TEXT NOT NULL,
  hr INTEGER NOT NULL,
  PRIMARY KEY (season, player_id)
);
CREATE TABLE IF NOT EXISTS batter_pitch_stats (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pitch_code TEXT NOT NULL,
  pa INTEGER NOT NULL,
  avg TEXT NOT NULL,
  hr INTEGER NOT NULL,
  PRIMARY KEY (season, player_id, pitch_code)
);
CREATE TABLE IF NOT EXISTS matchup_sync_runs (
  game_pk INTEGER PRIMARY KEY,
  season INTEGER NOT NULL,
  synced_at TEXT NOT NULL,
  feed_count INTEGER NOT NULL,
  failed_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS gameday_pitcher_arsenal (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pitch_code TEXT NOT NULL,
  pitch_name TEXT NOT NULL,
  pitches INTEGER NOT NULL,
  usage REAL NOT NULL,
  velo REAL,
  zones TEXT NOT NULL,
  PRIMARY KEY (season, player_id, pitch_code)
);
CREATE TABLE IF NOT EXISTS gameday_batter_season (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pa INTEGER NOT NULL,
  avg TEXT NOT NULL,
  hr INTEGER NOT NULL,
  PRIMARY KEY (season, player_id)
);
CREATE TABLE IF NOT EXISTS gameday_batter_pitch (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pitch_code TEXT NOT NULL,
  pa INTEGER NOT NULL,
  avg TEXT NOT NULL,
  hr INTEGER NOT NULL,
  zones TEXT NOT NULL,
  PRIMARY KEY (season, player_id, pitch_code)
);
CREATE TABLE IF NOT EXISTS gameday_batter_pitch_velocity (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pitch_code TEXT NOT NULL,
  velo_bucket INTEGER NOT NULL,
  pa INTEGER NOT NULL,
  at_bats INTEGER NOT NULL,
  hits INTEGER NOT NULL,
  hr INTEGER NOT NULL,
  strikeouts INTEGER NOT NULL DEFAULT 0,
  outs INTEGER NOT NULL DEFAULT 0,
  doubles INTEGER NOT NULL DEFAULT 0,
  triples INTEGER NOT NULL DEFAULT 0,
  total_bases INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (season, player_id, pitch_code, velo_bucket)
);
CREATE TABLE IF NOT EXISTS gameday_batter_pitch_context (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pitch_code TEXT NOT NULL,
  velo_bucket INTEGER NOT NULL,
  pitcher_throws TEXT NOT NULL,
  count_bucket TEXT NOT NULL,
  zone INTEGER NOT NULL,
  pa INTEGER NOT NULL,
  at_bats INTEGER NOT NULL,
  hits INTEGER NOT NULL,
  hr INTEGER NOT NULL,
  strikeouts INTEGER NOT NULL,
  outs INTEGER NOT NULL,
  doubles INTEGER NOT NULL,
  triples INTEGER NOT NULL,
  total_bases INTEGER NOT NULL,
  PRIMARY KEY (season, player_id, pitch_code, velo_bucket, pitcher_throws, count_bucket, zone)
);
CREATE INDEX IF NOT EXISTS idx_batter_pitch_context_lookup
  ON gameday_batter_pitch_context(season, player_id, pitch_code, pitcher_throws, velo_bucket);
CREATE TABLE IF NOT EXISTS gameday_pitcher_arsenal_context (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pitch_code TEXT NOT NULL,
  count_bucket TEXT NOT NULL,
  zone INTEGER NOT NULL,
  pitches INTEGER NOT NULL,
  PRIMARY KEY (season, player_id, pitch_code, count_bucket, zone)
);
CREATE TABLE IF NOT EXISTS gameday_pitcher_workload (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  appearances INTEGER NOT NULL,
  batters_faced INTEGER NOT NULL,
  strikeouts INTEGER NOT NULL,
  outs INTEGER NOT NULL,
  pitches INTEGER NOT NULL,
  recent_appearances INTEGER NOT NULL,
  recent_batters_faced INTEGER NOT NULL,
  recent_strikeouts INTEGER NOT NULL,
  recent_outs INTEGER NOT NULL,
  recent_pitches INTEGER NOT NULL,
  PRIMARY KEY (season, player_id)
);
CREATE TABLE IF NOT EXISTS gameday_batter_pitch_quality (
  season INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pitch_code TEXT NOT NULL,
  pitches INTEGER NOT NULL,
  swings INTEGER NOT NULL,
  whiffs INTEGER NOT NULL,
  chase_swings INTEGER NOT NULL,
  batted_balls INTEGER NOT NULL,
  hard_hits INTEGER NOT NULL,
  barrel_proxy INTEGER NOT NULL,
  strikeouts INTEGER NOT NULL,
  PRIMARY KEY (season, player_id, pitch_code)
);
CREATE TABLE IF NOT EXISTS gameday_umpire_game (
  season INTEGER NOT NULL,
  game_pk INTEGER NOT NULL,
  umpire_id INTEGER NOT NULL,
  umpire_name TEXT NOT NULL,
  batters_faced INTEGER NOT NULL,
  strikeouts INTEGER NOT NULL,
  walks INTEGER NOT NULL,
  pitches INTEGER NOT NULL,
  PRIMARY KEY (season, game_pk, umpire_id)
);
CREATE TABLE IF NOT EXISTS odds_slate_cache (
  cache_key TEXT PRIMARY KEY,
  checked_at REAL NOT NULL,
  changed_at REAL NOT NULL,
  payload TEXT NOT NULL,
  message TEXT
);
"""

MIGRATIONS = (
    (1, """
    CREATE TABLE IF NOT EXISTS games (
      game_pk INTEGER PRIMARY KEY,
      scheduled_start TEXT,
      official_date TEXT,
      away_team_id INTEGER,
      away_team_name TEXT,
      home_team_id INTEGER,
      home_team_name TEXT,
      venue_name TEXT,
      first_seen_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS player_game_observations (
      game_pk INTEGER NOT NULL,
      game_date TEXT NOT NULL,
      player_id INTEGER NOT NULL,
      player_name TEXT,
      team_id INTEGER,
      opponent_id INTEGER,
      role TEXT NOT NULL CHECK (role IN ('pitcher', 'batter')),
      is_start INTEGER NOT NULL DEFAULT 0 CHECK (is_start IN (0, 1)),
      throws TEXT,
      stands TEXT,
      batters_faced INTEGER,
      strikeouts INTEGER,
      outs INTEGER,
      pitches INTEGER,
      plate_appearances INTEGER,
      at_bats INTEGER,
      hits INTEGER,
      total_bases INTEGER,
      home_runs INTEGER,
      batter_strikeouts INTEGER,
      observed_at TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'mlb-gameday',
      PRIMARY KEY (game_pk, player_id, role)
    );
    CREATE INDEX IF NOT EXISTS idx_observations_player_date
      ON player_game_observations(player_id, role, game_date);
    CREATE TABLE IF NOT EXISTS pregame_snapshots (
      snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
      game_pk INTEGER NOT NULL,
      captured_at TEXT NOT NULL,
      scheduled_start TEXT,
      away_probable_pitcher_id INTEGER,
      home_probable_pitcher_id INTEGER,
      away_lineup_json TEXT,
      home_lineup_json TEXT,
      away_lineup_confirmed INTEGER NOT NULL DEFAULT 0,
      home_lineup_confirmed INTEGER NOT NULL DEFAULT 0,
      source TEXT NOT NULL DEFAULT 'mlb-gameday'
    );
    CREATE INDEX IF NOT EXISTS idx_pregame_game_time
      ON pregame_snapshots(game_pk, captured_at);
    """),
    (2, """
    CREATE TABLE IF NOT EXISTS market_snapshots (
      market_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
      captured_at TEXT NOT NULL,
      provider TEXT NOT NULL,
      platform_type TEXT NOT NULL CHECK (platform_type IN ('sportsbook', 'pickem')),
      game_pk INTEGER,
      player_id INTEGER,
      player_name TEXT NOT NULL,
      prop_type TEXT NOT NULL,
      line REAL NOT NULL,
      over_price INTEGER,
      under_price INTEGER,
      payout_json TEXT,
      is_closing INTEGER NOT NULL DEFAULT 0 CHECK (is_closing IN (0, 1)),
      source TEXT NOT NULL,
      imported_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_market_lookup
      ON market_snapshots(game_pk, player_id, prop_type, captured_at);
    CREATE TABLE IF NOT EXISTS model_predictions (
      prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL,
      as_of TEXT NOT NULL,
      game_pk INTEGER NOT NULL,
      scheduled_start TEXT,
      player_id INTEGER NOT NULL,
      player_name TEXT NOT NULL,
      prop_type TEXT NOT NULL,
      model_version TEXT NOT NULL,
      feature_version TEXT NOT NULL,
      projection REAL NOT NULL,
      median REAL NOT NULL,
      interval_low REAL NOT NULL,
      interval_high REAL NOT NULL,
      probability_over REAL,
      probability_under REAL,
      fair_over_price INTEGER,
      fair_under_price INTEGER,
      market_snapshot_id INTEGER,
      no_vig_over REAL,
      expected_value_over REAL,
      expected_value_under REAL,
      confidence TEXT NOT NULL,
      arsenal_coverage REAL NOT NULL,
      effective_sample_size REAL NOT NULL,
      lineup_confirmed INTEGER NOT NULL CHECK (lineup_confirmed IN (0, 1)),
      data_freshness_seconds REAL,
      decision TEXT NOT NULL,
      factors_json TEXT NOT NULL,
      inputs_json TEXT NOT NULL,
      FOREIGN KEY (market_snapshot_id) REFERENCES market_snapshots(market_snapshot_id)
    );
    CREATE INDEX IF NOT EXISTS idx_predictions_game_player
      ON model_predictions(game_pk, player_id, prop_type, as_of);
    CREATE TABLE IF NOT EXISTS prediction_results (
      prediction_id INTEGER PRIMARY KEY,
      settled_at TEXT NOT NULL,
      actual_value REAL NOT NULL,
      result_source TEXT NOT NULL,
      FOREIGN KEY (prediction_id) REFERENCES model_predictions(prediction_id)
    );
    """),
    (3, """
    CREATE TABLE IF NOT EXISTS model_evaluations (
      evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL,
      model_version TEXT NOT NULL,
      feature_version TEXT NOT NULL,
      evaluation_start TEXT,
      evaluation_end TEXT,
      report_json TEXT NOT NULL
    );
    """),
    (4, """
    CREATE TRIGGER IF NOT EXISTS immutable_player_observations_update BEFORE UPDATE ON player_game_observations BEGIN SELECT RAISE(ABORT, 'player_game_observations is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_player_observations_delete BEFORE DELETE ON player_game_observations BEGIN SELECT RAISE(ABORT, 'player_game_observations is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_pregame_update BEFORE UPDATE ON pregame_snapshots BEGIN SELECT RAISE(ABORT, 'pregame_snapshots is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_pregame_delete BEFORE DELETE ON pregame_snapshots BEGIN SELECT RAISE(ABORT, 'pregame_snapshots is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_market_update BEFORE UPDATE ON market_snapshots BEGIN SELECT RAISE(ABORT, 'market_snapshots is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_market_delete BEFORE DELETE ON market_snapshots BEGIN SELECT RAISE(ABORT, 'market_snapshots is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_prediction_update BEFORE UPDATE ON model_predictions BEGIN SELECT RAISE(ABORT, 'model_predictions is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_prediction_delete BEFORE DELETE ON model_predictions BEGIN SELECT RAISE(ABORT, 'model_predictions is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_result_update BEFORE UPDATE ON prediction_results BEGIN SELECT RAISE(ABORT, 'prediction_results is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_result_delete BEFORE DELETE ON prediction_results BEGIN SELECT RAISE(ABORT, 'prediction_results is immutable'); END;
    """),
    (5, """
    ALTER TABLE gameday_pitcher_arsenal ADD COLUMN swings INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE gameday_pitcher_arsenal ADD COLUMN whiffs INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE gameday_pitcher_arsenal ADD COLUMN chases INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE gameday_pitcher_arsenal ADD COLUMN strikeouts INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE player_game_observations ADD COLUMN walks_allowed INTEGER;
    ALTER TABLE player_game_observations ADD COLUMN hits_allowed INTEGER;
    ALTER TABLE player_game_observations ADD COLUMN runs_allowed INTEGER;
    ALTER TABLE player_game_observations ADD COLUMN earned_runs INTEGER;
    ALTER TABLE prediction_results ADD COLUMN actual_batters_faced REAL;
    ALTER TABLE prediction_results ADD COLUMN actual_pitches REAL;
    ALTER TABLE prediction_results ADD COLUMN actual_outs REAL;
    ALTER TABLE prediction_results ADD COLUMN actual_runs REAL;
    ALTER TABLE prediction_results ADD COLUMN actual_earned_runs REAL;
    ALTER TABLE prediction_results ADD COLUMN actual_hits REAL;
    ALTER TABLE prediction_results ADD COLUMN actual_walks REAL;
    """),
    (6, """
    CREATE TABLE IF NOT EXISTS pitcher_game_results (
      game_pk INTEGER NOT NULL,
      player_id INTEGER NOT NULL,
      batters_faced INTEGER,
      strikeouts INTEGER,
      outs INTEGER,
      pitches INTEGER,
      walks_allowed INTEGER,
      hits_allowed INTEGER,
      runs_allowed INTEGER,
      earned_runs INTEGER,
      observed_at TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'mlb-gameday-boxscore',
      PRIMARY KEY (game_pk, player_id)
    );
    CREATE INDEX IF NOT EXISTS idx_pitcher_results_player_game
      ON pitcher_game_results(player_id, game_pk);
    CREATE TRIGGER IF NOT EXISTS immutable_pitcher_results_update BEFORE UPDATE ON pitcher_game_results BEGIN SELECT RAISE(ABORT, 'pitcher_game_results is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_pitcher_results_delete BEFORE DELETE ON pitcher_game_results BEGIN SELECT RAISE(ABORT, 'pitcher_game_results is immutable'); END;
    """),
    (7, """
    CREATE TABLE IF NOT EXISTS gameday_batter_discipline (
      season INTEGER NOT NULL,
      player_id INTEGER NOT NULL,
      plate_appearances INTEGER NOT NULL,
      pitches_seen INTEGER NOT NULL,
      walks INTEGER NOT NULL,
      hit_by_pitch INTEGER NOT NULL,
      hits INTEGER NOT NULL,
      total_bases INTEGER NOT NULL,
      outs INTEGER NOT NULL,
      PRIMARY KEY (season, player_id)
    );
    CREATE TABLE IF NOT EXISTS workload_overrides (
      workload_override_id INTEGER PRIMARY KEY AUTOINCREMENT,
      captured_at TEXT NOT NULL,
      game_pk INTEGER NOT NULL,
      player_id INTEGER NOT NULL,
      player_name TEXT NOT NULL,
      pitch_limit REAL NOT NULL,
      source TEXT NOT NULL DEFAULT 'manual',
      note TEXT,
      imported_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_workload_override_lookup
      ON workload_overrides(game_pk, player_id, captured_at);
    CREATE TRIGGER IF NOT EXISTS immutable_workload_override_update BEFORE UPDATE ON workload_overrides BEGIN SELECT RAISE(ABORT, 'workload_overrides is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_workload_override_delete BEFORE DELETE ON workload_overrides BEGIN SELECT RAISE(ABORT, 'workload_overrides is immutable'); END;
    """),
    (8, """
    CREATE TABLE IF NOT EXISTS bullpen_snapshots (
      bullpen_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
      game_pk INTEGER NOT NULL,
      team_id INTEGER NOT NULL,
      team_name TEXT NOT NULL,
      scheduled_start TEXT,
      captured_at TEXT NOT NULL,
      player_id INTEGER NOT NULL,
      player_name TEXT NOT NULL,
      throws TEXT,
      role TEXT NOT NULL,
      readiness_score REAL NOT NULL,
      readiness_status TEXT NOT NULL,
      mix_weight REAL NOT NULL,
      pitches_today INTEGER NOT NULL,
      pitches_yesterday INTEGER NOT NULL,
      pitches_two_days_ago INTEGER NOT NULL,
      three_day_pitches INTEGER NOT NULL,
      consecutive_days INTEGER NOT NULL,
      days_rest INTEGER,
      recent_appearances INTEGER NOT NULL,
      recent_starts INTEGER NOT NULL,
      arsenal_available INTEGER NOT NULL CHECK (arsenal_available IN (0, 1)),
      source TEXT NOT NULL DEFAULT 'mlb-gameday-workload'
    );
    CREATE INDEX IF NOT EXISTS idx_bullpen_snapshot_lookup
      ON bullpen_snapshots(game_pk, team_id, captured_at, player_id);
    CREATE TRIGGER IF NOT EXISTS immutable_bullpen_snapshot_update BEFORE UPDATE ON bullpen_snapshots BEGIN SELECT RAISE(ABORT, 'bullpen_snapshots is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_bullpen_snapshot_delete BEFORE DELETE ON bullpen_snapshots BEGIN SELECT RAISE(ABORT, 'bullpen_snapshots is immutable'); END;
    """),
)

def connect(db_path=None):
    connection = sqlite3.connect(db_path or DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection

def initialize(db_path=None):
    """Create missing objects without replacing or deleting existing data."""
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)
        for column in (
            "strikeouts INTEGER NOT NULL DEFAULT 0",
            "outs INTEGER NOT NULL DEFAULT 0",
            "doubles INTEGER NOT NULL DEFAULT 0",
            "triples INTEGER NOT NULL DEFAULT 0",
            "total_bases INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                connection.execute(f"ALTER TABLE gameday_batter_pitch_velocity ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass
        applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
        for version, sql in MIGRATIONS:
            if version in applied:
                continue
            connection.executescript(sql)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                (version,),
            )
        connection.execute(f"PRAGMA user_version={MIGRATIONS[-1][0]}")
