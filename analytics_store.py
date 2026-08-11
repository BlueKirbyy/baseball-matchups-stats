"""Local, query-only analytics store for the Diamond Intel research board."""
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).with_name("diamond_intel.db")

SCHEMA = """
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
  PRIMARY KEY (season, player_id, pitch_code, velo_bucket)
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
"""

def connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def initialize():
    with connect() as connection:
        connection.executescript(SCHEMA)
        for column in ("strikeouts INTEGER NOT NULL DEFAULT 0", "outs INTEGER NOT NULL DEFAULT 0"):
            try:
                connection.execute(f"ALTER TABLE gameday_batter_pitch_velocity ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass
