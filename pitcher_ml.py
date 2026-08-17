"""Leakage-safe pitcher workload examples and shadow inference.

The live app keeps its transparent workload model in production.  This module
builds one pre-event row per historical starter-game, loads a versioned offline
artifact, and exposes a dependency-free workload challenger for the server.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from math import exp, log, sqrt
from pathlib import Path
import json

from analytics_store import DB_PATH, connect, initialize


MODEL_VERSION = "pitcher-workload-challenger-v1"
FEATURE_VERSION = "pitcher-game-pre-event-v1"
REGISTRY_PATH = Path(__file__).with_name("models") / "pitcher_workload_registry.json"
_REGISTRY_CACHE = {"path": None, "mtime_ns": None, "value": None}

LEAGUE = {
    "bf": 22.0, "pitches": 85.0, "outs": 16.5, "k": .225,
    "bb": .085, "on_base": .320, "tb_per_pa": .410,
}

FEATURE_NAMES = (
    "pitcher_log_starts", "pitcher_avg_bf", "pitcher_recent3_bf",
    "pitcher_recent5_bf", "pitcher_bf_sd", "pitcher_avg_pitches",
    "pitcher_recent3_pitches", "pitcher_recent5_pitches", "pitcher_pitch_sd",
    "pitcher_avg_outs", "pitcher_recent3_outs", "pitcher_pitches_per_batter",
    "pitcher_k_rate", "pitcher_bb_rate", "pitcher_baserunner_rate",
    "pitcher_early_exit_rate", "days_rest", "team_avg_bf",
    "team_avg_pitches", "team_early_exit_rate", "lineup_k_rate",
    "lineup_on_base_proxy", "lineup_tb_per_pa", "lineup_coverage",
    "bullpen_pitches_1d", "bullpen_pitches_3d", "bullpen_arms_1d",
    "season_progress",
)


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values, default):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else float(default)


def _shrunk_mean(values, prior, strength=3.0):
    values = [float(value) for value in values if value is not None]
    return (sum(values) + float(prior) * strength) / (len(values) + strength)


def _sd(values, center, floor):
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return float(floor)
    return max(float(floor), sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1)))


def _rate(numerator, denominator, prior, strength):
    return (float(numerator) + float(prior) * strength) / (float(denominator) + strength)


def _early_exit(row):
    return int(
        _number(row.get("batters_faced")) < 18
        or _number(row.get("outs")) < 15
        or _number(row.get("pitches")) < 70
    )


def _days_between(previous, current, default=5.0):
    try:
        return max(0.0, min(15.0, float((date.fromisoformat(current) - date.fromisoformat(previous)).days)))
    except (TypeError, ValueError):
        return float(default)


def _batter_counter():
    return {"pa": 0.0, "ab": 0.0, "hits": 0.0, "tb": 0.0, "k": 0.0, "non_ab": 0.0}


class PregameWorkloadState:
    """State ending before the target game; updated only after a whole game."""

    def __init__(self):
        self.pitcher_starts = defaultdict(list)
        self.team_starts = defaultdict(list)
        self.batters = defaultdict(_batter_counter)
        self.team_relief_games = defaultdict(list)

    def _bullpen_context(self, team_id, game_date):
        one_day_pitches = three_day_pitches = one_day_arms = 0.0
        for row in self.team_relief_games.get(team_id, []):
            days = _days_between(row["game_date"], game_date, 99)
            if 1 <= days <= 3:
                three_day_pitches += row["pitches"]
            if days == 1:
                one_day_pitches += row["pitches"]
                one_day_arms += row["arms"]
        return one_day_pitches, three_day_pitches, one_day_arms

    def feature_vector(self, starter, lineup, game_date):
        player_id, team_id = starter["player_id"], starter.get("team_id")
        starts = self.pitcher_starts[player_id]
        recent3, recent5 = starts[-3:], starts[-5:]
        team_starts = self.team_starts.get(team_id, [])[-30:]

        def values(rows, key):
            return [_number(row.get(key), None) for row in rows if row.get(key) is not None]

        bf = values(starts, "batters_faced")
        pitches = values(starts, "pitches")
        outs = values(starts, "outs")
        avg_bf = _shrunk_mean(bf, LEAGUE["bf"])
        avg_pitches = _shrunk_mean(pitches, LEAGUE["pitches"])
        avg_outs = _shrunk_mean(outs, LEAGUE["outs"])
        total_bf = sum(bf)
        total_pitches = sum(pitches)
        total_k = sum(_number(row.get("strikeouts")) for row in starts)
        total_bb = sum(_number(row.get("walks_allowed")) for row in starts)
        total_hits = sum(_number(row.get("hits_allowed")) for row in starts)

        lineup_k = lineup_on_base = lineup_tb = lineup_coverage = 0.0
        weight_total = 0.0
        for hitter in lineup:
            counter = self.batters[hitter["player_id"]]
            pa, ab = counter["pa"], counter["ab"]
            reliability = pa / (pa + 100.0) if pa else 0.0
            lineup_k += _rate(counter["k"], pa, LEAGUE["k"], 100.0)
            lineup_on_base += _rate(counter["hits"] + counter["non_ab"], pa, LEAGUE["on_base"], 100.0)
            lineup_tb += _rate(counter["tb"], pa, LEAGUE["tb_per_pa"], 100.0)
            lineup_coverage += reliability
            weight_total += 1.0
        if not weight_total:
            lineup_k, lineup_on_base, lineup_tb = LEAGUE["k"], LEAGUE["on_base"], LEAGUE["tb_per_pa"]
        else:
            lineup_k /= weight_total
            lineup_on_base /= weight_total
            lineup_tb /= weight_total
            lineup_coverage /= weight_total

        one_day, three_day, arms = self._bullpen_context(team_id, game_date)
        previous_date = starts[-1]["game_date"] if starts else None
        month = int(game_date[5:7]) if len(str(game_date)) >= 7 else 4
        result = {
            "pitcher_log_starts": log(len(starts) + 1.0),
            "pitcher_avg_bf": avg_bf,
            "pitcher_recent3_bf": _shrunk_mean(values(recent3, "batters_faced"), avg_bf, 2.0),
            "pitcher_recent5_bf": _shrunk_mean(values(recent5, "batters_faced"), avg_bf, 2.0),
            "pitcher_bf_sd": _sd(bf, avg_bf, 3.0),
            "pitcher_avg_pitches": avg_pitches,
            "pitcher_recent3_pitches": _shrunk_mean(values(recent3, "pitches"), avg_pitches, 2.0),
            "pitcher_recent5_pitches": _shrunk_mean(values(recent5, "pitches"), avg_pitches, 2.0),
            "pitcher_pitch_sd": _sd(pitches, avg_pitches, 10.0),
            "pitcher_avg_outs": avg_outs,
            "pitcher_recent3_outs": _shrunk_mean(values(recent3, "outs"), avg_outs, 2.0),
            "pitcher_pitches_per_batter": (total_pitches + 80.0 * 3.85) / (total_bf + 80.0),
            "pitcher_k_rate": _rate(total_k, total_bf, LEAGUE["k"], 80.0),
            "pitcher_bb_rate": _rate(total_bb, total_bf, LEAGUE["bb"], 100.0),
            "pitcher_baserunner_rate": _rate(total_hits + total_bb, total_bf, LEAGUE["on_base"], 100.0),
            "pitcher_early_exit_rate": _rate(sum(_early_exit(row) for row in starts), len(starts), .30, 6.0),
            "days_rest": _days_between(previous_date, game_date),
            "team_avg_bf": _shrunk_mean(values(team_starts, "batters_faced"), LEAGUE["bf"], 5.0),
            "team_avg_pitches": _shrunk_mean(values(team_starts, "pitches"), LEAGUE["pitches"], 5.0),
            "team_early_exit_rate": _rate(sum(_early_exit(row) for row in team_starts), len(team_starts), .30, 8.0),
            "lineup_k_rate": lineup_k,
            "lineup_on_base_proxy": lineup_on_base,
            "lineup_tb_per_pa": lineup_tb,
            "lineup_coverage": lineup_coverage,
            "bullpen_pitches_1d": min(one_day, 250.0) / 100.0,
            "bullpen_pitches_3d": min(three_day, 500.0) / 200.0,
            "bullpen_arms_1d": min(arms, 12.0) / 6.0,
            "season_progress": max(0.0, min(1.0, (month - 3) / 7.0)),
        }
        return {name: round(float(result[name]), 8) for name in FEATURE_NAMES}

    def update_game(self, rows, game_date):
        relief_by_team = defaultdict(lambda: {"pitches": 0.0, "arms": 0.0})
        for row in rows:
            if row["role"] == "batter":
                counter = self.batters[row["player_id"]]
                pa, ab = _number(row.get("plate_appearances")), _number(row.get("at_bats"))
                counter["pa"] += pa
                counter["ab"] += ab
                counter["hits"] += _number(row.get("hits"))
                counter["tb"] += _number(row.get("total_bases"))
                counter["k"] += _number(row.get("batter_strikeouts"))
                counter["non_ab"] += max(0.0, pa - ab)
            elif row["role"] == "pitcher" and row.get("is_start"):
                clean = dict(row)
                self.pitcher_starts[row["player_id"]].append(clean)
                self.team_starts[row.get("team_id")].append(clean)
            elif row["role"] == "pitcher":
                relief_by_team[row.get("team_id")]["pitches"] += _number(row.get("pitches"))
                relief_by_team[row.get("team_id")]["arms"] += 1.0
        for team_id, value in relief_by_team.items():
            self.team_relief_games[team_id].append({"game_date": game_date, **value})


def _observation_games(db):
    games, current = [], None
    for raw in db.execute(
        """SELECT * FROM player_game_observations
           ORDER BY game_date, game_pk, role DESC, player_id"""
    ):
        row = dict(raw)
        key = (row["game_date"], row["game_pk"])
        if key != current:
            games.append((key, []))
            current = key
        games[-1][1].append(row)
    return games


def build_training_examples(db_path=None, limit_games=None):
    """Append chronological starter-game examples from immutable observations."""
    db_path = db_path or DB_PATH
    initialize(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    inserted = outcomes = 0
    with connect(db_path) as db:
        games = _observation_games(db)
        if limit_games:
            games = games[-int(limit_games):]
        state, current_season = PregameWorkloadState(), None
        for (game_date, game_pk), rows in games:
            season = str(game_date)[:4]
            if current_season != season:
                state, current_season = PregameWorkloadState(), season
            starters = [row for row in rows if row["role"] == "pitcher" and row.get("is_start")]
            starting_hitters = [row for row in rows if row["role"] == "batter" and row.get("is_start")]
            for starter in starters:
                if any(starter.get(key) is None for key in ("batters_faced", "pitches", "outs", "strikeouts")):
                    continue
                lineup = [row for row in starting_hitters if row.get("team_id") == starter.get("opponent_id")]
                features = state.feature_vector(starter, lineup, game_date)
                cursor = db.execute(
                    """INSERT OR IGNORE INTO pitcher_game_ml_examples(
                         feature_version, game_pk, game_date, player_id, player_name,
                         team_id, opponent_id, features_json, batters_faced,
                         pitches, outs, strikeouts, early_exit, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        FEATURE_VERSION, game_pk, game_date, starter["player_id"],
                        starter.get("player_name"), starter.get("team_id"),
                        starter.get("opponent_id"), json.dumps(features, sort_keys=True),
                        int(starter["batters_faced"]), int(starter["pitches"]),
                        int(starter["outs"]), int(starter["strikeouts"]),
                        _early_exit(starter), created_at,
                    ),
                )
                inserted += cursor.rowcount
                outcome_payload = {
                    key: starter.get(key) for key in (
                        "batters_faced", "pitches", "outs", "strikeouts",
                        "walks_allowed", "hits_allowed", "runs_allowed", "earned_runs",
                    )
                }
                cursor = db.execute(
                    """INSERT OR IGNORE INTO settled_player_outcomes(
                         game_pk, game_date, player_id, player_name, target_group,
                         outcomes_json, settled_at, source
                       ) VALUES (?, ?, ?, ?, 'pitcher_game', ?, ?, 'mlb-gameday-observation')""",
                    (
                        game_pk, game_date, starter["player_id"], starter.get("player_name"),
                        json.dumps(outcome_payload, sort_keys=True), created_at,
                    ),
                )
                outcomes += cursor.rowcount
            state.update_game(rows, game_date)
        db.commit()
    return {"games": len(games), "inserted": inserted, "outcomes": outcomes,
            "feature_version": FEATURE_VERSION}


def current_feature_vector(side):
    """Map the live matchup payload onto the historical training schema."""
    state = PregameWorkloadState()
    player_id = int(side.get("pitcher_id") or 0)
    history = [dict(row) for row in side.get("appearance_history") or [] if row.get("is_start")]
    state.pitcher_starts[player_id] = history
    team_id = side.get("pitching_team_id")
    state.team_starts[team_id] = [dict(row) for row in side.get("team_workload_history") or history]

    lineup = []
    for hitter in side.get("batters") or []:
        discipline = hitter.get("discipline") or {}
        k_research = ((hitter.get("k_profile") or {}).get("research") or {})
        pa = _number(discipline.get("plate_appearances"), _number((hitter.get("season") or {}).get("pa")))
        ab = max(0.0, pa - _number(discipline.get("walks")) - _number(discipline.get("hit_by_pitch")))
        posterior_k = _number(k_research.get("posterior"), LEAGUE["k"])
        counter = state.batters[int(hitter["id"])]
        counter.update({
            "pa": pa, "ab": ab, "hits": _number(discipline.get("hits")),
            "tb": _number(discipline.get("total_bases")), "k": posterior_k * pa,
            "non_ab": max(0.0, pa - ab),
        })
        lineup.append({"player_id": int(hitter["id"])})

    bullpen = side.get("bullpen_context") or {}
    state.team_relief_games[team_id] = [{
        "game_date": "2000-01-01", "pitches": 0.0, "arms": 0.0,
    }]
    target_date = side.get("official_date") or datetime.now().date().isoformat()
    starter = {"player_id": player_id, "team_id": team_id}
    features = state.feature_vector(starter, lineup, target_date)
    features["bullpen_pitches_1d"] = min(_number(bullpen.get("pitches_yesterday")), 250.0) / 100.0
    features["bullpen_pitches_3d"] = min(_number(bullpen.get("three_day_pitches")), 500.0) / 200.0
    features["bullpen_arms_1d"] = min(_number(bullpen.get("arms_yesterday")), 12.0) / 6.0
    return features


def _tree_value(tree, row):
    node = 0
    leaves = tree.get("is_leaf")
    while not (leaves[node] if leaves is not None else tree["left"][node] == -1):
        feature = tree["feature"][node]
        missing_left = (tree.get("missing_go_to_left") or [0] * len(tree["left"]))[node]
        value = row[feature]
        node = tree["left"][node] if (value != value and missing_left) or (value == value and value <= tree["threshold"][node]) else tree["right"][node]
    return tree["value"][node]


def _feature_row(model, feature_map):
    return [float(feature_map.get(name, model["medians"][index]))
            for index, name in enumerate(model["feature_names"])]


def predict_exported_regressor(model, feature_map):
    row = _feature_row(model, feature_map)
    if model["kind"] == "ridge":
        scaled = [(value - model["means"][i]) / (model["scales"][i] or 1.0)
                  for i, value in enumerate(row)]
        return model["intercept"] + sum(coef * value for coef, value in zip(model["coefficients"], scaled))
    return model["initial_value"] + sum(_tree_value(tree, row) for tree in model["trees"])


def predict_exported_classifier(model, feature_map):
    raw = predict_exported_regressor(model, feature_map)
    calibration = model.get("calibration") or {"intercept": 0.0, "coefficient": 1.0}
    value = calibration["intercept"] + calibration["coefficient"] * raw
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def load_registry(path=None):
    path = Path(path or REGISTRY_PATH)
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime_ns
        if _REGISTRY_CACHE["path"] == path and _REGISTRY_CACHE["mtime_ns"] == mtime:
            return _REGISTRY_CACHE["value"]
        value = json.loads(path.read_text(encoding="utf-8"))
        _REGISTRY_CACHE.update({"path": path, "mtime_ns": mtime, "value": value})
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def shadow_workload_prediction(side, registry=None):
    features = current_feature_vector(side)
    registry = registry or load_registry()
    if not registry or registry.get("feature_version") != FEATURE_VERSION:
        return {
            "available": False, "status": "collecting",
            "message": "Workload challenger has not been trained yet.",
            "feature_version": FEATURE_VERSION, "features": features,
        }
    estimates, intervals, raw_estimates, guardrails = {}, {}, {}, []
    limits = {"batters_faced": (8.0, 36.0), "pitches": (25.0, 125.0), "outs": (3.0, 27.0)}
    for target, bounds in limits.items():
        record = registry["targets"][target]
        selected = record["selected"]
        raw = predict_exported_regressor(record["candidates"][selected], features)
        raw_estimates[target] = raw
        estimates[target] = max(bounds[0], min(bounds[1], raw))
        low = predict_exported_regressor(record["interval_models"]["low"], features)
        high = predict_exported_regressor(record["interval_models"]["high"], features)
        padding = float(record.get("interval_calibration") or 0.0)
        intervals[target] = [
            max(bounds[0], min(bounds[1], min(low, high) - padding)),
            max(bounds[0], min(bounds[1], max(low, high) + padding)),
        ]
    anchors = {
        "batters_faced": features["pitcher_recent5_bf"],
        "pitches": features["pitcher_recent5_pitches"],
        "outs": features["pitcher_recent3_outs"],
    }
    maximum_moves = {"batters_faced": 4.0, "pitches": 18.0, "outs": 4.5}
    interval_spreads = {"batters_faced": 6.0, "pitches": 25.0, "outs": 6.0}
    for target, anchor in anchors.items():
        low_allowed, high_allowed = anchor - maximum_moves[target], anchor + maximum_moves[target]
        guarded = max(low_allowed, min(high_allowed, estimates[target]))
        if abs(guarded - estimates[target]) > 1e-8:
            guardrails.append(
                f"{target.replace('_', ' ')} limited to ±{maximum_moves[target]:g} from recent-start workload"
            )
            estimates[target] = guarded
            bounds = limits[target]
            intervals[target] = [
                max(bounds[0], estimates[target] - interval_spreads[target]),
                min(bounds[1], estimates[target] + interval_spreads[target]),
            ]

    early = registry["targets"].get("early_exit") or {}
    early_probability = predict_exported_classifier(
        early["candidates"][early["selected"]], features,
    ) if early.get("selected") else None
    if early_probability is not None:
        early_anchor = features["pitcher_early_exit_rate"]
        guarded_early = max(0.02, min(0.85, max(early_anchor - .25, min(early_anchor + .25, early_probability))))
        if abs(guarded_early - early_probability) > 1e-8:
            guardrails.append("early exit limited to ±25 points from the prior-start rate")
            early_probability = guarded_early

    manual_limit = _number((side.get("workload_override") or {}).get("pitch_limit"), None)
    if manual_limit is not None and estimates["pitches"] > manual_limit:
        scale = max(.35, manual_limit / estimates["pitches"])
        estimates["pitches"] = manual_limit
        intervals["pitches"][1] = min(intervals["pitches"][1], manual_limit)
        estimates["batters_faced"] *= scale
        estimates["outs"] *= scale
        intervals["batters_faced"] = [value * scale for value in intervals["batters_faced"]]
        intervals["outs"] = [value * scale for value in intervals["outs"]]

    return {
        "available": True, "status": "shadow",
        "model_version": registry["model_version"],
        "feature_version": registry["feature_version"],
        "trained_through": registry.get("trained_through"),
        "expected_batters_faced": estimates["batters_faced"],
        "batters_faced_interval": intervals["batters_faced"],
        "expected_pitches": estimates["pitches"],
        "pitches_interval": intervals["pitches"],
        "expected_outs": estimates["outs"],
        "outs_interval": intervals["outs"],
        "early_exit_probability": early_probability,
        "raw_estimates": raw_estimates,
        "guardrails": guardrails,
        "manual_pitch_limit": manual_limit,
        "features": features,
        "metrics": {target: (registry["targets"].get(target) or {}).get("metrics")
                    for target in ("batters_faced", "pitches", "outs", "early_exit")},
        "note": "Shadow workload does not change the production ranking.",
    }
