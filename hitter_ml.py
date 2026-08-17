"""Leakage-safe hitter challenger models and pure-Python shadow inference.

The production site remains empirical-Bayes by default.  This module builds
pre-event plate-appearance examples chronologically, trains calibrated
challengers when scikit-learn is available, and exports a JSON artifact that
the normal no-dependency server can evaluate.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from math import exp, log, sqrt
from pathlib import Path
import json

from analytics_store import DB_PATH, connect, initialize

FEATURE_VERSION = "hitter-pa-pre-event-v1"
REGISTRY_PATH = Path(__file__).with_name("models") / "hitter_ml_registry.json"
TRAINING_DB_PATH = Path(__file__).with_name("hitter_training.db")
_REGISTRY_CACHE = {"path": None, "mtime_ns": None, "value": None}
LEAGUE = {
    "avg": .245, "slg": .400, "iso": .155, "k": .225, "hr": .030,
    "bb": .085, "xbh": .075,
}
FEATURE_NAMES = (
    "batter_log_pa", "batter_avg", "batter_slg", "batter_iso",
    "batter_k_rate", "batter_hr_rate", "batter_xbh_rate", "batter_bb_rate",
    "platoon_log_pa", "platoon_avg", "platoon_slg", "platoon_iso",
    "platoon_k_rate", "platoon_hr_rate", "pitcher_log_bf",
    "pitcher_k_rate", "pitcher_avg_allowed", "pitcher_slg_allowed",
    "pitcher_hr_rate", "pitcher_bb_rate", "fit_avg", "fit_slg",
    "fit_k_rate", "fit_coverage", "fastball_usage", "breaking_usage",
    "offspeed_usage", "pitcher_avg_velocity", "same_side",
    "lineup_order", "pitcher_is_starter", "season_progress",
)
TARGETS = ("hit", "extra_base_hit", "home_run", "strikeout")
HIT_EVENTS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
WALK_EVENTS = {"walk", "intent_walk", "hit_by_pitch"}
FASTBALLS = {"FF", "SI", "FC", "FA", "FT"}
BREAKING = {"SL", "CU", "KC", "ST", "SV", "SC", "CS"}
OFFSPEED = {"CH", "FS", "FO", "EP", "KN"}


def _counter():
    return {"pa": 0, "ab": 0, "h": 0, "tb": 0, "hr": 0, "k": 0,
            "xbh": 0, "bb": 0, "pitches": 0, "velo_sum": 0.0,
            "velo_n": 0, "bf": 0}


def _rate(numerator, denominator, default):
    return float(numerator) / denominator if denominator else default


def _posterior(numerator, denominator, prior, strength):
    return (float(numerator) + prior * strength) / (denominator + strength)


def _log1p(value):
    return log(max(0.0, float(value)) + 1.0)


def _pitch_code(play):
    for event in reversed(play.get("playEvents") or []):
        if event.get("isPitch"):
            return ((event.get("details") or {}).get("type") or {}).get("code") or "UNK"
    return "UNK"


def _lineup_orders(feed):
    orders = {}
    teams = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    for side in ("away", "home"):
        players = (teams.get(side) or {}).get("players") or {}
        for player in players.values():
            person = player.get("person") or {}
            raw = player.get("battingOrder")
            try:
                order = int(raw) // 100
            except (TypeError, ValueError):
                continue
            if person.get("id") and 1 <= order <= 9:
                orders[int(person["id"])] = order
    return orders


def _starting_pitchers(feed):
    starters = set()
    seen_halves = set()
    for play in ((feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or []:
        half = (play.get("about") or {}).get("halfInning")
        pitcher = (play.get("matchup") or {}).get("pitcher") or {}
        if half and half not in seen_halves and pitcher.get("id"):
            starters.add(int(pitcher["id"]))
            seen_halves.add(half)
    return starters


class PregameState:
    """Season-to-date counters; callers update only after a whole game is read."""

    def __init__(self):
        self.batter = defaultdict(_counter)
        self.batter_hand = defaultdict(_counter)
        self.batter_hand_pitch = defaultdict(_counter)
        self.pitcher = defaultdict(_counter)
        self.pitcher_pitch = defaultdict(_counter)

    def feature_vector(self, batter_id, pitcher_id, hand, bat_side, pitch_code,
                       lineup_order, is_starter, month):
        overall = self.batter[batter_id]
        split = self.batter_hand[(batter_id, hand)]
        pitcher = self.pitcher[pitcher_id]
        overall_avg = _posterior(overall["h"], overall["ab"], LEAGUE["avg"], 100)
        overall_slg = _posterior(overall["tb"], overall["ab"], LEAGUE["slg"], 100)
        split_avg = _posterior(split["h"], split["ab"], overall_avg, 80)
        split_slg = _posterior(split["tb"], split["ab"], overall_slg, 80)

        mix_rows = [(code, row) for (pid, code), row in self.pitcher_pitch.items() if pid == pitcher_id]
        total_pitches = sum(row["pitches"] for _, row in mix_rows)
        if total_pitches:
            fit_avg = fit_slg = fit_k = coverage = 0.0
            fastball = breaking = offspeed = velo_sum = velo_n = 0.0
            for code, row in mix_rows:
                usage = row["pitches"] / total_pitches
                versus = self.batter_hand_pitch[(batter_id, hand, code)]
                reliability = versus["pa"] / (versus["pa"] + 60.0) if versus["pa"] else 0.0
                fit_avg += usage * _posterior(versus["h"], versus["ab"], split_avg, 60)
                fit_slg += usage * _posterior(versus["tb"], versus["ab"], split_slg, 60)
                fit_k += usage * _posterior(versus["k"], versus["pa"], _posterior(split["k"], split["pa"], LEAGUE["k"], 80), 60)
                coverage += usage * reliability
                fastball += usage if code in FASTBALLS else 0.0
                breaking += usage if code in BREAKING else 0.0
                offspeed += usage if code in OFFSPEED else 0.0
                velo_sum += row["velo_sum"]
                velo_n += row["velo_n"]
        else:
            fit_avg, fit_slg = split_avg, split_slg
            fit_k = _posterior(split["k"], split["pa"], LEAGUE["k"], 80)
            coverage = fastball = breaking = offspeed = 0.0
            velo_sum = velo_n = 0.0

        values = {
            "batter_log_pa": _log1p(overall["pa"]),
            "batter_avg": overall_avg,
            "batter_slg": overall_slg,
            "batter_iso": max(0.0, overall_slg - overall_avg),
            "batter_k_rate": _posterior(overall["k"], overall["pa"], LEAGUE["k"], 100),
            "batter_hr_rate": _posterior(overall["hr"], overall["pa"], LEAGUE["hr"], 150),
            "batter_xbh_rate": _posterior(overall["xbh"], overall["pa"], LEAGUE["xbh"], 120),
            "batter_bb_rate": _posterior(overall["bb"], overall["pa"], LEAGUE["bb"], 100),
            "platoon_log_pa": _log1p(split["pa"]),
            "platoon_avg": split_avg,
            "platoon_slg": split_slg,
            "platoon_iso": max(0.0, split_slg - split_avg),
            "platoon_k_rate": _posterior(split["k"], split["pa"], LEAGUE["k"], 80),
            "platoon_hr_rate": _posterior(split["hr"], split["pa"], LEAGUE["hr"], 120),
            "pitcher_log_bf": _log1p(pitcher["bf"]),
            "pitcher_k_rate": _posterior(pitcher["k"], pitcher["bf"], LEAGUE["k"], 120),
            "pitcher_avg_allowed": _posterior(pitcher["h"], pitcher["ab"], LEAGUE["avg"], 120),
            "pitcher_slg_allowed": _posterior(pitcher["tb"], pitcher["ab"], LEAGUE["slg"], 120),
            "pitcher_hr_rate": _posterior(pitcher["hr"], pitcher["bf"], LEAGUE["hr"], 150),
            "pitcher_bb_rate": _posterior(pitcher["bb"], pitcher["bf"], LEAGUE["bb"], 100),
            "fit_avg": fit_avg, "fit_slg": fit_slg,
            "fit_k_rate": fit_k, "fit_coverage": coverage,
            "fastball_usage": fastball, "breaking_usage": breaking,
            "offspeed_usage": offspeed,
            "pitcher_avg_velocity": velo_sum / velo_n if velo_n else 0.0,
            "same_side": float(bool(hand and bat_side and hand == bat_side)),
            "lineup_order": float(lineup_order or 9) / 9.0,
            "pitcher_is_starter": float(bool(is_starter)),
            "season_progress": max(0.0, min(1.0, (int(month or 4) - 3) / 7.0)),
        }
        return {name: round(float(values[name]), 8) for name in FEATURE_NAMES}

    def update_game(self, plays):
        for item in plays:
            batter_id, pitcher_id = item["batter_id"], item["pitcher_id"]
            hand, code = item["pitcher_throws"], item["pitch_code"]
            for counter in (self.batter[batter_id], self.batter_hand[(batter_id, hand)],
                            self.batter_hand_pitch[(batter_id, hand, code)]):
                _update_outcome(counter, item)
            pitcher = self.pitcher[pitcher_id]
            pitcher["bf"] += 1
            _update_outcome(pitcher, item)
            for pitch in item["pitches"]:
                row = self.pitcher_pitch[(pitcher_id, pitch["code"])]
                row["pitches"] += 1
                if pitch["velo"] is not None:
                    row["velo_sum"] += pitch["velo"]
                    row["velo_n"] += 1


def _update_outcome(counter, item):
    counter["pa"] += 1
    counter["ab"] += item["at_bat"]
    counter["h"] += item["hit"]
    counter["tb"] += item["total_bases"]
    counter["hr"] += item["home_run"]
    counter["k"] += item["strikeout"]
    counter["xbh"] += item["extra_base_hit"]
    counter["bb"] += item["walk"]


def game_pa_rows(feed):
    orders = _lineup_orders(feed)
    starters = _starting_pitchers(feed)
    rows = []
    for index, play in enumerate((((feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or [])):
        result, matchup = play.get("result") or {}, play.get("matchup") or {}
        event = result.get("eventType")
        batter, pitcher = matchup.get("batter") or {}, matchup.get("pitcher") or {}
        if not event or not batter.get("id") or not pitcher.get("id") or event == "game_advisory":
            continue
        total_bases = HIT_EVENTS.get(event, 0)
        is_walk = int(event in WALK_EVENTS)
        at_bat = int(not is_walk and event not in {"catcher_interf", "sac_bunt", "sac_fly"})
        pitches = []
        for pitch in play.get("playEvents") or []:
            if not pitch.get("isPitch"):
                continue
            code = (((pitch.get("details") or {}).get("type") or {}).get("code") or "UNK")
            velo = (pitch.get("pitchData") or {}).get("startSpeed")
            try:
                velo = float(velo)
            except (TypeError, ValueError):
                velo = None
            pitches.append({"code": code, "velo": velo})
        rows.append({
            "pa_index": int((play.get("about") or {}).get("atBatIndex", index)),
            "batter_id": int(batter["id"]), "pitcher_id": int(pitcher["id"]),
            "pitcher_throws": ((matchup.get("pitchHand") or {}).get("code") or "U"),
            "bat_side": ((matchup.get("batSide") or {}).get("code") or "U"),
            "lineup_order": orders.get(int(batter["id"])),
            "pitcher_is_starter": int(int(pitcher["id"]) in starters),
            "pitch_code": _pitch_code(play), "pitches": pitches,
            "hit": int(event in HIT_EVENTS), "extra_base_hit": int(total_bases >= 2),
            "home_run": int(event == "home_run"),
            "strikeout": int(event in STRIKEOUT_EVENTS),
            "total_bases": total_bases, "walk": is_walk, "at_bat": at_bat,
        })
    return rows


def _cache_games(cache_dir):
    games = []
    for path in Path(cache_dir).glob("*.json"):
        try:
            with path.open(encoding="utf-8") as source:
                feed = json.load(source)
            data = feed.get("gameData") or {}
            if (data.get("status") or {}).get("abstractGameState") != "Final":
                continue
            game_type = (data.get("game") or {}).get("type")
            if game_type and game_type != "R":
                continue
            dt = (data.get("datetime") or {}).get("dateTime") or ""
            game_pk = (data.get("game") or {}).get("pk") or int(path.stem)
            games.append((dt, int(game_pk), path, feed))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(games, key=lambda item: (item[0], item[1]))


def build_training_examples(cache_dir=None, db_path=None, limit=None):
    """Append immutable examples whose features contain only prior-game data."""
    cache_dir = Path(cache_dir or Path(__file__).parent.parent / ".gameday_cache")
    initialize(db_path)
    games = _cache_games(cache_dir)
    if limit:
        games = games[-int(limit):]
    state, current_season, inserted = PregameState(), None, 0
    created_at = datetime.now(timezone.utc).isoformat()
    with connect(db_path or DB_PATH) as db:
        for dt, game_pk, _path, feed in games:
            game_date = ((feed.get("gameData") or {}).get("datetime") or {}).get("officialDate") or dt[:10]
            season = game_date[:4]
            if season != current_season:
                state, current_season = PregameState(), season
            plays = game_pa_rows(feed)
            try:
                month = int(game_date[5:7])
            except (TypeError, ValueError):
                month = 4
            for item in plays:
                features = state.feature_vector(
                    item["batter_id"], item["pitcher_id"], item["pitcher_throws"],
                    item["bat_side"], item["pitch_code"], item["lineup_order"],
                    item["pitcher_is_starter"], month,
                )
                cursor = db.execute(
                    """INSERT OR IGNORE INTO hitter_ml_examples
                       (feature_version, game_pk, pa_index, game_date, batter_id,
                        pitcher_id, pitcher_throws, lineup_order, features_json,
                        hit, extra_base_hit, home_run, strikeout, total_bases,
                        created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (FEATURE_VERSION, game_pk, item["pa_index"], game_date,
                     item["batter_id"], item["pitcher_id"], item["pitcher_throws"],
                     item["lineup_order"], json.dumps(features, sort_keys=True),
                     item["hit"], item["extra_base_hit"], item["home_run"],
                     item["strikeout"], item["total_bases"], created_at),
                )
                inserted += cursor.rowcount
            state.update_game(plays)
        db.commit()
    return {"games": len(games), "inserted": inserted, "feature_version": FEATURE_VERSION}


def _sigmoid(value):
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _tree_value(tree, row):
    node = 0
    leaves = tree.get("is_leaf")
    while not (leaves[node] if leaves is not None else tree["left"][node] == -1):
        feature = tree["feature"][node]
        missing_left = (tree.get("missing_go_to_left") or [0] * len(tree["left"]))[node]
        value = row[feature]
        node = tree["left"][node] if (value != value and missing_left) or (value == value and value <= tree["threshold"][node]) else tree["right"][node]
    return tree["value"][node]


def predict_exported(model, feature_map):
    """Evaluate an exported sklearn model without importing sklearn."""
    names = model["feature_names"]
    medians = model["medians"]
    row = [float(feature_map.get(name, medians[index])) for index, name in enumerate(names)]
    if model["kind"] == "logistic":
        scaled = [(value - model["means"][i]) / (model["scales"][i] or 1.0) for i, value in enumerate(row)]
        raw = model["intercept"] + sum(coef * value for coef, value in zip(model["coefficients"], scaled))
    else:
        raw = model["initial_log_odds"] + sum(_tree_value(tree, row) for tree in model["trees"])
    calibration = model.get("calibration") or {"intercept": 0.0, "coefficient": 1.0}
    return _sigmoid(calibration["intercept"] + calibration["coefficient"] * raw)


def load_registry(path=None):
    path = Path(path or REGISTRY_PATH)
    if not path.exists():
        return None
    try:
        mtime_ns = path.stat().st_mtime_ns
        if _REGISTRY_CACHE["path"] == path and _REGISTRY_CACHE["mtime_ns"] == mtime_ns:
            return _REGISTRY_CACHE["value"]
        value = json.loads(path.read_text(encoding="utf-8"))
        _REGISTRY_CACHE.update({"path": path, "mtime_ns": mtime_ns, "value": value})
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def current_feature_vector(hitter, side):
    """Map the live empirical profile onto the chronological training schema."""
    platoon = hitter.get("platoon") or {}
    season = hitter.get("season") or {}
    discipline = hitter.get("discipline") or {}
    summary = hitter.get("full_game_research") or hitter.get("arsenal_research") or {}
    workload = side.get("workload") or {}
    arsenal = side.get("arsenal") or []
    season_pa = float(season.get("pa") or platoon.get("overall_pa") or 0)
    season_avg = float(season.get("avg") or platoon.get("overall_avg") or LEAGUE["avg"])
    discipline_pa = float(discipline.get("plate_appearances") or season_pa)
    total_bases = float(discipline.get("total_bases") or season_avg * discipline_pa * 1.55)
    hits = float(discipline.get("hits") or season_avg * discipline_pa)
    pitcher_bf = float(workload.get("batters_faced") or 0)
    pitcher_k = float(workload.get("strikeouts") or 0)
    history = side.get("appearance_history") or []
    history_bf = sum(float(row.get("batters_faced") or 0) for row in history)
    history_hits = sum(float(row.get("hits_allowed") or 0) for row in history)
    history_walks = sum(float(row.get("walks_allowed") or 0) for row in history)
    pitcher_ab = max(0.0, history_bf - history_walks)
    usage = {"fastball": 0.0, "breaking": 0.0, "offspeed": 0.0}
    velo_weight = velo_total = 0.0
    for pitch in arsenal:
        weight = float(pitch.get("usage") or 0) / 100.0
        code = pitch.get("code")
        group = "fastball" if code in FASTBALLS else "breaking" if code in BREAKING else "offspeed"
        usage[group] += weight
        if pitch.get("velo") is not None:
            velo_total += weight * float(pitch["velo"])
            velo_weight += weight
    k_profile = hitter.get("k_profile") or {}
    k_research = k_profile.get("research") or {}
    values = {
        "batter_log_pa": _log1p(season_pa), "batter_avg": season_avg,
        "batter_slg": _rate(total_bases, max(hits / max(season_avg, .001), 1), LEAGUE["slg"]),
        "batter_iso": max(0.0, _rate(total_bases, max(hits / max(season_avg, .001), 1), LEAGUE["slg"]) - season_avg),
        "batter_k_rate": float(k_research.get("posterior") or LEAGUE["k"]),
        "batter_hr_rate": float(season.get("hr") or 0) / max(season_pa, 1),
        "batter_xbh_rate": LEAGUE["xbh"],
        "batter_bb_rate": float(discipline.get("walks") or 0) / max(discipline_pa, 1),
        "platoon_log_pa": _log1p(platoon.get("pa") or 0),
        "platoon_avg": float(platoon.get("posterior_avg") or LEAGUE["avg"]),
        "platoon_slg": float(platoon.get("posterior_slg") or LEAGUE["slg"]),
        "platoon_iso": float(platoon.get("posterior_iso") or LEAGUE["iso"]),
        "platoon_k_rate": float(k_research.get("posterior") or LEAGUE["k"]),
        "platoon_hr_rate": float(platoon.get("home_runs") or 0) / max(float(platoon.get("pa") or 0), 1),
        "pitcher_log_bf": _log1p(pitcher_bf),
        "pitcher_k_rate": _posterior(pitcher_k, pitcher_bf, LEAGUE["k"], 120),
        "pitcher_avg_allowed": _posterior(history_hits, pitcher_ab, LEAGUE["avg"], 120),
        "pitcher_slg_allowed": LEAGUE["slg"],
        "pitcher_hr_rate": LEAGUE["hr"],
        "pitcher_bb_rate": _posterior(history_walks, history_bf, LEAGUE["bb"], 100),
        "fit_avg": float(summary.get("expected_average") or LEAGUE["avg"]),
        "fit_slg": float(summary.get("expected_slg") or LEAGUE["slg"]),
        "fit_k_rate": float(summary.get("k_rate") or k_research.get("posterior") or LEAGUE["k"]),
        "fit_coverage": float(summary.get("coverage") or 0),
        "fastball_usage": usage["fastball"], "breaking_usage": usage["breaking"],
        "offspeed_usage": usage["offspeed"],
        "pitcher_avg_velocity": velo_total / velo_weight if velo_weight else 0.0,
        "same_side": float(bool(hitter.get("bat_side") and side.get("pitcher_throws") and hitter["bat_side"] == side["pitcher_throws"])),
        "lineup_order": float(hitter.get("lineup_order") or 9) / 9.0,
        "pitcher_is_starter": 1.0,
        "season_progress": max(0.0, min(1.0, (datetime.now().month - 3) / 7.0)),
    }
    return values


def shadow_prediction(hitter, side, registry=None):
    registry = registry or load_registry()
    if not registry or registry.get("feature_version") != FEATURE_VERSION:
        return {"available": False, "status": "unavailable", "message": "Run the hitter ML trainer to create a challenger artifact."}
    season_pa = float((hitter.get("season") or {}).get("pa") or 0)
    platoon_pa = float((hitter.get("platoon") or {}).get("pa") or 0)
    fit_coverage = float((hitter.get("full_game_research") or hitter.get("arsenal_research") or {}).get("coverage") or 0)
    live_blockers = []
    if season_pa < 75:
        live_blockers.append(f"only {season_pa:.0f} current-season PA")
    if platoon_pa < 30:
        live_blockers.append(f"only {platoon_pa:.0f} opposing-hand PA")
    if fit_coverage < .15:
        live_blockers.append(f"only {fit_coverage:.0%} arsenal coverage")
    if live_blockers:
        return {
            "available": True, "withheld": True, "status": "shadow",
            "model_version": registry.get("model_version"),
            "trained_through": registry.get("trained_through"),
            "blockers": live_blockers,
            "note": "Probability withheld outside the model's minimum live-evidence envelope.",
        }
    features = current_feature_vector(hitter, side)
    projected_pa = float(((hitter.get("full_game_research") or {}).get("exposure") or {}).get("projected_pa") or 4.2)
    targets = {}
    for target in TARGETS:
        record = (registry.get("targets") or {}).get(target) or {}
        candidates = record.get("candidates") or {}
        probabilities = {name: predict_exported(model, features) for name, model in candidates.items()}
        selected = record.get("selected")
        if selected not in probabilities:
            continue
        point = probabilities[selected]
        disagreement = abs(probabilities.get("logistic", point) - probabilities.get("gradient_boosting", point))
        ece = float(((record.get("metrics") or {}).get("selected") or {}).get("ece") or .04)
        half_width = max(.025, disagreement / 2.0, ece)
        targets[target] = {
            "per_pa": point, "low_per_pa": max(.001, point - half_width),
            "high_per_pa": min(.999, point + half_width),
            "candidate_disagreement": disagreement,
        }
    if not targets:
        return {"available": False, "status": "unavailable", "message": "No trained hitter targets are available."}

    def at_least_one(item):
        return {
            "probability": 1 - (1 - item["per_pa"]) ** projected_pa,
            "low": 1 - (1 - item["low_per_pa"]) ** projected_pa,
            "high": 1 - (1 - item["high_per_pa"]) ** projected_pa,
        }

    hit, hr = targets.get("hit"), targets.get("home_run")
    xbh = targets.get("extra_base_hit")
    expected_tb = projected_pa * ((hit or {}).get("per_pa", 0) + (xbh or {}).get("per_pa", 0) + 2 * (hr or {}).get("per_pa", 0))
    statuses = {(registry["targets"].get(target) or {}).get("status", "shadow") for target in targets}
    return {
        "available": True,
        "status": "promoted" if statuses == {"promoted"} else "shadow",
        "model_version": registry.get("model_version"),
        "trained_through": registry.get("trained_through"),
        "hit": at_least_one(hit) if hit else None,
        "home_run": at_least_one(hr) if hr else None,
        "strikeout": at_least_one(targets["strikeout"]) if targets.get("strikeout") else None,
        "expected_total_bases": expected_tb,
        "projected_pa": projected_pa,
        "note": "Shadow probabilities do not affect rankings." if "shadow" in statuses else "Calibrated challenger passed the promotion gate.",
    }
