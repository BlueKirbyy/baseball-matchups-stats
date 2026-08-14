"""Create local arsenal-fit profiles from MLB Gameday pitch feeds.

No Baseball Savant player-page scraping is used here. MLB's completed game feeds
provide pitch type, pitch velocity, plate coordinates, and plate outcomes.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import argparse
import json

from analytics_store import connect, initialize
from prediction_store import is_before_start, record_game, record_pregame_snapshot, utc_now

MLB = "https://statsapi.mlb.com/api/v1"
MLB_FEED = "https://statsapi.mlb.com/api/v1.1"
LOCAL_CACHE = Path(__file__).with_name(".gameday_cache")
MOVED_WORKSPACE_CACHE = Path(__file__).parent.parent / ".gameday_cache"
# Preserve an existing cache if the project folder was moved into a subfolder.
CACHE = LOCAL_CACHE if LOCAL_CACHE.exists() or not MOVED_WORKSPACE_CACHE.exists() else MOVED_WORKSPACE_CACHE
HITS = {"single", "double", "triple", "home_run"}
NOT_AB = {"walk", "intent_walk", "hit_by_pitch", "sac_bunt", "sac_fly", "catcher_interf"}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}
OUTS = {"field_out", "force_out", "grounded_into_double_play", "strikeout", "strikeout_double_play", "sac_bunt", "sac_fly", "double_play", "triple_play", "fielders_choice_out", "other_out"}
PLAYER_GAME_LOG_CACHE = {}

def batter_context_store():
    """Return batter → pitch → velo → hand → count → zone → events.

    Keep this factory explicit: omitting the zone dictionary makes the final
    lookup index an empty event list and crashes every feed with an IndexError.
    """
    return defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(list)
                    )
                )
            )
        )
    )

def get_json(url, query=None):
    target = url + ("?" + urlencode(query) if query else "")
    with urlopen(target, timeout=45) as response:
        return json.loads(response.read())

def mlb(path, query=None):
    return get_json(MLB + path, query)

def cached_feed(game_pk):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{game_pk}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        state = saved.get("gameData", {}).get("status", {}).get("abstractGameState")
        if state == "Final":
            return saved
    feed = get_json(f"{MLB_FEED}/game/{game_pk}/feed/live")
    path.write_text(json.dumps(feed))
    return feed

def zone_index(pitch):
    coordinates = pitch.get("pitchData", {}).get("coordinates", {})
    try:
        x, z = float(coordinates.get("pX")), float(coordinates.get("pZ"))
    except (TypeError, ValueError):
        return None
    column = 0 if x < -0.28 else 2 if x > 0.28 else 1
    row = 0 if z > 3.15 else 2 if z < 2.1 else 1
    return row * 3 + column

def count_bucket(pitch):
    """Group the pre-pitch count into a stable, non-sparse context split."""
    count = pitch.get("count", {})
    try:
        balls, strikes = int(count.get("balls", 0)), int(count.get("strikes", 0))
    except (TypeError, ValueError):
        return "unknown"
    if strikes > balls:
        return "pitcher_ahead"
    if balls > strikes:
        return "hitter_ahead"
    return "even"

def stat_line(events):
    pa = len(events)
    at_bats = sum(event not in NOT_AB for event in events)
    hits = sum(event in HITS for event in events)
    return pa, f"{hits / at_bats:.3f}" if at_bats else "—", sum(event == "home_run" for event in events)

def outcome_counts(events):
    return (
        len(events),
        sum(event not in NOT_AB for event in events),
        sum(event in HITS for event in events),
        sum(event == "home_run" for event in events),
        sum(event in STRIKEOUTS for event in events),
        sum(event in OUTS for event in events),
        sum(event == "double" for event in events),
        sum(event == "triple" for event in events),
        sum({"single": 1, "double": 2, "triple": 3, "home_run": 4}.get(event, 0) for event in events),
    )

def swung_at_pitch(pitch):
    description = pitch.get("details", {}).get("description", "").lower()
    return any(term in description for term in ("swing", "foul", "in play", "bunt"))

def missed_swing(pitch):
    description = pitch.get("details", {}).get("description", "").lower()
    return "swinging strike" in description or "missed bunt" in description

def chased_pitch(pitch):
    if not swung_at_pitch(pitch):
        return False
    coordinates = pitch.get("pitchData", {}).get("coordinates", {})
    try:
        x, z = float(coordinates.get("pX")), float(coordinates.get("pZ"))
    except (TypeError, ValueError):
        return False
    return abs(x) > 0.83 or z < 1.5 or z > 3.6

def barrel_proxy(pitch):
    """Conservative EV/launch-angle proxy; not Statcast's official Barrel metric."""
    hit = pitch.get("hitData", {})
    try:
        speed, angle = float(hit.get("launchSpeed")), float(hit.get("launchAngle"))
    except (TypeError, ValueError):
        return False
    return (speed >= 98 and 24 <= angle <= 31) or (speed >= 102 and 18 <= angle <= 36)

def umpire_game_line(feed):
    """Aggregate home-plate umpire outcomes from one completed MLB game feed."""
    officials = feed.get("liveData", {}).get("boxscore", {}).get("officials", [])
    home_plate = next((item.get("official", {}) for item in officials if item.get("officialType") == "Home Plate"), None)
    if not home_plate or not home_plate.get("id"):
        return None
    batters_faced = strikeouts = walks = pitches_thrown = 0
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        pitches = [event for event in play.get("playEvents", []) if event.get("isPitch")]
        if not pitches:
            continue
        event = play.get("result", {}).get("eventType") or play.get("result", {}).get("event") or ""
        batters_faced += 1
        strikeouts += int(event in STRIKEOUTS)
        walks += int(event in {"walk", "intent_walk"})
        pitches_thrown += len(pitches)
    if not batters_faced:
        return None
    return {"id": home_plate["id"], "name": home_plate.get("fullName", "Home plate umpire"), "batters_faced": batters_faced, "strikeouts": strikeouts, "walks": walks, "pitches": pitches_thrown}

def active_batters(team_id):
    roster = mlb(f"/teams/{team_id}/roster", {"rosterType": "active"}).get("roster", [])
    return {player["person"]["id"]: player["person"] for player in roster if player.get("position", {}).get("type") != "Pitcher"}

def completed_team_games(team_id, season, through_date):
    schedule = mlb("/schedule", {"sportId": 1, "teamId": team_id, "season": season, "gameType": "R", "startDate": f"{season}-03-01", "endDate": through_date})
    games = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            if game.get("status", {}).get("abstractGameState") == "Final":
                games.append(game["gamePk"])
    return games

def game_log_pks_from_payload(payload, before_date=None):
    """Extract only games strictly before an as-of date to prevent leakage."""
    game_pks = set()
    for group in payload.get("stats", []):
        for split in group.get("splits", []):
            game_date = str(split.get("date") or split.get("game", {}).get("gameDate") or "")[:10]
            if before_date and game_date and game_date >= before_date:
                continue
            game_pk = split.get("game", {}).get("gamePk") or split.get("gamePk")
            if game_pk:
                game_pks.add(game_pk)
    return game_pks


def player_game_logs(player_id, season, group, before_date=None):
    """Find a player's games independent of current team (trade-safe)."""
    cache_key = (player_id, season, group, before_date)
    if cache_key in PLAYER_GAME_LOG_CACHE:
        return set(PLAYER_GAME_LOG_CACHE[cache_key])
    payload = mlb(f"/people/{player_id}/stats", {"stats": "gameLog", "group": group, "season": season})
    game_pks = game_log_pks_from_payload(payload, before_date)
    PLAYER_GAME_LOG_CACHE[cache_key] = frozenset(game_pks)
    return game_pks


def pitcher_game_logs(player_id, season, before_date=None):
    return player_game_logs(player_id, season, "pitching", before_date)


def completed_game_observations(feed, observed_at=None):
    """Build immutable game-level rows for all players in a completed feed."""
    observed_at = observed_at or utc_now()
    game_data = feed.get("gameData", {})
    game_pk = game_data.get("game", {}).get("pk")
    game_date = game_data.get("datetime", {}).get("officialDate", "")
    teams = game_data.get("teams", {})
    team_ids = {side: (teams.get(side) or {}).get("id") for side in ("away", "home")}
    names = {}
    rows = {}
    first_pitcher = {"away": None, "home": None}
    starting_batters = {"away": [], "home": []}
    total_base_value = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        matchup = play.get("matchup", {})
        pitcher = matchup.get("pitcher", {})
        batter = matchup.get("batter", {})
        pitcher_id, batter_id = pitcher.get("id"), batter.get("id")
        if not pitcher_id or not batter_id:
            continue
        names[pitcher_id] = pitcher.get("fullName")
        names[batter_id] = batter.get("fullName")
        half = str(play.get("about", {}).get("halfInning", "")).lower()
        batting_side = "away" if half == "top" else "home"
        pitching_side = "home" if batting_side == "away" else "away"
        if first_pitcher[pitching_side] is None:
            first_pitcher[pitching_side] = pitcher_id
        if batter_id not in starting_batters[batting_side] and len(starting_batters[batting_side]) < 9:
            starting_batters[batting_side].append(batter_id)
        event = play.get("result", {}).get("eventType") or play.get("result", {}).get("event") or ""
        pitches = [item for item in play.get("playEvents", []) if item.get("isPitch")]
        pitcher_key = (pitcher_id, "pitcher")
        pitcher_row = rows.setdefault(pitcher_key, {
            "game_pk": game_pk, "game_date": game_date, "player_id": pitcher_id,
            "team_id": team_ids[pitching_side], "opponent_id": team_ids[batting_side], "role": "pitcher",
            "throws": (matchup.get("pitchHand") or {}).get("code"), "stands": None,
            "batters_faced": 0, "strikeouts": 0, "outs": 0, "pitches": 0,
            "plate_appearances": None, "at_bats": None, "hits": None, "total_bases": None,
            "home_runs": None, "batter_strikeouts": None,
        })
        pitcher_row["batters_faced"] += 1
        pitcher_row["strikeouts"] += int(event in STRIKEOUTS)
        pitcher_row["outs"] += int(event in OUTS)
        pitcher_row["pitches"] += len(pitches)
        batter_key = (batter_id, "batter")
        batter_row = rows.setdefault(batter_key, {
            "game_pk": game_pk, "game_date": game_date, "player_id": batter_id,
            "team_id": team_ids[batting_side], "opponent_id": team_ids[pitching_side], "role": "batter",
            "throws": None, "stands": (matchup.get("batSide") or {}).get("code"),
            "batters_faced": None, "strikeouts": None, "outs": None, "pitches": None,
            "plate_appearances": 0, "at_bats": 0, "hits": 0, "total_bases": 0,
            "home_runs": 0, "batter_strikeouts": 0,
        })
        batter_row["plate_appearances"] += 1
        batter_row["at_bats"] += int(event not in NOT_AB)
        batter_row["hits"] += int(event in HITS)
        batter_row["total_bases"] += total_base_value.get(event, 0)
        batter_row["home_runs"] += int(event == "home_run")
        batter_row["batter_strikeouts"] += int(event in STRIKEOUTS)
    output = []
    for (player_id, role), row in rows.items():
        row["player_name"] = names.get(player_id)
        row["is_start"] = int(
            (role == "batter" and any(player_id in players for players in starting_batters.values()))
            or first_pitcher.get("away") == player_id or first_pitcher.get("home") == player_id
        )
        row["observed_at"] = observed_at
        output.append(row)
    return output


def save_completed_game_observations(feed, db):
    columns = (
        "game_pk", "game_date", "player_id", "player_name", "team_id", "opponent_id", "role", "is_start",
        "throws", "stands", "batters_faced", "strikeouts", "outs", "pitches", "plate_appearances", "at_bats",
        "hits", "total_bases", "home_runs", "batter_strikeouts", "observed_at",
    )
    placeholders = ",".join("?" for _ in columns)
    for row in completed_game_observations(feed):
        db.execute(
            f"INSERT OR IGNORE INTO player_game_observations({','.join(columns)}, source) VALUES ({placeholders}, 'mlb-gameday')",
            tuple(row[column] for column in columns),
        )

def process_feed(feed, pitcher_ids, batter_ids, pitcher_data, pitcher_context_data, batter_events, batter_pitch_events, batter_pitch_zones, batter_velocity_events, batter_context_events, batter_quality, pitcher_workloads):
    game_lines = defaultdict(lambda: {"batters_faced": 0, "strikeouts": 0, "outs": 0, "pitches": 0})
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        matchup = play.get("matchup", {})
        pitcher_id = matchup.get("pitcher", {}).get("id")
        batter_id = matchup.get("batter", {}).get("id")
        pitches = [event for event in play.get("playEvents", []) if event.get("isPitch") and event.get("details", {}).get("type", {}).get("code")]
        if pitcher_id in pitcher_ids:
            event = play.get("result", {}).get("eventType") or play.get("result", {}).get("event") or ""
            if pitches:
                line = game_lines[pitcher_id]
                line["batters_faced"] += 1
                line["strikeouts"] += int(event in STRIKEOUTS)
                line["outs"] += int(event in OUTS)
                line["pitches"] += len(pitches)
            for pitch in pitches:
                details = pitch.get("details", {})
                code = details["type"]["code"]
                name = details["type"].get("description", code)
                speed = pitch.get("pitchData", {}).get("startSpeed")
                try: speed = float(speed)
                except (TypeError, ValueError): speed = None
                pitcher_data[pitcher_id][code]["name"] = name
                if speed is not None:
                    pitcher_data[pitcher_id][code]["speeds"].append(speed)
                zone = zone_index(pitch)
                if zone is not None:
                    pitcher_data[pitcher_id][code]["zones"][zone] += 1
                pitcher_context_data[pitcher_id][code][count_bucket(pitch)][zone if zone is not None else -1] += 1
        if batter_id in batter_ids and pitches:
            final_pitch = pitches[-1]
            event = play.get("result", {}).get("eventType") or play.get("result", {}).get("event") or ""
            code = final_pitch["details"]["type"]["code"]
            batter_events[batter_id].append(event)
            batter_pitch_events[batter_id][code].append(event)
            speed = final_pitch.get("pitchData", {}).get("startSpeed")
            try:
                velo_bucket = round(float(speed))
                batter_velocity_events[batter_id][code][velo_bucket].append(event)
                pitcher_throws = str((matchup.get("pitchHand") or {}).get("code") or "U")
                batter_context_events[batter_id][code][velo_bucket][pitcher_throws][count_bucket(final_pitch)][zone_index(final_pitch) if zone_index(final_pitch) is not None else -1].append(event)
            except (TypeError, ValueError):
                pass
            zone = zone_index(final_pitch)
            if zone is not None:
                batter_pitch_zones[batter_id][code][zone] += 1
            for pitch in pitches:
                pitch_code = pitch["details"]["type"]["code"]
                quality = batter_quality[batter_id][pitch_code]
                quality["pitches"] += 1
                quality["swings"] += int(swung_at_pitch(pitch))
                quality["whiffs"] += int(missed_swing(pitch))
                quality["chase_swings"] += int(chased_pitch(pitch))
                hit = pitch.get("hitData", {})
                try:
                    launch_speed = float(hit.get("launchSpeed"))
                except (TypeError, ValueError):
                    launch_speed = None
                if launch_speed is not None:
                    quality["batted_balls"] += 1
                    quality["hard_hits"] += int(launch_speed >= 95)
                    quality["barrel_proxy"] += int(barrel_proxy(pitch))
            batter_quality[batter_id][code]["strikeouts"] += int(event in STRIKEOUTS)
    official_date = feed.get("gameData", {}).get("datetime", {}).get("officialDate", "")
    for pitcher_id, line in game_lines.items():
        if line["batters_faced"]:
            pitcher_workloads[pitcher_id].append({"date": official_date, **line})

def sync_game(game_pk, season, workers):
    """Build one matchup profile; cached completed feeds are reused across games."""
    target = cached_feed(game_pk)
    game_data = target.get("gameData", {})
    probable = game_data.get("probablePitchers", {})
    teams = game_data.get("teams", {})
    pitcher_ids = {pitcher["id"] for pitcher in probable.values() if pitcher}
    if len(pitcher_ids) != 2:
        raise ValueError("Both probable starters must be listed before matchup data can be built.")
    home_id, away_id = teams["home"]["id"], teams["away"]["id"]
    batters = active_batters(home_id) | active_batters(away_id)
    target_date = game_data.get("datetime", {}).get("officialDate") or datetime.now(timezone.utc).date().isoformat()
    historical_end = (datetime.fromisoformat(target_date).date() - timedelta(days=1)).isoformat()
    game_ids = set(completed_team_games(home_id, season, historical_end)) | set(completed_team_games(away_id, season, historical_end))
    game_ids.discard(game_pk)
    # Team schedules are fast, while player logs preserve pre-trade history.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        history_futures = {
            executor.submit(player_game_logs, player_id, season, "pitching" if player_id in pitcher_ids else "hitting", target_date): player_id
            for player_id in pitcher_ids | set(batters)
        }
        for future in as_completed(history_futures):
            try:
                game_ids.update(future.result())
            except Exception as error:
                print(f"Player history {history_futures[future]} unavailable: {str(error)[:100]}")
    print(f"Game {game_pk}: reading {len(game_ids)} completed feeds for {len(batters)} active hitters and two probable starters…")
    feeds, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(cached_feed, completed_game_pk): completed_game_pk for completed_game_pk in game_ids}
        for future in as_completed(futures):
            completed_game_pk = futures[future]
            try: feeds.append(future.result())
            except Exception as error: failures.append(f"{completed_game_pk}: {error}")
    pitcher_data = defaultdict(lambda: defaultdict(lambda: {"name": "Pitch", "speeds": [], "zones": [0] * 9}))
    pitcher_context_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    batter_events = defaultdict(list)
    batter_pitch_events = defaultdict(lambda: defaultdict(list))
    batter_pitch_zones = defaultdict(lambda: defaultdict(lambda: [0] * 9))
    batter_velocity_events = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    batter_context_events = batter_context_store()
    batter_quality = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pitcher_workloads = defaultdict(list)
    for feed in feeds:
        process_feed(feed, pitcher_ids, batters, pitcher_data, pitcher_context_data, batter_events, batter_pitch_events, batter_pitch_zones, batter_velocity_events, batter_context_events, batter_quality, pitcher_workloads)
    # A probable starter can have no 2026 pitch history (injury, rookie, or a
    # future game-date). Use only that pitcher's actual prior-season game logs
    # as a transparent fallback instead of dropping the whole matchup.
    fallback_data, fallback_failures = {}, []
    missing_pitchers = [pitcher_id for pitcher_id in pitcher_ids if not pitcher_data[pitcher_id]]
    for pitcher_id in missing_pitchers:
        try:
            prior_pks = pitcher_game_logs(pitcher_id, season - 1)
            prior_data = defaultdict(lambda: defaultdict(lambda: {"name": "Pitch", "speeds": [], "zones": [0] * 9}))
            prior_context_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                prior_feeds = list(executor.map(cached_feed, prior_pks))
            for prior_feed in prior_feeds:
                process_feed(prior_feed, {pitcher_id}, set(), prior_data, prior_context_data, defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(lambda: [0] * 9)), defaultdict(lambda: defaultdict(lambda: defaultdict(list))), batter_context_store(), defaultdict(lambda: defaultdict(lambda: defaultdict(int))), defaultdict(list))
            if prior_data[pitcher_id]:
                fallback_data[pitcher_id] = {"arsenal": prior_data[pitcher_id], "context": prior_context_data[pitcher_id]}
            else:
                fallback_failures.append(pitcher_id)
        except Exception:
            fallback_failures.append(pitcher_id)
    with connect() as db:
        for completed_feed in feeds:
            save_completed_game_observations(completed_feed, db)
        for pitcher_id in pitcher_ids:
            db.execute("DELETE FROM gameday_pitcher_arsenal WHERE player_id=?", (pitcher_id,))
            source_season = season if pitcher_data[pitcher_id] else season - 1
            pitches = pitcher_data[pitcher_id] or fallback_data.get(pitcher_id, {}).get("arsenal", {})
            contexts = pitcher_context_data[pitcher_id] or fallback_data.get(pitcher_id, {}).get("context", {})
            total = sum(len(value["speeds"]) for value in pitches.values())
            for code, value in pitches.items():
                count = len(value["speeds"])
                if not count: continue
                db.execute("INSERT INTO gameday_pitcher_arsenal VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (source_season, pitcher_id, code, value["name"], count, round(count / total * 100, 1), round(sum(value["speeds"]) / count, 1), json.dumps(value["zones"])))
            db.execute("DELETE FROM gameday_pitcher_arsenal_context WHERE player_id=?", (pitcher_id,))
            for code, count_rows in contexts.items():
                for context_count, zones in count_rows.items():
                    for zone, pitches_thrown in zones.items():
                        db.execute("INSERT INTO gameday_pitcher_arsenal_context VALUES (?, ?, ?, ?, ?, ?)", (source_season, pitcher_id, code, context_count, zone, pitches_thrown))
        for batter_id in batters:
            season_line = stat_line(batter_events[batter_id])
            db.execute("INSERT OR REPLACE INTO gameday_batter_season VALUES (?, ?, ?, ?, ?)", (season, batter_id, *season_line))
            db.execute("DELETE FROM gameday_batter_pitch WHERE season=? AND player_id=?", (season, batter_id))
            db.execute("DELETE FROM gameday_batter_pitch_velocity WHERE season=? AND player_id=?", (season, batter_id))
            db.execute("DELETE FROM gameday_batter_pitch_context WHERE season=? AND player_id=?", (season, batter_id))
            for code, events in batter_pitch_events[batter_id].items():
                db.execute("INSERT INTO gameday_batter_pitch VALUES (?, ?, ?, ?, ?, ?, ?)", (season, batter_id, code, *stat_line(events), json.dumps(batter_pitch_zones[batter_id][code])))
            for code, speeds in batter_velocity_events[batter_id].items():
                for velo_bucket, events in speeds.items():
                    db.execute("INSERT INTO gameday_batter_pitch_velocity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (season, batter_id, code, velo_bucket, *outcome_counts(events)))
            for code, speeds in batter_context_events[batter_id].items():
                for velo_bucket, hands in speeds.items():
                    for pitcher_throws, counts in hands.items():
                        for context_count, zones in counts.items():
                            for zone, events in zones.items():
                                db.execute("INSERT INTO gameday_batter_pitch_context VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (season, batter_id, code, velo_bucket, pitcher_throws, context_count, zone, *outcome_counts(events)))
            db.execute("DELETE FROM gameday_batter_pitch_quality WHERE season=? AND player_id=?", (season, batter_id))
            for code, quality in batter_quality[batter_id].items():
                db.execute("INSERT INTO gameday_batter_pitch_quality VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (season, batter_id, code, quality["pitches"], quality["swings"], quality["whiffs"], quality["chase_swings"], quality["batted_balls"], quality["hard_hits"], quality["barrel_proxy"], quality["strikeouts"]))
        for pitcher_id in pitcher_ids:
            lines = sorted(pitcher_workloads[pitcher_id], key=lambda line: line["date"])
            recent = lines[-3:]
            totals = {key: sum(line[key] for line in lines) for key in ("batters_faced", "strikeouts", "outs", "pitches")}
            recent_totals = {key: sum(line[key] for line in recent) for key in ("batters_faced", "strikeouts", "outs", "pitches")}
            db.execute("INSERT OR REPLACE INTO gameday_pitcher_workload VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (season, pitcher_id, len(lines), totals["batters_faced"], totals["strikeouts"], totals["outs"], totals["pitches"], len(recent), recent_totals["batters_faced"], recent_totals["strikeouts"], recent_totals["outs"], recent_totals["pitches"]))
        for completed_feed in feeds:
            umpire = umpire_game_line(completed_feed)
            if not umpire:
                continue
            completed_game_pk = completed_feed.get("gameData", {}).get("game", {}).get("pk")
            if completed_game_pk:
                db.execute("INSERT OR REPLACE INTO gameday_umpire_game VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (season, completed_game_pk, umpire["id"], umpire["name"], umpire["batters_faced"], umpire["strikeouts"], umpire["walks"], umpire["pitches"]))
        db.execute("INSERT OR REPLACE INTO matchup_sync_runs VALUES (?, ?, ?, ?, ?)", (game_pk, season, datetime.now(timezone.utc).isoformat(), len(feeds), len(failures)))
    recorded_at = utc_now()
    record_game({
        "game_pk": game_pk,
        "scheduled_start": game_data.get("datetime", {}).get("dateTime"),
        "official_date": target_date,
        "away_team_id": away_id,
        "away_team_name": teams.get("away", {}).get("name"),
        "home_team_id": home_id,
        "home_team_name": teams.get("home", {}).get("name"),
        "venue_name": game_data.get("venue", {}).get("name"),
    }, recorded_at)
    boxscore_teams = target.get("liveData", {}).get("boxscore", {}).get("teams", {})
    snapshot_lineups = {}
    for side in ("away", "home"):
        batting_order = boxscore_teams.get(side, {}).get("battingOrder", [])
        snapshot_lineups[side] = list(batting_order[:9]) if len(batting_order) >= 9 else []
    scheduled_start = game_data.get("datetime", {}).get("dateTime")
    if is_before_start(recorded_at, scheduled_start):
        record_pregame_snapshot(game_pk, recorded_at, scheduled_start, probable, snapshot_lineups)
    fallback_count = len(fallback_data)
    print(f"Game {game_pk}: saved {sum(len(x) for x in pitcher_data.values()) + sum(len(x) for x in fallback_data.values())} arsenal pitches from {len(feeds)} completed games.")
    if fallback_count: print(f"Game {game_pk}: used prior-season pitch history for {fallback_count} starter(s).")
    if failures: print(f"{len(failures)} game feeds failed; rerun safely to retry them.")

def todays_game_pks():
    # Match the board's local-calendar slate instead of rolling over at UTC midnight.
    date = datetime.now().astimezone().date().isoformat()
    schedule = mlb("/schedule", {"sportId": 1, "date": date})
    game_pks = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            # The schedule hydration can lag the game feed. Include every game
            # here; sync_game verifies probable starters from the fresher feed.
            game_pks.append(game["gamePk"])
    return game_pks

def main():
    parser = argparse.ArgumentParser(description="Build local pitcher-arsenal and hitter-fit profiles from MLB Gameday feeds")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--game-pk", type=int, help="Upcoming MLB game ID to analyze")
    choice.add_argument("--all", action="store_true", help="Attempt every game on today's slate; games without two live-feed probable starters are skipped")
    parser.add_argument("--season", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    initialize()
    game_pks = todays_game_pks() if args.all else [args.game_pk]
    if not game_pks:
        raise SystemExit("No games with two announced probable starters were found today.")
    print(f"Syncing {len(game_pks)} matchup profile(s). Cached game feeds will be reused.")
    for game_pk in game_pks:
        try:
            sync_game(game_pk, args.season, args.workers)
        except Exception as error:
            print(f"Game {game_pk}: skipped ({type(error).__name__}: {error})")

if __name__ == "__main__":
    main()
