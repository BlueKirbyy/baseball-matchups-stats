"""Create local arsenal-fit profiles from MLB Gameday pitch feeds.

No Baseball Savant player-page scraping is used here. MLB's completed game feeds
provide pitch type, pitch velocity, plate coordinates, and plate outcomes.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import argparse
import json

from analytics_store import connect, initialize

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
        return json.loads(path.read_text())
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

def pitcher_game_logs(player_id, season):
    """Find a pitcher's actual prior-season games for a targeted fallback."""
    payload = mlb(f"/people/{player_id}/stats", {"stats": "gameLog", "group": "pitching", "season": season})
    game_pks = set()
    for group in payload.get("stats", []):
        for split in group.get("splits", []):
            game_pk = split.get("game", {}).get("gamePk") or split.get("gamePk")
            if game_pk:
                game_pks.add(game_pk)
    return game_pks

def process_feed(feed, pitcher_ids, batter_ids, pitcher_data, batter_events, batter_pitch_events, batter_pitch_zones, batter_velocity_events, batter_quality, pitcher_workloads):
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
        if batter_id in batter_ids and pitches:
            final_pitch = pitches[-1]
            event = play.get("result", {}).get("eventType") or play.get("result", {}).get("event") or ""
            code = final_pitch["details"]["type"]["code"]
            batter_events[batter_id].append(event)
            batter_pitch_events[batter_id][code].append(event)
            speed = final_pitch.get("pitchData", {}).get("startSpeed")
            try:
                batter_velocity_events[batter_id][code][round(float(speed))].append(event)
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
    through_date = datetime.now(timezone.utc).date().isoformat()
    game_ids = set(completed_team_games(home_id, season, through_date)) | set(completed_team_games(away_id, season, through_date))
    game_ids.discard(game_pk)
    print(f"Game {game_pk}: reading {len(game_ids)} completed feeds for {len(batters)} active hitters and two probable starters…")
    feeds, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(cached_feed, completed_game_pk): completed_game_pk for completed_game_pk in game_ids}
        for future in as_completed(futures):
            completed_game_pk = futures[future]
            try: feeds.append(future.result())
            except Exception as error: failures.append(f"{completed_game_pk}: {error}")
    pitcher_data = defaultdict(lambda: defaultdict(lambda: {"name": "Pitch", "speeds": [], "zones": [0] * 9}))
    batter_events = defaultdict(list)
    batter_pitch_events = defaultdict(lambda: defaultdict(list))
    batter_pitch_zones = defaultdict(lambda: defaultdict(lambda: [0] * 9))
    batter_velocity_events = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    batter_quality = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pitcher_workloads = defaultdict(list)
    for feed in feeds:
        process_feed(feed, pitcher_ids, batters, pitcher_data, batter_events, batter_pitch_events, batter_pitch_zones, batter_velocity_events, batter_quality, pitcher_workloads)
    # A probable starter can have no 2026 pitch history (injury, rookie, or a
    # future game-date). Use only that pitcher's actual prior-season game logs
    # as a transparent fallback instead of dropping the whole matchup.
    fallback_data, fallback_failures = {}, []
    missing_pitchers = [pitcher_id for pitcher_id in pitcher_ids if not pitcher_data[pitcher_id]]
    for pitcher_id in missing_pitchers:
        try:
            prior_pks = pitcher_game_logs(pitcher_id, season - 1)
            prior_data = defaultdict(lambda: defaultdict(lambda: {"name": "Pitch", "speeds": [], "zones": [0] * 9}))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                prior_feeds = list(executor.map(cached_feed, prior_pks))
            for prior_feed in prior_feeds:
                process_feed(prior_feed, {pitcher_id}, set(), prior_data, defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(lambda: [0] * 9)), defaultdict(lambda: defaultdict(lambda: defaultdict(list))), defaultdict(lambda: defaultdict(lambda: defaultdict(int))), defaultdict(list))
            if prior_data[pitcher_id]:
                fallback_data[pitcher_id] = prior_data[pitcher_id]
            else:
                fallback_failures.append(pitcher_id)
        except Exception:
            fallback_failures.append(pitcher_id)
    with connect() as db:
        for pitcher_id in pitcher_ids:
            db.execute("DELETE FROM gameday_pitcher_arsenal WHERE player_id=?", (pitcher_id,))
            source_season = season if pitcher_data[pitcher_id] else season - 1
            pitches = pitcher_data[pitcher_id] or fallback_data.get(pitcher_id, {})
            total = sum(len(value["speeds"]) for value in pitches.values())
            for code, value in pitches.items():
                count = len(value["speeds"])
                if not count: continue
                db.execute("INSERT INTO gameday_pitcher_arsenal VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (source_season, pitcher_id, code, value["name"], count, round(count / total * 100, 1), round(sum(value["speeds"]) / count, 1), json.dumps(value["zones"])))
        for batter_id in batters:
            season_line = stat_line(batter_events[batter_id])
            db.execute("INSERT OR REPLACE INTO gameday_batter_season VALUES (?, ?, ?, ?, ?)", (season, batter_id, *season_line))
            db.execute("DELETE FROM gameday_batter_pitch WHERE season=? AND player_id=?", (season, batter_id))
            db.execute("DELETE FROM gameday_batter_pitch_velocity WHERE season=? AND player_id=?", (season, batter_id))
            for code, events in batter_pitch_events[batter_id].items():
                db.execute("INSERT INTO gameday_batter_pitch VALUES (?, ?, ?, ?, ?, ?, ?)", (season, batter_id, code, *stat_line(events), json.dumps(batter_pitch_zones[batter_id][code])))
            for code, speeds in batter_velocity_events[batter_id].items():
                for velo_bucket, events in speeds.items():
                    db.execute("INSERT INTO gameday_batter_pitch_velocity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (season, batter_id, code, velo_bucket, *outcome_counts(events)))
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
    fallback_count = len(fallback_data)
    print(f"Game {game_pk}: saved {sum(len(x) for x in pitcher_data.values()) + sum(len(x) for x in fallback_data.values())} arsenal pitches from {len(feeds)} completed games.")
    if fallback_count: print(f"Game {game_pk}: used prior-season pitch history for {fallback_count} starter(s).")
    if failures: print(f"{len(failures)} game feeds failed; rerun safely to retry them.")

def todays_game_pks():
    date = datetime.now(timezone.utc).date().isoformat()
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
            print(f"Game {game_pk}: skipped ({error})")

if __name__ == "__main__":
    main()
