"""No-key local server for the Diamond Intel MLB matchup board."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
from analytics_store import connect, initialize

PORT = 8000
MLB = "https://statsapi.mlb.com/api/v1"
MLB_GAME_FEED = "https://statsapi.mlb.com/api/v1.1"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

def get_json(path, query=None):
    url = MLB + path + ("?" + urlencode(query) if query else "")
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read())

def get_game_feed(game_pk):
    """The live game feed is on the separately versioned v1.1 MLB endpoint."""
    with urlopen(f"{MLB_GAME_FEED}/game/{game_pk}/feed/live", timeout=20) as response:
        return json.loads(response.read())

def get_espn_json(path, query=None):
    url = ESPN + path + ("?" + urlencode(query) if query else "")
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read())

def team_batters(team_id):
    roster = get_json(f"/teams/{team_id}/roster", {"rosterType": "active"}).get("roster", [])
    return [p["person"] for p in roster if p.get("position", {}).get("type") != "Pitcher"]

def game_context(feed, db, season):
    venue = feed.get("gameData", {}).get("venue", {})
    field = venue.get("fieldInfo", {})
    roof = field.get("roofType") or "Unknown"
    left, center, right = field.get("leftLine"), field.get("center"), field.get("rightLine")
    dimensions = " / ".join(f"{distance} ft" for distance in (left, center, right) if distance)
    geometry = []
    if roof.lower() != "open":
        geometry.append(f"{roof} roof limits weather impact")
    if any(distance and distance <= 315 for distance in (left, right)):
        geometry.append("short corner dimensions")
    if center and center >= 410:
        geometry.append("spacious center field")
    weather = feed.get("gameData", {}).get("weather", {})
    wind = weather.get("wind", "")
    temperature = weather.get("temp")
    weather_read = []
    if roof.lower() != "open":
        weather_read.append("Weather impact reduced by roof")
    else:
        if isinstance(temperature, (int, float)) or str(temperature).isdigit():
            temp = int(temperature)
            weather_read.append("warmer air favors carry" if temp >= 82 else "cooler air modestly suppresses carry" if temp <= 55 else "temperature is neutral")
        if "out" in wind.lower(): weather_read.append("wind out favors hitters")
        elif "in" in wind.lower(): weather_read.append("wind in favors pitchers")
    officials = feed.get("liveData", {}).get("boxscore", {}).get("officials", [])
    home_plate = next((item.get("official", {}) for item in officials if item.get("officialType") == "Home Plate"), None)
    umpire = {"status": "Home-plate umpire not announced yet."}
    if home_plate and home_plate.get("id"):
        row = db.execute("SELECT umpire_name, COUNT(*) AS games, SUM(batters_faced) AS bf, SUM(strikeouts) AS strikeouts, SUM(walks) AS walks FROM gameday_umpire_game WHERE season=? AND umpire_id=?", (season, home_plate["id"])).fetchone()
        league = db.execute("SELECT SUM(batters_faced) AS bf, SUM(strikeouts) AS strikeouts, SUM(walks) AS walks FROM gameday_umpire_game WHERE season=?", (season,)).fetchone()
        games = row["games"] if row and row["bf"] else 0
        if games >= 8 and league and league["bf"]:
            k_rate, bb_rate = row["strikeouts"] / row["bf"], row["walks"] / row["bf"]
            league_k, league_bb = league["strikeouts"] / league["bf"], league["walks"] / league["bf"]
            if k_rate - league_k >= .012 and bb_rate <= league_bb + .002:
                tendency = "Pitcher-friendly"
            elif k_rate - league_k <= -.012 and bb_rate >= league_bb - .002:
                tendency = "Hitter-friendly"
            else:
                tendency = "Neutral"
            umpire = {"name": home_plate.get("fullName", "Home plate umpire"), "tendency": tendency, "games": games, "k_rate": k_rate, "bb_rate": bb_rate, "league_k_rate": league_k, "league_bb_rate": league_bb, "status": f"Based on {games} cached regular-season games."}
        else:
            umpire = {"name": home_plate.get("fullName", "Home plate umpire"), "status": f"Limited cached sample ({games} games); no tendency label yet."}
    return {"park": {"name": venue.get("name", "Park TBD"), "roof": roof, "turf": field.get("turfType", "Unknown"), "dimensions": dimensions or "Dimensions unavailable", "read": "; ".join(geometry) or "Standard geometry context"}, "weather": {"condition": weather.get("condition", "Weather pending"), "temp": temperature, "wind": wind or "Wind pending", "read": "; ".join(weather_read) or "Weather impact pending"}, "umpire": umpire}

def matchup_research(game_pk):
    feed = get_game_feed(game_pk)
    data = feed.get("gameData", {})
    probable = data.get("probablePitchers", {})
    teams = data.get("teams", {})
    home_pitcher, away_pitcher = probable.get("home"), probable.get("away")
    if not home_pitcher or not away_pitcher:
        return {"ready": False, "message": "Both probable starters must be announced before matchup research is available."}
    home_batters, away_batters = team_batters(teams["home"]["id"]), team_batters(teams["away"]["id"])
    season = datetime.now(timezone.utc).year
    with connect() as db:
        latest_sync = db.execute("SELECT * FROM matchup_sync_runs WHERE game_pk=? AND season=?", (game_pk, season)).fetchone()
        context = game_context(feed, db, season)
        def profile(pitcher, hitters):
            arsenal = [dict(row) for row in db.execute("SELECT season AS source_season, pitch_name AS pitch, pitch_code AS code, usage, velo, zones FROM gameday_pitcher_arsenal WHERE player_id=? AND season=(SELECT MAX(season) FROM gameday_pitcher_arsenal WHERE player_id=? AND season<=?) ORDER BY pitches DESC", (pitcher["id"], pitcher["id"], season))]
            if not arsenal:
                return None
            workload_row = db.execute("SELECT appearances, batters_faced, strikeouts, outs, pitches, recent_appearances, recent_batters_faced, recent_strikeouts, recent_outs, recent_pitches FROM gameday_pitcher_workload WHERE season=? AND player_id=?", (season, pitcher["id"])).fetchone()
            workload = dict(workload_row) if workload_row else None
            for pitch in arsenal:
                pitch["zones"] = json.loads(pitch["zones"])
            codes = [pitch["code"] for pitch in arsenal[:5]]
            hitter_rows = []
            for batter in hitters:
                season_row = db.execute("SELECT pa, avg, hr FROM gameday_batter_season WHERE season=? AND player_id=?", (season, batter["id"])).fetchone()
                pitch_rows = db.execute(f"SELECT pitch_code, pa, avg, hr, zones FROM gameday_batter_pitch WHERE season=? AND player_id=? AND pitch_code IN ({','.join('?' for _ in codes)})", (season, batter["id"], *codes)).fetchall() if codes else []
                exact_rows = {row["pitch_code"]: row for row in pitch_rows}
                quality_rows = db.execute(f"SELECT pitch_code, pitches, swings, whiffs, chase_swings, batted_balls, hard_hits, barrel_proxy, strikeouts FROM gameday_batter_pitch_quality WHERE season=? AND player_id=? AND pitch_code IN ({','.join('?' for _ in codes)})", (season, batter["id"], *codes)).fetchall() if codes else []
                quality_by_pitch = {row["pitch_code"]: dict(row) for row in quality_rows}
                by_pitch = {}
                for pitch in arsenal[:5]:
                    center = round(float(pitch["velo"] or 0))
                    velocity_row = db.execute("SELECT SUM(pa) AS pa, SUM(at_bats) AS at_bats, SUM(hits) AS hits, SUM(hr) AS hr, SUM(strikeouts) AS strikeouts, SUM(outs) AS outs FROM gameday_batter_pitch_velocity WHERE season=? AND player_id=? AND pitch_code=? AND velo_bucket BETWEEN ? AND ?", (season, batter["id"], pitch["code"], center - 2, center + 2)).fetchone()
                    if velocity_row and (velocity_row["pa"] or 0) >= 3 and (velocity_row["at_bats"] or 0):
                        by_pitch[pitch["code"]] = {"pa": velocity_row["pa"], "avg": f"{velocity_row['hits'] / velocity_row['at_bats']:.3f}", "hr": velocity_row["hr"], "strikeouts": velocity_row["strikeouts"], "outs": velocity_row["outs"], "quality": quality_by_pitch.get(pitch["code"]), "range": "±2 mph"}
                    elif pitch["code"] in exact_rows:
                        row = exact_rows[pitch["code"]]
                        outcome_row = db.execute("SELECT SUM(strikeouts) AS strikeouts, SUM(outs) AS outs FROM gameday_batter_pitch_velocity WHERE season=? AND player_id=? AND pitch_code=?", (season, batter["id"], pitch["code"])).fetchone()
                        by_pitch[pitch["code"]] = {"pa": row["pa"], "avg": row["avg"], "hr": row["hr"], "strikeouts": outcome_row["strikeouts"] if outcome_row else None, "outs": outcome_row["outs"] if outcome_row else None, "quality": quality_by_pitch.get(pitch["code"]), "range": "all velo"}
                hitter_rows.append({"name": batter["fullName"], "season": dict(season_row) if season_row else {"pa": "—", "avg": "—", "hr": "—"}, "vs_pitches": by_pitch})
            batter_ids = [batter["id"] for batter in hitters]
            placeholders = ",".join("?" for _ in batter_ids)
            opponent_row = db.execute(f"SELECT SUM(pa) AS pa, SUM(strikeouts) AS strikeouts FROM gameday_batter_pitch_velocity WHERE season=? AND player_id IN ({placeholders})", (season, *batter_ids)).fetchone() if batter_ids else None
            opponent_k_rate = (opponent_row["strikeouts"] / opponent_row["pa"]) if opponent_row and opponent_row["pa"] and opponent_row["strikeouts"] is not None else None
            return {"pitcher": pitcher["fullName"], "arsenal": arsenal, "arsenal_season": arsenal[0]["source_season"], "workload": workload, "opponent_k_rate": opponent_k_rate, "batters": hitter_rows}
        home = profile(home_pitcher, away_batters)
        away = profile(away_pitcher, home_batters)
    if not home and not away:
        return {"ready": False, "message": "Neither probable starter has a saved MLB pitch profile yet. Rerun the matchup sync after both starters have appeared in an MLB game."}
    home = home or {"pitcher": home_pitcher["fullName"], "arsenal": [], "arsenal_season": "—", "batters": []}
    away = away or {"pitcher": away_pitcher["fullName"], "arsenal": [], "arsenal_season": "—", "batters": []}
    sync_note = f"Profile updated {latest_sync['synced_at'][:16].replace('T', ' ')} UTC from {latest_sync['feed_count']} completed games." if latest_sync else "Saved MLB Gameday pitch profiles matched by probable starter."
    return {"ready": True, "lineup_note": f"{sync_note} Active non-pitchers are shown until official lineups post.", "context": context, "home": home, "away": away}

def games_for_today():
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    scoreboard = get_espn_json("/scoreboard", {"dates": date, "limit": 100})
    mlb_schedule = get_json("/schedule", {"sportId": 1, "date": f"{date[:4]}-{date[4:6]}-{date[6:]}", "hydrate": "probablePitcher"})
    mlb_by_teams = {}
    for day in mlb_schedule.get("dates", []):
        for game in day.get("games", []):
            teams = game["teams"]
            key = frozenset((teams["away"]["team"]["name"], teams["home"]["team"]["name"]))
            mlb_by_teams[key] = game
    games = []
    for event in scoreboard.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = {item.get("homeAway"): item for item in competition.get("competitors", [])}
        away, home = competitors.get("away", {}), competitors.get("home", {})
        away_team, home_team = away.get("team", {}), home.get("team", {})
        mlb_game = mlb_by_teams.get(frozenset((away_team.get("displayName"), home_team.get("displayName"))), {})
        mlb_teams = mlb_game.get("teams", {})
        weather = competition.get("weather", {})
        games.append({
            "gamePk": mlb_game.get("gamePk"), "venue": competition.get("venue", {}).get("fullName", "Venue TBD"),
            "start": event.get("date", "").replace("T", " ").replace("Z", " UTC"),
            "status": event.get("status", {}).get("type", {}).get("detail", "Scheduled"),
            "away": {"name": away_team.get("displayName", "Away"), "abbr": away_team.get("abbreviation", "AWY"), "pitcher": mlb_teams.get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")},
            "home": {"name": home_team.get("displayName", "Home"), "abbr": home_team.get("abbreviation", "HME"), "pitcher": mlb_teams.get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")},
            "weather": {"condition": weather.get("displayValue"), "temp": weather.get("temperature"), "wind": weather.get("wind")},
        })
    return {"date": f"{date[:4]}-{date[4:6]}-{date[6:]}", "games": games}

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/matchups":
            try:
                body = json.dumps(games_for_today()).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
            except Exception:
                self.send_error(502, "Could not reach the public MLB schedule service")
            return
        if self.path.startswith("/api/matchups/") and self.path.endswith("/research"):
            try:
                game_pk = int(self.path.split("/")[3])
                body = json.dumps(matchup_research(game_pk)).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
            except Exception as error:
                body = json.dumps({"ready": False, "message": f"Research service error: {error}"}).encode()
                self.send_response(502); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
            return
        if self.path == "/.env": self.send_error(404); return
        return super().do_GET()

if __name__ == "__main__":
    initialize()
    print(f"Diamond Intel is running at http://localhost:{PORT}")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
