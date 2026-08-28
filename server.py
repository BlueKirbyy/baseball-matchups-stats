"""No-key local server for the Diamond Intel MLB matchup board."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen
from analytics_store import connect, initialize
from market_data import normalize_row
from hitter_ml import shadow_prediction
from park_factors import venue_factor
from modeling import (
    HITTER_PLATOON_PRIOR_PA, LEAGUE_HITTER_AVERAGE, LEAGUE_HITTER_SLG,
    blend_full_game_hitter_matchup, calculate_hitter_recent_form,
    hitter_arsenal_summary, hitter_k_risk,
    hitter_market_context, hitter_pitch_summary, park_weather_fit,
    pitcher_k_projection, shrunk_rate,
)
from prediction_store import (
    add_market_snapshot, add_workload_override, latest_bullpen_snapshot,
    latest_markets, latest_workload_override,
    record_game, record_ml_feature_snapshot, record_pregame_snapshot,
    ensure_pregame_prediction, is_before_start, pending_prediction_game_pks,
    prediction_tracking_summary, save_prediction, settle_game_predictions, utc_now,
)

PORT = int(os.getenv("PORT", "8000"))
MLB = "https://statsapi.mlb.com/api/v1"
MLB_GAME_FEED = "https://statsapi.mlb.com/api/v1.1"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
ODDS_CACHE = {"key": None, "checked_at": 0, "by_game": {}, "message": None}
SETTLEMENT_CACHE = {"checked_at": 0.0, "settled": 0}

def get_json(path, query=None):
    url = MLB + path + ("?" + urlencode(query) if query else "")
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read())

def get_game_feed(game_pk):
    """The live game feed is on the separately versioned v1.1 MLB endpoint."""
    with urlopen(f"{MLB_GAME_FEED}/game/{game_pk}/feed/live", timeout=20) as response:
        return json.loads(response.read())


def final_pitching_lines(feed):
    """Extract official pitcher results used to settle immutable forecasts."""
    lines = {}
    teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
    for side in ("away", "home"):
        players = (teams.get(side) or {}).get("players", {})
        for player in players.values() if isinstance(players, dict) else []:
            person = player.get("person", {}) if isinstance(player, dict) else {}
            stats = ((player.get("stats") or {}).get("pitching") or {}) if isinstance(player, dict) else {}
            player_id = person.get("id")
            if not player_id or not stats or stats.get("strikeOuts") is None:
                continue
            lines[int(player_id)] = {
                "strikeouts": number(stats.get("strikeOuts")),
                "batters_faced": number(stats.get("battersFaced")),
                "pitches": number(stats.get("numberOfPitches")),
                "outs": number(stats.get("outs")),
                "runs": number(stats.get("runs")),
                "earned_runs": number(stats.get("earnedRuns")),
                "hits": number(stats.get("hits")),
                "walks": number(stats.get("baseOnBalls")),
            }
    return lines


def settle_finished_predictions(force=False):
    """Best-effort free settlement from MLB; never blocks the board on failure."""
    now = time.time()
    if not force and now - SETTLEMENT_CACHE["checked_at"] < 5 * 60:
        return SETTLEMENT_CACHE["settled"]
    settled = 0
    for game_pk in pending_prediction_game_pks():
        try:
            feed = get_game_feed(game_pk)
            state = feed.get("gameData", {}).get("status", {}).get("abstractGameState")
            if state == "Final":
                settled += len(settle_game_predictions(game_pk, final_pitching_lines(feed)))
        except Exception:
            continue
    SETTLEMENT_CACHE.update({"checked_at": now, "settled": settled})
    return settled

def get_espn_json(path, query=None):
    url = ESPN + path + ("?" + urlencode(query) if query else "")
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read())

def local_game_time(value):
    """Format ESPN's UTC event timestamp in the timezone of this local server."""
    try:
        game_time = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return game_time.astimezone().strftime("%a, %b %d · %I:%M %p %Z").replace(" 0", " ")
    except (TypeError, ValueError):
        return "Time TBD"


def espn_event_is_unstarted(event):
    """Return whether an ESPN event can still begin on today's slate."""
    status_type = (event.get("status") or {}).get("type") or {}
    status_text = " ".join(
        str(status_type.get(key) or "") for key in ("name", "shortDetail", "detail", "description")
    ).lower()
    if any(marker in status_text for marker in ("final", "postponed", "canceled", "cancelled")):
        return False
    return status_type.get("state") == "pre"


def active_slate_date_and_scoreboard(now=None, scoreboard_loader=None):
    """Keep today until no local-day event remains unstarted, then load tomorrow."""
    scoreboard_loader = scoreboard_loader or get_espn_json
    local_day = (now or datetime.now().astimezone()).astimezone().date()

    def load(day):
        key = day.strftime("%Y%m%d")
        return scoreboard_loader("/scoreboard", {"dates": key, "limit": 100})

    scoreboard = load(local_day)
    events = scoreboard.get("events", [])
    is_lookahead = not events or not any(espn_event_is_unstarted(event) for event in events)
    slate_day = local_day + timedelta(days=1) if is_lookahead else local_day
    if is_lookahead:
        scoreboard = load(slate_day)
    return slate_day, scoreboard, is_lookahead

def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def hitter_power_metrics(row):
    """Derive descriptive power stats from the same final-pitch PA sample.

    A fresh sync is required after these fields were introduced. Old rows are
    recognized by impossible total-base counts and intentionally return None.
    """
    row = dict(row or {})
    at_bats = number(row.get("at_bats"))
    hits = number(row.get("hits"))
    total_bases = number(row.get("total_bases"))
    if not at_bats or hits is None or total_bases is None or total_bases < hits:
        return None
    average = hits / at_bats
    slugging = total_bases / at_bats
    doubles = int(number(row.get("doubles")) or 0)
    triples = int(number(row.get("triples")) or 0)
    home_runs = int(number(row.get("hr")) or 0)
    return {
        "slg": round(slugging, 3),
        "iso": round(max(0.0, slugging - average), 3),
        "xbh": doubles + triples + home_runs,
        "doubles": doubles,
        "triples": triples,
        "hr": home_runs,
    }

def hitter_context_metrics(base_row, split_rows, pitcher_context_rows):
    """Blend hitter count/zone/handedness splits to this starter's pitch shape.

    Each sparse bucket is shrunk toward the hitter's same-pitch, same-velocity
    sample. The starter's historical count/zone mix supplies the weights.
    """
    base = hitter_power_metrics(base_row)
    if not base:
        return None
    base_row = dict(base_row or {})
    at_bats = number(base_row.get("at_bats")) or 0.0
    hits = number(base_row.get("hits")) or 0.0
    if at_bats <= 0:
        return None
    base_avg = hits / at_bats
    split_by_bucket = {(str(row["count_bucket"]), int(row["zone"])): dict(row) for row in split_rows if int(row["zone"]) >= 0}
    usable_context = [dict(row) for row in pitcher_context_rows if int(row["zone"]) >= 0 and str(row["count_bucket"]) != "unknown"]
    total_weight = sum(number(row.get("pitches")) or 0.0 for row in usable_context)
    if not total_weight:
        return {"adjusted_avg": round(base_avg, 3), "adjusted_slg": base["slg"], "adjusted_iso": base["iso"], "coverage": 0.0}
    prior_ab = 20.0
    adjusted_avg = adjusted_slg = coverage = 0.0
    for profile in usable_context:
        weight = (number(profile.get("pitches")) or 0.0) / total_weight
        split = split_by_bucket.get((str(profile["count_bucket"]), int(profile["zone"])))
        if split:
            split_ab = number(split.get("at_bats")) or 0.0
            split_hits = number(split.get("hits")) or 0.0
            split_tb = number(split.get("total_bases")) or 0.0
            reliability = split_ab / (split_ab + prior_ab) if split_ab else 0.0
            avg = (split_hits + prior_ab * base_avg) / (split_ab + prior_ab)
            slg = (split_tb + prior_ab * base["slg"]) / (split_ab + prior_ab)
        else:
            reliability, avg, slg = 0.0, base_avg, base["slg"]
        adjusted_avg += weight * avg
        adjusted_slg += weight * slg
        coverage += weight * reliability
    return {
        "adjusted_avg": round(adjusted_avg, 3),
        "adjusted_slg": round(adjusted_slg, 3),
        "adjusted_iso": round(max(0.0, adjusted_slg - adjusted_avg), 3),
        "coverage": round(coverage, 3),
    }

def implied_probability(american_price):
    """Convert an ESPN American moneyline to its unadjusted implied probability."""
    price = number(american_price)
    if price is None or price == 0:
        return None
    return (-price / (-price + 100)) if price < 0 else (100 / (price + 100))

def summarize_espn_odds(rows, home, away):
    """Keep only the total and moneyline from ESPN's public scoreboard response.

    ESPN's endpoint is a best-effort public web feed, not a guaranteed betting
    API. Its `odds` array generally contains one entry per provider.
    """
    totals, over_prices, under_prices = [], [], []
    home_probs, away_probs, providers = [], [], set()
    home_favorite_votes, away_favorite_votes = 0, 0
    for odds in rows if isinstance(rows, list) else []:
        if not isinstance(odds, dict):
            continue
        provider = odds.get("provider", {})
        provider_name = provider.get("name") if isinstance(provider, dict) else None
        if provider_name:
            providers.add(str(provider_name))
        total = number(odds.get("overUnder"))
        if total is not None:
            totals.append(total)
        over_price, under_price = number(odds.get("overOdds")), number(odds.get("underOdds"))
        if over_price is not None:
            over_prices.append(over_price)
        if under_price is not None:
            under_prices.append(under_price)
        home_odds = odds.get("homeTeamOdds", {})
        away_odds = odds.get("awayTeamOdds", {})
        if isinstance(home_odds, dict) and home_odds.get("favorite") is True:
            home_favorite_votes += 1
        if isinstance(away_odds, dict) and away_odds.get("favorite") is True:
            away_favorite_votes += 1

        def moneyline(team_odds):
            if not isinstance(team_odds, dict):
                return None
            candidates = [team_odds]
            candidates.extend(
                team_odds.get(key) for key in ("current", "close", "open")
                if isinstance(team_odds.get(key), dict)
            )
            for candidate in candidates:
                for key in ("moneyLine", "moneyline", "american", "odds"):
                    value = number(candidate.get(key))
                    if value is not None and value != 0:
                        return value
            return None

        home_probability = implied_probability(moneyline(home_odds))
        away_probability = implied_probability(moneyline(away_odds))
        if home_probability is not None:
            home_probs.append(home_probability)
        if away_probability is not None:
            away_probs.append(away_probability)
    home_probability, away_probability = median(home_probs), median(away_probs)
    favorite = None
    if home_probability is not None and away_probability is not None:
        probability_sum = home_probability + away_probability
        fair_home = home_probability / probability_sum if probability_sum else home_probability
        fair_away = away_probability / probability_sum if probability_sum else away_probability
        favorite = {
            "team": home if fair_home >= fair_away else away,
            "probability": max(fair_home, fair_away),
        }
    elif home_favorite_votes != away_favorite_votes:
        # Some ESPN providers expose only the favorite flag, not a moneyline.
        # Use a deliberately modest default edge so team-run allocation still
        # has direction without pretending an unavailable price is known.
        favorite = {
            "team": home if home_favorite_votes > away_favorite_votes else away,
            "probability": .55,
            "probability_source": "favorite flag; price unavailable",
        }
    if not totals and not favorite:
        return None
    source = ", ".join(sorted(providers)) if providers else "ESPN public feed"
    return {
        "total": median(totals),
        "over_american": median(over_prices),
        "under_american": median(under_prices),
        "favorite": favorite,
        "total_books": len(totals),
        "moneyline_books": min(len(home_probs), len(away_probs)),
        "source": source,
        "updated": "ESPN public feed",
    }

def odds_ttl_seconds(games):
    """Refresh less often early in the day and more often near first pitch."""
    now = datetime.now(timezone.utc)
    starts = []
    for game in games:
        raw = game.get("start_time", "")
        try:
            starts.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except (TypeError, ValueError):
            pass
    future = [start for start in starts if start > now]
    if not future:
        return 24 * 60 * 60
    seconds = (min(future) - now).total_seconds()
    if seconds <= 15 * 60:
        return 60
    if seconds <= 3 * 60 * 60:
        return 180
    return 900

def read_odds_cache(cache_key):
    with connect() as db:
        row = db.execute("SELECT checked_at, changed_at, payload, message FROM odds_slate_cache WHERE cache_key=?", (cache_key,)).fetchone()
    if not row:
        return None
    try:
        return {"checked_at": row["checked_at"], "changed_at": row["changed_at"], "by_game": {int(game_pk): odds for game_pk, odds in json.loads(row["payload"]).items()}, "message": row["message"]}
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

def write_odds_cache(cache_key, checked_at, changed_at, by_game, message):
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO odds_slate_cache VALUES (?, ?, ?, ?, ?)", (cache_key, checked_at, changed_at, json.dumps(by_game), message))

def odds_for_slate(games, slate_date):
    """Persist ESPN's latest public odds snapshot and reuse it adaptively."""
    now = time.time()
    ttl = odds_ttl_seconds(games)
    cache_key = f"espn-public-odds:{slate_date}"
    if ODDS_CACHE["key"] == cache_key and now - ODDS_CACHE["checked_at"] < ttl:
        return ODDS_CACHE["by_game"], ODDS_CACHE["message"]
    saved = read_odds_cache(cache_key)
    if saved and now - saved["checked_at"] < ttl:
        ODDS_CACHE.update({"key": cache_key, **saved})
        return saved["by_game"], saved["message"]
    try:
        by_game = {}
        for game in games:
            if not game.get("gamePk"):
                continue
            summary = summarize_espn_odds(game.get("espn_odds", []), game["home"]["name"], game["away"]["name"])
            if summary:
                by_game[game["gamePk"]] = summary
        message = None if by_game else "ESPN has not published MLB game totals or moneylines for this slate yet."
        changed_at = now if not saved or saved["by_game"] != by_game or saved["message"] != message else saved["changed_at"]
        write_odds_cache(cache_key, now, changed_at, by_game, message)
    except Exception as error:
        if saved:
            by_game, message = saved["by_game"], f"Showing saved odds; live refresh failed ({str(error)[:120]})."
        else:
            by_game, message = {}, f"Could not read ESPN's public odds feed: {str(error)[:120]}"
    ODDS_CACHE.update({"key": cache_key, "checked_at": now, "by_game": by_game, "message": message})
    return by_game, message


def saved_game_odds(game_pk, official_date):
    """Read the latest persisted slate market without extra calls per profile."""
    if not official_date:
        return None
    saved = read_odds_cache(f"espn-public-odds:{official_date}")
    return (saved or {}).get("by_game", {}).get(int(game_pk))

def team_batters(team_id):
    roster = get_json(f"/teams/{team_id}/roster", {"rosterType": "active"}).get("roster", [])
    return [p["person"] for p in roster if p.get("position", {}).get("type") != "Pitcher"]

def confirmed_starting_lineup(feed, side):
    """Return MLB's posted nine-player batting order, or None until it exists."""
    team = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
    batting_order = team.get("battingOrder", [])
    players = team.get("players", {})
    if not isinstance(batting_order, list) or len(batting_order) < 9 or not isinstance(players, dict):
        return None
    lineup = []
    seen = set()
    for spot, player_id in enumerate(batting_order, start=1):
        try:
            player_id = int(player_id)
        except (TypeError, ValueError):
            continue
        if player_id in seen:
            continue
        player = players.get(f"ID{player_id}", {})
        person = player.get("person", {}) if isinstance(player, dict) else {}
        if not person.get("id") or not person.get("fullName"):
            continue
        lineup.append({
            "id": person["id"], "fullName": person["fullName"], "lineup_order": spot,
            "bat_side": (player.get("batSide") or {}).get("code"),
        })
        seen.add(player_id)
        if len(lineup) == 9:
            return lineup
    return None

def game_context(feed, db, season):
    venue = feed.get("gameData", {}).get("venue", {})
    field = venue.get("fieldInfo", {})
    roof = field.get("roofType") or "Unknown"
    distance_fields = {
        "LF": field.get("leftLine"), "LCF": field.get("leftCenter"),
        "CF": field.get("center"), "RCF": field.get("rightCenter"),
        "RF": field.get("rightLine"),
    }
    distances = {sector: number(distance) for sector, distance in distance_fields.items() if number(distance) is not None}
    dimensions = " / ".join(f"{sector} {distance:.0f} ft" for sector, distance in distances.items())
    geometry = []
    if roof.lower() != "open":
        geometry.append(f"{roof} roof limits weather impact")
    if any(distances.get(sector, 999) <= 315 for sector in ("LF", "RF")):
        geometry.append("short corner dimensions")
    if distances.get("CF", 0) >= 410:
        geometry.append("spacious center field")
    empirical_park = venue_factor(venue.get("id"), None)
    if empirical_park:
        geometry.append(
            f"Statcast {empirical_park['years']} baseline: "
            f"TB {empirical_park['total_bases_multiplier']:.2f}× / "
            f"HR {empirical_park['home_run_multiplier']:.2f}×"
        )
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
    return {"park": {"name": venue.get("name", "Park TBD"), "venue_id": venue.get("id"), "roof": roof, "turf": field.get("turfType", "Unknown"), "dimensions": dimensions or "Dimensions unavailable", "distances": distances, "park_factor": empirical_park, "read": "; ".join(geometry) or "Standard geometry context"}, "weather": {"condition": weather.get("condition", "Weather pending"), "temp": temperature, "wind": wind or "Wind pending", "read": "; ".join(weather_read) or "Weather impact pending"}, "umpire": umpire}


def hitter_spray_profile(db, season, batter_id, bat_side, pitcher_throws, arsenal):
    """Build a pitch-mix-weighted physical-field spray profile.

    Exact batter-side and pitcher-hand rows are preferred. Thin hand samples
    fall back to the batter's same-side spray history and are shrunk toward the
    overall five-sector distribution.
    """
    if bat_side == "S":
        expected_side = "L" if pitcher_throws == "R" else "R" if pitcher_throws == "L" else "S"
    else:
        expected_side = bat_side or "U"
    query = """SELECT bat_side, pitcher_throws, pitch_code, sector, batted_balls,
                      hard_hits, barrel_proxy, home_runs, exit_velocity_sum,
                      launch_angle_sum
                 FROM gameday_batter_spray
                WHERE season=? AND player_id=?"""
    rows = [dict(row) for row in db.execute(
        query + (" AND bat_side=?" if expected_side != "U" else ""),
        (season, batter_id, expected_side) if expected_side != "U" else (season, batter_id),
    )]
    if not rows and expected_side != "U":
        rows = [dict(row) for row in db.execute(query, (season, batter_id))]
        expected_side = "U"
    if expected_side == "U" and rows:
        hand_rows = [row for row in rows if row["pitcher_throws"] == pitcher_throws]
        inference_pool = hand_rows if sum(row["batted_balls"] for row in hand_rows) >= 10 else rows
        side_counts = {
            side: sum(row["batted_balls"] for row in inference_pool if row["bat_side"] == side)
            for side in ("L", "R")
        }
        inferred_side = max(side_counts, key=side_counts.get)
        if side_counts[inferred_side] > 0:
            expected_side = inferred_side
            rows = [row for row in rows if row["bat_side"] == expected_side]
    exact = [row for row in rows if row["pitcher_throws"] == pitcher_throws]
    exact_count = sum(row["batted_balls"] for row in exact)
    selected = exact if exact_count >= 35 else rows
    side_label = f"{expected_side}HB" if expected_side != "U" else "all batter sides"
    source = f"{side_label} vs {pitcher_throws}HP" if exact_count >= 35 else f"{side_label} all-pitcher-hand fallback"
    sectors = ("LF", "LCF", "CF", "RCF", "RF")
    total_batted_balls = sum(row["batted_balls"] for row in selected)
    if not total_batted_balls:
        return {
            "available": False, "bat_side": expected_side, "source": "spray data pending",
            "batted_balls": 0, "sectors": {sector: .2 for sector in sectors},
            "pull_rate": None, "center_rate": None, "opposite_rate": None,
        }
    overall_counts = {sector: sum(row["batted_balls"] for row in selected if row["sector"] == sector) for sector in sectors}
    overall_shares = {sector: overall_counts[sector] / total_batted_balls for sector in sectors}
    by_pitch = {}
    for row in selected:
        by_pitch.setdefault(row["pitch_code"], {sector: 0 for sector in sectors})
        by_pitch[row["pitch_code"]][row["sector"]] += row["batted_balls"]
    weighted = {sector: 0.0 for sector in sectors}
    total_usage = sum(max(0.0, number(pitch.get("usage")) or 0.0) for pitch in arsenal) or 100.0
    for pitch in arsenal:
        usage = max(0.0, number(pitch.get("usage")) or 0.0) / total_usage
        counts = by_pitch.get(pitch.get("code")) or {}
        pitch_total = sum(counts.values())
        for sector in sectors:
            share = (counts.get(sector, 0) + 25.0 * overall_shares[sector]) / (pitch_total + 25.0)
            weighted[sector] += usage * share
    weighted_total = sum(weighted.values()) or 1.0
    weighted = {sector: weighted[sector] / weighted_total for sector in sectors}
    if expected_side == "L":
        pull = weighted["RCF"] + weighted["RF"]
        opposite = weighted["LF"] + weighted["LCF"]
    elif expected_side == "R":
        pull = weighted["LF"] + weighted["LCF"]
        opposite = weighted["RCF"] + weighted["RF"]
    else:
        pull = opposite = None
    return {
        "available": True, "bat_side": expected_side, "source": source,
        "batted_balls": total_batted_balls, "exact_hand_batted_balls": exact_count,
        "sectors": weighted, "pull_rate": pull, "center_rate": weighted["CF"],
        "opposite_rate": opposite,
    }


def saved_pitcher_arsenal(db, player_id, season):
    arsenal = [dict(row) for row in db.execute(
        """SELECT season AS source_season, pitch_name AS pitch, pitch_code AS code,
                  usage, velo, zones, swings, whiffs, chases, strikeouts
           FROM gameday_pitcher_arsenal
           WHERE player_id=? AND season=(
             SELECT MAX(season) FROM gameday_pitcher_arsenal
             WHERE player_id=? AND season<=?
           ) ORDER BY pitches DESC""",
        (player_id, player_id, season),
    )]
    for pitch in arsenal:
        try:
            pitch["zones"] = json.loads(pitch["zones"])
        except (TypeError, json.JSONDecodeError):
            pitch["zones"] = []
    return arsenal


def pitcher_appearance_history(db, player_id, official_date):
    return [dict(row) for row in db.execute(
        """SELECT observation.game_pk, observation.game_date, observation.is_start,
                  COALESCE(result.batters_faced, observation.batters_faced) AS batters_faced,
                  COALESCE(result.strikeouts, observation.strikeouts) AS strikeouts,
                  COALESCE(result.outs, observation.outs) AS outs,
                  COALESCE(result.pitches, observation.pitches) AS pitches,
                  observation.throws,
                  COALESCE(result.walks_allowed, observation.walks_allowed) AS walks_allowed,
                  COALESCE(result.hits_allowed, observation.hits_allowed) AS hits_allowed,
                  COALESCE(result.runs_allowed, observation.runs_allowed) AS runs_allowed,
                  COALESCE(result.earned_runs, observation.earned_runs) AS earned_runs
           FROM player_game_observations observation
           LEFT JOIN pitcher_game_results result
             ON result.game_pk=observation.game_pk AND result.player_id=observation.player_id
           WHERE observation.player_id=? AND observation.role='pitcher'
             AND (? IS NULL OR observation.game_date<?)
           ORDER BY observation.game_date, observation.game_pk""",
        (player_id, official_date, official_date),
    )]


def hitter_k_profile_for_hand(db, season, batter_id, pitcher_throws):
    hand_row = None
    if pitcher_throws:
        hand_row = db.execute(
            """SELECT SUM(pa) AS pa, SUM(strikeouts) AS strikeouts
               FROM gameday_batter_pitch_context
               WHERE season=? AND player_id=? AND pitcher_throws=?""",
            (season, batter_id, pitcher_throws),
        ).fetchone()
    if hand_row and (hand_row["pa"] or 0):
        row, source = hand_row, f"vs {pitcher_throws}HP"
    else:
        row = db.execute(
            """SELECT SUM(pa) AS pa, SUM(strikeouts) AS strikeouts
               FROM gameday_batter_pitch_velocity
               WHERE season=? AND player_id=?""",
            (season, batter_id),
        ).fetchone()
        source = "all pitchers fallback"
    pa = int(row["pa"] or 0) if row else 0
    strikeouts = int(row["strikeouts"] or 0) if row else 0
    profile = {
        "pa": pa, "strikeouts": strikeouts,
        "rate": round(strikeouts / pa, 3) if pa else None,
        "source": source,
    }
    profile["research"] = hitter_k_risk(profile)
    return profile


def hitter_platoon_profile(db, season, batter_id, pitcher_throws):
    """Build a handedness anchor before applying pitch/velocity refinements."""
    overall = db.execute(
        """SELECT SUM(pa) AS pa, SUM(at_bats) AS at_bats, SUM(hits) AS hits,
                  SUM(total_bases) AS total_bases, SUM(hr) AS hr,
                  SUM(strikeouts) AS strikeouts
           FROM gameday_batter_pitch_velocity
           WHERE season=? AND player_id=?""",
        (season, batter_id),
    ).fetchone()
    hand = None
    if pitcher_throws:
        hand = db.execute(
            """SELECT SUM(pa) AS pa, SUM(at_bats) AS at_bats, SUM(hits) AS hits,
                      SUM(total_bases) AS total_bases, SUM(hr) AS hr,
                      SUM(strikeouts) AS strikeouts
               FROM gameday_batter_pitch_context
               WHERE season=? AND player_id=? AND pitcher_throws=?""",
            (season, batter_id, pitcher_throws),
        ).fetchone()

    overall = dict(overall) if overall else {}
    hand = dict(hand) if hand and (hand["pa"] or 0) else overall
    overall_ab = max(0, int(overall.get("at_bats") or 0))
    overall_pa = max(0, int(overall.get("pa") or 0))
    overall_avg = (overall.get("hits") or 0) / overall_ab if overall_ab else LEAGUE_HITTER_AVERAGE
    overall_slg = (overall.get("total_bases") or 0) / overall_ab if overall_ab else LEAGUE_HITTER_SLG
    hand_ab = max(0, int(hand.get("at_bats") or 0))
    hand_pa = max(0, int(hand.get("pa") or 0))
    hand_hits = max(0, int(hand.get("hits") or 0))
    hand_tb = max(0, int(hand.get("total_bases") or 0))
    posterior_avg = shrunk_rate(hand_hits, hand_ab, overall_avg, HITTER_PLATOON_PRIOR_PA)
    posterior_slg = (
        (hand_tb + overall_slg * HITTER_PLATOON_PRIOR_PA) /
        (hand_ab + HITTER_PLATOON_PRIOR_PA)
        if hand_ab + HITTER_PLATOON_PRIOR_PA else overall_slg
    )
    source = f"vs {pitcher_throws}HP" if pitcher_throws and hand is not overall else "all pitchers fallback"
    return {
        "pitcher_throws": pitcher_throws,
        "label": source,
        "source": source,
        "pa": hand_pa,
        "at_bats": hand_ab,
        "hits": hand_hits,
        "home_runs": max(0, int(hand.get("hr") or 0)),
        "strikeouts": max(0, int(hand.get("strikeouts") or 0)),
        "raw_avg": hand_hits / hand_ab if hand_ab else None,
        "raw_slg": hand_tb / hand_ab if hand_ab else None,
        "posterior_avg": posterior_avg,
        "posterior_slg": posterior_slg,
        "posterior_iso": max(0.0, posterior_slg - posterior_avg),
        "overall_pa": overall_pa,
        "overall_avg": overall_avg,
        "overall_slg": overall_slg,
    }


def hitter_recent_form_profile(db, batter_id, scheduled_start, official_date):
    """Load only completed games scheduled before this matchup's first pitch."""
    as_of = scheduled_start or (f"{official_date}T23:59:59+00:00" if official_date else None)
    if not as_of:
        return calculate_hitter_recent_form([], None)
    rows = [dict(row) for row in db.execute(
        """SELECT game_pk, scheduled_start, game_date, plate_appearances,
                  at_bats, hits, walks, hit_by_pitch, sacrifice_flies,
                  total_bases, doubles, triples, home_runs, strikeouts
             FROM batter_game_form
            WHERE player_id=? AND game_date<=?
            ORDER BY COALESCE(scheduled_start, game_date), game_pk""",
        (batter_id, official_date or str(as_of)[:10]),
    )]
    return calculate_hitter_recent_form(rows, as_of)


def hitter_profile_for_pitcher(db, season, batter, pitcher_id, arsenal,
                               pitcher_throws, market_context, game_environment=None):
    """Build the same pitch/velocity/context hitter evidence for any pitcher."""
    batter_id = batter["id"]
    codes = [pitch["code"] for pitch in arsenal[:5]]
    source_season = arsenal[0]["source_season"] if arsenal else season
    pitcher_context_by_pitch = {}
    for row in db.execute(
        """SELECT pitch_code, count_bucket, zone, pitches
           FROM gameday_pitcher_arsenal_context
           WHERE season=? AND player_id=?""",
        (source_season, pitcher_id),
    ):
        pitcher_context_by_pitch.setdefault(row["pitch_code"], []).append(row)

    # The caller normally supplies the pitcher id separately on the temporary
    # batter object. Keeping it out of the public response avoids ambiguity.
    def context_for(pitch, base_row, velocity_range):
        if not pitcher_throws or not base_row or not pitcher_context_by_pitch.get(pitch["code"]):
            return None
        center = round(float(pitch["velo"] or 0))
        if velocity_range == "±2 mph":
            velocity_clause, velocity_values = "velo_bucket BETWEEN ? AND ?", (center - 2, center + 2)
        else:
            velocity_clause, velocity_values = "1=1", ()
        rows = db.execute(
            f"""SELECT count_bucket, zone, SUM(pa) AS pa, SUM(at_bats) AS at_bats,
                       SUM(hits) AS hits, SUM(total_bases) AS total_bases
                FROM gameday_batter_pitch_context
                WHERE season=? AND player_id=? AND pitch_code=?
                  AND pitcher_throws=? AND {velocity_clause}
                GROUP BY count_bucket, zone""",
            (season, batter_id, pitch["code"], pitcher_throws, *velocity_values),
        ).fetchall()
        context = hitter_context_metrics(base_row, rows, pitcher_context_by_pitch[pitch["code"]])
        if context:
            context["pitcher_throws"] = pitcher_throws
            context["range"] = velocity_range
        return context

    season_row = db.execute(
        "SELECT pa, avg, hr FROM gameday_batter_season WHERE season=? AND player_id=?",
        (season, batter_id),
    ).fetchone()
    discipline_row = db.execute(
        """SELECT plate_appearances, pitches_seen, walks, hit_by_pitch,
                  hits, total_bases, outs
           FROM gameday_batter_discipline WHERE season=? AND player_id=?""",
        (season, batter_id),
    ).fetchone()
    pitch_rows = db.execute(
        f"""SELECT pitch_code, pa, avg, hr, zones FROM gameday_batter_pitch
            WHERE season=? AND player_id=? AND pitch_code IN ({','.join('?' for _ in codes)})""",
        (season, batter_id, *codes),
    ).fetchall() if codes else []
    exact_rows = {row["pitch_code"]: row for row in pitch_rows}
    quality_rows = db.execute(
        f"""SELECT pitch_code, pitches, swings, whiffs, chase_swings,
                   batted_balls, hard_hits, barrel_proxy, strikeouts
            FROM gameday_batter_pitch_quality
            WHERE season=? AND player_id=? AND pitch_code IN ({','.join('?' for _ in codes)})""",
        (season, batter_id, *codes),
    ).fetchall() if codes else []
    quality_by_pitch = {row["pitch_code"]: dict(row) for row in quality_rows}
    platoon = hitter_platoon_profile(db, season, batter_id, pitcher_throws)
    by_pitch = {}
    for pitch in arsenal[:5]:
        center = round(float(pitch["velo"] or 0))
        velocity_row = None
        evidence_source = "all pitcher hands"
        if pitcher_throws:
            velocity_row = db.execute(
                """SELECT SUM(pa) AS pa, SUM(at_bats) AS at_bats, SUM(hits) AS hits,
                          SUM(hr) AS hr, SUM(strikeouts) AS strikeouts, SUM(outs) AS outs,
                          SUM(doubles) AS doubles, SUM(triples) AS triples,
                          SUM(total_bases) AS total_bases
                   FROM gameday_batter_pitch_context
                   WHERE season=? AND player_id=? AND pitch_code=? AND pitcher_throws=?
                     AND velo_bucket BETWEEN ? AND ?""",
                (season, batter_id, pitch["code"], pitcher_throws, center - 2, center + 2),
            ).fetchone()
            evidence_source = f"vs {pitcher_throws}HP"
        if not velocity_row or (velocity_row["pa"] or 0) < 3:
            velocity_row = db.execute(
                """SELECT SUM(pa) AS pa, SUM(at_bats) AS at_bats, SUM(hits) AS hits,
                          SUM(hr) AS hr, SUM(strikeouts) AS strikeouts, SUM(outs) AS outs,
                          SUM(doubles) AS doubles, SUM(triples) AS triples,
                          SUM(total_bases) AS total_bases
                   FROM gameday_batter_pitch_velocity
                   WHERE season=? AND player_id=? AND pitch_code=?
                     AND velo_bucket BETWEEN ? AND ?""",
                (season, batter_id, pitch["code"], center - 2, center + 2),
            ).fetchone()
            evidence_source = "all pitcher hands fallback"
        if velocity_row and (velocity_row["pa"] or 0) >= 3 and (velocity_row["at_bats"] or 0):
            velocity_range = "±2 mph"
            by_pitch[pitch["code"]] = {
                "pa": velocity_row["pa"],
                "avg": f"{velocity_row['hits'] / velocity_row['at_bats']:.3f}",
                "hr": velocity_row["hr"], "strikeouts": velocity_row["strikeouts"],
                "outs": velocity_row["outs"], "advanced": hitter_power_metrics(velocity_row),
                "context": context_for(pitch, velocity_row, velocity_range),
                "quality": quality_by_pitch.get(pitch["code"]), "range": velocity_range,
                "evidence_source": evidence_source,
            }
        elif pitch["code"] in exact_rows:
            row = exact_rows[pitch["code"]]
            outcome_row = None
            if pitcher_throws:
                outcome_row = db.execute(
                    """SELECT SUM(pa) AS pa, SUM(at_bats) AS at_bats, SUM(hits) AS hits,
                              SUM(hr) AS hr, SUM(strikeouts) AS strikeouts, SUM(outs) AS outs,
                              SUM(doubles) AS doubles, SUM(triples) AS triples,
                              SUM(total_bases) AS total_bases
                       FROM gameday_batter_pitch_context
                       WHERE season=? AND player_id=? AND pitch_code=? AND pitcher_throws=?""",
                    (season, batter_id, pitch["code"], pitcher_throws),
                ).fetchone()
                evidence_source = f"vs {pitcher_throws}HP"
            if not outcome_row or (outcome_row["pa"] or 0) < 3:
                outcome_row = db.execute(
                    """SELECT SUM(pa) AS pa, SUM(at_bats) AS at_bats, SUM(hits) AS hits,
                              SUM(hr) AS hr, SUM(strikeouts) AS strikeouts, SUM(outs) AS outs,
                              SUM(doubles) AS doubles, SUM(triples) AS triples,
                              SUM(total_bases) AS total_bases
                       FROM gameday_batter_pitch_velocity
                       WHERE season=? AND player_id=? AND pitch_code=?""",
                    (season, batter_id, pitch["code"]),
                ).fetchone()
                evidence_source = "all pitcher hands fallback"
            velocity_range = "all velo"
            outcome_ab = (outcome_row["at_bats"] or 0) if outcome_row else 0
            by_pitch[pitch["code"]] = {
                "pa": outcome_row["pa"] if outcome_row else row["pa"],
                "avg": f"{outcome_row['hits'] / outcome_ab:.3f}" if outcome_ab else row["avg"],
                "hr": outcome_row["hr"] if outcome_row else row["hr"],
                "strikeouts": outcome_row["strikeouts"] if outcome_row else None,
                "outs": outcome_row["outs"] if outcome_row else None,
                "advanced": hitter_power_metrics(outcome_row),
                "context": context_for(pitch, outcome_row, velocity_range),
                "quality": quality_by_pitch.get(pitch["code"]), "range": velocity_range,
                "evidence_source": evidence_source,
            }
    public_batter = dict(batter)
    hitter = {
        "id": batter_id, "name": batter["fullName"],
        "lineup_order": batter.get("lineup_order"), "bat_side": batter.get("bat_side"),
        "season": dict(season_row) if season_row else {"pa": "—", "avg": "—", "hr": "—"},
        "discipline": dict(discipline_row) if discipline_row else {},
        "vs_pitches": by_pitch,
        "platoon": platoon,
        "recent_form": batter.get("recent_form") or {},
        "k_profile": hitter_k_profile_for_hand(db, season, batter_id, pitcher_throws),
    }
    spray = hitter_spray_profile(
        db, season, batter_id, hitter.get("bat_side"), pitcher_throws, arsenal[:5],
    )
    game_environment = game_environment or {}
    hitter["spray_profile"] = spray
    hitter["environment"] = park_weather_fit(
        spray, game_environment.get("park"), game_environment.get("weather"),
    )
    for stat in by_pitch.values():
        stat["research"] = hitter_pitch_summary(stat, prior={
            "avg": platoon["posterior_avg"], "slg": platoon["posterior_slg"],
            "iso": platoon["posterior_iso"],
        })
    hitter["arsenal_research"] = hitter_arsenal_summary(
        hitter, arsenal[:5], market_context=market_context,
    )
    hitter.update({key: value for key, value in public_batter.items() if key not in hitter})
    return hitter

def matchup_research(game_pk):
    feed = get_game_feed(game_pk)
    data = feed.get("gameData", {})
    probable = data.get("probablePitchers", {})
    teams = data.get("teams", {})
    home_pitcher, away_pitcher = probable.get("home"), probable.get("away")
    if not home_pitcher or not away_pitcher:
        return {"ready": False, "message": "Both probable starters must be announced before matchup research is available."}
    home_lineup, away_lineup = confirmed_starting_lineup(feed, "home"), confirmed_starting_lineup(feed, "away")
    home_batters = home_lineup or team_batters(teams["home"]["id"])
    away_batters = away_lineup or team_batters(teams["away"]["id"])
    season = datetime.now(timezone.utc).year
    scheduled_start = data.get("datetime", {}).get("dateTime")
    official_date = data.get("datetime", {}).get("officialDate")
    game_market = saved_game_odds(game_pk, official_date)
    captured_at = utc_now()
    if is_before_start(captured_at, scheduled_start):
        record_pregame_snapshot(
            game_pk, captured_at, scheduled_start, probable,
            {"away": away_lineup or [], "home": home_lineup or []},
        )
    with connect() as db:
        latest_sync = db.execute("SELECT * FROM matchup_sync_runs WHERE game_pk=? AND season=?", (game_pk, season)).fetchone()
        context = game_context(feed, db, season)
        data_freshness_seconds = None
        if latest_sync:
            try:
                synced_at = datetime.fromisoformat(latest_sync["synced_at"].replace("Z", "+00:00"))
                data_freshness_seconds = max(0.0, (datetime.now(timezone.utc) - synced_at).total_seconds())
            except (TypeError, ValueError):
                pass
        def profile(pitcher, hitters, lineup_confirmed, batting_team, pitching_team):
            arsenal = saved_pitcher_arsenal(db, pitcher["id"], season)
            if not arsenal:
                return None
            workload_row = db.execute("SELECT appearances, batters_faced, strikeouts, outs, pitches, recent_appearances, recent_batters_faced, recent_strikeouts, recent_outs, recent_pitches FROM gameday_pitcher_workload WHERE season=? AND player_id=?", (season, pitcher["id"])).fetchone()
            workload = dict(workload_row) if workload_row else None
            appearance_history = pitcher_appearance_history(db, pitcher["id"], official_date)
            pitcher_throws = next((row.get("throws") for row in reversed(appearance_history) if row.get("throws")), None)
            market_context = hitter_market_context(game_market, batting_team)
            prepared_hitters = [
                {**batter, "recent_form": hitter_recent_form_profile(
                    db, batter["id"], scheduled_start, official_date,
                )}
                for batter in hitters
            ]
            hitter_rows = [
                hitter_profile_for_pitcher(
                    db, season, batter, pitcher["id"], arsenal,
                    pitcher_throws, market_context, context,
                )
                for batter in prepared_hitters
            ]
            batter_ids = [batter["id"] for batter in prepared_hitters]
            placeholders = ",".join("?" for _ in batter_ids)
            opponent_row = db.execute(f"SELECT SUM(pa) AS pa, SUM(strikeouts) AS strikeouts FROM gameday_batter_pitch_velocity WHERE season=? AND player_id IN ({placeholders})", (season, *batter_ids)).fetchone() if batter_ids else None
            opponent_k_rate = (opponent_row["strikeouts"] / opponent_row["pa"]) if opponent_row and opponent_row["pa"] and opponent_row["strikeouts"] is not None else None
            team_workload_history = [dict(row) for row in db.execute(
                """SELECT game_date, player_id, team_id, batters_faced, strikeouts,
                          outs, pitches, walks_allowed, hits_allowed, runs_allowed,
                          earned_runs, is_start
                     FROM player_game_observations
                    WHERE role='pitcher' AND is_start=1 AND team_id=? AND game_date<?
                    ORDER BY game_date, game_pk""",
                (pitching_team["id"], official_date),
            )]
            snapshots = latest_bullpen_snapshot(game_pk, pitching_team["id"], captured_at)
            bullpen_context = {
                "pitches_yesterday": sum(number(row.get("pitches_yesterday")) or 0 for row in snapshots),
                "three_day_pitches": sum(number(row.get("three_day_pitches")) or 0 for row in snapshots),
                "arms_yesterday": sum((number(row.get("pitches_yesterday")) or 0) > 0 for row in snapshots),
            }
            side = {"pitcher_id": pitcher["id"], "pitcher": pitcher["fullName"], "pitcher_throws": pitcher_throws, "pitching_team_id": pitching_team["id"], "official_date": official_date, "arsenal": arsenal, "arsenal_season": arsenal[0]["source_season"], "workload": workload, "appearance_history": appearance_history, "team_workload_history": team_workload_history, "bullpen_context": bullpen_context, "opponent_k_rate": opponent_k_rate, "lineup_confirmed": lineup_confirmed, "data_freshness_seconds": data_freshness_seconds, "batting_team": batting_team, "market_context": market_context, "batters": hitter_rows}
            side["workload_override"] = latest_workload_override(game_pk, pitcher["id"], captured_at)
            markets = latest_markets(game_pk, player_id=pitcher["id"], player_name=pitcher["fullName"], as_of=captured_at)
            market_predictions = [pitcher_k_projection(side, market, captured_at, scheduled_start) for market in markets]
            priced = [item for item in market_predictions if item.get("market") and item["market"].get("expected_value_over") is not None]
            side["projection"] = max(
                priced or market_predictions,
                key=lambda item: max(
                    (item.get("market") or {}).get("expected_value_over") or -99,
                    (item.get("market") or {}).get("expected_value_under") or -99,
                ),
            ) if market_predictions else pitcher_k_projection(side, None, captured_at, scheduled_start)
            side["market_projections"] = market_predictions
            challenger = (side.get("projection") or {}).get("workload_challenger") or {}
            if challenger.get("features"):
                side["ml_feature_snapshot_id"] = record_ml_feature_snapshot(
                    game_pk, captured_at, scheduled_start, pitcher["id"],
                    pitcher["fullName"], "pitcher_workload",
                    challenger.get("model_version") or "pitcher-workload-challenger-v1",
                    challenger.get("feature_version") or "pitcher-game-pre-event-v1",
                    challenger["features"], lineup_confirmed=lineup_confirmed,
                )
            relievers = []
            for snapshot in snapshots:
                reliever_id = snapshot["player_id"]
                reliever_arsenal = saved_pitcher_arsenal(db, reliever_id, season)
                reliever_hand = snapshot.get("throws")
                fits = {}
                if reliever_arsenal:
                    for batter in prepared_hitters:
                        reliever_hitter = hitter_profile_for_pitcher(
                            db, season, batter, reliever_id, reliever_arsenal,
                            reliever_hand, market_context, context,
                        )
                        fits[batter["id"]] = {
                            "summary": reliever_hitter["arsenal_research"],
                            "k_profile": reliever_hitter["k_profile"],
                        }
                relievers.append({
                    **snapshot,
                    "arsenal": reliever_arsenal,
                    "fits": fits,
                })
            total_mix = sum(max(0.0, number(reliever.get("mix_weight")) or 0.0) for reliever in relievers)
            for reliever in relievers:
                reliever["mix_share"] = (
                    (number(reliever.get("mix_weight")) or 0.0) / total_mix
                    if total_mix else 0.0
                )
            projection = side.get("projection") or {}
            expected_bf = projection.get("expected_batters_faced") or 22.0
            bf_interval = projection.get("batters_faced_interval")
            for hitter in hitter_rows:
                bullpen_entries = []
                for reliever in relievers:
                    fit = reliever["fits"].get(hitter["id"], {})
                    bullpen_entries.append({
                        "player_id": reliever["player_id"],
                        "name": reliever["player_name"],
                        "weight": reliever.get("mix_weight", 0.0),
                        "status": reliever.get("readiness_status"),
                        "role": reliever.get("role"),
                        "summary": fit.get("summary"),
                        "k_profile": fit.get("k_profile"),
                    })
                hitter["full_game_research"] = blend_full_game_hitter_matchup(
                    hitter, hitter["arsenal_research"], bullpen_entries,
                    expected_bf, bf_interval, market_context,
                    (projection.get("performance_outlook") or {}),
                )
                try:
                    hitter["ml_research"] = shadow_prediction(hitter, side)
                except (KeyError, TypeError, ValueError, OSError) as error:
                    hitter["ml_research"] = {
                        "available": False, "status": "unavailable",
                        "message": f"Hitter challenger unavailable: {error}",
                    }
                hitter["recent_form_snapshot_id"] = record_ml_feature_snapshot(
                    game_pk, captured_at, scheduled_start, hitter["id"], hitter["name"],
                    "hitter_spotlight", "hitter-spotlight-recent-form-v1",
                    "hitter-recent-form-pregame-v1",
                    {
                        "recent_form": hitter.get("recent_form") or {},
                        "opportunities": (hitter.get("full_game_research") or {}).get("opportunities") or {},
                    },
                    lineup_confirmed=lineup_confirmed,
                )
            public_relievers = []
            for reliever in relievers:
                hitter_fits = []
                for hitter in hitter_rows:
                    fit = reliever["fits"].get(hitter["id"], {})
                    summary = fit.get("summary") or {}
                    hitter_fits.append({
                        "id": hitter["id"], "name": hitter["name"],
                        "lineup_order": hitter.get("lineup_order"),
                        "tier": summary.get("tier", "watchlist"),
                        "tone": summary.get("tone", "neutral"),
                        "expected_average": summary.get("expected_average"),
                        "expected_slg": summary.get("expected_slg"),
                        "expected_iso": summary.get("expected_iso"),
                        "coverage": summary.get("coverage", 0.0),
                        "effective_sample_size": summary.get("effective_sample_size", 0.0),
                    })
                public_relievers.append({
                    key: value for key, value in reliever.items() if key != "fits"
                } | {"hitter_fits": hitter_fits})
            status_counts = {
                status: sum(reliever.get("readiness_status") == status for reliever in relievers)
                for status in ("fresh", "available", "limited", "unlikely", "unknown")
            }
            weighted_readiness = (
                sum((number(reliever.get("readiness_score")) or 0.0) * reliever["mix_share"] for reliever in relievers)
                if relievers else None
            )
            if weighted_readiness is None:
                readiness_label = "Not synced"
            elif weighted_readiness >= 75:
                readiness_label = "Fresh"
            elif weighted_readiness >= 50:
                readiness_label = "Available"
            else:
                readiness_label = "Taxed"
            side["bullpen"] = {
                "team_id": pitching_team["id"], "team": pitching_team["name"],
                "captured_at": snapshots[0]["captured_at"] if snapshots else None,
                "readiness_score": weighted_readiness,
                "readiness_label": readiness_label,
                "status_counts": status_counts,
                "modeled_mix": sum(reliever["mix_share"] for reliever in relievers if reliever["arsenal"]),
                "relievers": public_relievers,
                "message": None if snapshots else "No saved bullpen snapshot. Rerun python3 sync_statcast.py --all.",
            }
            return side
        home = profile(home_pitcher, away_batters, bool(away_lineup), teams["away"]["name"], teams["home"])
        away = profile(away_pitcher, home_batters, bool(home_lineup), teams["home"]["name"], teams["away"])
    if not home and not away:
        return {"ready": False, "message": "Neither probable starter has a saved MLB pitch profile yet. Rerun the matchup sync after both starters have appeared in an MLB game."}
    home = home or {"pitcher_id": home_pitcher["id"], "pitcher": home_pitcher["fullName"], "arsenal": [], "arsenal_season": "—", "lineup_confirmed": bool(away_lineup), "batters": []}
    away = away or {"pitcher_id": away_pitcher["id"], "pitcher": away_pitcher["fullName"], "arsenal": [], "arsenal_season": "—", "lineup_confirmed": bool(home_lineup), "batters": []}
    for side in (home, away):
        projection = side.get("projection")
        if projection:
            side["prediction_record"] = ensure_pregame_prediction(
                game_pk, projection, market=projection.get("market"),
            )
    sync_note = f"Profile updated {latest_sync['synced_at'][:16].replace('T', ' ')} UTC from {latest_sync['feed_count']} completed games." if latest_sync else "Saved MLB Gameday pitch profiles matched by probable starter."
    confirmed = []
    if away_lineup:
        confirmed.append("away")
    if home_lineup:
        confirmed.append("home")
    lineup_note = f"{sync_note} "
    if len(confirmed) == 2:
        lineup_note += "Both official MLB starting lineups are posted; hitters are shown in batting order."
    elif confirmed:
        lineup_note += f"The {confirmed[0]} official MLB starting lineup is posted; the other side remains an active-roster view."
    else:
        lineup_note += "Official starting lineups are not posted yet, so active non-pitchers are shown by evidence coverage."
    return {
        "ready": True, "lineup_note": lineup_note, "context": context,
        "market": game_market, "tracking": prediction_tracking_summary(),
        "home": home, "away": away,
    }

def games_for_today(now=None, scoreboard_loader=None, schedule_loader=None):
    # MLB's schedule should follow the user/server calendar day, not UTC.
    # Keep today visible while at least one game has not started, then expose tomorrow's
    # look-ahead slate even when starters and lineups are still unannounced.
    settle_finished_predictions()
    slate_day, scoreboard, is_lookahead = active_slate_date_and_scoreboard(now, scoreboard_loader)
    slate_date = slate_day.isoformat()
    schedule_loader = schedule_loader or get_json
    mlb_schedule = schedule_loader("/schedule", {"sportId": 1, "date": slate_date, "hydrate": "probablePitcher"})
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
            "start": local_game_time(event.get("date")),
            "start_time": event.get("date", ""),
            "status": event.get("status", {}).get("type", {}).get("detail", "Scheduled"),
            "espn_odds": competition.get("odds", []),
            "away": {"name": away_team.get("displayName", "Away"), "abbr": away_team.get("abbreviation", "AWY"), "pitcher": mlb_teams.get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")},
            "home": {"name": home_team.get("displayName", "Home"), "abbr": home_team.get("abbreviation", "HME"), "pitcher": mlb_teams.get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")},
            "weather": {"condition": weather.get("displayValue"), "temp": weather.get("temperature"), "wind": weather.get("wind")},
        })
        if mlb_game.get("gamePk"):
            record_game({
                "game_pk": mlb_game["gamePk"], "scheduled_start": event.get("date"),
                "official_date": slate_date,
                "away_team_id": mlb_teams.get("away", {}).get("team", {}).get("id"),
                "away_team_name": away_team.get("displayName"),
                "home_team_id": mlb_teams.get("home", {}).get("team", {}).get("id"),
                "home_team_name": home_team.get("displayName"),
                "venue_name": competition.get("venue", {}).get("fullName"),
            })
    odds_by_game, odds_message = odds_for_slate(games, slate_date)
    for game in games:
        game["odds"] = odds_by_game.get(game["gamePk"])
        game.pop("espn_odds", None)
    return {
        "date": slate_date,
        "is_lookahead": is_lookahead,
        "slate_label": "Tomorrow's look-ahead slate" if is_lookahead else "Today's MLB slate",
        "odds_message": odds_message,
        "games": games,
    }

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # This is a local development app whose HTML changes frequently. Do
        # not let an open tab keep an obsolete dashboard after the server has
        # been stopped, edited, and restarted.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 100_000:
            raise ValueError("JSON body must be between 1 and 100000 bytes")
        return json.loads(self.rfile.read(length))

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            if self.path == "/api/workload-overrides":
                payload = self._json_body()
                required = ("game_pk", "player_id", "player_name", "pitch_limit", "captured_at")
                missing = [field for field in required if payload.get(field) in (None, "")]
                if missing:
                    raise ValueError(f"Missing required fields: {', '.join(missing)}")
                with connect() as db:
                    game = db.execute("SELECT scheduled_start FROM games WHERE game_pk=?", (int(payload["game_pk"]),)).fetchone()
                if game and game["scheduled_start"] and not is_before_start(payload["captured_at"], game["scheduled_start"]):
                    raise ValueError("A pitch limit must be captured before scheduled first pitch")
                identifier = add_workload_override({
                    "captured_at": payload["captured_at"],
                    "game_pk": int(payload["game_pk"]),
                    "player_id": int(payload["player_id"]),
                    "player_name": str(payload["player_name"]).strip(),
                    "pitch_limit": float(payload["pitch_limit"]),
                    "source": str(payload.get("source") or "manual").strip(),
                    "note": str(payload.get("note") or "").strip() or None,
                })
                self._send_json(201, {"workload_override_id": identifier})
                return
            if self.path == "/api/markets":
                payload = self._json_body()
                row = normalize_row(payload, "manual-api")
                if row.get("game_pk"):
                    with connect() as db:
                        game = db.execute("SELECT scheduled_start FROM games WHERE game_pk=?", (row["game_pk"],)).fetchone()
                    if game and game["scheduled_start"] and not is_before_start(row["captured_at"], game["scheduled_start"]):
                        raise ValueError("A sportsbook line must be captured before scheduled first pitch")
                identifier = add_market_snapshot(row)
                self._send_json(201, {"market_snapshot_id": identifier, "market": row})
                return
            if self.path == "/api/predictions":
                payload = self._json_body()
                game_pk, player_id = int(payload["game_pk"]), int(payload["player_id"])
                research = matchup_research(game_pk)
                sides = [research.get("home", {}), research.get("away", {})]
                side = next((item for item in sides if item.get("pitcher_id") == player_id), None)
                if not side or not side.get("projection"):
                    raise ValueError("No pitcher projection is available for that game and player")
                identifier = save_prediction(game_pk, side["projection"])
                self._send_json(201, {"prediction_id": identifier, "prediction": side["projection"]})
                return
            self._send_json(404, {"error": "Not found"})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            self._send_json(500, {"error": f"Request failed: {str(error)[:160]}"})

    def do_GET(self):
        request_path = urlsplit(self.path).path
        if request_path == "/api/matchups":
            try:
                body = json.dumps(games_for_today()).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
            except Exception:
                self.send_error(502, "Could not reach the public MLB schedule service")
            return
        if request_path == "/api/model-status":
            settle_finished_predictions()
            self._send_json(200, prediction_tracking_summary())
            return
        if request_path.startswith("/api/matchups/") and request_path.endswith("/research"):
            try:
                game_pk = int(request_path.split("/")[3])
                body = json.dumps(matchup_research(game_pk)).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
            except Exception as error:
                body = json.dumps({"ready": False, "message": f"Research service error: {error}"}).encode()
                self.send_response(502); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
            return
        if request_path == "/matchup":
            self.path = "/index.html"
            return super().do_GET()
        if request_path in ("/", "/index.html", "/dashboard"):
            self.path = "/index.html"
            return super().do_GET()
        self.send_error(404)

if __name__ == "__main__":
    initialize()
    print(f"Diamond Intel is running at http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
