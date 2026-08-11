# Diamond Intel

This version separates research-data collection from the browser. The board only
reads saved analytics from `diamond_intel.db`. The sync uses MLB Gameday pitch
feeds to calculate exact pitch type, velocity, a 3×3 plate-location profile,
and opposing-hitter strikeout and out results for each probable starter.

## Start the board

```bash
python3 server.py
```

Then open http://localhost:8000.

## Sync research data

In a second Terminal window, sync the game you want. The `gamePk` is in the
Baseball Savant preview URL after `game_pk=`:

```bash
python3 sync_statcast.py --game-pk 823917
```

To build profiles for every game with two announced probable starters today:

```bash
python3 sync_statcast.py --all
```

The first all-slate sync can take a while because it downloads completed game
feeds for every team. Later runs reuse the `.gameday_cache` files.

The first sync downloads the two teams’ completed regular-season game feeds;
later runs reuse the local feed cache. It calculates a starter’s pitch usage,
average velocity, and location pattern, then calculates each active hitter’s
performance against those exact pitch codes. It also powers the pitcher
outlook’s usage-weighted K opportunity and out-conversion estimates. Once it
finishes, refresh the board and choose that game.

After updating to a version with the pitcher outlook, run `python3
sync_statcast.py --all` once so the new strikeout and out fields are populated.

The current version also calculates pitcher workload, a workload-adjusted K
projection, and hitter pitch-quality signals (K%, whiff%, hard-hit%, and a
conservative barrel proxy) from those same feeds. Run the all-slate sync once
again after this update to populate them. MLB Gameday does not provide official
Statcast xwOBA, so the board intentionally does not invent an xwOBA value.

The all-slate sync also accumulates home-plate umpire outcomes from completed
MLB feeds. An umpire is labeled pitcher-friendly, hitter-friendly, or neutral
only after the live game feed names the home-plate umpire and the local cache
contains at least eight of that umpire's games; otherwise the board reports the
assignment or limited-sample status without guessing.
