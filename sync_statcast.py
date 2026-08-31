"""Compatibility command for the Baseball Research local matchup sync.

This no longer scrapes Baseball Savant player pages. It delegates to the MLB
Gameday pitch-feed sync, retaining the familiar command name.
"""
from sync_matchup_data import main

if __name__ == "__main__":
    main()
