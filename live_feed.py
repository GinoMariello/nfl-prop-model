"""
Live scores for the Prop Board, pulled outside the page (the published board can't reach ESPN itself).

    python3 live_feed.py          # writes site/live.json; prints CHANGED or UNCHANGED

ESPN's public scoreboard plus a trimmed summary (box score, drives with plays, win probability)
for every game that has started. The board's Live tab reads site/live.json when it can't reach
ESPN directly; the live-scores scheduled task runs this every 7 minutes during game windows and
republishes the board with it.
"""
import hashlib
import json
import os
import socket
import urllib.request
from datetime import datetime, timezone

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "live.json")
socket.setdefaulttimeout(30)


def get(path: str) -> dict:
    with urllib.request.urlopen(ESPN + path) as r:
        return json.load(r)


def pick(d: dict, *keys):
    return {k: d[k] for k in keys if k in d}


def trim_scoreboard(sb: dict) -> dict:
    """Only what the Live tab reads: each event's id, status and the two teams with scores."""
    events = []
    for e in sb.get("events", []):
        c = e["competitions"][0]
        events.append({"id": e["id"], "competitions": [{
            "status": c["status"],
            "competitors": [{"homeAway": t["homeAway"], "score": t.get("score"),
                             "team": pick(t["team"], "id", "displayName", "abbreviation")} for t in c["competitors"]],
        }]})
    return {"events": events}


def trim_play(p: dict) -> dict:
    return {**pick(p, "id", "text", "scoringPlay", "statYardage"),
            "period": pick(p.get("period", {}), "number"), "clock": pick(p.get("clock", {}), "displayValue"),
            "type": pick(p.get("type", {}), "text"),
            "start": pick(p.get("start", {}), "yardLine", "distance", "yardsToEndzone", "downDistanceText"),
            "end": pick(p.get("end", {}), "yardLine")}


def trim_drive(d: dict) -> dict:
    return {**pick(d, "id", "displayResult", "shortDisplayResult", "description", "isScore"),
            "team": pick(d.get("team", {}), "id", "abbreviation"),
            "start": pick(d.get("start", {}), "yardLine", "text"), "end": pick(d.get("end", {}), "yardLine", "text"),
            "plays": [trim_play(p) for p in d.get("plays", [])]}


def trim_summary(s: dict) -> dict:
    """Box score stat tables, drives and win probability -- the fields boxStats/allPlays/the field view use."""
    players = [{"statistics": [{"name": c.get("name"), "keys": c.get("keys", []),
                                "athletes": [{"athlete": pick(a.get("athlete", {}), "displayName"), "stats": a.get("stats", [])}
                                             for a in c.get("athletes", [])]}
                               for c in t.get("statistics", [])]}
               for t in s.get("boxscore", {}).get("players", [])]
    drives = s.get("drives", {})
    return {"boxscore": {"players": players},
            "drives": {"previous": [trim_drive(d) for d in drives.get("previous", [])],
                       **({"current": trim_drive(drives["current"])} if drives.get("current") else {})},
            "winprobability": [pick(w, "homeWinPercentage", "playId") for w in s.get("winprobability", [])]}


def build() -> dict:
    sb = trim_scoreboard(get("scoreboard"))
    summaries = {}
    for e in sb["events"]:
        if e["competitions"][0]["status"]["type"]["state"] != "pre":
            summaries[e["id"]] = trim_summary(get(f"summary?event={e['id']}"))
    return {"scoreboard": sb, "summaries": summaries}


def main() -> None:
    feed = build()
    digest = lambda f: hashlib.sha1(json.dumps({k: f.get(k) for k in ("scoreboard", "summaries")},
                                               separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    old = None
    if os.path.exists(OUT):
        with open(OUT) as f:
            old = json.load(f)
    changed = old is None or digest(old) != digest(feed)
    feed["at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(OUT, "w") as f:
        json.dump(feed, f, separators=(",", ":"))
    live = sum(e["competitions"][0]["status"]["type"]["state"] == "in" for e in feed["scoreboard"]["events"])
    print(f"{'CHANGED' if changed else 'UNCHANGED'} {live} live games, {len(feed['summaries'])} with box scores, "
          f"{os.path.getsize(OUT) // 1024} KB -> {OUT}")


if __name__ == "__main__":
    main()
