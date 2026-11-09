"""
A tiny local MCP server that hands ESPN's live NFL data to the private Prop Board.

The desktop MCP host starts it (server "espn"); the private board calls
it as host:espn every 10 seconds. Two read-only tools, both returning live_feed.py's trimmed JSON:
    scoreboard           -- this week's games, status and scores
    summary {event}      -- one started game's box score, drives and win probability
Standard library only: MCP over stdio is newline-delimited JSON-RPC 2.0.
"""
import json
import sys

import live_feed

TOOLS = [
    {"name": "scoreboard", "description": "This week's NFL games from ESPN: status and scores.",
     "inputSchema": {"type": "object", "properties": {}}, "annotations": {"readOnlyHint": True}},
    {"name": "summary", "description": "One NFL game's box score, drives and win probability from ESPN.",
     "inputSchema": {"type": "object", "properties": {"event": {"type": "string", "description": "ESPN event id"}},
                     "required": ["event"]}, "annotations": {"readOnlyHint": True}},
]


def call(name: str, args: dict) -> dict:
    if name == "scoreboard":
        return live_feed.trim_scoreboard(live_feed.get("scoreboard"))
    if name == "summary":
        event = str(args.get("event", ""))
        if not event.isdigit():
            raise ValueError("event must be an ESPN event id")
        return live_feed.trim_summary(live_feed.get(f"summary?event={event}"))
    raise ValueError(f"unknown tool {name}")


def handle(msg: dict):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:   # a notification (e.g. notifications/initialized): no reply
        return None
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "espn", "version": "1.0.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params", {})
        try:
            data = call(p.get("name"), p.get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(data, separators=(",", ":"))}], "structuredContent": data}
        except Exception as ex:   # a failed ESPN fetch is a tool error, not a protocol error
            result = {"content": [{"type": "text", "text": f"{type(ex).__name__}: {ex}"}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
