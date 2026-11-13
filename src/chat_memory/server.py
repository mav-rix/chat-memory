"""MCP server exposing the memory store over stdio."""

from __future__ import annotations

import functools
import json
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .embed import load_embedder
from .store import KINDS, Store, StoreError, default_db_path

# The instructions are the most important part of this server: they decide
# what gets remembered. Keep them short, concrete and testable.
INSTRUCTIONS = f"""\
Persistent memory that survives between chat sessions. Use it quietly; don't announce each call.

WHEN TO LOOK: when the request may depend on past sessions (the user's setup, devices, projects,
earlier decisions or preferences), call recall once with a few good keywords before answering.
The strongest hits come back in full, so you rarely need get_memory. Skip memory entirely for
self-contained requests. Call get_profile only when the user asks what you know about them or
the session is open-ended personal help.

SAVE (remember) only things likely to matter in a future session:
  - durable facts about the user, their setup, people and projects
  - preferences and corrections ("don't do X", "always Y") and the reason given
  - decisions and their why; outcomes of finished work; where things live
  - at the end of substantial work, one kind="summary" note: goal, what was done, next steps
DO NOT SAVE: passing state (free disk space, today's command output), things the files or
git history already record, guesses, chit-chat, raw tool output, or anything secret
(passwords, tokens, keys, account numbers). The server refuses obvious secrets.

HOW TO WRITE A MEMORY: one fact per memory, self-contained, with absolute dates (not "yesterday").
Title = short label. Content = What / Why / Where / How to apply, as relevant.
kind is one of: {", ".join(KINDS)}.
For a subject that will evolve (a project's status, a device's setup), pass a stable topic_key
such as "project/chat-memory" or "device/printer"; saving with the same key updates in place.

DUPLICATES AND CONFLICTS: remember checks for similar memories. If it answers needs_review,
update the existing memory instead (or pass force=true if it is genuinely different). If it
saved but lists similar memories that the new one replaces, call supersede(old_id, new_id).
When the user says something is no longer true, update or supersede; use forget only when it
should never have been saved or the user asks. restore undoes forget or supersede.

TIDYING UP: when the user asks, call review and go through near_duplicates with them; act on
each suggestion only after they agree.
"""

mcp = MCPServer(name="chat-memory", version=__version__, instructions=INSTRUCTIONS)


@functools.cache
def store() -> Store:
    return Store(default_db_path(), embedder=load_embedder())


def _log(tool: str, chars: int, ok: bool) -> None:
    try:
        store().log_call(tool, chars, ok)
    except Exception:  # the usage log must never break a tool call
        pass


def _tool_errors(fn):
    """Turn StoreError into ToolError, and log each call's size (not its content)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except StoreError as e:
            _log(fn.__name__, len(str(e)), ok=False)
            raise ToolError(str(e)) from e
        _log(fn.__name__, len(json.dumps(result, ensure_ascii=False, default=str)), ok=True)
        return result

    return wrapper


def fixed_cost_chars() -> int:
    """Text every session pays for before any call: instructions plus tool definitions."""
    import asyncio

    tools = asyncio.run(mcp.list_tools())
    defs = [t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t for t in tools]
    return len(INSTRUCTIONS) + len(json.dumps(defs, default=str))


@mcp.tool()
@_tool_errors
def remember(
    title: str,
    content: str,
    kind: str = "fact",
    topic_key: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Save a memory for future sessions. Returns action: created, updated (same topic_key),
    duplicate, unchanged, or needs_review (a near-identical memory exists; nothing saved)."""
    return store().remember(title, content, kind, topic_key, tags, project, force)


@mcp.tool()
@_tool_errors
def recall(
    query: str,
    limit: int = 8,
    project: str | None = None,
    kind: str | None = None,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """Search memories by keywords and meaning (when the embedding model is installed).
    Ranked results; the strongest come with full content, the rest with a preview
    (use get_memory for those)."""
    return store().recall(query, limit, project, kind, include_superseded)


@mcp.tool()
@_tool_errors
def get_memory(memory_id: int, with_history: bool = False) -> dict[str, Any]:
    """Get one memory in full, optionally with its earlier versions."""
    return store().get(memory_id, with_history)


@mcp.tool()
@_tool_errors
def update_memory(
    memory_id: int,
    title: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    kind: str | None = None,
    pinned: bool | None = None,
) -> dict[str, Any]:
    """Change an existing memory. The previous text is kept as a version. pinned=true puts it
    in every session's profile."""
    return store().update(memory_id, title, content, tags, kind, pinned)


@mcp.tool()
@_tool_errors
def supersede(old_id: int, new_id: int) -> dict[str, Any]:
    """Mark old_id as replaced by new_id. The old memory is hidden from recall but kept."""
    return store().supersede(old_id, new_id)


@mcp.tool()
@_tool_errors
def forget(memory_id: int, reason: str = "") -> dict[str, Any]:
    """Hide a memory that should not have been saved or that the user asked to remove."""
    return store().forget(memory_id, reason)


@mcp.tool()
@_tool_errors
def restore(memory_id: int) -> dict[str, Any]:
    """Bring back a forgotten or superseded memory."""
    return store().restore(memory_id)


@mcp.tool()
@_tool_errors
def list_recent(limit: int = 10, project: str | None = None) -> list[dict[str, Any]]:
    """Most recently saved or updated memories."""
    return store().recent(limit, project)


@mcp.tool()
@_tool_errors
def get_profile(project: str | None = None) -> dict[str, Any]:
    """Session-start briefing: pinned memories, preferences and the most-used facts."""
    return {"memories": store().profile(project), "stats": store().stats()}


@mcp.tool()
@_tool_errors
def review(stale_days: int = 180) -> dict[str, Any]:
    """Memories worth tidying, for when the user asks to clean up memory: near_duplicates
    (pairs that look like the same fact or a changed fact, each with a suggestion), stale
    (unused for stale_days), superseded, repeated (saved many times) and forgotten."""
    return store().review(stale_days)


def _print_review(r: dict[str, Any]) -> None:
    line = lambda m: f"  [{m['id']}] {m['title']}: {m['preview']}"  # noqa: E731
    if r["near_duplicates"]:
        print(f"Near-duplicates and possible conflicts ({len(r['near_duplicates'])}):")
        for p in r["near_duplicates"]:
            print(f"- {p['label']} ({p['similarity']})")
            print(line(p["older"]))
            print(line(p["newer"]))
            print(f"  → {p['suggestion']}")
    for key, heading in (
        ("stale", "Not used for a long time"),
        ("repeated", "Saved repeatedly"),
        ("superseded", "Superseded (hidden from search)"),
        ("forgotten", "Forgotten (removed for good by: chat-memory purge)"),
    ):
        if r[key]:
            print(f"{heading} ({len(r[key])}):")
            for m in r[key]:
                print(line(m))
    if not any(r.values()):
        print("Nothing to tidy.")


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    from . import transfer

    p = argparse.ArgumentParser(prog="chat-memory", description="Local MCP memory server.")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run the MCP server over stdio (the default)")
    sub.add_parser("download-model", help="fetch the embedding model for semantic search (~90 MB)")
    sub.add_parser("reindex", help="embed any memories that are missing embeddings")
    sub.add_parser("stats", help="show memory counts and search mode")
    us = sub.add_parser("usage", help="estimated tokens used by memory tool calls")
    us.add_argument("--days", type=int, default=30)
    ex = sub.add_parser("export", help="back up memories as JSON Lines")
    ex.add_argument("file", nargs="?", default="-", help="output file (default: standard output)")
    ex.add_argument("--include-forgotten", action="store_true", help="also export forgotten memories")
    im = sub.add_parser("import", help="merge a chat-memory export or a Claude Code memory folder")
    im.add_argument("path", type=Path, help="export file (.jsonl) or a folder of Claude Code memory files")
    im.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    rv = sub.add_parser("review", help="list memories worth tidying")
    rv.add_argument("--stale-days", type=int, default=180)
    rv.add_argument("--json", action="store_true", help="print JSON instead of text")
    pg = sub.add_parser("purge", help="permanently delete memories forgotten a while ago")
    pg.add_argument("--days", type=int, default=30, help="forgotten more than this many days ago (default 30)")
    pg.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    args = p.parse_args(argv)

    if args.cmd not in (None, "serve"):
        # Windows consoles and redirected output may not be UTF-8; never crash on a "→".
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        _run(args, json, sys, transfer)
    except StoreError as e:
        sys.exit(str(e))


def _run(args, json, sys, transfer) -> None:
    from pathlib import Path

    if args.cmd in (None, "serve"):
        mcp.run()
    elif args.cmd == "download-model":
        from .embed import download_model

        try:
            path = download_model()
        except ImportError:
            sys.exit("Install the semantic extra first: pip install 'chat-memory[semantic]'")
        except Exception as e:  # network or checksum problems
            sys.exit(f"Download failed: {e}")
        print(f"Model ready in {path}. Semantic search turns on the next time the server starts.")
        if load_embedder() is None:
            print("Note: onnxruntime/tokenizers/numpy are missing; install 'chat-memory[semantic]'.")
    elif args.cmd == "reindex":
        if store().embedder is None:
            sys.exit("Semantic search is not available (run: chat-memory download-model).")
        print(f"Embedded {store().backfill_embeddings()} memories.")
    elif args.cmd == "stats":
        print(json.dumps({"database": str(default_db_path()), **store().stats()}, indent=2))
    elif args.cmd == "usage":
        u = store().usage(args.days)
        fixed = round(fixed_cost_chars() / 4)
        print(f"Fixed cost per session (instructions + tool definitions): ~{fixed} tokens")
        print(f"Tool calls in the last {u['days']} days: ~{u['est_tokens_total']} tokens returned")
        for t in u["tools"]:
            err = f", {t['errors']} errors" if t["errors"] else ""
            print(f"  {t['tool']:<14} {t['calls']:>4} calls  ~{t['est_tokens']:>6} tokens  "
                  f"(~{t['avg_tokens_per_call']} per call{err})")
        print("Estimates use 4 characters ≈ 1 token.")
    elif args.cmd == "export":
        if args.file == "-":
            transfer.export_jsonl(store(), sys.stdout, args.include_forgotten)
            return
        path = Path(args.file).expanduser()
        n = transfer.export_to_file(store(), path, args.include_forgotten)
        print(f"Exported {n} memories to {path} (readable only by you).", file=sys.stderr)
        if transfer.inside_git_repo(path):
            print("Warning: that file is inside a git repository. It holds your memories; "
                  "don't commit it.", file=sys.stderr)
    elif args.cmd == "import":
        if not args.path.exists():
            sys.exit(f"No such file or folder: {args.path}")
        report = store().import_records(transfer.read_any(args.path), dry_run=args.dry_run)
        transfer.print_report(report, args.dry_run)
        if not args.dry_run and report["created"] + report["updated"] and store().embedder:
            print(f"Embedded {store().backfill_embeddings()} memories for search.")
    elif args.cmd == "review":
        r = store().review(args.stale_days)
        print(json.dumps(r, indent=2)) if args.json else _print_review(r)
    elif args.cmd == "purge":
        s = store()
        n = s.purgeable(args.days)
        if not n:
            print(f"Nothing forgotten more than {args.days} days ago.")
            return
        if not args.yes:
            answer = input(f"Permanently delete {n} forgotten memories (and their history)? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("Nothing deleted.")
                return
        print(f"Deleted {s.purge(args.days)} memories.")


if __name__ == "__main__":
    main()
