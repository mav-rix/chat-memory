"""MCP server exposing the memory store over stdio."""

from __future__ import annotations

import functools
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .store import KINDS, Store, StoreError, default_db_path

# The instructions are the most important part of this server: they decide
# what gets remembered. Keep them short, concrete and testable.
INSTRUCTIONS = f"""\
Persistent memory that survives between chat sessions. Use it quietly; don't announce each call.

START OF A SESSION: call get_profile once to load pinned facts and preferences. When a topic
comes up that past sessions may have covered, call recall before answering.

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
should never have been saved or the user asks.
"""

mcp = MCPServer(name="chat-memory", version=__version__, instructions=INSTRUCTIONS)


@functools.cache
def store() -> Store:
    return Store(default_db_path())


def _tool_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except StoreError as e:
            raise ToolError(str(e)) from e

    return wrapper


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
    """Search memories by keywords. Returns short previews; use get_memory for full text."""
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
    """Memories worth tidying: unused for stale_days, superseded, or saved repeatedly.
    Use when the user asks to clean up memory."""
    return store().review(stale_days)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
