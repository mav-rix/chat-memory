"""Export and import.

Export format: JSON Lines. The first line is a header
``{"format": "chat-memory", "version": 1, "exported_at": ...}``; each following line is one
memory with its version history. Embeddings are not exported (they're rebuilt on import).

Import also reads a Claude Code memory folder (Markdown files with front matter), so
memories kept there can be brought across.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .store import Store, StoreError, _now

FORMAT = "chat-memory"
FORMAT_VERSION = 1


def inside_git_repo(path: Path) -> bool:
    p = path.resolve().parent
    return any((d / ".git").exists() for d in (p, *p.parents))


def export_jsonl(store: Store, out, include_forgotten: bool = False) -> int:
    header = {"format": FORMAT, "version": FORMAT_VERSION, "exported_at": _now()}
    out.write(json.dumps(header) + "\n")
    n = 0
    for rec in store.export_records(include_forgotten):
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    return n


def export_to_file(store: Store, path: Path, include_forgotten: bool = False) -> int:
    """Write an export readable only by the owner (it holds personal data)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):  # POSIX; on Windows the user profile's permissions apply
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        return export_jsonl(store, f, include_forgotten)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        try:
            header = json.loads(first)
        except json.JSONDecodeError:
            header = None
        if not isinstance(header, dict) or header.get("format") != FORMAT:
            raise StoreError(f"{path} is not a chat-memory export (missing header line).")
        if int(header.get("version", 0)) > FORMAT_VERSION:
            raise StoreError(f"{path} was written by a newer chat-memory; upgrade first.")
        for n, line in enumerate(f, 2):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise StoreError(f"{path}, line {n}: not valid JSON ({e.msg}).") from e
            if isinstance(rec, dict):
                rec["_source"] = f"line {n}"
                yield rec


# ── Claude Code memory folders ─────────────────────────────────────────────
_CC_KINDS = {"user": "fact", "feedback": "preference", "project": "project", "reference": "fact"}


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if m := re.match(r"^\s*(name|description|type)\s*:\s*(.+?)\s*$", line):
            meta.setdefault(m[1], m[2].strip("\"'"))
    return meta, text[end + 4 :].lstrip("\n")


def read_claude_code_dir(directory: Path) -> Iterator[dict[str, Any]]:
    """Each ``*.md`` memory file (except the MEMORY.md index) becomes one record."""
    for path in sorted(directory.glob("*.md")):
        if path.name == "MEMORY.md":
            continue
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        if not body.strip():
            continue
        name = meta.get("name") or path.stem
        when = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
        yield {
            "_source": path.name,
            "title": meta.get("description") or name,
            "content": body.strip(),
            "kind": _CC_KINDS.get(meta.get("type", ""), "fact"),
            "topic_key": f"claude-code/{name}",
            "tags": ["claude-code"],
            "created_at": when,
            "updated_at": when,
        }


def read_any(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_dir():
        return read_claude_code_dir(path)
    return read_jsonl(path)


def print_report(report: dict[str, Any], dry_run: bool, out=None) -> None:
    out = out or sys.stdout
    verb = "Would import" if dry_run else "Imported"
    print(
        f"{verb}: {report['created']} new, {report['updated']} updated (newer version of a topic), "
        f"{report['duplicate']} already present, {report['older']} older than what you have.",
        file=out,
    )
    if report["refused"]:
        print(f"Refused {len(report['refused'])} (looked like secrets or were empty):", file=out)
        for r in report["refused"]:
            print(f"  {r['source']}: {r['reason']}", file=out)
    if report["history_dropped"]:
        print(f"Dropped {report['history_dropped']} old versions that looked like secrets.", file=out)
