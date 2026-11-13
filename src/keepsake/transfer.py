"""Export and import.

Export format: JSON Lines. The first line is a header
``{"format": "keepsake", "version": 1, "exported_at": ...}``; each following line is one
memory with its version history. Embeddings are not exported (they're rebuilt on import).

Import also reads a Claude Code memory folder (Markdown files with front matter) and other
tools' SQLite memory databases (opened read-only; columns mapped by name, or by hand).
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

FORMAT = "keepsake"
_FORMATS = (FORMAT, "chat-memory")  # exports made before the rename still import
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
        if not isinstance(header, dict) or header.get("format") not in _FORMATS:
            raise StoreError(f"{path} is not a keepsake export (missing header line).")
        if int(header.get("version", 0)) > FORMAT_VERSION:
            raise StoreError(f"{path} was written by a newer keepsake; upgrade first.")
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


# ── other SQLite memory databases ──────────────────────────────────────────
# Column names that other memory tools commonly use, best guess first.
FIELD_GUESSES: dict[str, tuple[str, ...]] = {
    "content": ("content", "text", "body", "memory", "observation", "note", "notes", "document",
                "message", "value", "description", "data"),
    "title": ("title", "name", "summary", "subject", "label", "topic", "heading"),
    "kind": ("kind", "type", "memory_type", "category", "observation_type", "entity_type"),
    "tags": ("tags", "labels", "keywords"),
    "topic_key": ("topic_key",),
    "project": ("project", "scope", "namespace", "workspace"),
    "created_at": ("created_at", "created_at_iso", "created", "timestamp", "inserted_at", "date", "time"),
    "updated_at": ("updated_at", "updated_at_iso", "modified_at", "updated", "last_modified", "modified"),
    "deleted_at": ("deleted_at", "deleted", "is_deleted", "archived"),
}
_SHADOW = re.compile(r"_(data|idx|content|docsize|config|chunks|rowids|vector_chunks\d*|info)$")


def is_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def _open_readonly(path: Path):
    import sqlite3

    db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    except sqlite3.DatabaseError as e:
        db.close()
        raise StoreError(
            f"Couldn't read {path} ({e}). If the other memory tool is running, close it first, "
            "or copy the .db file (with any -wal file next to it) and import the copy."
        ) from e
    return db


def _tables(db) -> list[dict[str, Any]]:
    out = []
    for (name, sql) in db.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall():
        if name.startswith("sqlite_") or _SHADOW.search(name) or (sql or "").upper().startswith("CREATE VIRTUAL"):
            continue
        cols = [(r["name"], (r["type"] or "").upper()) for r in db.execute(f'PRAGMA table_info("{name}")')]
        rows = db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        out.append({"table": name, "rows": rows, "columns": cols})
    return out


def guess_mapping(columns: list[tuple[str, str]]) -> dict[str, str]:
    """Map keepsake fields to the columns most likely to hold them."""
    names = {c.lower(): c for c, _ in columns}
    mapping: dict[str, str] = {}
    for field, options in FIELD_GUESSES.items():
        for opt in options:
            if opt in names and names[opt] not in mapping.values():
                mapping[field] = names[opt]
                break
    return mapping


def _pick_table(tables: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [t for t in tables if "content" in guess_mapping(t["columns"])]
    if not usable and len(tables) == 1:
        return tables[0]  # the only table: ask which column holds the text
    return max(usable, key=lambda t: (t["rows"], len(guess_mapping(t["columns"]))), default=None)


def inspect_sqlite(path: Path) -> dict[str, Any]:
    """Tables, columns, row counts and the suggested mapping. Never any row content."""
    db = _open_readonly(path)
    try:
        tables = _tables(db)
    finally:
        db.close()
    best = _pick_table(tables)
    return {
        "tables": tables,
        "suggested_table": best["table"] if best else None,
        "suggested_mapping": guess_mapping(best["columns"]) if best else {},
    }


def _time(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d{9,13}(\.\d+)?", value)):
        t = float(value)
        t = t / 1000 if t > 1e11 else t  # milliseconds
        return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds")
    return str(value).replace(" ", "T", 1) if re.match(r"\d{4}-\d\d-\d\d \d", str(value)) else str(value)


def _tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return [str(t) for t in json.loads(value)]
        except json.JSONDecodeError:
            pass
    return [t for t in re.split(r"[,;\s]+", str(value)) if t]


def read_sqlite(
    path: Path, table: str | None = None, mapping: dict[str, str] | None = None, kind: str | None = None
) -> Iterator[dict[str, Any]]:
    """Records from another tool's SQLite database, opened read-only."""
    from .store import KINDS

    db = _open_readonly(path)
    try:
        tables = {t["table"]: t for t in _tables(db)}
        chosen = tables.get(table) if table else _pick_table(list(tables.values()))
        if chosen is None:
            raise StoreError(
                f"No table '{table}' in {path}." if table
                else f"Couldn't find a table of memories in {path}; run `keepsake inspect` and pass --table."
            )
        cols = {c for c, _ in chosen["columns"]}
        m = {**guess_mapping(chosen["columns"]), **(mapping or {})}
        if bad := [f"{k}={v}" for k, v in m.items() if v not in cols]:
            raise StoreError(f"Table {chosen['table']} has no column for: {', '.join(bad)}.")
        if "content" not in m:
            raise StoreError(f"Which column of {chosen['table']} holds the memory text? Pass --map content=COLUMN.")
        for n, row in enumerate(db.execute(f'SELECT * FROM "{chosen["table"]}"'), 1):
            get = lambda f: row[m[f]] if f in m else None  # noqa: E731
            deleted = get("deleted_at")
            if deleted not in (None, "", 0, "0", False):
                continue  # deleted in the old tool: not migrated
            content = str(get("content") or "").strip()
            title = str(get("title") or "").strip() or content.split("\n", 1)[0][:80]
            old_kind = str(get("kind") or "").strip().lower()
            tags = _tags(get("tags"))
            if old_kind and old_kind not in KINDS:
                tags.append(f"type:{old_kind}")
            yield {
                "_source": f"{chosen['table']} row {n}",
                "title": title,
                "content": content,
                "kind": kind or (old_kind if old_kind in KINDS else "fact"),
                "tags": tags + ["imported"],
                "topic_key": get("topic_key"),
                "project": get("project"),
                "created_at": _time(get("created_at")),
                "updated_at": _time(get("updated_at")) or _time(get("created_at")),
            }
    finally:
        db.close()


def read_any(path: Path, **sqlite_options: Any) -> Iterator[dict[str, Any]]:
    if path.is_dir():
        return read_claude_code_dir(path)
    if is_sqlite(path):
        return read_sqlite(path, **sqlite_options)
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
    if report.get("redacted"):
        print(f"Removed card expiry dates/PINs/security codes from {report['redacted']} memories.", file=out)
    if report["history_dropped"]:
        print(f"Dropped {report['history_dropped']} old versions that looked like secrets.", file=out)
