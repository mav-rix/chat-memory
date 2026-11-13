"""SQLite storage for chat memory.

One file, no server. Full-text search uses SQLite's built-in FTS5.

Design (borrowed ideas, our own code):
  * topic keys: one evolving subject keeps one row; updates keep old versions
    (idea from Engram, MIT).
  * normalized-hash exact-duplicate detection (Engram).
  * superseding instead of deleting (mcp-memory-service, Apache-2.0).
  * near-duplicate check before saving, with the decision left to the AI.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .redact import find_secrets, strip_private

SCHEMA_VERSION = 1

KINDS = ("fact", "preference", "project", "decision", "event", "person", "howto", "summary")

# Word-overlap thresholds (Jaccard) for the near-duplicate check.
DUPLICATE_THRESHOLD = 0.8  # at or above: probably the same memory, don't save without force
RELATED_THRESHOLD = 0.25  # at or above: related, worth showing to the AI (tune with real data)

_STOPWORDS = frozenset(
    """a an and are as at be been but by can did do does for from had has have he her his i if in
    into is it its me my no not of on or our she so than that the their them then there these they
    this to too us was we were what when which who will with you your""".split()
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_key       TEXT,
    kind            TEXT    NOT NULL DEFAULT 'fact',
    title           TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    tags            TEXT    NOT NULL DEFAULT '',
    project         TEXT,
    normalized_hash TEXT    NOT NULL,
    duplicate_count INTEGER NOT NULL DEFAULT 1,
    revision_count  INTEGER NOT NULL DEFAULT 1,
    use_count       INTEGER NOT NULL DEFAULT 0,
    pinned          INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    last_used_at    TEXT,
    superseded_by   INTEGER REFERENCES memories(id),
    deleted_at      TEXT,
    delete_reason   TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_topic
    ON memories(topic_key) WHERE topic_key IS NOT NULL AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_mem_hash    ON memories(normalized_hash);
CREATE INDEX IF NOT EXISTS idx_mem_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_mem_updated ON memories(updated_at DESC);

CREATE TABLE IF NOT EXISTS memory_versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id  INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    version    INTEGER NOT NULL,
    title      TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    UNIQUE(memory_id, version)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    title, content, tags,
    content='memories', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, title, content, tags) VALUES (new.id, new.title, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
        VALUES ('delete', old.id, old.title, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF title, content, tags ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
        VALUES ('delete', old.id, old.title, old.content, old.tags);
    INSERT INTO memories_fts(rowid, title, content, tags) VALUES (new.id, new.title, new.content, new.tags);
END;
"""

_ACTIVE = "deleted_at IS NULL AND superseded_by IS NULL"


class StoreError(Exception):
    """A request the store refuses (bad input, secrets, unknown id)."""


def default_db_path() -> Path:
    if env := os.environ.get("CHAT_MEMORY_DB"):
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(base).expanduser() / "chat-memory" / "memory.db"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _normalized_hash(title: str, content: str) -> str:
    norm = " ".join(f"{title}\n{content}".lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()


def similarity(a: str, b: str) -> float:
    """Word-overlap (Jaccard) similarity between two texts, 0..1."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _fts_query(text: str, max_terms: int = 16) -> str | None:
    """Turn free text into a safe FTS5 OR-query (user text never reaches FTS syntax)."""
    terms: list[str] = []
    for w in re.findall(r"[A-Za-z0-9]+", text):
        lw = w.lower()
        if len(lw) > 1 and lw not in _STOPWORDS and lw not in terms:
            terms.append(lw)
        if len(terms) >= max_terms:
            break
    return " OR ".join(f'"{t}"' for t in terms) if terms else None


def _clean_tags(tags: list[str] | str | None) -> str:
    if not tags:
        return ""
    items = tags.split(",") if isinstance(tags, str) else tags
    return ",".join(sorted({t.strip().lower() for t in items if t.strip()}))


@dataclass
class Store:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            new_file = not self.path.exists()
        else:
            new_file = False
        self.db = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(_SCHEMA)
        self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        if new_file:
            os.chmod(self.path, 0o600)

    def close(self) -> None:
        self.db.close()

    # ── helpers ────────────────────────────────────────────────────────────
    def _row(self, memory_id: int) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise StoreError(f"No memory with id {memory_id}.")
        return row

    @staticmethod
    def _summary(row: sqlite3.Row, preview: int = 200) -> dict[str, Any]:
        content = row["content"]
        return {
            "id": row["id"],
            "title": row["title"],
            "kind": row["kind"],
            "topic_key": row["topic_key"],
            "project": row["project"],
            "tags": row["tags"].split(",") if row["tags"] else [],
            "preview": content if len(content) <= preview else content[:preview].rstrip() + "…",
            "updated_at": row["updated_at"],
            "pinned": bool(row["pinned"]),
        }

    @staticmethod
    def _full(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["tags"] = row["tags"].split(",") if row["tags"] else []
        d["pinned"] = bool(row["pinned"])
        d.pop("normalized_hash", None)
        return d

    def _check_text(self, title: str, content: str) -> tuple[str, str]:
        title, content = strip_private(title).strip(), strip_private(content).strip()
        if not title or not content:
            raise StoreError("Both title and content are required.")
        if found := find_secrets(f"{title}\n{content}"):
            raise StoreError(
                "Refused: this looks like it contains a secret (" + ", ".join(found) + "). "
                "Never store credentials. Remove it, or wrap the sensitive part in <private>…</private>."
            )
        return title, content

    def _save_version(self, row: sqlite3.Row) -> None:
        self.db.execute(
            "INSERT INTO memory_versions(memory_id, version, title, content, created_at) VALUES (?,?,?,?,?)",
            (row["id"], row["revision_count"], row["title"], row["content"], row["updated_at"]),
        )

    def similar(self, text: str, limit: int = 3, exclude_id: int | None = None) -> list[dict[str, Any]]:
        """Active memories most similar to ``text`` by word overlap (FTS5 finds candidates)."""
        q = _fts_query(text, max_terms=24)
        if not q:
            return []
        rows = self.db.execute(
            f"""SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.rowid
                WHERE memories_fts MATCH ? AND m.deleted_at IS NULL AND m.superseded_by IS NULL
                ORDER BY bm25(memories_fts) LIMIT 25""",
            (q,),
        ).fetchall()
        scored = []
        for r in rows:
            if r["id"] == exclude_id:
                continue
            s = similarity(text, f"{r['title']}\n{r['content']}")
            if s >= RELATED_THRESHOLD:
                scored.append((s, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**self._summary(r), "similarity": round(s, 2)} for s, r in scored[:limit]]

    # ── write operations ───────────────────────────────────────────────────
    def remember(
        self,
        title: str,
        content: str,
        kind: str = "fact",
        topic_key: str | None = None,
        tags: list[str] | str | None = None,
        project: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        title, content = self._check_text(title, content)
        kind = kind.strip().lower() or "fact"
        if kind not in KINDS:
            raise StoreError(f"Unknown kind '{kind}'. Use one of: {', '.join(KINDS)}.")
        topic_key = topic_key.strip().lower() if topic_key and topic_key.strip() else None
        tags_s, now = _clean_tags(tags), _now()
        h = _normalized_hash(title, content)

        # 1. Same topic key → update that memory in place (keeping history).
        if topic_key:
            row = self.db.execute(
                "SELECT * FROM memories WHERE topic_key = ? AND deleted_at IS NULL", (topic_key,)
            ).fetchone()
            if row:
                if row["normalized_hash"] == h:
                    self.db.execute(
                        "UPDATE memories SET duplicate_count = duplicate_count + 1, updated_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
                    return {"action": "unchanged", "id": row["id"], "message": "Topic already says exactly this."}
                self.db.execute("BEGIN")
                self._save_version(row)
                self.db.execute(
                    """UPDATE memories SET title=?, content=?, kind=?, tags=?, project=COALESCE(?, project),
                       normalized_hash=?, revision_count=revision_count+1, updated_at=?, superseded_by=NULL
                       WHERE id=?""",
                    (title, content, kind, tags_s or row["tags"], project, h, now, row["id"]),
                )
                self.db.execute("COMMIT")
                return {
                    "action": "updated",
                    "id": row["id"],
                    "message": f"Updated topic '{topic_key}' (previous version kept).",
                }

        # 2. Exact duplicate of an active memory → just count it.
        row = self.db.execute(
            f"SELECT * FROM memories WHERE normalized_hash = ? AND {_ACTIVE}", (h,)
        ).fetchone()
        if row:
            self.db.execute(
                "UPDATE memories SET duplicate_count = duplicate_count + 1, updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            return {"action": "duplicate", "id": row["id"], "message": "Already remembered; nothing new saved."}

        # 3. Near-duplicates → let the AI decide unless it already has (force).
        similar = self.similar(f"{title}\n{content}")
        if not force and similar and similar[0]["similarity"] >= DUPLICATE_THRESHOLD:
            return {
                "action": "needs_review",
                "id": None,
                "similar": similar,
                "message": (
                    "Not saved: a very similar memory exists. Update it with update_memory, "
                    "or call remember again with force=true if this is genuinely different."
                ),
            }

        cur = self.db.execute(
            """INSERT INTO memories(topic_key, kind, title, content, tags, project, normalized_hash,
                                    created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (topic_key, kind, title, content, tags_s, project, h, now, now),
        )
        result: dict[str, Any] = {"action": "created", "id": cur.lastrowid}
        if similar:
            result["similar"] = similar
            result["message"] = (
                "Saved. Related memories exist; if the new one replaces any of them, "
                "call supersede(old_id, new_id)."
            )
        return result

    def update(
        self,
        memory_id: int,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | str | None = None,
        kind: str | None = None,
        pinned: bool | None = None,
    ) -> dict[str, Any]:
        row = self._row(memory_id)
        if row["deleted_at"]:
            raise StoreError(f"Memory {memory_id} was forgotten; save a new one instead.")
        new_title, new_content = self._check_text(title or row["title"], content or row["content"])
        if kind is not None and kind.strip().lower() not in KINDS:
            raise StoreError(f"Unknown kind '{kind}'. Use one of: {', '.join(KINDS)}.")
        text_changed = (new_title, new_content) != (row["title"], row["content"])
        self.db.execute("BEGIN")
        if text_changed:
            self._save_version(row)
        self.db.execute(
            """UPDATE memories SET title=?, content=?, tags=?, kind=?, pinned=?, normalized_hash=?,
               revision_count = revision_count + ?, updated_at=? WHERE id=?""",
            (
                new_title,
                new_content,
                _clean_tags(tags) if tags is not None else row["tags"],
                kind.strip().lower() if kind else row["kind"],
                int(pinned) if pinned is not None else row["pinned"],
                _normalized_hash(new_title, new_content),
                1 if text_changed else 0,
                _now(),
                memory_id,
            ),
        )
        self.db.execute("COMMIT")
        return {"action": "updated", "id": memory_id}

    def supersede(self, old_id: int, new_id: int) -> dict[str, Any]:
        if old_id == new_id:
            raise StoreError("A memory can't supersede itself.")
        self._row(old_id)
        self._row(new_id)
        self.db.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE id = ?", (new_id, _now(), old_id)
        )
        return {"action": "superseded", "id": old_id, "superseded_by": new_id}

    def forget(self, memory_id: int, reason: str = "") -> dict[str, Any]:
        """Soft delete: hidden everywhere, kept on disk until ``purge``."""
        self._row(memory_id)
        self.db.execute(
            "UPDATE memories SET deleted_at = ?, delete_reason = ? WHERE id = ?",
            (_now(), reason.strip() or None, memory_id),
        )
        return {"action": "forgotten", "id": memory_id}

    def purge(self, older_than_days: int = 30) -> int:
        """Permanently remove memories forgotten more than ``older_than_days`` ago."""
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat(timespec="seconds")
        cur = self.db.execute(
            "DELETE FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,)
        )
        return cur.rowcount

    # ── read operations ────────────────────────────────────────────────────
    def recall(
        self,
        query: str,
        limit: int = 8,
        project: str | None = None,
        kind: str | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        q = _fts_query(query)
        if not q:
            return []
        where = ["memories_fts MATCH ?", "m.deleted_at IS NULL"]
        args: list[Any] = [q]
        if not include_superseded:
            where.append("m.superseded_by IS NULL")
        if project:
            where.append("(m.project = ? OR m.project IS NULL)")
            args.append(project)
        if kind:
            where.append("m.kind = ?")
            args.append(kind.strip().lower())
        # bm25 is negative (lower = better); pinned memories get a nudge.
        rows = self.db.execute(
            f"""SELECT m.*, bm25(memories_fts, 3.0, 1.0, 2.0) - (m.pinned * 2.0) AS score
                FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid
                WHERE {' AND '.join(where)} ORDER BY score LIMIT ?""",
            (*args, max(1, min(limit, 50))),
        ).fetchall()
        if rows:
            now = _now()
            self.db.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(now, r["id"]) for r in rows],
            )
        out = []
        for r in rows:
            item = self._summary(r)
            if r["superseded_by"]:
                item["superseded_by"] = r["superseded_by"]
            out.append(item)
        return out

    def get(self, memory_id: int, with_history: bool = False) -> dict[str, Any]:
        row = self._row(memory_id)
        self.db.execute(
            "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?", (_now(), memory_id)
        )
        d = self._full(row)
        if with_history:
            d["history"] = [
                dict(v)
                for v in self.db.execute(
                    "SELECT version, title, content, created_at FROM memory_versions "
                    "WHERE memory_id = ? ORDER BY version DESC",
                    (memory_id,),
                )
            ]
        return d

    def recent(self, limit: int = 10, project: str | None = None) -> list[dict[str, Any]]:
        args: list[Any] = []
        where = _ACTIVE
        if project:
            where += " AND (project = ? OR project IS NULL)"
            args.append(project)
        rows = self.db.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY updated_at DESC LIMIT ?",
            (*args, max(1, min(limit, 50))),
        ).fetchall()
        return [self._summary(r) for r in rows]

    def profile(self, project: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Pinned memories and preferences first, then the most-used: the session-start briefing."""
        args: list[Any] = []
        where = _ACTIVE
        if project:
            where += " AND (project = ? OR project IS NULL)"
            args.append(project)
        rows = self.db.execute(
            f"""SELECT * FROM memories WHERE {where}
                ORDER BY pinned DESC, (kind = 'preference') DESC, use_count DESC, updated_at DESC
                LIMIT ?""",
            (*args, max(1, min(limit, 50))),
        ).fetchall()
        return [self._summary(r, preview=300) for r in rows]

    def review(self, stale_days: int = 180, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
        """Memories worth a human or AI look: stale, superseded, or saved repeatedly."""
        cutoff = (datetime.now(UTC) - timedelta(days=stale_days)).isoformat(timespec="seconds")
        q = lambda sql, *a: [self._summary(r) for r in self.db.execute(sql, (*a, limit)).fetchall()]  # noqa: E731
        return {
            "stale": q(
                f"""SELECT * FROM memories WHERE {_ACTIVE} AND pinned = 0
                    AND COALESCE(last_used_at, created_at) < ? ORDER BY COALESCE(last_used_at, created_at) LIMIT ?""",
                cutoff,
            ),
            "superseded": q(
                "SELECT * FROM memories WHERE superseded_by IS NOT NULL AND deleted_at IS NULL "
                "ORDER BY updated_at DESC LIMIT ?"
            ),
            "repeated": q(
                f"SELECT * FROM memories WHERE {_ACTIVE} AND duplicate_count > 2 "
                "ORDER BY duplicate_count DESC LIMIT ?"
            ),
        }

    def stats(self) -> dict[str, int]:
        r = self.db.execute(
            f"""SELECT COUNT(*) AS total,
                   SUM({_ACTIVE}) AS active,
                   SUM(superseded_by IS NOT NULL AND deleted_at IS NULL) AS superseded,
                   SUM(deleted_at IS NOT NULL) AS forgotten
                FROM memories"""
        ).fetchone()
        return {k: int(r[k] or 0) for k in ("total", "active", "superseded", "forgotten")}
