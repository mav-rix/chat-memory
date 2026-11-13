"""SQLite storage for keepsake.

One file, no server. Full-text search uses SQLite's built-in FTS5.

Design (borrowed ideas, our own code):
  * topic keys: one evolving subject keeps one row; updates keep old versions
    (idea from Engram, MIT).
  * normalized-hash exact-duplicate detection (Engram).
  * superseding instead of deleting (mcp-memory-service, Apache-2.0).
  * near-duplicate check before saving, with the decision left to the AI.
  * optional semantic search: embeddings stored in the row and compared by
    brute force (fast enough for tens of thousands of memories), mixed with
    keyword scores 0.7 / 0.3 (weights from mcp-memory-service).
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from .identifiers import extract as extract_identifiers
from .identifiers import redact_card_details
from .redact import find_secrets, strip_private

SCHEMA_VERSION = 4
WINDOWS = os.name == "nt"

KINDS = ("fact", "preference", "project", "decision", "event", "person", "howto", "summary", "case")

# Word-overlap thresholds (Jaccard) for the near-duplicate check.
DUPLICATE_THRESHOLD = 0.8  # at or above: probably the same memory, don't save without force
RELATED_THRESHOLD = 0.25  # at or above: related, worth showing to the AI (tune with real data)

# Cosine thresholds for all-MiniLM-L6-v2, calibrated on sample pairs:
# paraphrased duplicates 0.82-0.92, updates/conflicts 0.61-0.74, related 0.21-0.38, unrelated <= 0.11.
SEM_DUPLICATE = 0.80
SEM_RELATED = 0.45
SEM_RECALL_FLOOR = 0.20
SEMANTIC_WEIGHT, KEYWORD_WEIGHT = 0.7, 0.3
PINNED_BONUS = 0.05
# recall() returns full text (not a preview) for up to this many top hits within this budget.
FULL_TEXT_HITS = 3
FULL_TEXT_BUDGET = 8000
# review(): pairs at or above this cosine are listed as possible conflicts (changed facts
# scored 0.61-0.74 in calibration); keyword-only mode uses word overlap instead.
REVIEW_COSINE = 0.60
REVIEW_JACCARD = 0.50


class EmbedderLike(Protocol):
    name: str

    def embed(self, texts: list[str]) -> Any: ...  # (n, dims) float32, L2-normalised

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
    delete_reason   TEXT,
    embedding       BLOB,
    embedding_model TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_topic
    ON memories(topic_key) WHERE topic_key IS NOT NULL AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_mem_hash    ON memories(normalized_hash);
CREATE INDEX IF NOT EXISTS idx_mem_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_mem_updated ON memories(updated_at DESC);

-- Identifiers (IPs, emails, card and order numbers...) in normalised form, for exact matching.
CREATE TABLE IF NOT EXISTS identifiers (
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    kind      TEXT    NOT NULL,
    value     TEXT    NOT NULL,
    raw       TEXT    NOT NULL,
    PRIMARY KEY (memory_id, kind, value)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_ident_value ON identifiers(kind, value);

-- Usage log: which tool ran and how much text it returned. Never the content itself.
CREATE TABLE IF NOT EXISTS tool_calls (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      TEXT    NOT NULL,
    tool    TEXT    NOT NULL,
    chars   INTEGER NOT NULL,
    ok      INTEGER NOT NULL DEFAULT 1
);

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


def setting(name: str, default: str = "") -> str:
    """KEEPSAKE_<name>, falling back to the pre-rename CHAT_MEMORY_<name>."""
    return os.environ.get(f"KEEPSAKE_{name}") or os.environ.get(f"CHAT_MEMORY_{name}") or default


def default_db_path() -> Path:
    if env := setting("DB"):
        return Path(env).expanduser()
    path = data_home() / "keepsake" / "memory.db"
    legacy = data_home() / "chat-memory" / "memory.db"  # name before 0.6
    return legacy if not path.exists() and legacy.exists() else path


def data_home() -> Path:
    """Per-user data folder: %LOCALAPPDATA% on Windows, XDG_DATA_HOME or ~/.local/share elsewhere."""
    if WINDOWS and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"])
    return Path(os.environ.get("XDG_DATA_HOME") or "~/.local/share").expanduser()


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


def _iso(value: Any, default: str | None) -> str | None:
    """A valid ISO timestamp (normalised to UTC) or ``default``."""
    if not value:
        return default
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return default
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def _int(value: Any, default: int) -> int:
    try:
        return max(default, int(value))
    except (TypeError, ValueError):
        return default


def _clean_tags(tags: list[str] | str | None) -> str:
    if not tags:
        return ""
    items = tags.split(",") if isinstance(tags, str) else tags
    return ",".join(sorted({t.strip().lower() for t in items if t.strip()}))


@dataclass
class Store:
    path: Path
    embedder: EmbedderLike | None = None
    _writes: int = field(default=0, init=False, repr=False)
    _vec_key: tuple | None = field(default=None, init=False, repr=False)
    _vec_ids: Any = field(default=None, init=False, repr=False)
    _vec_mat: Any = field(default=None, init=False, repr=False)

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
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(memories)")}
        if cols and "embedding" not in cols:  # v1 → v2
            self.db.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
            self.db.execute("ALTER TABLE memories ADD COLUMN embedding_model TEXT")
        old_version = self.db.execute("PRAGMA user_version").fetchone()[0]
        self.db.executescript(_SCHEMA)
        if cols and old_version < 4:
            self.reindex_identifiers()
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
        for internal in ("normalized_hash", "embedding", "embedding_model"):
            d.pop(internal, None)
        return d

    # ── embeddings ─────────────────────────────────────────────────────────
    @staticmethod
    def _vec_text(title: str, content: str) -> str:
        return f"{title}\n{content}"

    # Embeddings use the content only: different titles on the same fact pulled paraphrase
    # scores down to the level of genuine updates (0.74 vs 0.82 without titles). Titles
    # still count through keyword search.

    def _embed(self, text: str) -> Any:
        return self.embedder.embed([text])[0] if self.embedder else None

    def _embedding_cols(self, vec: Any) -> tuple[bytes | None, str | None]:
        if vec is None:
            return None, None
        return vec.astype("float32").tobytes(), self.embedder.name

    def backfill_embeddings(self) -> int:
        """Embed memories that have no embedding (or one from a different model)."""
        if not self.embedder:
            return 0
        done = 0
        while True:
            rows = self.db.execute(
                """SELECT id, title, content FROM memories WHERE deleted_at IS NULL
                   AND (embedding IS NULL OR embedding_model IS NOT ?) LIMIT 256""",
                (self.embedder.name,),
            ).fetchall()
            if not rows:
                break
            vecs = self.embedder.embed([r["content"] for r in rows])
            self.db.executemany(
                "UPDATE memories SET embedding = ?, embedding_model = ? WHERE id = ?",
                [(*self._embedding_cols(v), r["id"]) for v, r in zip(vecs, rows)],
            )
            self._writes += 1
            done += len(rows)
        return done

    def _vectors(self) -> tuple[Any, Any]:
        """(ids, matrix) of all non-deleted embeddings, cached until the database changes."""
        import numpy as np

        self.backfill_embeddings()
        key = (self.db.execute("PRAGMA data_version").fetchone()[0], self._writes)
        if key != self._vec_key:
            rows = self.db.execute(
                "SELECT id, embedding FROM memories WHERE deleted_at IS NULL AND embedding_model = ?",
                (self.embedder.name,),
            ).fetchall()
            self._vec_ids = np.array([r["id"] for r in rows], dtype=np.int64)
            self._vec_mat = (
                np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
                if rows
                else np.zeros((0, 1), dtype=np.float32)
            )
            self._vec_key = key
        return self._vec_ids, self._vec_mat

    def _semantic(self, vec: Any, k: int, floor: float) -> dict[int, float]:
        """Top-k memory ids by cosine similarity to ``vec`` (vectors are normalised)."""
        import numpy as np

        ids, mat = self._vectors()
        if not len(ids):
            return {}
        sims = mat @ vec
        top = np.argsort(-sims)[:k] if len(sims) <= k else np.argpartition(-sims, k)[:k]
        return {int(ids[i]): float(sims[i]) for i in top if sims[i] >= floor}

    def _check_text(self, title: str, content: str) -> tuple[str, str, list[str]]:
        """Clean text for storage: drop <private> parts and card expiry/PIN/security codes,
        and refuse credentials. Returns (title, content, what was redacted)."""
        title, content = strip_private(title).strip(), strip_private(content).strip()
        title, r1 = redact_card_details(title)
        content, r2 = redact_card_details(content)
        redacted = list(dict.fromkeys(r1 + r2))
        if not title or not content:
            raise StoreError("Both title and content are required.")
        if found := find_secrets(f"{title}\n{content}"):
            raise StoreError(
                "Refused: this looks like it contains a secret (" + ", ".join(found) + "). "
                "Never store credentials. Remove it, or wrap the sensitive part in <private>…</private>."
            )
        return title, content, redacted

    def _index(self, memory_id: int, title: str, content: str) -> None:
        self.db.execute("DELETE FROM identifiers WHERE memory_id = ?", (memory_id,))
        self.db.executemany(
            "INSERT OR IGNORE INTO identifiers(memory_id, kind, value, raw) VALUES (?,?,?,?)",
            [(memory_id, k, v, raw) for k, v, raw in extract_identifiers(f"{title}\n{content}")],
        )

    def reindex_identifiers(self) -> int:
        rows = self.db.execute("SELECT id, title, content FROM memories").fetchall()
        for r in rows:
            self._index(r["id"], r["title"], r["content"])
        return len(rows)

    def _save_version(self, row: sqlite3.Row) -> None:
        self.db.execute(
            "INSERT INTO memory_versions(memory_id, version, title, content, created_at) VALUES (?,?,?,?,?)",
            (row["id"], row["revision_count"], row["title"], row["content"], row["updated_at"]),
        )

    def similar(
        self, text: str, limit: int = 3, exclude_id: int | None = None, vec: Any = None
    ) -> list[dict[str, Any]]:
        """Active memories most similar to ``text``.

        Candidates come from keyword search and, when available, embeddings. Each gets a
        word-overlap score and a cosine score; ``duplicate`` is set when either crosses its
        duplicate threshold.
        """
        cos: dict[int, float] = {}
        if self.embedder:
            vec = self._embed(text) if vec is None else vec
            cos = self._semantic(vec, k=10, floor=SEM_RELATED)
        candidate_ids = set(cos)
        if q := _fts_query(text, max_terms=24):
            candidate_ids |= {
                r[0]
                for r in self.db.execute(
                    "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT 25",
                    (q,),
                )
            }
        candidate_ids.discard(exclude_id)
        if not candidate_ids:
            return []
        marks = ",".join("?" * len(candidate_ids))
        rows = self.db.execute(
            f"SELECT * FROM memories WHERE id IN ({marks}) AND {_ACTIVE}", tuple(candidate_ids)
        ).fetchall()
        scored = []
        for r in rows:
            jac = similarity(text, self._vec_text(r["title"], r["content"]))
            c = cos.get(r["id"], 0.0)
            if jac >= RELATED_THRESHOLD or c >= SEM_RELATED:
                dup = jac >= DUPLICATE_THRESHOLD or c >= SEM_DUPLICATE
                scored.append((max(jac, c), dup, r))
        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return [
            {**self._summary(r), "similarity": round(s, 2), "duplicate": dup} for s, dup, r in scored[:limit]
        ]

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
        title, content, redacted = self._check_text(title, content)
        note = (
            {"redacted": redacted, "note": "Removed before saving: " + ", ".join(redacted) + "."}
            if redacted
            else {}
        )
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
                    return {"action": "unchanged", "id": row["id"], "message": "Topic already says exactly this.", **note}
                emb, emb_model = self._embedding_cols(self._embed(content))
                self.db.execute("BEGIN")
                self._save_version(row)
                self.db.execute(
                    """UPDATE memories SET title=?, content=?, kind=?, tags=?, project=COALESCE(?, project),
                       normalized_hash=?, revision_count=revision_count+1, updated_at=?, superseded_by=NULL,
                       embedding=?, embedding_model=? WHERE id=?""",
                    (title, content, kind, tags_s or row["tags"], project, h, now, emb, emb_model, row["id"]),
                )
                self._index(row["id"], title, content)
                self.db.execute("COMMIT")
                self._writes += 1
                return {
                    "action": "updated",
                    "id": row["id"],
                    "message": f"Updated topic '{topic_key}' (previous version kept).",
                    **note,
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
            return {"action": "duplicate", "id": row["id"], "message": "Already remembered; nothing new saved.", **note}

        # 3. Near-duplicates → let the AI decide unless it already has (force).
        vec = self._embed(content)
        similar = self.similar(self._vec_text(title, content), vec=vec)
        if not force and similar and similar[0]["duplicate"]:
            return {
                "action": "needs_review",
                "id": None,
                "similar": similar,
                "message": (
                    "Not saved: a very similar memory exists. Update it with update_memory, "
                    "or call remember again with force=true if this is genuinely different."
                ),
                **note,
            }

        cur = self.db.execute(
            """INSERT INTO memories(topic_key, kind, title, content, tags, project, normalized_hash,
                                    created_at, updated_at, embedding, embedding_model)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (topic_key, kind, title, content, tags_s, project, h, now, now, *self._embedding_cols(vec)),
        )
        self._index(cur.lastrowid, title, content)
        self._writes += 1
        result: dict[str, Any] = {"action": "created", "id": cur.lastrowid, **note}
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
        new_title, new_content, redacted = self._check_text(title or row["title"], content or row["content"])
        if kind is not None and kind.strip().lower() not in KINDS:
            raise StoreError(f"Unknown kind '{kind}'. Use one of: {', '.join(KINDS)}.")
        text_changed = (new_title, new_content) != (row["title"], row["content"])
        content_changed = new_content != row["content"]
        if content_changed and self.embedder:
            emb, emb_model = self._embedding_cols(self._embed(new_content))
        else:
            emb, emb_model = row["embedding"], row["embedding_model"]
        self.db.execute("BEGIN")
        if text_changed:
            self._save_version(row)
        self._writes += 1
        self.db.execute(
            """UPDATE memories SET title=?, content=?, tags=?, kind=?, pinned=?, normalized_hash=?,
               revision_count = revision_count + ?, updated_at=?, embedding=?, embedding_model=?
               WHERE id=?""",
            (
                new_title,
                new_content,
                _clean_tags(tags) if tags is not None else row["tags"],
                kind.strip().lower() if kind else row["kind"],
                int(pinned) if pinned is not None else row["pinned"],
                _normalized_hash(new_title, new_content),
                1 if text_changed else 0,
                _now(),
                None if content_changed and not self.embedder else emb,
                None if content_changed and not self.embedder else emb_model,
                memory_id,
            ),
        )
        if text_changed:
            self._index(memory_id, new_title, new_content)
        self.db.execute("COMMIT")
        result: dict[str, Any] = {"action": "updated", "id": memory_id}
        if redacted:
            result["redacted"] = redacted
        return result

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
        self._writes += 1
        return {"action": "forgotten", "id": memory_id}

    def purgeable(self, older_than_days: int = 30) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat(timespec="seconds")
        return self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,)
        ).fetchone()[0]

    def purge(self, older_than_days: int = 30) -> int:
        """Permanently remove memories forgotten more than ``older_than_days`` ago."""
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat(timespec="seconds")
        doomed = "SELECT id FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?"
        # A memory replaced by one being purged stays hidden: it becomes forgotten itself
        # (purged in a later run) rather than resurfacing as if it were current.
        self.db.execute(
            f"""UPDATE memories SET deleted_at = COALESCE(deleted_at, ?),
                   delete_reason = COALESCE(delete_reason, 'its replacement was purged'),
                   superseded_by = NULL WHERE superseded_by IN ({doomed})""",
            (_now(), cutoff),
        )
        cur = self.db.execute(
            "DELETE FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,)
        )
        self._writes += 1
        return cur.rowcount

    def restore(self, memory_id: int) -> dict[str, Any]:
        """Undo forget or supersede: the memory becomes active again."""
        row = self._row(memory_id)
        if not row["deleted_at"] and not row["superseded_by"]:
            raise StoreError(f"Memory {memory_id} is already active.")
        if row["topic_key"] and row["deleted_at"]:
            clash = self.db.execute(
                "SELECT id FROM memories WHERE topic_key = ? AND deleted_at IS NULL AND id != ?",
                (row["topic_key"], memory_id),
            ).fetchone()
            if clash:
                raise StoreError(
                    f"Memory {clash[0]} now holds topic '{row['topic_key']}'; update that one instead."
                )
        self.db.execute(
            "UPDATE memories SET deleted_at = NULL, delete_reason = NULL, superseded_by = NULL, "
            "updated_at = ? WHERE id = ?",
            (_now(), memory_id),
        )
        self._writes += 1
        return {"action": "restored", "id": memory_id}

    # ── export / import ────────────────────────────────────────────────────
    def export_records(self, include_forgotten: bool = False) -> Iterator[dict[str, Any]]:
        """Every memory (forgotten ones only on request) with its version history."""
        where = "" if include_forgotten else "WHERE deleted_at IS NULL"
        rows = self.db.execute(f"SELECT * FROM memories {where} ORDER BY id").fetchall()
        for row in rows:
            d = self._full(row)
            d["history"] = [
                dict(v)
                for v in self.db.execute(
                    "SELECT version, title, content, created_at FROM memory_versions "
                    "WHERE memory_id = ? ORDER BY version",
                    (row["id"],),
                )
            ]
            yield d

    def import_records(self, records: Iterable[dict[str, Any]], dry_run: bool = False) -> dict[str, Any]:
        """Merge records (from ``export_records`` or another source) into this store.

        Every record passes the same secret filter as ``remember``. Exact duplicates are
        skipped; for a topic_key that already exists the newer version wins and the other is
        kept in history. Superseded links are re-pointed to the new ids. All or nothing.
        """
        report: dict[str, Any] = {
            "created": 0, "updated": 0, "duplicate": 0, "older": 0, "refused": [], "history_dropped": 0,
            "redacted": 0,
        }
        idmap: dict[Any, int] = {}
        links: list[tuple[int, Any]] = []
        self.db.execute("BEGIN")
        try:
            for n, rec in enumerate(records, 1):
                source = str(rec.get("_source") or f"record {n}")
                try:
                    title, content, redacted = self._check_text(
                        str(rec.get("title") or ""), str(rec.get("content") or "")
                    )
                    report["redacted"] += bool(redacted)
                except StoreError as e:
                    report["refused"].append({"source": source, "reason": str(e).split(". ")[0] + "."})
                    continue
                kind = str(rec.get("kind") or "fact").strip().lower()
                kind = kind if kind in KINDS else "fact"
                tk = rec.get("topic_key")
                tk = str(tk).strip().lower() if tk and str(tk).strip() else None
                deleted = _iso(rec.get("deleted_at"), None)
                created = _iso(rec.get("created_at"), _now())
                updated = _iso(rec.get("updated_at"), created)
                h = _normalized_hash(title, content)

                same = self.db.execute(
                    "SELECT id FROM memories WHERE normalized_hash = ? AND (deleted_at IS NULL) = ?",
                    (h, int(deleted is None)),
                ).fetchone()
                if same:
                    idmap[rec.get("id")] = same[0]
                    report["duplicate"] += 1
                    continue

                if tk and deleted is None:
                    row = self.db.execute(
                        "SELECT * FROM memories WHERE topic_key = ? AND deleted_at IS NULL", (tk,)
                    ).fetchone()
                    if row:
                        idmap[rec.get("id")] = row["id"]
                        if updated > row["updated_at"]:
                            self._save_version(row)
                            self.db.execute(
                                """UPDATE memories SET title=?, content=?, kind=?, tags=?, normalized_hash=?,
                                   revision_count=revision_count+1, updated_at=?, embedding=NULL,
                                   embedding_model=NULL WHERE id=?""",
                                (title, content, kind, _clean_tags(rec.get("tags")) or row["tags"], h, updated,
                                 row["id"]),
                            )
                            self._index(row["id"], title, content)
                            report["updated"] += 1
                        else:
                            report["older"] += 1
                        continue

                history = []
                for v in rec.get("history") or []:
                    try:
                        vt, vc, _ = self._check_text(str(v.get("title") or ""), str(v.get("content") or ""))
                        history.append((int(v["version"]), vt, vc, _iso(v.get("created_at"), created)))
                    except (StoreError, KeyError, TypeError, ValueError):
                        report["history_dropped"] += 1
                revisions = max([_int(rec.get("revision_count"), 1)] + [ver + 1 for ver, *_ in history])
                cur = self.db.execute(
                    """INSERT INTO memories(topic_key, kind, title, content, tags, project, normalized_hash,
                           duplicate_count, revision_count, use_count, pinned, created_at, updated_at,
                           last_used_at, deleted_at, delete_reason)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        tk, kind, title, content, _clean_tags(rec.get("tags")), rec.get("project") or None, h,
                        _int(rec.get("duplicate_count"), 1), revisions, _int(rec.get("use_count"), 0),
                        int(bool(rec.get("pinned"))), created, updated, _iso(rec.get("last_used_at"), None),
                        deleted, (str(rec["delete_reason"]) if deleted and rec.get("delete_reason") else None),
                    ),
                )
                new_id = cur.lastrowid
                self._index(new_id, title, content)
                idmap[rec.get("id")] = new_id
                self.db.executemany(
                    "INSERT OR IGNORE INTO memory_versions(memory_id, version, title, content, created_at) "
                    "VALUES (?,?,?,?,?)",
                    [(new_id, *v) for v in history],
                )
                if rec.get("superseded_by") is not None:
                    links.append((new_id, rec["superseded_by"]))
                report["created"] += 1

            for new_id, old_target in links:
                target = idmap.get(old_target)
                if target and target != new_id:
                    self.db.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (target, new_id))
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("ROLLBACK" if dry_run else "COMMIT")
        if not dry_run:
            self._writes += 1
        return report

    # ── read operations ────────────────────────────────────────────────────
    def recall(
        self,
        query: str,
        limit: int = 8,
        project: str | None = None,
        kind: str | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Hybrid search: keyword (BM25) and, when available, meaning (cosine) scores, mixed
        0.3 / 0.7. Without embeddings this is plain keyword search."""
        limit = max(1, min(limit, 50))
        where = ["m.deleted_at IS NULL"]
        args: list[Any] = []
        if not include_superseded:
            where.append("m.superseded_by IS NULL")
        if project:
            where.append("(m.project = ? OR m.project IS NULL)")
            args.append(project)
        if kind:
            where.append("m.kind = ?")
            args.append(kind.strip().lower())
        filters = " AND ".join(where)

        # Keyword candidates. bm25() is negative; lower is better.
        kw: dict[int, float] = {}
        if q := _fts_query(query):
            kw = {
                r[0]: r[1]
                for r in self.db.execute(
                    f"""SELECT m.id, bm25(memories_fts, 3.0, 1.0, 2.0)
                        FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid
                        WHERE memories_fts MATCH ? AND {filters}
                        ORDER BY bm25(memories_fts, 3.0, 1.0, 2.0) LIMIT 50""",
                    (q, *args),
                )
            }
        # Meaning candidates.
        sem: dict[int, float] = {}
        if self.embedder and query.strip():
            sem = self._semantic(self._embed(query), k=50, floor=SEM_RECALL_FLOOR)
        ids = set(kw) | set(sem)
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = {
            r["id"]: r
            for r in self.db.execute(
                f"SELECT * FROM memories m WHERE m.id IN ({marks}) AND {filters}", (*ids, *args)
            )
        }
        best_kw = min(kw.values(), default=0.0)

        def score(mid: int) -> float:
            k = kw[mid] / best_kw if mid in kw and best_kw < 0 else (1.0 if mid in kw else 0.0)
            s = (SEMANTIC_WEIGHT * sem.get(mid, 0.0) + KEYWORD_WEIGHT * k) if self.embedder else k
            return s + (PINNED_BONUS if rows[mid]["pinned"] else 0.0)

        ranked = sorted(rows, key=score, reverse=True)[:limit]
        scores = {mid: score(mid) for mid in ranked}
        rows = [rows[mid] for mid in ranked]
        if rows:
            now = _now()
            self.db.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(now, r["id"]) for r in rows],
            )
        out = []
        budget = FULL_TEXT_BUDGET
        best = scores[rows[0]["id"]] if rows else 0.0
        for n, r in enumerate(rows):
            item = {**self._summary(r), "score": round(scores[r["id"]], 3)}
            # Full text for the strongest hits saves the AI a get_memory round trip, which
            # costs far more (the whole conversation is re-read) than the extra text.
            if n < FULL_TEXT_HITS and scores[r["id"]] >= 0.6 * best and len(r["content"]) <= budget:
                item.pop("preview")
                item["content"] = r["content"]
                budget -= len(r["content"])
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

    def near_duplicates(self, limit: int = 20) -> list[dict[str, Any]]:
        """Pairs of active memories that look like duplicates or conflicting versions of one fact.

        With embeddings, every pair is compared (in blocks, so memory stays small). Without,
        each memory's keyword neighbours are compared by word overlap.
        """
        active = {r["id"]: r for r in self.db.execute(f"SELECT * FROM memories WHERE {_ACTIVE}")}
        pairs: dict[tuple[int, int], float] = {}
        if self.embedder:
            import numpy as np

            ids, mat = self._vectors()
            keep = np.array([int(i) in active for i in ids], dtype=bool)
            ids, mat = ids[keep], mat[keep]
            cols = np.arange(len(ids))
            for start in range(0, len(ids), 512):
                sims = mat[start : start + 512] @ mat.T
                rows = np.arange(start, start + len(sims))
                sims[cols[None, :] <= rows[:, None]] = -1.0  # each pair once, never with itself
                flat = sims.ravel()
                k = min(limit, int((flat >= REVIEW_COSINE).sum()))
                if not k:
                    continue
                for f in np.argpartition(-flat, k - 1)[:k]:
                    bi, j = divmod(int(f), len(ids))
                    pairs[(int(ids[start + bi]), int(ids[j]))] = float(flat[f])
            dup_at = SEM_DUPLICATE
        else:
            recent = sorted(active.values(), key=lambda r: r["updated_at"], reverse=True)[:5000]
            for r in recent:
                text = self._vec_text(r["title"], r["content"])
                for hit in self.similar(text, limit=3, exclude_id=r["id"]):
                    if hit["similarity"] >= REVIEW_JACCARD:
                        pairs[tuple(sorted((r["id"], hit["id"])))] = hit["similarity"]
            dup_at = DUPLICATE_THRESHOLD
        out = []
        for (a, b), sim in sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
            older, newer = sorted((active[a], active[b]), key=lambda r: (r["updated_at"], r["id"]))
            dup = sim >= dup_at
            out.append(
                {
                    "similarity": round(sim, 2),
                    "label": "duplicate" if dup else "possible_conflict",
                    "older": self._summary(older, preview=160),
                    "newer": self._summary(newer, preview=160),
                    "suggestion": (
                        f"Same fact: merge anything useful into {newer['id']}, then supersede({older['id']}, {newer['id']})."
                        if dup
                        else f"Check whether {newer['id']} replaces {older['id']}; if so supersede({older['id']}, {newer['id']})."
                    ),
                }
            )
        return out

    def review(self, stale_days: int = 180, limit: int = 20) -> dict[str, Any]:
        """Memories worth a human or AI look: near-duplicates and conflicts, stale, superseded,
        or saved repeatedly."""
        cutoff = (datetime.now(UTC) - timedelta(days=stale_days)).isoformat(timespec="seconds")
        q = lambda sql, *a: [self._summary(r, 160) for r in self.db.execute(sql, (*a, limit)).fetchall()]  # noqa: E731
        return {
            "near_duplicates": self.near_duplicates(limit),
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
            "forgotten": q(
                "SELECT * FROM memories WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC LIMIT ?"
            ),
        }

    # ── exact matching ─────────────────────────────────────────────────────
    def match(self, items: list[str], limit: int = 10) -> list[dict[str, Any]]:
        """Exact lookups, for checking data received from elsewhere.

        Identifiers in each item (IPs, emails, card/gift card, phone, order and reference
        numbers, UUIDs, hashes) are compared in normalised form against the index. An item with
        no identifier (a name, say) is matched as a whole phrase, case-insensitively, on word
        boundaries. Never fuzzy: 203.0.113.45 does not match 203.0.113.46. Forgotten memories
        are excluded; superseded ones are included and marked, since old records still count.
        """
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            item = str(item).strip()
            if not item:
                continue
            found = extract_identifiers(item)
            if found:
                for kind, value, raw in found:
                    if (kind, value) in seen:
                        continue
                    seen.add((kind, value))
                    rows = self.db.execute(
                        """SELECT m.*, i.raw AS hit FROM identifiers i JOIN memories m ON m.id = i.memory_id
                           WHERE i.kind = ? AND i.value = ? AND m.deleted_at IS NULL
                           ORDER BY m.updated_at DESC LIMIT ?""",
                        (kind, value, limit),
                    ).fetchall()
                    out.append(self._match_result(raw, kind, rows))
            elif len(item) >= 3 and ("text", item.lower()) not in seen:
                seen.add(("text", item.lower()))
                pattern = re.compile(r"(?<!\w)" + re.escape(item) + r"(?!\w)", re.IGNORECASE)
                rows = [
                    r
                    for r in self.db.execute(
                        """SELECT m.*, NULL AS hit FROM memories m WHERE m.deleted_at IS NULL
                           AND (instr(lower(m.content), lower(?)) OR instr(lower(m.title), lower(?)))
                           ORDER BY m.updated_at DESC""",
                        (item, item),
                    )
                    if pattern.search(r["title"]) or pattern.search(r["content"])
                ][:limit]
                out.append(self._match_result(item, "text", rows))
        return out

    @staticmethod
    def _match_result(query: str, kind: str, rows: list[sqlite3.Row]) -> dict[str, Any]:
        def snippet(r: sqlite3.Row) -> str:
            text, needle = r["content"], (r["hit"] or query)
            i = text.lower().find(needle.lower())
            if i < 0:
                return text[:160]
            start = max(0, i - 60)
            return ("…" if start else "") + text[start : i + len(needle) + 60] + ("…" if i + len(needle) + 60 < len(text) else "")

        return {
            "query": query,
            "kind": kind,
            "found": bool(rows),
            "memories": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "kind": r["kind"],
                    "topic_key": r["topic_key"],
                    "updated_at": r["updated_at"],
                    "superseded": r["superseded_by"] is not None,
                    "as_written": r["hit"],
                    "snippet": snippet(r),
                }
                for r in rows
            ],
        }

    # ── usage log ──────────────────────────────────────────────────────────
    def log_call(self, tool: str, chars: int, ok: bool = True) -> None:
        self.db.execute(
            "INSERT INTO tool_calls(at, tool, chars, ok) VALUES (?,?,?,?)", (_now(), tool, chars, int(ok))
        )

    def usage(self, days: int = 30) -> dict[str, Any]:
        """Per-tool call counts and returned text, as a rough token estimate (4 chars ≈ 1 token)."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        rows = self.db.execute(
            """SELECT tool, COUNT(*) AS calls, SUM(chars) AS chars, SUM(1 - ok) AS errors
               FROM tool_calls WHERE at >= ? GROUP BY tool ORDER BY SUM(chars) DESC""",
            (cutoff,),
        ).fetchall()
        tools = [
            {"tool": r["tool"], "calls": r["calls"], "errors": r["errors"],
             "est_tokens": round(r["chars"] / 4), "avg_tokens_per_call": round(r["chars"] / 4 / r["calls"])}
            for r in rows
        ]
        return {"days": days, "tools": tools, "est_tokens_total": sum(t["est_tokens"] for t in tools)}

    def stats(self) -> dict[str, Any]:
        r = self.db.execute(
            f"""SELECT COUNT(*) AS total,
                   SUM({_ACTIVE}) AS active,
                   SUM(superseded_by IS NOT NULL AND deleted_at IS NULL) AS superseded,
                   SUM(deleted_at IS NOT NULL) AS forgotten,
                   SUM(embedding IS NOT NULL AND deleted_at IS NULL) AS embedded
                FROM memories"""
        ).fetchone()
        out: dict[str, Any] = {k: int(r[k] or 0) for k in ("total", "active", "superseded", "forgotten", "embedded")}
        out["search"] = f"hybrid ({self.embedder.name})" if self.embedder else "keyword only"
        return out
