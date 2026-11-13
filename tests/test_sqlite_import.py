"""Importing other tools' SQLite memory databases. Made-up data only."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from keepsake import transfer
from keepsake.server import main
from keepsake.store import Store, StoreError


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "ours.db")
    yield s
    s.close()


def make_db(path, script, rows=()):
    db = sqlite3.connect(path)
    db.executescript(script)
    for sql, params in rows:
        db.execute(sql, params)
    db.commit()
    db.close()
    return path


@pytest.fixture
def service_style(tmp_path):
    """Shaped like a vector-memory service: no title, epoch timestamps, comma tags, an FTS index."""
    return make_db(
        tmp_path / "service.db",
        """CREATE TABLE memories (id INTEGER PRIMARY KEY, content_hash TEXT, content TEXT, tags TEXT,
               memory_type TEXT, metadata TEXT, created_at REAL, updated_at REAL, deleted_at REAL);
           CREATE VIRTUAL TABLE memory_content_fts USING fts5(content);
           CREATE TABLE settings (key TEXT, value TEXT);""",
        [
            ("INSERT INTO memories VALUES (1,'h1',?,?,?,'{}',?,?,NULL)",
             ("Case FR-2001: refund abuse from 203.0.113.50 on ORD-2026-300001.\nMore detail.",
              "fraud,refund", "note", 1767225600.0, 1767312000.0)),
            ("INSERT INTO memories VALUES (2,'h2',?,?,?,'{}',?,NULL,NULL)",
             ("Gift card 6006491234567890123 PIN 4821 drained in 3 minutes.", "", "decision", 1767225600000, )),
            ("INSERT INTO memories VALUES (3,'h3','Removed in the old tool.','','note','{}',1767225600,NULL,1767225700)", ()),
            ("INSERT INTO settings VALUES ('theme','dark')", ()),
        ],
    )


@pytest.fixture
def observation_style(tmp_path):
    """Shaped like a session/observation store: titles, types, topic keys, text dates."""
    return make_db(
        tmp_path / "obs.db",
        """CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at TEXT);
           CREATE TABLE observations (id INTEGER PRIMARY KEY, session_id TEXT, type TEXT, title TEXT,
               content TEXT, project TEXT, scope TEXT, topic_key TEXT, created_at TEXT, updated_at TEXT,
               deleted_at TEXT);""",
        [
            ("INSERT INTO sessions VALUES ('s1','2026-01-01 09:00:00')", ()),
            ("INSERT INTO observations VALUES (1,'s1','case','Case FR-2002','Mule account linked to "
             "jane.example@example.com and +44 7700 900123.','fraud','team','case/fr-2002',"
             "'2026-01-02 10:00:00','2026-01-03 11:00:00',NULL)", ()),
        ],
    )


def test_inspect_suggests_table_and_columns_without_content(service_style):
    info = transfer.inspect_sqlite(service_style)
    assert info["suggested_table"] == "memories"
    assert info["suggested_mapping"] == {
        "content": "content", "kind": "memory_type", "tags": "tags",
        "created_at": "created_at", "updated_at": "updated_at", "deleted_at": "deleted_at",
    }
    assert {t["table"] for t in info["tables"]} == {"memories", "settings"}  # FTS index hidden
    assert "203.0.113.50" not in repr(info) and "refund abuse" not in repr(info)


def test_import_service_style(store, service_style):
    before = hashlib.sha256(service_style.read_bytes()).hexdigest()
    report = store.import_records(transfer.read_sqlite(service_style))
    assert report["created"] == 2 and report["redacted"] == 1  # deleted row skipped, PIN removed
    assert hashlib.sha256(service_style.read_bytes()).hexdigest() == before  # old db untouched
    case = store.get(store.match(["203.0.113.50"])[0]["memories"][0]["id"])
    assert case["title"] == "Case FR-2001: refund abuse from 203.0.113.50 on ORD-2026-300001."
    assert set(case["tags"]) == {"fraud", "refund", "type:note", "imported"}
    assert case["created_at"].startswith("2026-01-01") and case["updated_at"].startswith("2026-01-02")
    gift = store.get(store.match(["6006491234567890123"])[0]["memories"][0]["id"])
    assert "4821" not in gift["content"] and gift["kind"] == "decision"
    assert gift["created_at"].startswith("2026-01-01")  # milliseconds understood


def test_import_observation_style(store, observation_style):
    report = store.import_records(transfer.read_sqlite(observation_style))
    assert report["created"] == 1
    m = store.get(store.match(["07700 900123"])[0]["memories"][0]["id"])
    assert (m["kind"], m["topic_key"], m["project"]) == ("case", "case/fr-2002", "fraud")
    assert m["created_at"] == "2026-01-02T10:00:00+00:00"


def test_manual_mapping_and_kind(store, tmp_path):
    path = make_db(
        tmp_path / "custom.db",
        "CREATE TABLE kb (id INTEGER PRIMARY KEY, heading TEXT, note_text TEXT, logged TEXT);",
        [("INSERT INTO kb VALUES (1,'Case FR-2003','Card 4111 1111 1111 1111 exp 12/27 used from 192.0.2.30.','2026-02-01T08:00:00')", ())],
    )
    with pytest.raises(StoreError, match="holds the memory text"):
        list(transfer.read_sqlite(path))
    recs = transfer.read_sqlite(path, table="kb", mapping={"content": "note_text", "title": "heading",
                                                           "created_at": "logged"}, kind="case")
    assert store.import_records(recs)["created"] == 1
    m = store.get(store.match(["4111111111111111"])[0]["memories"][0]["id"])
    assert m["kind"] == "case" and "12/27" not in m["content"] and m["title"] == "Case FR-2003"


def test_bad_mapping_is_explained(tmp_path, service_style):
    with pytest.raises(StoreError, match="no column for: title=heading"):
        list(transfer.read_sqlite(service_style, mapping={"title": "heading"}))
    with pytest.raises(StoreError, match="No table 'nope'"):
        list(transfer.read_sqlite(service_style, table="nope"))


def test_reimport_adds_nothing(store, service_style):
    store.import_records(transfer.read_sqlite(service_style))
    again = store.import_records(transfer.read_sqlite(service_style))
    assert again["created"] == 0 and again["duplicate"] == 2


def test_cli_inspect_and_import(tmp_path, monkeypatch, capsys, service_style):
    monkeypatch.setenv("KEEPSAKE_DB", str(tmp_path / "a.db"))
    monkeypatch.setenv("KEEPSAKE_SEMANTIC", "0")
    from keepsake import server

    server.store.cache_clear()
    main(["inspect", str(service_style)])
    out = capsys.readouterr().out
    assert "memories (3 rows)  ← suggested" in out and "--map content=content" in out
    assert "203.0.113.50" not in out
    main(["import", str(service_style), "--dry-run"])
    assert "Would import: 2 new" in capsys.readouterr().out
    main(["import", str(service_style), "--kind", "case"])
    out = capsys.readouterr().out
    assert "Imported: 2 new" in out and "Removed card expiry dates/PINs/security codes from 1" in out
    assert server.store().match(["ORD-2026-300001"])[0]["memories"][0]["kind"] == "case"
    with pytest.raises(SystemExit):
        main(["import", str(tmp_path / "a.db")])  # its own database
    server.store.cache_clear()
