"""Export/import, restore, purge and review tests. Made-up data only."""

from __future__ import annotations

import io
import json
import os
import stat

import pytest

from chat_memory import transfer
from chat_memory.server import main
from chat_memory.store import Store, StoreError

from test_semantic import FakeEmbedder


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db")
    yield s
    s.close()


def _roundtrip(src: Store, dst: Store, **kw):
    buf = io.StringIO()
    transfer.export_jsonl(src, buf, **kw)
    path = dst.path.parent / "export.jsonl"
    path.write_text(buf.getvalue())
    return dst.import_records(transfer.read_jsonl(path))


# ── export / import ────────────────────────────────────────────────────────
def test_roundtrip_keeps_everything(tmp_path, store):
    a = store.remember("Bike", "Bike lock code is kept in the blue tin.", kind="howto", tags=["home"])
    store.update(a["id"], content="Bike lock key hangs by the back door.", pinned=True)
    old = store.remember("Club night", "Chess club meets on Mondays.")
    new = store.remember("Club night", "Chess club now meets on Thursdays.", force=True)
    store.supersede(old["id"], new["id"])
    store.remember("Car", "Car is a blue hatchback.", topic_key="device/car")
    store.remember("Car", "Car is a green estate.", topic_key="device/car")

    dst = Store(tmp_path / "other" / "m.db")
    report = _roundtrip(store, dst)
    assert report["created"] == 4 and not report["refused"]

    bike = dst.recall("bike key back door")[0]
    full = dst.get(bike["id"], with_history=True)
    assert full["pinned"] and full["kind"] == "howto" and full["tags"] == ["home"]
    assert full["history"][0]["content"].startswith("Bike lock code")
    assert full["revision_count"] == 2
    car = dst.get(dst.recall("car estate")[0]["id"], with_history=True)
    assert car["topic_key"] == "device/car" and len(car["history"]) == 1
    # superseded link re-pointed to the new ids
    hits = dst.recall("chess club", include_superseded=True)
    sup = [h for h in hits if "superseded_by" in h]
    assert len(sup) == 1 and dst.get(sup[0]["superseded_by"])["content"].endswith("Thursdays.")


def test_import_twice_adds_nothing(tmp_path, store):
    store.remember("Plants", "Water the fern on Sundays.")
    dst = Store(tmp_path / "o" / "m.db")
    _roundtrip(store, dst)
    again = _roundtrip(store, dst)
    assert again["created"] == 0 and again["duplicate"] == 1
    assert dst.stats()["active"] == 1


def test_topic_conflict_newer_wins(tmp_path, store):
    dst = Store(tmp_path / "o" / "m.db")
    dst.remember("Router", "Router is in the hallway.", topic_key="device/router")
    newer = [{"title": "Router", "content": "Router moved to the study.", "topic_key": "device/router",
              "updated_at": "2099-01-01T00:00:00+00:00"}]
    older = [{"title": "Router", "content": "Router was in the loft.", "topic_key": "device/router",
              "updated_at": "2000-01-01T00:00:00+00:00"}]
    assert dst.import_records(older)["older"] == 1
    assert dst.import_records(newer)["updated"] == 1
    m = dst.get(dst.recall("router")[0]["id"], with_history=True)
    assert m["content"] == "Router moved to the study." and m["history"][0]["content"] == "Router is in the hallway."


def test_import_refuses_secrets_and_keeps_the_rest(store):
    fake = "gh" + "p_" + "A" * 36
    recs = [
        {"title": "Token", "content": f"My token is {fake}", "_source": "line 2"},
        {"title": "Kettle", "content": "Kettle descaler is under the sink.",
         "history": [{"version": 1, "title": "Kettle", "content": f"old {fake}"}]},
    ]
    report = store.import_records(recs)
    assert report["created"] == 1 and report["history_dropped"] == 1
    assert report["refused"][0]["source"] == "line 2" and fake not in json.dumps(report)


def test_dry_run_changes_nothing(store):
    report = store.import_records([{"title": "Bins", "content": "Bins go out on Tuesday."}], dry_run=True)
    assert report["created"] == 1 and store.stats()["total"] == 0


def test_bad_record_rolls_back_everything(store):
    class Boom(dict):
        def get(self, *a):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        store.import_records([{"title": "Ok", "content": "First one is fine."}, Boom()])
    assert store.stats()["total"] == 0


def test_forgotten_only_exported_on_request(tmp_path, store):
    r = store.remember("Old", "An old fact about the shed.")
    store.forget(r["id"], "wrong")
    assert list(store.export_records()) == []
    dst = Store(tmp_path / "o" / "m.db")
    _roundtrip(store, dst, include_forgotten=True)
    assert dst.stats()["forgotten"] == 1 and dst.stats()["active"] == 0


def test_reader_rejects_other_files(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"hello": 1}\n')
    with pytest.raises(StoreError):
        list(transfer.read_jsonl(p))


def test_claude_code_folder_import(tmp_path, store):
    d = tmp_path / "memory"
    d.mkdir()
    (d / "MEMORY.md").write_text("- [x](x.md) — index only\n")
    (d / "style.md").write_text(
        "---\nname: answer-style\ndescription: Prefers short answers\nmetadata:\n  type: feedback\n---\n\n"
        "Keep answers short.\n**Why:** reads on a phone.\n"
    )
    (d / "printer.md").write_text("---\nname: printer\ndescription: Office printer\ntype: reference\n---\nLaser, on the landing.\n")
    report = store.import_records(transfer.read_any(d))
    assert report["created"] == 2
    style = store.get(store.recall("short answers")[0]["id"])
    assert style["kind"] == "preference" and style["topic_key"] == "claude-code/answer-style"
    assert "reads on a phone" in style["content"]


def test_export_file_is_private_and_cli_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHAT_MEMORY_DB", str(tmp_path / "a.db"))
    monkeypatch.setenv("CHAT_MEMORY_SEMANTIC", "0")
    from chat_memory import server

    server.store.cache_clear()
    server.store().remember("Bins", "Recycling goes out on Tuesday night.")
    out = tmp_path / "backup.jsonl"
    main(["export", str(out)])
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(out).st_mode) == 0o600

    server.store.cache_clear()
    monkeypatch.setenv("CHAT_MEMORY_DB", str(tmp_path / "b.db"))
    main(["import", str(out)])
    assert "1 new" in capsys.readouterr().out
    assert server.store().stats()["active"] == 1
    server.store.cache_clear()


# ── restore / purge ────────────────────────────────────────────────────────
def test_restore_forgotten_and_superseded(store):
    a = store.remember("Shed", "Shed key is on the hook.")
    store.forget(a["id"])
    store.restore(a["id"])
    assert store.recall("shed key")
    b = store.remember("Shed", "Shed key moved to the drawer.", force=True)
    store.supersede(a["id"], b["id"])
    store.restore(a["id"])
    assert len(store.recall("shed key")) == 2
    with pytest.raises(StoreError):
        store.restore(a["id"])


def test_restore_refuses_topic_clash(store):
    a = store.remember("Car", "Car is blue.", topic_key="device/car")
    store.forget(a["id"])
    store.remember("Car", "Car is green.", topic_key="device/car")
    with pytest.raises(StoreError, match="now holds topic"):
        store.restore(a["id"])


def test_purge_handles_superseded_links(store):
    old = store.remember("Gym", "Gym opens at six.")
    new = store.remember("Gym", "Gym opens at seven now.", force=True)
    store.supersede(old["id"], new["id"])
    store.forget(new["id"])
    store.db.execute("UPDATE memories SET deleted_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (new["id"],))
    assert store.purgeable() == 1 and store.purge() == 1
    assert store.recall("gym") == []  # the old fact did not resurface
    assert store.stats()["forgotten"] == 1


# ── review ─────────────────────────────────────────────────────────────────
def test_review_finds_near_duplicates_keyword_mode(store):
    a = store.remember("Boiler", "The boiler service is due every March with Acme Heating.")
    store.remember("Boiler service", "Boiler service due every March, Acme Heating.", force=True)
    store.remember("Printer", "Printer toner is in the hall cupboard.")
    pairs = store.review()["near_duplicates"]
    assert len(pairs) == 1 and pairs[0]["older"]["id"] == a["id"]
    assert "supersede" in pairs[0]["suggestion"]


def test_review_finds_near_duplicates_with_embeddings(tmp_path):
    s = Store(tmp_path / "m.db", embedder=FakeEmbedder())
    s.remember("Bakery", "Rye bakery is in Bristol on Mill Street.")
    s.remember("Bakery", "Rye bakery is in Bath on Mill Street.", force=True)
    s.remember("Printer", "Printer toner is in the hall cupboard.")
    gone = s.remember("Gone", "Rye bakery Bristol Mill Street old note.", force=True)
    s.forget(gone["id"])
    pairs = s.review()["near_duplicates"]
    assert len(pairs) == 1 and {pairs[0]["older"]["title"], pairs[0]["newer"]["title"]} == {"Bakery"}
    assert s.review()["forgotten"][0]["id"] == gone["id"]


def test_cli_review_prints(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHAT_MEMORY_DB", str(tmp_path / "a.db"))
    monkeypatch.setenv("CHAT_MEMORY_SEMANTIC", "0")
    from chat_memory import server

    server.store.cache_clear()
    main(["review"])
    assert "Nothing to tidy" in capsys.readouterr().out
    server.store.cache_clear()


# ── usage log ──────────────────────────────────────────────────────────────
def test_tool_calls_are_logged_without_content(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_MEMORY_DB", str(tmp_path / "a.db"))
    monkeypatch.setenv("CHAT_MEMORY_SEMANTIC", "0")
    from chat_memory import server

    server.store.cache_clear()
    server.remember("Bins", "Recycling goes out on Tuesday night.")
    server.recall("recycling")
    with pytest.raises(Exception):
        server.get_memory(999)
    rows = server.store().db.execute("SELECT * FROM tool_calls").fetchall()
    assert [r["tool"] for r in rows] == ["remember", "recall", "get_memory"]
    assert "Recycling" not in json.dumps([dict(r) for r in rows])
    u = server.store().usage()
    assert {t["tool"] for t in u["tools"]} == {"remember", "recall", "get_memory"}
    assert server.fixed_cost_chars() > 1000
    server.store.cache_clear()
