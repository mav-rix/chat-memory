"""Store tests. All data here is made up; never put real memories in tests."""

from __future__ import annotations

import os
import stat

import pytest

from keepsake.redact import find_secrets, strip_private
from keepsake.store import Store, StoreError, similarity


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "sub" / "memory.db")
    yield s
    s.close()


# ── saving and duplicates ──────────────────────────────────────────────────
def test_create_and_recall(store):
    r = store.remember("Garden layout", "Tomatoes are planted along the south fence.", kind="fact")
    assert r["action"] == "created"
    hits = store.recall("where are the tomatoes planted")
    assert [h["id"] for h in hits] == [r["id"]]


def test_exact_duplicate_is_counted_not_saved(store):
    a = store.remember("Coffee", "Prefers  oat milk in coffee.")
    b = store.remember("coffee", "prefers oat milk in COFFEE.")  # case/space differences
    assert b["action"] == "duplicate" and b["id"] == a["id"]
    assert store.stats()["active"] == 1
    assert store.get(a["id"])["duplicate_count"] == 2


def test_near_duplicate_needs_review_then_force(store):
    store.remember("Bike service", "The blue bike was serviced at the corner shop in March 2026.")
    r = store.remember("Bike service", "The blue bike was serviced at the corner shop in March 2026 again.")
    assert r["action"] == "needs_review" and r["id"] is None
    assert r["similar"][0]["similarity"] >= 0.8
    forced = store.remember(
        "Bike service", "The blue bike was serviced at the corner shop in March 2026 again.", force=True
    )
    assert forced["action"] == "created"


def test_related_memory_is_saved_and_listed(store):
    old = store.remember("Router", "Home router admin page is on the hallway router, model X100.")
    r = store.remember("Router replaced", "The hallway router X100 was replaced by a mesh system.")
    assert r["action"] == "created"
    assert any(s["id"] == old["id"] for s in r.get("similar", []))


def test_topic_key_updates_in_place_with_history(store):
    a = store.remember("Thermostat", "Set to 20C.", kind="preference", topic_key="Home/Thermostat")
    b = store.remember("Thermostat", "Set to 19C in winter.", kind="preference", topic_key="home/thermostat")
    assert b["action"] == "updated" and b["id"] == a["id"]
    m = store.get(a["id"], with_history=True)
    assert m["content"] == "Set to 19C in winter." and m["revision_count"] == 2
    assert m["history"][0]["content"] == "Set to 20C."
    same = store.remember("Thermostat", "Set to 19C in winter.", kind="preference", topic_key="home/thermostat")
    assert same["action"] == "unchanged"


def test_unknown_kind_and_empty_text_rejected(store):
    with pytest.raises(StoreError):
        store.remember("x", "y", kind="gossip")
    with pytest.raises(StoreError):
        store.remember("  ", "content")


# ── secrets ────────────────────────────────────────────────────────────────
# Token-shaped strings are built at runtime so no credential-like text sits in
# the repository (keeps secret scanners quiet and avoids training bad habits).
FAKE_SECRETS = [
    "gh" + "p_" + "A" * 36,
    "github" + "_pat_" + "B" * 50,
    "s" + "k-ant-" + "C" * 30,
    "AK" + "IA" + "D" * 16,
    "AI" + "za" + "E" * 35,
    "-----BEGIN " + "RSA PRIVATE KEY-----",
    "https://user:" + "hunter2" + "@example.com/repo.git",
    "pass" + "word = " + "correcthorse",
]


@pytest.mark.parametrize("secret", FAKE_SECRETS)
def test_secrets_are_refused(store, secret):
    assert find_secrets(secret)
    with pytest.raises(StoreError, match="secret"):
        store.remember("Note", f"Remember this: {secret}")
    assert store.stats()["total"] == 0


def test_update_cannot_sneak_in_a_secret(store):
    r = store.remember("Deploy", "Deploys run from the main branch.")
    with pytest.raises(StoreError):
        store.update(r["id"], content="Use token " + "gh" + "o_" + "Z" * 36)


def test_private_tags_are_redacted(store):
    assert strip_private("a <private>hidden</private> b") == "a [REDACTED] b"
    r = store.remember("Locker", "Gym locker 12, code <PRIVATE>4321</private>.")
    assert "4321" not in store.get(r["id"])["content"]


def test_harmless_text_passes():
    assert not find_secrets("The password manager is Bitwarden; reset links expire after 1 hour.")


# ── lifecycle ──────────────────────────────────────────────────────────────
def test_supersede_hides_old_from_recall(store):
    old = store.remember("Phone", "Main phone is the grey Pixel.")
    new = store.remember("Phone", "Main phone is now the green Galaxy.", force=True)
    store.supersede(old["id"], new["id"])
    ids = [h["id"] for h in store.recall("main phone")]
    assert ids == [new["id"]]
    assert old["id"] in [h["id"] for h in store.recall("main phone", include_superseded=True)]


def test_forget_and_purge(store):
    r = store.remember("Typo", "Saved by mistake.")
    store.forget(r["id"], "mistake")
    assert store.recall("mistake") == []
    assert store.purge(older_than_days=30) == 0  # too recent to purge
    store.db.execute("UPDATE memories SET deleted_at = '2000-01-01T00:00:00+00:00'")
    assert store.purge(older_than_days=30) == 1


def test_profile_puts_pinned_and_preferences_first(store):
    store.remember("Trivia", "The office plant is a fern.")
    p = store.remember("Tone", "Likes short, direct answers.", kind="preference")
    pin = store.remember("Name", "Goes by Sam.", kind="person")
    store.update(pin["id"], pinned=True)
    ids = [m["id"] for m in store.profile()]
    assert ids[:2] == [pin["id"], p["id"]]


def test_review_finds_stale(store):
    r = store.remember("Old hobby", "Collected stamps for a while.")
    store.db.execute("UPDATE memories SET created_at = '2020-01-01T00:00:00+00:00'")
    assert [m["id"] for m in store.review(stale_days=180)["stale"]] == [r["id"]]


def test_project_filter(store):
    a = store.remember("Build cmd", "Run make all to build.", project="alpha")
    store.remember("Build cmd", "Run cargo build to build the tool.", project="beta")
    assert [h["id"] for h in store.recall("build", project="alpha")] == [a["id"]]


def test_fts_syntax_in_query_is_safe(store):
    store.remember("Quotes", "Mentions AND, OR and NEAR in text.")
    for q in ['"unbalanced', "a AND OR (", "*", "NEAR(", ""]:
        store.recall(q)  # must not raise


def test_database_file_is_private(store):
    if os.name == "nt":
        pytest.skip("POSIX permissions; Windows relies on the user profile's ACLs")
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600


def test_similarity_basics():
    assert similarity("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert similarity("apples and pears", "spanner and wrench") == 0.0


def test_recall_gives_full_text_for_top_hits(tmp_path):
    from keepsake.store import Store

    s = Store(tmp_path / "m.db")
    long = "The greenhouse heater runs on a timer. " * 20
    s.remember("Greenhouse heater", long)
    s.remember("Shed", "The shed is next to the greenhouse, mostly empty.")
    hits = s.recall("greenhouse heater timer")
    assert hits[0]["content"] == long.strip() and "preview" not in hits[0]


def test_windows_paths_use_localappdata(monkeypatch, tmp_path):
    from keepsake import embed, store as store_mod

    monkeypatch.delenv("KEEPSAKE_DB", raising=False)
    monkeypatch.delenv("KEEPSAKE_MODEL_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(store_mod, "WINDOWS", True)
    monkeypatch.setattr(embed, "WINDOWS", True)
    assert store_mod.default_db_path() == tmp_path / "keepsake" / "memory.db"
    assert embed.model_dir().is_relative_to(tmp_path / "keepsake" / "models")


def test_names_from_before_the_rename_still_work(monkeypatch, tmp_path):
    from keepsake import embed, store as store_mod

    for v in ("KEEPSAKE_DB", "CHAT_MEMORY_DB", "KEEPSAKE_MODEL_DIR", "CHAT_MEMORY_MODEL_DIR"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("CHAT_MEMORY_DB", str(tmp_path / "old-setting.db"))
    assert store_mod.default_db_path() == tmp_path / "old-setting.db"
    monkeypatch.delenv("CHAT_MEMORY_DB")
    monkeypatch.setattr(store_mod, "WINDOWS", False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert store_mod.default_db_path() == tmp_path / "keepsake" / "memory.db"
    (tmp_path / "chat-memory").mkdir()
    (tmp_path / "chat-memory" / "memory.db").write_bytes(b"")
    assert store_mod.default_db_path() == tmp_path / "chat-memory" / "memory.db"  # existing data found
    monkeypatch.setattr(embed, "WINDOWS", False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert "keepsake" in embed.model_dir().parts
