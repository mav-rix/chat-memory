"""Semantic-search tests. Made-up data only.

Tests marked ``real_model`` use all-MiniLM-L6-v2 and are skipped when it isn't installed
(run ``keepsake download-model``). The rest use a tiny deterministic fake embedder.
"""

from __future__ import annotations

import json
import re
import zlib

import numpy as np
import pytest

from keepsake.embed import load_embedder
from keepsake.store import Store

REAL = load_embedder()
real_model = pytest.mark.skipif(REAL is None, reason="embedding model not installed")


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims: same words → same direction."""

    name = "fake-bow"

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 64] += 1.0
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)


@pytest.fixture
def real(tmp_path):
    s = Store(tmp_path / "m.db", embedder=REAL)
    yield s
    s.close()


# ── with the real model ────────────────────────────────────────────────────
@real_model
def test_recall_by_meaning_without_shared_words(real):
    veg = real.remember("Vegetable patch", "The south fence bed has courgettes and runner beans.")
    real.remember("Printer", "The office printer is a laser model on the landing.")
    hits = real.recall("what's growing in the garden?")
    assert hits and hits[0]["id"] == veg["id"]


@real_model
def test_paraphrase_is_caught_as_duplicate(real):
    real.remember("Answer style", "Sam prefers short answers in bullet points.", kind="preference")
    r = real.remember("Reply format", "Sam likes brief, bulleted replies.", kind="preference")
    assert r["action"] == "needs_review"
    assert r["similar"][0]["duplicate"] is True


@real_model
def test_changed_fact_is_saved_and_flagged_as_similar(real):
    old = real.remember("Bakery location", "Rye & Shine bakery is in Bristol.")
    new = real.remember("Bakery location", "Rye & Shine bakery moved to Bath.")
    assert new["action"] == "created"
    assert new["similar"][0]["id"] == old["id"] and new["similar"][0]["duplicate"] is False


@real_model
def test_keyword_hit_still_ranks(real):
    a = real.remember("Router model", "Hallway router is model X100.")
    real.remember("Internet", "Broadband contract renews every spring.")
    assert real.recall("X100")[0]["id"] == a["id"]


@real_model
def test_forgotten_memory_leaves_semantic_results(real):
    r = real.remember("Allotment", "Allotment plot 7 is by the river gate.")
    assert real.recall("where is my garden plot")
    real.forget(r["id"])
    assert real.recall("where is my garden plot") == []


# ── with the fake embedder (always runs) ───────────────────────────────────
def test_other_connection_writes_are_seen(tmp_path):
    a = Store(tmp_path / "m.db", embedder=FakeEmbedder())
    b = Store(tmp_path / "m.db", embedder=FakeEmbedder())
    a.remember("Kettle", "The kettle descaler is under the sink.")
    assert len(a.recall("descaler sink")) == 1  # warms a's vector cache
    b.remember("Toaster", "The toaster crumb tray slides out the back.")
    assert [h["title"] for h in a.recall("toaster crumb tray")][0] == "Toaster"


def test_v1_database_is_upgraded_and_backfilled(tmp_path):
    path = tmp_path / "m.db"
    old = Store(path)  # keyword-only
    old.remember("Bins", "Recycling goes out on Tuesday night.")
    old.db.execute("ALTER TABLE memories DROP COLUMN embedding")
    old.db.execute("ALTER TABLE memories DROP COLUMN embedding_model")
    old.db.execute("PRAGMA user_version=1")
    old.close()

    new = Store(path, embedder=FakeEmbedder())
    assert new.stats()["embedded"] == 0
    assert new.backfill_embeddings() == 1
    assert new.stats()["embedded"] == 1 and new.stats()["search"].startswith("hybrid")
    assert new.recall("recycling tuesday")[0]["title"] == "Bins"


def test_model_change_triggers_reembedding(tmp_path):
    s = Store(tmp_path / "m.db", embedder=FakeEmbedder())
    s.remember("Plants", "Water the fern on Sundays.")

    class OtherModel(FakeEmbedder):
        name = "other-model"

    s.embedder = OtherModel()
    assert s.backfill_embeddings() == 1


def test_update_refreshes_embedding(tmp_path):
    s = Store(tmp_path / "m.db", embedder=FakeEmbedder())
    r = s.remember("Car", "The car is parked in bay twelve.")
    s.update(r["id"], content="The bicycle is locked by the station.")
    assert s.recall("bicycle station")[0]["id"] == r["id"]


def test_results_are_json_serialisable(tmp_path):
    s = Store(tmp_path / "m.db", embedder=FakeEmbedder())
    r = s.remember("Notebook", "The blue notebook holds the recipes.")
    json.dumps(s.get(r["id"], with_history=True))
    json.dumps(s.recall("recipes notebook"))
    json.dumps(s.stats())
