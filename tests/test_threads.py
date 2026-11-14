"""The MCP server runs tool calls on worker threads that come and go; the store must cope.

Regression test for a live failure: tools worked for a few minutes, then every call failed
with "SQLite objects created in a thread can only be used in that same thread".
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEPSAKE_DB", str(tmp_path / "m.db"))
    monkeypatch.setenv("KEEPSAKE_SEMANTIC", "0")
    from keepsake import server as srv

    srv.store.cache_clear()
    yield srv
    srv.store.cache_clear()


def test_calls_from_a_new_thread_work(server):
    server.remember("Kettle", "The kettle descaler is under the sink.")  # opens the store here
    out = {}
    t = threading.Thread(target=lambda: out.update(hits=server.recall("descaler"),
                                                   saved=server.remember("Bins", "Bins go out on Tuesday.")))
    t.start()
    t.join()
    assert out["hits"][0]["title"] == "Kettle" and out["saved"]["action"] == "created"


def test_many_concurrent_calls(server):
    def work(i):
        server.remember(f"Case {i}", f"Order ORD-2026-{i:06d} from 203.0.113.{i % 250}.", topic_key=f"case/{i}")
        server.update_memory(server.match([f"ORD-2026-{i:06d}"])[0]["memories"][0]["id"], pinned=True)
        return server.match([f"ORD-2026-{i:06d}"])[0]["found"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(work, range(120)))
    assert server.store().stats()["active"] == 120
    rows = server.store().db.execute("SELECT COUNT(*) FROM tool_calls WHERE ok = 1").fetchone()[0]
    assert rows == 120 * 4  # every call logged, none lost


def test_unexpected_errors_are_reported_not_hidden(server, monkeypatch, capsys):
    from mcp.server.mcpserver.exceptions import ToolError

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(server.store(), "recall", boom)
    with pytest.raises(ToolError, match="RuntimeError: disk on fire"):
        server.recall("anything")
    assert "disk on fire" in capsys.readouterr().err  # traceback in the server log
