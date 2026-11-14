"""Case-work features: card-detail redaction, identifier extraction, exact matching, work mode.

Made-up data only: documentation IP ranges (RFC 5737/3849), card-network test numbers,
example.com addresses and the UK drama phone range.
"""

from __future__ import annotations

import pytest

from keepsake.identifiers import extract, redact_card_details
from keepsake.server import instructions, main
from keepsake.store import Store

TEST_CARD = "4111 1111 1111 1111"  # Visa test number
GIFT_CARD = "6006491234567890123"  # 19 digits, not Luhn-valid, like many gift cards


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db")
    yield s
    s.close()


# ── redaction ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text, gone, label",
    [
        (f"Card {TEST_CARD} exp 12/27", "12/27", "card expiry"),
        (f"Card {TEST_CARD} expiry date: 03/2028", "03/2028", "card expiry"),
        (f"Card {TEST_CARD} 12/27 used twice", "12/27", "card expiry"),
        ("Valid thru 1/29", "1/29", "card expiry"),
        ("CVV 123 read out on the call", "123", "card security code"),
        ("security code is 9876", "9876", "card security code"),
        (f"Gift card {GIFT_CARD} PIN: 4821", "4821", "PIN"),
        ("pin is 8K2X91", "8K2X91", "PIN"),
    ],
)
def test_card_details_are_removed(text, gone, label):
    out, labels = redact_card_details(text)
    assert gone not in out and "[REDACTED]" in out and label in labels


def test_card_and_gift_numbers_are_kept():
    text = f"Card {TEST_CARD} exp 12/27, gift card {GIFT_CARD} pin 4821"
    out, _ = redact_card_details(text)
    assert TEST_CARD in out and GIFT_CARD in out


@pytest.mark.parametrize(
    "text",
    ["Pin the memory to the top", "Delivery expected 12/27 (December)", "Expected value 3/4 of total",
     "Version 12/27 of the form"],
)
def test_ordinary_text_is_untouched(text):
    assert redact_card_details(text) == (text, [])


# ── extraction ─────────────────────────────────────────────────────────────
def kinds(text):
    return {(k, v) for k, v, _ in extract(text)}


def test_extracts_each_kind():
    got = kinds(
        "From 203.0.113.45 and 2001:DB8::1, jane.example@Example.com, +44 7700 900123, "
        f"card {TEST_CARD}, gift {GIFT_CARD}, order ORD-2026-118842, ref 55120934, "
        "session 3FA85F64-5717-4562-B3FC-2C963F66AFA6, "
        "device 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    )
    assert {
        ("ip", "203.0.113.45"),
        ("ip", "2001:db8::1"),
        ("email", "jane.example@example.com"),
        ("phone", "700900123"),
        ("card", "4111111111111111"),
        ("card", GIFT_CARD),
        ("ref", "ord2026118842"),
        ("number", "55120934"),
        ("uuid", "3fa85f64-5717-4562-b3fc-2c963f66afa6"),
        ("hash", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"),
    } <= got


def test_formats_normalise_to_the_same_value():
    assert kinds("4111-1111-1111-1111") == kinds("4111111111111111") == kinds(TEST_CARD)
    assert kinds("ORD2026118842") == kinds("ord-2026-118842") == kinds("#ORD_2026_118842")
    assert kinds("+44 7700 900123") == kinds("07700 900123")


def test_dates_amounts_and_versions_are_not_identifiers():
    assert kinds("Paid £1,250.00 on 13/11/2026 at 10:30:00, version 0.4.1") == set()


def test_unlabelled_expiry_is_not_part_of_the_card_number():
    assert ("card", "4111111111111111") in kinds(f"{TEST_CARD} 12/27")


# ── storing and matching ───────────────────────────────────────────────────
def test_remember_redacts_and_reports(store):
    r = store.remember("Case 7", f"Card {TEST_CARD} exp 12/27 CVV 123 used from 203.0.113.45.", kind="case")
    assert set(r["redacted"]) == {"card expiry", "card security code"}
    content = store.get(r["id"])["content"]
    assert TEST_CARD in content and "12/27" not in content and "123 " not in content


def test_match_is_exact_never_near(store):
    a = store.remember("Case 1 mule account", "Orders ORD-2026-118842 and ORD-2026-118843 from 203.0.113.45.")
    b = store.remember("Case 2 refund abuse", "Refund on ORD-2026-220019 from 203.0.113.46, customer Jane Example.")
    res = {r["query"]: r for r in store.match(["203.0.113.45", "203.0.113.99", "ord2026220019", "Jane Example", "Jane"])}
    assert [m["id"] for m in res["203.0.113.45"]["memories"]] == [a["id"]]
    assert res["203.0.113.99"]["found"] is False
    assert [m["id"] for m in res["ord2026220019"]["memories"]] == [b["id"]]
    assert res["ord2026220019"]["memories"][0]["as_written"] == "ORD-2026-220019"
    assert [m["id"] for m in res["Jane Example"]["memories"]] == [b["id"]]
    assert res["Jane"]["found"] is True  # whole word "Jane" appears
    assert store.match(["Jan"])[0]["found"] is False  # never a partial word


def test_match_finds_identifiers_inside_a_pasted_line(store):
    store.remember("Case 3", f"Gift card {GIFT_CARD} redeemed by jane.example@example.com.")
    res = store.match([f"partner report: {GIFT_CARD}, JANE.EXAMPLE@EXAMPLE.COM, 198.51.100.7"])
    assert {(r["kind"], r["found"]) for r in res} == {("email", True), ("card", True), ("ip", False)}


def test_match_marks_superseded_and_skips_forgotten(store):
    old = store.remember("Case 4", "Device 3fa85f64-5717-4562-b3fc-2c963f66afa6 first seen.")
    new = store.remember("Case 4 update", "Device 3fa85f64-5717-4562-b3fc-2c963f66afa6 linked to case 5.", force=True)
    store.supersede(old["id"], new["id"])
    gone = store.remember("Wrong", "Email wrong@example.com saved by mistake.")
    store.forget(gone["id"])
    hits = store.match(["3fa85f64-5717-4562-b3fc-2c963f66afa6"])[0]["memories"]
    assert {(m["id"], m["superseded"]) for m in hits} == {(old["id"], True), (new["id"], False)}
    assert store.match(["wrong@example.com"])[0]["found"] is False


def test_updates_reindex(store):
    r = store.remember("Case 6", "Seen from 203.0.113.10.")
    store.update(r["id"], content="Seen from 203.0.113.20 (corrected).")
    assert store.match(["203.0.113.10"])[0]["found"] is False
    assert store.match(["203.0.113.20"])[0]["found"] is True


def test_old_database_is_indexed_on_upgrade(tmp_path):
    path = tmp_path / "m.db"
    s = Store(path)
    s.remember("Case 8", "Order ORD-77-1234 from 192.0.2.8.")
    s.db.execute("DROP TABLE identifiers")
    s.db.execute("PRAGMA user_version=3")
    s.close()
    assert Store(path).match(["192.0.2.8"])[0]["found"] is True


def test_import_indexes_and_redacts(store):
    report = store.import_records([{"title": "Case 9", "content": "Gift card 6006491234567890123 PIN 4821, IP 192.0.2.9."}])
    assert report["created"] == 1 and report["redacted"] == 1
    assert store.match(["192.0.2.9"])[0]["found"] is True
    assert "4821" not in store.recall("gift card")[0].get("content", "")


# ── work mode and CLI ──────────────────────────────────────────────────────
def test_work_mode_instructions():
    work, personal = instructions("work"), instructions("personal")
    assert "card or gift card numbers" in work and "call match" in work and "account numbers" not in work
    assert "account numbers" in personal and "WORK MODE" not in personal


def test_cli_search(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KEEPSAKE_DB", str(tmp_path / "a.db"))
    monkeypatch.setenv("KEEPSAKE_SEMANTIC", "0")
    from keepsake import server

    server.store.cache_clear()
    server.store().remember("Case 11", "Refund abuse on ORD-2026-555011.")
    main(["search", "refund", "abuse"])
    assert "Case 11" in capsys.readouterr().out
    main(["search", "zebra"])
    assert "No matches" in capsys.readouterr().out
    server.store.cache_clear()


def test_cli_instructions(capsys):
    main(["instructions", "--mode", "work"])
    assert "WORK MODE" in capsys.readouterr().out


def test_cli_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KEEPSAKE_DB", str(tmp_path / "a.db"))
    monkeypatch.setenv("KEEPSAKE_SEMANTIC", "0")
    from keepsake import server

    server.store.cache_clear()
    server.store().remember("Case 10", "Order ORD-2026-555001 from 203.0.113.77 by Jane Example.")
    f = tmp_path / "partner.csv"
    f.write_text("order,ip,name,status\nORD-2026-555001,203.0.113.78,Jane Example,shipped\n")
    main(["match", str(f)])
    out = capsys.readouterr().out
    assert "FOUND      ref     ORD-2026-555001" in out and "not found  ip      203.0.113.78" in out
    assert "Jane Example" not in out and "1 of 2 values found" in out
    main(["match", str(f), "--text"])
    out = capsys.readouterr().out
    assert "FOUND      text    Jane Example" in out
    server.store.cache_clear()
