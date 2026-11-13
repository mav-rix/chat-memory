"""Identifiers for exact matching, and removal of card details that are never needed.

Case work (fraud, support, disputes) means checking data from other parties against what you
already know, so identifiers are kept in full and indexed in a normalised form:

    ip        IPv4/IPv6, canonical form          203.0.113.45, 2001:db8::1
    email     lower-cased                         jane@example.com
    card      digits only (12-19 digits; card     4111111111111111
              and gift card numbers, kept whole)
    phone     last 9 digits, so +44 7700 900123   700900123
              and 07700 900123 match
    uuid      lower-cased                         3fa85f64-5717-4562-b3fc-2c963f66afa6
    hash      lower-cased hex, 32+ characters     9f86d0...
    ref       letters and digits mixed, lower-    ORD-2026-118842 -> ord2026118842
              cased, without - and _
    number    6-11 digit numbers                  55120934

Card expiry dates, PINs and security codes are never useful for matching and are the most
sensitive card data, so ``redact_card_details`` removes them before anything is stored.
"""

from __future__ import annotations

import ipaddress
import re

REDACTED = "[REDACTED]"

# ── card details that are removed ──────────────────────────────────────────
_SEP = r"\s*(?:is|was|:|=|#|-)?\s*"
_CARD_NUMBER = r"(?:\d{4}(?:[ -]\d{4}){2,3}(?:[ -]\d{1,3}(?![/\d]))?|\d{4}[ -]\d{6}[ -]\d{5}|\d{12,19})"
_REDACTIONS: list[tuple[str, re.Pattern[str]]] = [
    (
        "card security code",
        re.compile(rf"(?i)\b(cvv2?|cvc2?|cv2|cid|csc|security\s+code){_SEP}\d{{3,4}}\b"),
    ),
    (
        "PIN",
        re.compile(
            rf"(?i)\b(pin(?:\s*(?:code|number|no\.?))?){_SEP}(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{{3,16}}\b"
        ),
    ),
    (
        "card expiry",
        re.compile(
            rf"(?i)\b(exp(?:iry|iration|ires|\.)?(?:\s*date)?|valid\s*(?:thru|through|until)|"
            rf"expires?\s*(?:on|end)?){_SEP}(?:0?[1-9]|1[0-2])\s*[/.-]\s*(?:\d{{4}}|\d{{2}})\b"
        ),
    ),
]
# An unlabelled MM/YY straight after a card number ("4111 1111 1111 1111 12/27").
_EXPIRY_AFTER_CARD = re.compile(rf"(?<!\w)({_CARD_NUMBER})(\W{{1,3}})(?:0[1-9]|1[0-2])\s*/\s*(?:\d{{4}}|\d{{2}})\b")


def redact_card_details(text: str) -> tuple[str, list[str]]:
    """Remove card expiry dates, PINs and security codes, keeping their labels so the
    text still reads naturally ("PIN: [REDACTED]"). Returns the text and what was removed."""
    found: list[str] = []
    for label, pattern in _REDACTIONS:
        text, n = pattern.subn(lambda m: f"{m[1]}: {REDACTED}", text)
        if n:
            found.append(label)
    text, n = _EXPIRY_AFTER_CARD.subn(lambda m: f"{m[1]}{m[2]}{REDACTED}", text)
    if n and "card expiry" not in found:
        found.append("card expiry")
    return text, found


# ── identifiers that are indexed ───────────────────────────────────────────
def _ip(raw: str) -> str | None:
    try:
        return ipaddress.ip_address(raw).compressed
    except ValueError:
        return None


def _digits(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def _phone(raw: str) -> str | None:
    d = _digits(raw)
    return d[-9:] if 8 <= len(d) <= 15 else None


def _ref(raw: str) -> str | None:
    v = re.sub(r"[-_]", "", raw.lstrip("#")).lower()
    return v if len(v) >= 5 else None


# (kind, pattern, normaliser), in priority order: a span claimed by an earlier kind is
# not reused by a later one (an email's parts are not refs, a UUID is not a hash).
_KINDS: list[tuple[str, re.Pattern[str], callable]] = [
    ("email", re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), str.lower),
    ("uuid", re.compile(r"(?<![\w-])[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}(?![\w-])"), str.lower),
    ("ip", re.compile(r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])"), _ip),
    ("ip", re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)"), _ip),
    ("phone", re.compile(r"(?<![\w+])\+\d[\d ()-]{6,18}\d(?!\w)"), _phone),
    ("card", re.compile(rf"(?<![\w+]){_CARD_NUMBER}(?!\w)"), _digits),
    ("phone", re.compile(r"(?<![\w+])0\d{3,4}[ -]?\d{3}[ -]?\d{3,4}(?![\w-])"), _phone),
    ("hash", re.compile(r"(?<!\w)[0-9a-fA-F]{32,128}(?!\w)"), str.lower),
    (
        "ref",
        re.compile(r"(?<![\w.@-])#?(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9_-]{4,63}(?![\w@-])"),
        _ref,
    ),
    ("number", re.compile(r"(?<![\w.,/:+-])\d{6,11}(?![\w/:-]|[.,]\d)"), lambda s: s),
]


def extract(text: str) -> list[tuple[str, str, str]]:
    """All identifiers in ``text`` as (kind, normalised value, text as written), de-duplicated."""
    taken: list[tuple[int, int]] = []
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for kind, pattern, norm in _KINDS:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            value = norm(m[0])
            if not value:
                continue
            taken.append(m.span())
            if (kind, value) not in seen:
                seen.add((kind, value))
                out.append((kind, value, m[0]))
    return out
