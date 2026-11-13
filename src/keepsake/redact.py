"""Keep secrets out of memory.

Two layers:
  * ``strip_private`` replaces ``<private>...</private>`` sections with ``[REDACTED]``.
  * ``find_secrets`` spots things that look like credentials, so the store can
    refuse to save them at all. A refused save is better than a leaked token.
"""

from __future__ import annotations

import re

_PRIVATE_TAG = re.compile(r"(?is)<private>.*?</private>")

# (label, pattern). Kept deliberately broad: a false positive costs a retry,
# a false negative puts a credential on disk.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("API key (sk-...)", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    ("Stripe key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("URL with embedded credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@")),
    (
        "password or secret assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
            r"\s*[:=]\s*['\"]?[^\s'\"]{6,}"
        ),
    ),
]


def strip_private(text: str) -> str:
    return _PRIVATE_TAG.sub("[REDACTED]", text)


def find_secrets(text: str) -> list[str]:
    """Return the labels of any credential-like strings found in ``text``."""
    return [label for label, pattern in _SECRET_PATTERNS if pattern.search(text)]
