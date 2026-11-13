# chat-memory

A small, local [MCP](https://modelcontextprotocol.io) server that gives AI chats a memory that
lasts between sessions. It needs no cloud service, no GPU and no background service: just one
SQLite file and Python.

- **Local only.** Memories live in a single SQLite file on your machine
  (default `~/.local/share/chat-memory/memory.db`, readable only by you).
- **Hybrid search.** Keyword search (SQLite FTS5), plus optional meaning-based search with a
  small CPU model (`all-MiniLM-L6-v2`, ONNX). "Where's my vegetable patch?" finds the allotment.
- **No duplicates piling up.** Exact duplicates are counted, not re-saved. Near-duplicates are
  shown to the AI so it can update the existing memory instead. Subjects that evolve use a
  stable `topic_key` and are updated in place, with every earlier version kept.
- **Nothing silently lost.** Out-of-date memories are *superseded* (hidden but kept), and
  `forget` is a soft delete.
- **Secrets refused.** Anything that looks like a token, key, password or credential-bearing
  URL is rejected before it reaches disk. Wrap sensitive parts in `<private>…</private>` to
  have them redacted.

The AI you're chatting with decides what is worth remembering, guided by the server's
instructions (see `INSTRUCTIONS` in `src/chat_memory/server.py`). The server stores, searches
and guards against duplicates. That split keeps it light enough for an ordinary laptop.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/mav-rix/chat-memory.git
cd chat-memory
python3 -m venv .venv
.venv/bin/pip install -e .
```

### Optional: semantic search

```bash
.venv/bin/pip install -e '.[semantic]'
.venv/bin/chat-memory download-model   # ~90 MB, pinned revision, SHA-256 checked
```

The model is stored in `~/.cache/chat-memory/models/` and loads on the first search, so a
session that never searches stays at about 75 MB of RAM (about 275 MB once loaded). Existing
memories are embedded automatically the first time they're needed, or run
`chat-memory reindex`. Set `CHAT_MEMORY_SEMANTIC=0` to switch it off.

### Claude Code

```bash
claude mcp add --scope user chat-memory -- "$PWD/.venv/bin/chat-memory"
```

### Claude Desktop and other MCP clients

Add to the client's MCP config (adjust the path):

```json
{
  "mcpServers": {
    "chat-memory": {
      "command": "/path/to/chat-memory/.venv/bin/chat-memory"
    }
  }
}
```

Set `CHAT_MEMORY_DB` to store the database somewhere else (see `.env.example`).

## Tools

| Tool | What it does |
|---|---|
| `get_profile` | Session-start briefing: pinned memories, preferences, most-used facts |
| `remember` | Save a memory (handles topic keys, duplicates and near-duplicates) |
| `recall` | Hybrid keyword + meaning search, returning ranked previews |
| `get_memory` | One memory in full, optionally with its version history |
| `update_memory` | Edit a memory (previous text kept), or pin it |
| `supersede` | Mark an old memory as replaced by a newer one |
| `forget` | Soft-delete a memory |
| `list_recent` | Latest saved or updated memories |
| `review` | Stale, superseded and repeatedly saved memories, for tidying up |

## Your data stays private

The database is never inside the repository. Backups are just a copy of the `.db` file; keep
them out of the repo too. `.gitignore` blocks database files anyway, as a second line of
defence.

## How duplicates are judged

| Signal | Threshold | Effect |
|---|---|---|
| Same normalised text | exact | counted, not re-saved |
| Same `topic_key` | exact | updated in place, old version kept |
| Word overlap (Jaccard) | ≥ 0.80 | not saved; AI asked to update instead (`needs_review`) |
| Meaning (cosine) | ≥ 0.80 | not saved; `needs_review` |
| Word overlap ≥ 0.25 or meaning ≥ 0.45 | | saved, but listed as similar so the AI can `supersede` |

The cosine thresholds were calibrated on sample pairs with `all-MiniLM-L6-v2`: paraphrased
duplicates scored 0.82–0.92, changed facts 0.61–0.74, related topics 0.21–0.38, unrelated
≤ 0.11. Embeddings use the memory's content only; titles count through keyword search.

Search scores mix 0.7 × meaning + 0.3 × keyword (BM25, normalised), with a small boost for
pinned memories. On a 2021 laptop CPU, 10,000 memories search in about 45 ms.

## Command line

```bash
chat-memory                  # run the MCP server (stdio)
chat-memory download-model   # fetch the embedding model
chat-memory reindex          # embed memories that lack embeddings
chat-memory stats            # counts and search mode
```

## Development

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
git config core.hooksPath .githooks   # runs gitleaks before each commit
```

The pre-commit hook needs [gitleaks](https://github.com/gitleaks/gitleaks) on your `PATH`.

## Roadmap

1. ~~Keyword search, topic keys, duplicate and secret guards~~ (0.1)
2. ~~Optional meaning-based search, mixed with keyword scores~~ (0.2)
3. Export and import, and tidier review tooling

## Credits

The design borrows ideas from these projects (no code copied):

- [Engram](https://github.com/Gentleman-Programming/engram) (MIT): topic keys, normalized-hash
  duplicate counting, `<private>` redaction, version history
- [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) (Apache-2.0):
  superseding instead of deleting, similarity bands for duplicates vs. related memories,
  the 0.7 / 0.3 hybrid weighting, the choice of `all-MiniLM-L6-v2`
- The [reference memory server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)
  (MIT): one fact per observation, a prompt that guides what to remember

## License

MIT. The optional embedding model,
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), is
Apache-2.0 and is downloaded separately, not bundled.
