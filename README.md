# chat-memory

A small, local [MCP](https://modelcontextprotocol.io) server that gives AI chats a memory that
lasts between sessions. It needs no cloud service, no GPU and no background service: just one
SQLite file and Python.

- **Local only.** Memories live in a single SQLite file on your machine
  (default `~/.local/share/chat-memory/memory.db`, readable only by you).
- **Full-text search** with SQLite FTS5.
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
| `recall` | Keyword search, returning short previews |
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

## Development

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
git config core.hooksPath .githooks   # runs gitleaks before each commit
```

The pre-commit hook needs [gitleaks](https://github.com/gitleaks/gitleaks) on your `PATH`.

## Roadmap

1. Keyword search, topic keys, duplicate and secret guards (this release)
2. Optional meaning-based search with a small CPU embedding model (`all-MiniLM-L6-v2`, ONNX),
   mixed with keyword scores
3. Export and import, and tidier review tooling

## Credits

The design borrows ideas from these projects (no code copied):

- [Engram](https://github.com/Gentleman-Programming/engram) (MIT): topic keys, normalized-hash
  duplicate counting, `<private>` redaction, version history
- [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) (Apache-2.0):
  superseding instead of deleting, similarity bands for duplicates vs. related memories
- The [reference memory server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)
  (MIT): one fact per observation, a prompt that guides what to remember

## License

MIT
