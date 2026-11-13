# chat-memory

A small, local [MCP](https://modelcontextprotocol.io) server that gives AI chats a memory that
lasts between sessions. It needs no cloud service, no GPU and no background service: just one
SQLite file and Python.

- **Local only.** Memories live in a single SQLite file on your machine
  (default `~/.local/share/chat-memory/memory.db`, or `%LOCALAPPDATA%\chat-memory` on
  Windows; readable only by you).
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

Requires Python 3.11+ and works on Linux, macOS and Windows. (Windows: see
[the Windows section](#windows) for the commands.)

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

The model is stored in `~/.cache/chat-memory/models/` (Windows: `%LOCALAPPDATA%\chat-memory\models`) and loads on the first search, so a
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

### Windows

In PowerShell. Check `py --version` shows 3.11 or newer first (otherwise
`winget install Python.Python.3.12`).

```powershell
git clone https://github.com/mav-rix/chat-memory.git
cd chat-memory
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[semantic]"        # or just -e . for keyword-only search
.venv\Scripts\chat-memory download-model         # optional, ~90 MB
claude mcp add --scope user chat-memory -- "$PWD\.venv\Scripts\chat-memory.exe"
claude mcp list                                   # should show chat-memory: Connected
```

- **Where things live:** the database is `%LOCALAPPDATA%\chat-memory\memory.db` and the model
  is under `%LOCALAPPDATA%\chat-memory\models`. They're protected by your Windows user
  account's folder permissions (the Linux/macOS "owner only" file mode doesn't apply).
- **Claude Desktop:** in its JSON config, double the backslashes:
  `"command": "C:\\Users\\you\\chat-memory\\.venv\\Scripts\\chat-memory.exe"`.
- **Don't move the folder** after registering it; Claude Code runs the server from there.
  If you do, run `claude mcp remove chat-memory --scope user` and add it again.
- **Keep the database out of OneDrive/Dropbox.** Syncing a live SQLite file can corrupt it.
  Use `export`/`import` to copy memories between machines.

### Behind a company network

- **pip:** use your company's package mirror or proxy (`pip install --proxy ...` or
  `pip config set global.index-url ...`).
- **Model download:** set `HTTPS_PROXY` if needed. Python uses the Windows certificate store,
  so company TLS inspection usually works. If huggingface.co is blocked, everything still
  works with keyword search; or download `onnx/model.onnx` and `tokenizer.json` from the pinned
  revision another way, put them in one folder and point `CHAT_MEMORY_MODEL_DIR` at it.
- **Managed Claude Code:** your IT team may only allow approved MCP servers. If
  `claude mcp list` doesn't show it, that's likely why. Check that keeping work chat memories
  in a local file is allowed.

## Tools

| Tool | What it does |
|---|---|
| `get_profile` | Session-start briefing: pinned memories, preferences, most-used facts |
| `remember` | Save a memory (handles topic keys, duplicates and near-duplicates) |
| `recall` | Hybrid keyword + meaning search; the strongest hits come back in full, the rest as previews |
| `get_memory` | One memory in full, optionally with its version history |
| `update_memory` | Edit a memory (previous text kept), or pin it |
| `supersede` | Mark an old memory as replaced by a newer one |
| `forget` | Soft-delete a memory |
| `restore` | Bring back a forgotten or superseded memory |
| `list_recent` | Latest saved or updated memories |
| `review` | Near-duplicates and possible conflicts (with suggestions), plus stale, superseded, repeated and forgotten memories |

## Your data stays private

The database is never inside the repository. `.gitignore` blocks database files and
`.jsonl` exports anyway, as a second line of defence.

## Backups, moving machines, tidying up

```bash
chat-memory export ~/memory-backup.jsonl     # readable only by you; warns if inside a git repo
chat-memory import ~/memory-backup.jsonl --dry-run
chat-memory import ~/memory-backup.jsonl
chat-memory import ~/.claude/projects/<project>/memory   # bring in Claude Code's memory files
```

Exports are JSON Lines: a header line, then one memory per line with its version history
(embeddings are rebuilt on import). Import is all or nothing and safe to repeat: exact
duplicates are skipped, a `topic_key` that already exists keeps whichever version is newer
(the other goes into history), superseded links are kept, and every record passes the same
secret filter as `remember`.

`chat-memory review` lists pairs that look like the same fact (cosine ≥ 0.80) or a changed fact
(≥ 0.60), each with a suggested `supersede`, plus stale, repeated, superseded and forgotten
memories. In a chat, ask the AI to tidy up your memory; it calls `review` and asks before
changing anything. `chat-memory purge` permanently deletes memories forgotten more than 30
days ago (it asks first).

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
chat-memory usage            # estimated tokens used by memory tool calls
chat-memory review           # what needs tidying
chat-memory export [FILE]    # back up as JSON Lines (stdout if no file)
chat-memory import PATH      # merge an export or a Claude Code memory folder (--dry-run)
chat-memory purge            # permanently delete long-forgotten memories (--days, --yes)
```

## Token use

Every tool call means the AI re-reads the whole conversation, so the number of round trips
matters far more than the size of each answer. `recall` therefore returns the full text of its
best hits (up to 3, within 8,000 characters), and the instructions ask for a single `recall`
only when past context could matter, not a `get_profile` at every session start. In tests
with Claude Code this cut tokens on memory-backed questions by roughly 45%, and added
nothing measurable to questions that need no memory (Claude Code loads MCP tool definitions
on demand). `chat-memory usage` shows what the tools have returned; it logs only tool names
and sizes, never content.

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
3. ~~Export and import, near-duplicate and conflict review, restore~~ (0.3)
4. ~~Usage log, fewer round trips~~ (0.4)

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
