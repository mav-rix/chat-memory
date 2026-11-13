# keepsake

A small, local [MCP](https://modelcontextprotocol.io) server that gives AI chats a memory that
lasts between sessions. It needs no cloud service, no GPU and no background service: just one
SQLite file and Python.

- **Local only.** Memories live in a single SQLite file on your machine
  (default `~/.local/share/keepsake/memory.db`, or `%LOCALAPPDATA%\keepsake` on
  Windows; readable only by you).
- **Hybrid search.** Keyword search (SQLite FTS5), plus optional meaning-based search with a
  small CPU model (`all-MiniLM-L6-v2`, ONNX). "Where's my vegetable patch?" finds the allotment.
- **No duplicates piling up.** Exact duplicates are counted, not re-saved. Near-duplicates are
  shown to the AI so it can update the existing memory instead. Subjects that evolve use a
  stable `topic_key` and are updated in place, with every earlier version kept.
- **Nothing silently lost.** Out-of-date memories are *superseded* (hidden but kept), and
  `forget` is a soft delete.
- **Secrets refused.** Anything that looks like a token, key, password or credential-bearing
  URL is rejected before it reaches disk. Card expiry dates, PINs and security codes are
  removed automatically. Wrap other sensitive parts in `<private>…</private>` to have them
  redacted.
- **Exact matching.** IPs, emails, card and gift card numbers, phone numbers, order and
  reference numbers, UUIDs and hashes are indexed, so `match` can check a list of values from
  elsewhere and report exact hits only: `203.0.113.45` never matches `203.0.113.46`.

*Renamed from `chat-memory` in 0.6. Existing databases, models, exports and
`CHAT_MEMORY_*` settings are still found automatically; `KEEPSAKE_*` settings take priority.*

The AI you're chatting with decides what is worth remembering, guided by the server's
instructions (see `INSTRUCTIONS` in `src/keepsake/server.py`). The server stores, searches
and guards against duplicates. That split keeps it light enough for an ordinary laptop.

## Install

Requires Python 3.11+ and works on Linux, macOS and Windows. (Windows: see
[the Windows section](#windows) for the commands.)

```bash
git clone https://github.com/mav-rix/keepsake.git
cd keepsake
python3 -m venv .venv
.venv/bin/pip install -e .
```

### Optional: semantic search

```bash
.venv/bin/pip install -e '.[semantic]'
.venv/bin/keepsake download-model   # ~90 MB, pinned revision, SHA-256 checked
```

**Privacy:** the model runs entirely on your CPU. The only network access is
`download-model` fetching two files (an HTTPS download from huggingface.co; nothing is sent
except the request itself). Your memories and searches never leave the machine, and ONNX
Runtime's optional telemetry is switched off. On a locked-down network, download the two
files another way and set `KEEPSAKE_MODEL_DIR` (see below).

The model is stored in `~/.cache/keepsake/models/` (Windows: `%LOCALAPPDATA%\keepsake\models`) and loads on the first search, so a
session that never searches stays at about 75 MB of RAM (about 275 MB once loaded). Existing
memories are embedded automatically the first time they're needed, or run
`keepsake reindex`. Set `KEEPSAKE_SEMANTIC=0` to switch it off.

### Claude Code

```bash
claude mcp add --scope user keepsake -- "$PWD/.venv/bin/keepsake"
```

### GitHub Copilot (VS Code, Copilot CLI)

keepsake works with any model Copilot offers (OpenAI, Anthropic, Google...): MCP is
model-neutral. Add it to Copilot's MCP config, `~/.copilot/mcp-config.json`
(Windows: `%USERPROFILE%\.copilot\mcp-config.json`), which VS Code's Copilot agent and the
Copilot CLI both read:

```json
{
  "mcpServers": {
    "keepsake": {
      "type": "stdio",
      "command": "C:\\Users\\you\\keepsake\\.venv\\Scripts\\keepsake.exe",
      "args": [],
      "env": { "KEEPSAKE_MODE": "work" },
      "tools": ["*"]
    }
  }
}
```

(Older VS Code setups use **MCP: Open User Configuration**, with the same entry under a
`"servers"` key instead of `"mcpServers"`.) Then, in Copilot Chat, check the tools list shows
keepsake.

- **Business/Enterprise seats:** an organisation policy, *MCP servers in Copilot*, controls
  whether MCP servers can be used at all, and is off by default on some plans. Ask your admin.
- **Instructions:** VS Code passes the server's instructions to the model. If a model still
  doesn't save or recall when it should, run `keepsake instructions --mode work` and paste
  the output into your Copilot custom instructions.
- **Models differ:** how readily a model calls `remember` or `match` varies between models;
  asking directly ("check these against my cases", "log this case") always works.
- **Data flow:** recalled memories become part of the conversation, which Copilot sends to
  the model you've selected, under your organisation's Copilot agreement.

### Claude Desktop and other MCP clients

Add to the client's MCP config (adjust the path):

```json
{
  "mcpServers": {
    "keepsake": {
      "command": "/path/to/keepsake/.venv/bin/keepsake"
    }
  }
}
```

Set `KEEPSAKE_DB` to store the database somewhere else (see `.env.example`).

### Windows

In PowerShell. Check `py --version` shows 3.11 or newer first (otherwise
`winget install Python.Python.3.12`).

```powershell
git clone https://github.com/mav-rix/keepsake.git
cd keepsake
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[semantic]"        # or just -e . for keyword-only search
.venv\Scripts\keepsake download-model         # optional, ~90 MB
claude mcp add --scope user keepsake -- "$PWD\.venv\Scripts\keepsake.exe"
# work mode: add  -e KEEPSAKE_MODE=work  after "--scope user keepsake"
claude mcp list                                   # should show keepsake: Connected
```

- **Where things live:** the database is `%LOCALAPPDATA%\keepsake\memory.db` and the model
  is under `%LOCALAPPDATA%\keepsake\models`. They're protected by your Windows user
  account's folder permissions (the Linux/macOS "owner only" file mode doesn't apply).
- **Claude Desktop:** in its JSON config, double the backslashes:
  `"command": "C:\\Users\\you\\keepsake\\.venv\\Scripts\\keepsake.exe"`.
- **Don't move the folder** after registering it; Claude Code runs the server from there.
  If you do, run `claude mcp remove keepsake --scope user` and add it again.
- **Keep the database out of OneDrive/Dropbox.** Syncing a live SQLite file can corrupt it.
  Use `export`/`import` to copy memories between machines.

### Behind a company network

- **pip:** use your company's package mirror or proxy (`pip install --proxy ...` or
  `pip config set global.index-url ...`).
- **Model download:** set `HTTPS_PROXY` if needed. Python uses the Windows certificate store,
  so company TLS inspection usually works. If huggingface.co is blocked, everything still
  works with keyword search; or download `onnx/model.onnx` and `tokenizer.json` from the pinned
  revision another way, put them in one folder and point `KEEPSAKE_MODEL_DIR` at it.
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
| `match` | Exact lookup of identifiers or names (e.g. data received from another party) |
| `list_recent` | Latest saved or updated memories |
| `review` | Near-duplicates and possible conflicts (with suggestions), plus stale, superseded, repeated and forgotten memories |

## Your data stays private

The database is never inside the repository. `.gitignore` blocks database files and
`.jsonl` exports anyway, as a second line of defence.

## Work mode: case work and exact matching

For investigation work (fraud, disputes, support cases), where you need to check data from
other parties against what you've seen before, run the server in work mode:

```bash
claude mcp add --scope user keepsake -e KEEPSAKE_MODE=work -- "$PWD/.venv/bin/keepsake"
```

Work mode changes only what the AI is asked to save: customer names, emails, phone numbers,
IP addresses, order IDs, transaction references, device IDs and card or gift card numbers are
saved in full (matching depends on exact values), one evolving record per case
(`kind="case"`, `topic_key="case/<id>"`). In every mode:

- **Card expiry dates, PINs and security codes (CVV/CVC) are removed** before saving and
  replaced with `[REDACTED]`; the response says what was removed. Card and gift card numbers
  are kept. Labelled values are caught (`exp 12/27`, `PIN: 4821`, `CVV 123`), as is an
  unlabelled `MM/YY` straight after a card number.
- **Credentials are refused**, as always.

### How matching works

When a memory is saved, the identifiers in it are indexed in a normalised form, so different
ways of writing the same value match:

| Kind | Examples that match each other |
|---|---|
| IP (v4/v6) | `2001:DB8::1` = `2001:db8:0:0:0:0:0:1` |
| Email | `Jane.Example@Example.com` = `jane.example@example.com` |
| Card / gift card (12–19 digits) | `4111 1111 1111 1111` = `4111-1111-1111-1111` = `4111111111111111` |
| Phone (compared on the last 9 digits) | `+44 7700 900123` = `07700 900123` |
| Order / reference (letters and digits) | `ORD-2026-118842` = `ord2026118842` = `#ORD_2026_118842` |
| Number (6–11 digits), UUID, hash (32+ hex) | exact, case-insensitive |

Anything else, such as a customer name, is matched as a whole phrase, ignoring case
(`Jane Example` matches; `Jan` doesn't match `Jane`). Superseded memories are included and
marked, because old records still count as evidence; forgotten ones are not.

In a chat, give the AI the values and it calls `match`. From the command line, any text or
CSV file works; every cell is checked:

```bash
keepsake match partner-report.csv          # identifiers only
keepsake match partner-report.csv --text   # also names and other plain cells
```

```
FOUND      ref     ORD-2026-118843  →  [12] Fraud case FR-1042
not found  ip      203.0.113.46
1 of 2 values found (exact matches only).
```

**Before using this for real case data,** check it fits your organisation's rules: card
numbers and customer details will sit in a local SQLite file (not encrypted by keepsake, so
rely on disk encryption such as BitLocker or FileVault), and recalled memories are sent to the
AI as part of the conversation. Keep work and personal memories in separate databases.

## Backups, moving machines, tidying up

```bash
keepsake export ~/memory-backup.jsonl     # readable only by you; warns if inside a git repo
keepsake import ~/memory-backup.jsonl --dry-run
keepsake import ~/memory-backup.jsonl
keepsake import ~/.claude/projects/<project>/memory   # bring in Claude Code's memory files
keepsake import other-tool.db --dry-run                # another tool's SQLite memory database
```

### Moving from another SQLite-based memory tool

Many memory tools keep their data in SQLite, each with its own tables. `inspect` shows the
tables, columns and row counts, and which columns it would use. It never prints row
contents, so its output is safe to share with whoever is helping you migrate:

```bash
keepsake inspect old-memory.db
#   memories (1520 rows)  ← suggested
#     id INTEGER, content TEXT, tags TEXT, memory_type TEXT, created_at REAL, ...
#   Suggested: --table memories --map content=content --map kind=memory_type ...
keepsake import old-memory.db --dry-run
keepsake import old-memory.db                          # add --kind case for case records
keepsake import old-memory.db --table kb --map content=note_text --map title=heading
```

Fields you can map: `content` (required), `title` (else the first line of the content),
`kind`, `tags` (JSON list or comma-separated), `topic_key`, `project`, `created_at` and
`updated_at` (ISO text, or Unix seconds/milliseconds), and `deleted_at` (rows with a value are
skipped). A type that isn't a keepsake kind is kept as a `type:<name>` tag; every imported
memory is tagged `imported`. Full-text (FTS5/FTS4) tables are read when they hold the
memories themselves; ones that only index another table, or hold no text, are skipped. The
old database is opened read-only and never changed; close
the old tool first (or import a copy). Records go through the same checks as any save, so
card expiry dates, PINs and security codes are removed and credentials are refused. Importing
again adds nothing new.

Exports are JSON Lines: a header line, then one memory per line with its version history
(embeddings are rebuilt on import). Import is all or nothing and safe to repeat: exact
duplicates are skipped, a `topic_key` that already exists keeps whichever version is newer
(the other goes into history), superseded links are kept, and every record passes the same
secret filter as `remember`.

`keepsake review` lists pairs that look like the same fact (cosine ≥ 0.80) or a changed fact
(≥ 0.60), each with a suggested `supersede`, plus stale, repeated, superseded and forgotten
memories. In a chat, ask the AI to tidy up your memory; it calls `review` and asks before
changing anything. `keepsake purge` permanently deletes memories forgotten more than 30
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
keepsake                  # run the MCP server (stdio)
keepsake download-model   # fetch the embedding model
keepsake reindex          # embed memories that lack embeddings
keepsake stats            # counts and search mode
keepsake usage            # estimated tokens used by memory tool calls
keepsake review           # what needs tidying
keepsake match FILE       # exact hits for identifiers in a text/CSV file (--text for names)
keepsake export [FILE]    # back up as JSON Lines (stdout if no file)
keepsake import PATH      # merge an export, a Claude Code memory folder or a SQLite db (--dry-run)
keepsake inspect DB       # another tool's tables/columns and the suggested mapping (no content)
keepsake instructions     # the AI instructions, for clients' custom-instructions settings
keepsake purge            # permanently delete long-forgotten memories (--days, --yes)
```

## Token use

Every tool call means the AI re-reads the whole conversation, so the number of round trips
matters far more than the size of each answer. `recall` therefore returns the full text of its
best hits (up to 3, within 8,000 characters), and the instructions ask for a single `recall`
only when past context could matter, not a `get_profile` at every session start. In tests
with Claude Code this cut tokens on memory-backed questions by roughly 45%, and added
nothing measurable to questions that need no memory (Claude Code loads MCP tool definitions
on demand). `keepsake usage` shows what the tools have returned; it logs only tool names
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
5. ~~Work mode, card-detail redaction, exact identifier matching~~ (0.5)

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
