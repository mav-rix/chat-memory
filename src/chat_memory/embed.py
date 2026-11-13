"""Optional semantic embeddings: all-MiniLM-L6-v2 on CPU via ONNX Runtime.

Install the extra (``pip install 'chat-memory[semantic]'``), then fetch the model once with
``chat-memory download-model``. The model is never downloaded implicitly by the MCP server:
a 90 MB download in the middle of a tool call would time out and surprise people.

Files come from a pinned Hugging Face revision and are checked against SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"  # Apache-2.0
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSIONS = 384
WINDOWS = os.name == "nt"
MAX_TOKENS = 256

MODEL_FILES = {
    "onnx/model.onnx": ("model.onnx", "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"),  # gitleaks:allow (public file checksum)
    "tokenizer.json": ("tokenizer.json", "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),  # gitleaks:allow (public file checksum)
}


def model_dir() -> Path:
    if env := os.environ.get("CHAT_MEMORY_MODEL_DIR"):
        return Path(env).expanduser()
    if WINDOWS and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return base / "chat-memory" / "models" / MODEL_NAME / MODEL_REVISION[:12]


def semantic_disabled() -> bool:
    return os.environ.get("CHAT_MEMORY_SEMANTIC", "").strip().lower() in ("0", "false", "no", "off")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(out=sys.stderr) -> Path:
    """Fetch the pinned model files, verifying each hash. Safe to re-run."""
    target = model_dir()
    target.mkdir(parents=True, exist_ok=True)
    for remote, (local, expected) in MODEL_FILES.items():
        dest = target / local
        if dest.exists() and _sha256(dest) == expected:
            print(f"ok      {local}", file=out)
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{remote}"
        print(f"fetch   {local} …", file=out)
        fd, tmp = tempfile.mkstemp(dir=target, prefix=".part-")
        try:
            with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=60) as r:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            got = _sha256(Path(tmp))
            if got != expected:
                raise RuntimeError(f"{local}: checksum mismatch (expected {expected}, got {got}); not installed.")
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        print(f"ok      {local}", file=out)
    return target


class Embedder:
    """Sentence embeddings (mean-pooled, L2-normalised float32 vectors)."""

    name = f"{MODEL_NAME}@{MODEL_REVISION[:12]}"
    dimensions = DIMENSIONS

    def __init__(self, directory: Path | None = None) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(directory or model_dir())
        self._np = np
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(max_length=MAX_TOKENS)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(str(d / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}

    def embed(self, texts: list[str]):
        np = self._np
        if not texts:
            return np.zeros((0, DIMENSIONS), dtype=np.float32)
        out = []
        for start in range(0, len(texts), 32):
            enc = self._tok.encode_batch(texts[start : start + 32])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0]  # (batch, tokens, 384)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.append(pooled.astype(np.float32))
        return np.vstack(out)


class LazyEmbedder:
    """Loads the model on first use. Each chat session runs its own server process, and the
    loaded model costs roughly 250 MB of RAM, so sessions that never search don't pay for it."""

    name = Embedder.name
    dimensions = DIMENSIONS

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._real: Embedder | None = None

    def embed(self, texts: list[str]):
        if self._real is None:
            self._real = Embedder(self._dir)
        return self._real.embed(texts)


def load_embedder() -> LazyEmbedder | None:
    """An embedder if semantic search is available and enabled, else None (keyword-only).
    Nothing heavy is imported or loaded until the first embedding is needed."""
    import importlib.util

    if semantic_disabled():
        return None
    d = model_dir()
    if not all((d / local).exists() for local, _ in MODEL_FILES.values()):
        return None
    if any(importlib.util.find_spec(m) is None for m in ("numpy", "onnxruntime", "tokenizers")):
        return None
    return LazyEmbedder(d)
