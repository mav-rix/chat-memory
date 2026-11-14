"""Optional semantic embeddings: all-MiniLM-L6-v2 on CPU via ONNX Runtime.

Install the extra (``pip install 'keepsake[semantic]'``), then fetch the model once with
``keepsake download-model``. The model is never downloaded implicitly by the MCP server:
a 90 MB download in the middle of a tool call would time out and surprise people.

Files come from a pinned Hugging Face revision and are checked against SHA-256 hashes.
Downloading is the only network access: the model runs entirely on the local CPU, nothing is
ever uploaded, and ONNX Runtime's optional telemetry is switched off.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
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
    from .store import setting

    if env := setting("MODEL_DIR"):
        return Path(env).expanduser()
    if WINDOWS and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    path = base / "keepsake" / "models" / MODEL_NAME / MODEL_REVISION[:12]
    legacy = base / "chat-memory" / "models" / MODEL_NAME / MODEL_REVISION[:12]  # name before 0.6
    return legacy if not path.exists() and legacy.exists() else path


def semantic_disabled() -> bool:
    from .store import setting

    return setting("SEMANTIC").strip().lower() in ("0", "false", "no", "off")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HF_BASE = "https://huggingface.co"


def _file_url(base: str, remote: str) -> str:
    """URL of one model file. ``base`` is a Hugging Face-style endpoint (huggingface.co, or an
    Artifactory/Nexus Hugging Face repository), or a template containing {path} (e.g.
    onnx/model.onnx) or {name} (e.g. model.onnx) for a plain file store."""
    if "{path}" in base or "{name}" in base:
        return base.replace("{path}", remote).replace("{name}", remote.rsplit("/", 1)[-1])
    return f"{base.rstrip('/')}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{remote}"


def _fetch(url: str, dest_fd: int) -> None:
    from . import __version__

    headers = {"User-Agent": f"keepsake/{__version__}"}
    from .store import setting

    if token := setting("MODEL_MIRROR_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with os.fdopen(dest_fd, "wb") as f, urllib.request.urlopen(req, timeout=60) as r:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        hint = ""
        if e.code in (401, 403, 407):
            hint = (
                " This often means a company network filter or a login is needed. Use an approved "
                "internal copy instead: --mirror URL (with KEEPSAKE_MODEL_MIRROR_TOKEN if it needs a "
                "token) or --from FOLDER. Keyword search and match work without the model."
            )
        raise RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}.{hint}") from None


def _find_local(folder: Path, remote: str, local: str) -> Path | None:
    for candidate in (folder / remote, folder / local):
        if candidate.is_file():
            return candidate
    matches = sorted(folder.rglob(local))
    return matches[0] if matches else None


def download_model(out=None, mirror: str | None = None, source_dir: Path | None = None) -> Path:
    """Install the pinned model files, verifying each SHA-256. Safe to re-run.

    Sources, in order: ``source_dir`` (files someone gave you), ``mirror`` / KEEPSAKE_MODEL_MIRROR
    / HF_ENDPOINT (an internal copy), else huggingface.co. Whatever the source, a file is only
    installed if its hash matches the pinned one.
    """
    from .store import setting

    out = out or sys.stderr
    base = mirror or setting("MODEL_MIRROR") or os.environ.get("HF_ENDPOINT") or HF_BASE
    target = model_dir()
    target.mkdir(parents=True, exist_ok=True)
    for remote, (local, expected) in MODEL_FILES.items():
        dest = target / local
        if dest.exists() and _sha256(dest) == expected:
            print(f"ok      {local}", file=out)
            continue
        fd, tmp = tempfile.mkstemp(dir=target, prefix=".part-")
        try:
            if source_dir is not None:
                found = _find_local(Path(source_dir), remote, local)
                if found is None:
                    os.close(fd)
                    raise RuntimeError(f"{local} not found in {source_dir}.")
                print(f"copy    {local} from {found}", file=out)
                with os.fdopen(fd, "wb") as f, open(found, "rb") as src:
                    shutil.copyfileobj(src, f, 1 << 20)
            else:
                url = _file_url(base, remote)
                print(f"fetch   {local} from {url.split('?')[0]} …", file=out)
                _fetch(url, fd)
            got = _sha256(Path(tmp))
            if got != expected:
                raise RuntimeError(
                    f"{local}: checksum mismatch (expected {expected}, got {got}); not installed. "
                    "The file isn't the pinned version (or was altered)."
                )
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

        # Official Windows builds of ONNX Runtime can emit usage telemetry (no input data, per
        # its docs). Turn it off anyway: memories may hold sensitive case data.
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
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
