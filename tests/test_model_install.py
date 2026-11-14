"""Model installation from huggingface.co-style mirrors, file stores and folders.

Uses tiny stand-in files (with their own hashes) and a local HTTP server; nothing is
downloaded from the internet.
"""

from __future__ import annotations

import hashlib
import http.server
import threading

import pytest

from keepsake import embed
from keepsake.server import main

FILES = {"onnx/model.onnx": b"fake onnx bytes", "tokenizer.json": b'{"fake": true}'}


@pytest.fixture(autouse=True)
def fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        embed,
        "MODEL_FILES",
        {remote: (remote.rsplit("/", 1)[-1], hashlib.sha256(data).hexdigest()) for remote, data in FILES.items()},
    )
    monkeypatch.setenv("KEEPSAKE_MODEL_DIR", str(tmp_path / "installed"))
    for v in ("KEEPSAKE_MODEL_MIRROR", "KEEPSAKE_MODEL_MIRROR_TOKEN", "HF_ENDPOINT"):
        monkeypatch.delenv(v, raising=False)
    return tmp_path / "installed"


def installed_ok(target):
    return {p.name: p.read_bytes() for p in target.iterdir()} == {
        "model.onnx": FILES["onnx/model.onnx"], "tokenizer.json": FILES["tokenizer.json"]
    }


@pytest.fixture
def server():
    """Serves FILES under any path prefix; records requests; /deny/ answers 403 like a filter."""
    seen: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append({"path": self.path, "auth": self.headers.get("Authorization")})
            if self.path.startswith("/deny/"):
                self.send_response(403)
                self.send_header("X-Direct-Response", "true")
                self.end_headers()
                self.wfile.write(b"<html>blocked</html>")
                return
            for remote, data in FILES.items():
                if self.path.endswith("/" + remote) or self.path.endswith("/" + remote.rsplit("/", 1)[-1]):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()


def test_from_folder_in_either_layout(tmp_path, fake_model):
    nested = tmp_path / "given" / "onnx"
    nested.mkdir(parents=True)
    (nested / "model.onnx").write_bytes(FILES["onnx/model.onnx"])
    (tmp_path / "given" / "tokenizer.json").write_bytes(FILES["tokenizer.json"])
    embed.download_model(source_dir=tmp_path / "given")
    assert installed_ok(fake_model)


def test_from_folder_rejects_altered_file(tmp_path, fake_model):
    d = tmp_path / "given"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"something else")
    (d / "tokenizer.json").write_bytes(FILES["tokenizer.json"])
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        embed.download_model(source_dir=d)
    assert not (fake_model / "model.onnx").exists()
    assert not list(fake_model.glob(".part-*"))  # no leftovers


def test_from_folder_missing_file(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="model.onnx not found"):
        embed.download_model(source_dir=tmp_path / "empty")


def test_huggingface_style_mirror_with_token(server, fake_model, monkeypatch):
    base, seen = server
    monkeypatch.setenv("KEEPSAKE_MODEL_MIRROR_TOKEN", "test-token-value")
    embed.download_model(mirror=base + "/artifactory/api/huggingfaceml/hf-remote")
    assert installed_ok(fake_model)
    assert seen[0]["path"] == (
        f"/artifactory/api/huggingfaceml/hf-remote/{embed.MODEL_REPO}/resolve/{embed.MODEL_REVISION}/onnx/model.onnx"
    )
    assert seen[0]["auth"] == "Bearer test-token-value"


def test_template_mirror_and_env_setting(server, fake_model, monkeypatch):
    base, seen = server
    monkeypatch.setenv("KEEPSAKE_MODEL_MIRROR", base + "/generic/minilm/{name}")
    embed.download_model()
    assert installed_ok(fake_model)
    assert [s["path"] for s in seen] == ["/generic/minilm/model.onnx", "/generic/minilm/tokenizer.json"]
    assert seen[0]["auth"] is None


def test_hf_endpoint_is_honoured(server, fake_model, monkeypatch):
    base, seen = server
    monkeypatch.setenv("HF_ENDPOINT", base)
    embed.download_model()
    assert installed_ok(fake_model)


def test_blocked_download_explains_the_options(server):
    base, _ = server
    with pytest.raises(RuntimeError) as e:
        embed.download_model(mirror=base + "/deny")
    msg = str(e.value)
    assert "HTTP 403" in msg and "--mirror" in msg and "--from" in msg and "<html>" not in msg


def test_cli_from_folder(tmp_path, fake_model, capsys):
    d = tmp_path / "given"
    d.mkdir()
    for remote, data in FILES.items():
        (d / remote.rsplit("/", 1)[-1]).write_bytes(data)
    main(["download-model", "--from", str(d)])
    assert "Model ready" in capsys.readouterr().out and installed_ok(fake_model)


def test_tls_uses_the_os_trust_store_when_available():
    import truststore

    assert isinstance(embed._ssl_context(), truststore.SSLContext)


def test_tls_relax_strict_is_opt_in(monkeypatch):
    import builtins
    import ssl

    real_import = builtins.__import__

    def no_truststore(name, *a, **k):
        if name == "truststore":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_truststore)
    monkeypatch.delenv("KEEPSAKE_TLS_RELAX_STRICT", raising=False)
    strict = embed._ssl_context()
    assert strict.verify_mode == ssl.CERT_REQUIRED and strict.check_hostname
    monkeypatch.setenv("KEEPSAKE_TLS_RELAX_STRICT", "1")
    relaxed = embed._ssl_context()
    assert relaxed.verify_mode == ssl.CERT_REQUIRED and relaxed.check_hostname  # still verified
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        assert not relaxed.verify_flags & ssl.VERIFY_X509_STRICT
