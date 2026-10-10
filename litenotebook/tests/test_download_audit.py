"""Downloader re-audit (Phase 16): what may be fetched, from where, and when
it may be used.

* every catalog asset is HTTPS on a publisher host, size-pinned and
  SHA-256-pinned; MiniLM is pinned to one upstream commit and every one of
  its files has a checksum;
* the downloader refuses non-catalog hosts *before connecting*, including
  on redirects;
* the MiniLM fetch child and the worker both use the pinned commit;
* nothing downloaded is executed or selected before verification (covered
  in test_ai_download / test_ai_runtime / test_audio_speech; spot-checked
  here through the catalog invariants).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from litenotebook.ai import catalog as cat
from litenotebook.ai.download import Downloader, DownloadError
from litenotebook.config import EMBED_MODEL, EMBED_REVISION

CATALOG = Path(cat.__file__).parent / "catalog"
SHA = re.compile(r"^[0-9a-f]{64}$")


def _assets(node):  # type: ignore[no-untyped-def]
    if isinstance(node, dict):
        if "url" in node and ("sha256" in node or "size" in node):
            yield node
        for value in node.values():
            yield from _assets(value)
    elif isinstance(node, list):
        for value in node:
            yield from _assets(value)


@pytest.mark.parametrize("name", ["runtimes.json", "models.json", "speech.json"])
def test_every_catalog_asset_is_https_publisher_size_and_sha_pinned(name):
    data = json.loads((CATALOG / name).read_text("utf-8"))
    assets = list(_assets(data))
    assert assets, name
    for asset in assets:
        assert cat.host_allowed(asset["url"]), asset["url"]
        assert asset["url"].startswith("https://")
        assert SHA.match(asset["sha256"] or ""), asset["url"]
        assert int(asset["size"]) > 0
        # pinned release paths, never "latest"
        assert "/latest/" not in asset["url"] and "/main/" not in asset["url"]


def test_minilm_is_pinned_to_one_commit_with_every_file_checksummed():
    entry = cat.load_embeddings()[0]
    assert entry.repository == EMBED_MODEL
    assert entry.revision == EMBED_REVISION and re.fullmatch(
        r"[0-9a-f]{40}", entry.revision
    )
    assert all(f.sha256 and SHA.match(f.sha256) for f in entry.files)
    assert sum(f.size for f in entry.files) == entry.size


def test_minilm_fetch_uses_the_pinned_commit(monkeypatch, capsys):
    import huggingface_hub

    from litenotebook.ai import minilm_fetch

    seen = []
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, path, revision=None, **k: seen.append((repo, path, revision)),
    )
    assert minilm_fetch.main() == 0
    assert seen and all(rev == EMBED_REVISION for _, _, rev in seen)


def test_redirect_to_a_foreign_host_is_refused_before_connecting():
    contacted: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        contacted.append(request.url.host)
        if request.url.host == "github.com":
            return httpx.Response(
                302, headers={"location": "https://evil.example/x.zip"}
            )
        return httpx.Response(200, content=b"never")

    client = httpx.Client(
        transport=httpx.MockTransport(transport), follow_redirects=True
    )
    downloader = Downloader(client, retries=0)
    with pytest.raises(DownloadError, match="unexpected site"):
        downloader.client.get(
            "https://github.com/ggml-org/llama.cpp/releases/download/b1/x.zip"
        )
    assert contacted == ["github.com"]  # evil.example was never contacted


def test_non_catalog_and_plain_http_urls_are_refused(tmp_path):
    downloader = Downloader(
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x"))
        )
    )
    for url in (
        "http://github.com/a/b.zip",
        "https://evil.example/a.zip",
        "https://github.com.evil.example/a.zip",
        "https://huggingface.co@evil.example/a.zip",
    ):
        with pytest.raises(DownloadError):
            downloader.fetch(url, tmp_path / "a.zip", sha256="0" * 64, size=1)
    assert not list(tmp_path.iterdir())
