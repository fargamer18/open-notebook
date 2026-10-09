"""Phase 12: verified, resumable downloads and hostile runtime archives.

Every download comes from a local HTTP server; nothing reaches the internet.
"""

import io
import tarfile
import threading
import zipfile
from pathlib import Path

import httpx
import pytest

from ai_fixtures import FileServer, sha256
from litenotebook.ai import download as dl
from litenotebook.ai.catalog import host_allowed
from litenotebook.ai.download import (
    ChecksumMismatch,
    DownloadCancelled,
    Downloader,
    DownloadError,
    InsufficientSpace,
    NoConnection,
    check_space,
)
from litenotebook.ai.extract import Limits, UnsafeArchiveError, extract

BODY = bytes(range(256)) * 4096  # 1 MiB


@pytest.fixture
def server():
    s = FileServer()
    yield s
    s.close()


def local(**options) -> Downloader:
    options.setdefault("backoff_s", 0.01)
    options.setdefault("reserve_bytes", 0)
    return Downloader(
        require_https=False, allowed_hosts=frozenset({"127.0.0.1"}), **options
    )


def test_download_verifies_and_renames_atomically(server, tmp_path):
    url = server.add("/m.gguf", BODY)
    seen = []
    result = local().fetch(
        url,
        tmp_path / "m.gguf",
        sha256=sha256(BODY),
        size=len(BODY),
        progress=seen.append,
    )
    assert result.path.read_bytes() == BODY and result.sha256 == sha256(BODY)
    assert not (tmp_path / "m.gguf.part").exists()
    assert seen and seen[-1].done == len(BODY) and seen[-1].total == len(BODY)


def test_resume_uses_range_from_the_partial_file(server, tmp_path):
    url = server.add("/m.gguf", BODY)
    (tmp_path / "m.gguf.part").write_bytes(BODY[:300_000])
    result = local().fetch(
        url, tmp_path / "m.gguf", sha256=sha256(BODY), size=len(BODY)
    )
    assert result.resumed_from == 300_000
    assert server.routes["/m.gguf"].hits[0]["Range"] == "bytes=300000-"
    assert (tmp_path / "m.gguf").read_bytes() == BODY


def test_server_without_range_support_restarts_from_zero(server, tmp_path):
    url = server.add("/m.gguf", BODY, ranges=False)
    (tmp_path / "m.gguf.part").write_bytes(b"x" * 1000)  # wrong bytes, ignored
    local().fetch(url, tmp_path / "m.gguf", sha256=sha256(BODY), size=len(BODY))
    assert (tmp_path / "m.gguf").read_bytes() == BODY


def test_a_206_for_the_wrong_offset_is_not_appended(server, tmp_path):
    url = server.add("/m.gguf", BODY, wrong_range=True)
    (tmp_path / "m.gguf.part").write_bytes(BODY[:5000])
    # the mismatched part is discarded and the retry starts from zero
    result = local(retries=1).fetch(
        url, tmp_path / "m.gguf", sha256=sha256(BODY), size=len(BODY)
    )
    assert result.path.read_bytes() == BODY
    hits = [h.get("Range") for h in server.routes["/m.gguf"].hits]
    assert hits == ["bytes=5000-", None]


def test_truncated_transfer_is_retried_and_resumed(server, tmp_path):
    url = server.add("/m.gguf", BODY, truncate_first=2)
    result = local().fetch(
        url, tmp_path / "m.gguf", sha256=sha256(BODY), size=len(BODY)
    )
    assert result.path.read_bytes() == BODY
    assert len(server.routes["/m.gguf"].hits) == 3  # two cut short, then complete


def test_5xx_is_retried_then_succeeds(server, tmp_path):
    url = server.add("/m.gguf", BODY, fail_first=2)
    local().fetch(url, tmp_path / "m.gguf", sha256=sha256(BODY), size=len(BODY))
    assert len(server.routes["/m.gguf"].hits) == 3


def test_persistent_5xx_fails_with_a_clear_message(server, tmp_path):
    url = server.add("/m.gguf", BODY, fail_first=99)
    with pytest.raises(DownloadError, match="HTTP 500"):
        local(retries=2).fetch(url, tmp_path / "m.gguf", sha256=None, size=len(BODY))
    assert not (tmp_path / "m.gguf").exists()


def test_checksum_mismatch_deletes_the_partial_and_never_installs(server, tmp_path):
    url = server.add("/m.gguf", BODY)
    with pytest.raises(ChecksumMismatch):
        local().fetch(url, tmp_path / "m.gguf", sha256="0" * 64, size=len(BODY))
    assert not (tmp_path / "m.gguf").exists()
    assert not (tmp_path / "m.gguf.part").exists()


def test_more_bytes_than_the_catalog_size_is_rejected(server, tmp_path):
    url = server.add("/m.gguf", BODY)
    with pytest.raises(DownloadError, match="more data"):
        local().fetch(url, tmp_path / "m.gguf", sha256=None, size=1000)


def test_timeout_reports_and_keeps_the_partial_for_later(server, tmp_path):
    url = server.add("/m.gguf", BODY, delay_s=2.0)
    with pytest.raises(DownloadError, match="timed out"):
        local(retries=0, timeout_s=0.3).fetch(
            url, tmp_path / "m.gguf", sha256=sha256(BODY), size=len(BODY)
        )
    assert not (tmp_path / "m.gguf").exists()


def test_cancel_stops_and_keeps_the_partial_for_resume(server, tmp_path):
    url = server.add("/m.gguf", BODY * 8)
    cancel = threading.Event()

    def progress(p):
        if p.done > 100_000:
            cancel.set()

    with pytest.raises(DownloadCancelled):
        local().fetch(
            url,
            tmp_path / "m.gguf",
            sha256=None,
            size=len(BODY) * 8,
            progress=progress,
            cancel=cancel,
        )
    assert not (tmp_path / "m.gguf").exists()
    assert (tmp_path / "m.gguf.part").stat().st_size > 0


def test_not_enough_disk_space_fails_before_downloading(server, tmp_path, monkeypatch):
    url = server.add("/m.gguf", BODY)
    usage = dl.shutil._ntuple_diskusage(10**9, 10**9 - 5000, 5000)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: usage)
    with pytest.raises(InsufficientSpace):
        local().fetch(url, tmp_path / "m.gguf", sha256=None, size=len(BODY))
    assert server.routes["/m.gguf"].hits == []
    with pytest.raises(InsufficientSpace):
        check_space(tmp_path, 1000, reserve=10_000)


def test_no_connection_is_reported_as_offline(tmp_path):
    with pytest.raises(NoConnection, match="internet"):
        local(retries=0).fetch(
            "http://127.0.0.1:9/none", tmp_path / "x", sha256=None, size=10
        )


def test_only_catalog_hosts_and_https_are_allowed(tmp_path):
    real = Downloader(
        client=httpx.Client(transport=httpx.MockTransport(lambda r: 1 / 0))
    )
    for url in (
        "http://huggingface.co/x",  # not https
        "https://example.com/x",
        "https://huggingface.co.evil.test/x",
    ):
        with pytest.raises(DownloadError, match="outside the catalog"):
            real.fetch(url, tmp_path / "x", sha256=None, size=1)
    assert host_allowed("https://github.com/ggml-org/llama.cpp/releases/x")
    assert host_allowed("https://cas-bridge.xethub.hf.co/x", redirect=True)
    assert not host_allowed("https://cas-bridge.xethub.hf.co/x")  # only as a redirect


def test_redirect_to_another_site_is_refused(server, tmp_path):
    other = FileServer()
    try:
        target = other.add("/evil", BODY)
        url = server.add(
            "/m.gguf", b"", redirect=target.replace("127.0.0.1", "localhost")
        )
        with pytest.raises(DownloadError, match="unexpected site"):
            local(retries=0).fetch(
                url, tmp_path / "m.gguf", sha256=None, size=len(BODY)
            )
    finally:
        other.close()


# --- hostile runtime archives --------------------------------------------------------


def tar_bytes(entries) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for info, data in entries:
            tf.addfile(info, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def member(name, data=b"x", kind=tarfile.REGTYPE, linkname="", mode=0o644):
    info = tarfile.TarInfo(name)
    info.type, info.linkname, info.mode = kind, linkname, mode
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    return info, data if kind == tarfile.REGTYPE else None


def write(tmp_path, data: bytes, suffix=".tar.gz") -> Path:
    path = tmp_path / f"a{suffix}"
    path.write_bytes(data)
    return path


def test_real_layout_extracts_with_links_as_copies(tmp_path):
    archive = write(
        tmp_path,
        tar_bytes(
            [
                member("llama-b1/"[:-1] + "/", kind=tarfile.DIRTYPE),
                member("llama-b1/llama-server", b"#!bin", mode=0o755),
                member("llama-b1/libggml.so.0", b"lib"),
                member(
                    "llama-b1/libggml.so", kind=tarfile.SYMTYPE, linkname="libggml.so.0"
                ),
            ]
        ),
    )
    out = tmp_path / "out"
    names = extract(archive, "tar.gz", out)
    assert "llama-b1/libggml.so" in names
    link = out / "llama-b1" / "libggml.so"
    assert link.is_file() and not link.is_symlink() and link.read_bytes() == b"lib"
    assert (out / "llama-b1" / "llama-server").stat().st_mode & 0o100


@pytest.mark.parametrize(
    "entry",
    [
        member("../evil"),
        member("/abs/evil"),
        member("a/../../evil"),
        member("C:/evil"),
        member("a\\..\\evil"),
        member("CON"),
        member("ok/link", kind=tarfile.SYMTYPE, linkname="/etc/passwd"),
        member("ok/link", kind=tarfile.SYMTYPE, linkname="../../outside"),
        member("ok/hard", kind=tarfile.LNKTYPE, linkname="../outside"),
        member("dev", kind=tarfile.CHRTYPE),
        member("fifo", kind=tarfile.FIFOTYPE),
    ],
    ids=[
        "dotdot",
        "absolute",
        "nested-dotdot",
        "drive",
        "backslash",
        "reserved",
        "symlink-abs",
        "symlink-up",
        "hardlink-up",
        "device",
        "fifo",
    ],
)
def test_malicious_tar_entries_are_rejected(tmp_path, entry):
    archive = write(tmp_path, tar_bytes([member("ok/file"), entry]))
    with pytest.raises(UnsafeArchiveError):
        extract(archive, "tar.gz", tmp_path / "out")
    assert not (tmp_path / "evil").exists() and not (tmp_path.parent / "evil").exists()


def test_link_to_a_missing_target_is_rejected(tmp_path):
    archive = write(
        tmp_path,
        tar_bytes(
            [
                member("a/link", kind=tarfile.SYMTYPE, linkname="nothing-here"),
            ]
        ),
    )
    with pytest.raises(UnsafeArchiveError):
        extract(archive, "tar.gz", tmp_path / "out")


def test_duplicate_and_case_colliding_names_are_rejected(tmp_path):
    for second in ("a/file", "A/FILE"):
        archive = write(tmp_path, tar_bytes([member("a/file"), member(second)]))
        with pytest.raises(UnsafeArchiveError, match="Duplicate"):
            extract(archive, "tar.gz", tmp_path / f"out-{second[0]}")


def test_size_and_count_limits_stop_archive_bombs(tmp_path):
    archive = write(tmp_path, tar_bytes([member("big", b"\0" * 5000)]))
    with pytest.raises(UnsafeArchiveError, match="too large"):
        extract(archive, "tar.gz", tmp_path / "o1", Limits(max_entry_bytes=1000))
    archive = write(tmp_path, tar_bytes([member(f"f{i}") for i in range(20)]))
    with pytest.raises(UnsafeArchiveError, match="too many"):
        extract(archive, "tar.gz", tmp_path / "o2", Limits(max_files=10))
    archive = write(
        tmp_path, tar_bytes([member(f"f{i}", b"\0" * 600) for i in range(5)])
    )
    with pytest.raises(UnsafeArchiveError, match="size limit"):
        extract(archive, "tar.gz", tmp_path / "o3", Limits(max_total_bytes=2000))


def zip_bytes(entries) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            zf.writestr(info, data)
    return buf.getvalue()


def test_windows_style_flat_zip_extracts(tmp_path):
    archive = write(
        tmp_path,
        zip_bytes(
            [
                ("llama-server.exe", b"MZ", 0o100644),
                ("ggml.dll", b"dll", 0o100644),
            ]
        ),
        ".zip",
    )
    names = extract(archive, "zip", tmp_path / "out")
    assert sorted(names) == ["ggml.dll", "llama-server.exe"]


@pytest.mark.parametrize(
    "name,data,mode",
    [
        ("../evil.exe", b"x", 0o100644),
        ("C:\\evil.exe", b"x", 0o100644),
        ("//server/share/evil", b"x", 0o100644),
        ("aux.dll", b"x", 0o100644),
        ("link", b"/etc/passwd", 0o120777),
    ],
    ids=["dotdot", "drive-backslash", "unc", "reserved", "symlink-abs"],
)
def test_malicious_zip_entries_are_rejected(tmp_path, name, data, mode):
    archive = write(
        tmp_path, zip_bytes([("ok.txt", b"ok", 0o100644), (name, data, mode)]), ".zip"
    )
    with pytest.raises(UnsafeArchiveError):
        extract(archive, "zip", tmp_path / "out")


def test_not_an_archive_is_rejected(tmp_path):
    bad = write(tmp_path, b"not an archive")
    with pytest.raises(UnsafeArchiveError):
        extract(bad, "tar.gz", tmp_path / "o1")
    with pytest.raises(UnsafeArchiveError):
        extract(bad, "zip", tmp_path / "o2")
    with pytest.raises(UnsafeArchiveError, match="Unsupported"):
        extract(bad, "rar", tmp_path / "o3")
