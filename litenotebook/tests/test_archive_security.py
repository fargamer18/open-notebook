"""Phase 10: hostile archives, limits, and portable (Linux/Windows) paths."""

import json
import stat
import zipfile
from dataclasses import replace

import pytest

from archive_fixture import build_rich_notebook, edit_rows, read_entries, write_archive
from litenotebook.archive.export import export_notebook
from litenotebook.archive.format import ArchiveError, Limits, UnsupportedArchiveError
from litenotebook.archive.importer import import_archive
from litenotebook.archive.reader import verify_archive
from litenotebook.platform_utils.filesystem import (
    InstanceLock,
    create_exclusive,
    in_use,
)
from litenotebook.platform_utils.paths import (
    archive_name_problem,
    download_name,
    portable_key,
)


@pytest.fixture
def export(conn, settings, tmp_path):
    rich = build_rich_notebook(conn, settings, tmp_path)
    path = export_notebook(conn, settings, rich.notebook_id, tmp_path / "ok.zip").path
    return {
        "path": path,
        "entries": read_entries(path),
        "nb": f"notebooks/{rich.notebook_id}",
        "id": rich.notebook_id,
    }


def check(path, settings, limits=None):
    return verify_archive(
        path, limits or Limits.from_settings(settings), current_schema=8
    )


def rejected(path, settings, match, limits=None):
    with pytest.raises(ArchiveError, match=match):
        check(path, settings, limits)


def nothing_imported(conn, settings, path, count_before):
    with pytest.raises(ArchiveError):
        import_archive(conn, settings, path)
    assert conn.execute("SELECT count(*) FROM notebooks").fetchone()[0] == count_before


# --- entry names --------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, reason",
    [
        ("../evil.json", "traversal"),
        ("notebooks/1/../../evil.json", "traversal"),
        ("/etc/passwd", "absolute"),
        ("C:/Windows/evil.json", "drive-letter"),
        ("C:\\Windows\\evil.json", "backslash"),
        ("c:evil.json", "drive-letter"),
        ("//server/share/evil.json", "UNC"),
        ("\\\\server\\share\\evil.json", "backslash"),
        ("notebooks\\1\\notebook.json", "backslash"),
        ("notebooks/1/nul\x00.json", "control"),
        ("notebooks/1/ctrl\r\n.json", "control"),
        ("notebooks/1/./notebook.json", "traversal"),
        ("notebooks//1/notebook.json", "traversal"),
        ("notebooks/1/CON.json", "reserved"),
        ("notebooks/1/aux", "reserved"),
        ("notebooks/1/file.json:stream", "colon"),
        ("notebooks/1/trailing. ", "space or dot"),
    ],
)
def test_unsafe_entry_names_are_rejected(export, settings, tmp_path, name, reason):
    entries = list(export["entries"].items()) + [(name, b"{}")]
    bad = write_archive(tmp_path / "bad.zip", entries, fix_checksums=False)
    rejected(bad, settings, "Unsafe archive entry|Unexpected entry")
    assert (
        archive_name_problem(name) is not None
        and reason.split()[0].lower() in (archive_name_problem(name) or "").lower()
    )


def test_unexpected_or_nested_entries_are_rejected(export, settings, tmp_path):
    nb = export["nb"]
    for name in (
        f"{nb}/sources/9.zip",
        f"{nb}/evil.exe",
        "readme.txt",
        f"{nb}/tables/../x.jsonl",
        f"{nb}/sources/01.pdf",
        "notebooks/0/notebook.json",
    ):
        bad = write_archive(
            tmp_path / "x.zip", {**export["entries"], name: b"x"}, fix_checksums=False
        )
        rejected(bad, settings, "Unexpected entry|Unsafe archive entry")


def test_symlinks_directories_and_encryption_are_rejected(export, settings, tmp_path):
    nb = export["nb"]
    link = write_archive(
        tmp_path / "l.zip",
        export["entries"],
        infos={f"{nb}/notebook.json": {"external_attr": (stat.S_IFLNK | 0o777) << 16}},
    )
    rejected(link, settings, "Symbolic links")
    with zipfile.ZipFile(tmp_path / "d.zip", "w") as zf:
        for name, blob in export["entries"].items():
            zf.writestr(name, blob)
        zf.writestr(zipfile.ZipInfo(f"{nb}/"), b"")
    rejected(tmp_path / "d.zip", settings, "Unsafe archive entry|directory")
    # zipfile clears flag_bits when writing, so set the "encrypted" bit in the
    # central directory record directly, as a hostile tool would.
    import struct

    raw = bytearray(write_archive(tmp_path / "e0.zip", export["entries"]).read_bytes())
    target = f"{nb}/notebook.json".encode()
    pos = raw.find(b"PK\x01\x02")
    while pos != -1:
        name_len = struct.unpack("<H", raw[pos + 28 : pos + 30])[0]
        if raw[pos + 46 : pos + 46 + name_len] == target:
            flags = struct.unpack("<H", raw[pos + 8 : pos + 10])[0]
            raw[pos + 8 : pos + 10] = struct.pack("<H", flags | 0x1)
            break
        pos = raw.find(b"PK\x01\x02", pos + 4)
    enc = tmp_path / "e.zip"
    enc.write_bytes(bytes(raw))
    with zipfile.ZipFile(enc) as zf:
        assert zf.getinfo(f"{nb}/notebook.json").flag_bits & 0x1
    rejected(enc, settings, "Encrypted entries are not supported")


def test_duplicate_and_case_colliding_names(export, settings, tmp_path):
    nb = export["nb"]
    items = list(export["entries"].items())
    dup = write_archive(
        tmp_path / "dup.zip",
        [*items, (f"{nb}/notebook.json", b"{}")],
        fix_checksums=False,
    )
    rejected(dup, settings, "Duplicate entry")
    upper = f"{nb}/notebook.json".replace("notebook.json", "NOTEBOOK.json")
    case = write_archive(
        tmp_path / "case.zip", [*items, (upper, b"{}")], fix_checksums=False
    )
    rejected(case, settings, "Duplicate entry|Unexpected entry")
    assert portable_key("Notes/Ä.json") == portable_key(
        "notes/a\u0308.json".replace("a\u0308", "\u00c4").lower()
    )


def test_malformed_and_truncated_zips(export, settings, tmp_path):
    for blob in (
        b"",
        b"not a zip at all",
        b"PK\x03\x04" + b"\x00" * 40,
        export["path"].read_bytes()[:-200],
    ):
        path = tmp_path / "m.zip"
        path.write_bytes(blob)
        rejected(path, settings, "not a valid ZIP|corrupt|truncated|missing")


# --- checksums, manifest, rows ------------------------------------------------------


def test_checksum_mismatches(export, settings, tmp_path, conn):
    nb, entries = export["nb"], dict(export["entries"])
    pdf = next(n for n in entries if n.endswith(".pdf"))
    entries[pdf] = entries[pdf][:-3] + b"xyz"
    before = conn.execute("SELECT count(*) FROM notebooks").fetchone()[0]
    bad = write_archive(tmp_path / "a.zip", entries, fix_checksums=False)
    rejected(bad, settings, "Checksum mismatch")
    nothing_imported(conn, settings, bad, before)
    entries = dict(export["entries"])  # checksums.json edited, manifest not
    sums = json.loads(entries["checksums.json"])
    sums["files"][f"{nb}/notebook.json"] = "0" * 64
    entries["checksums.json"] = json.dumps(sums).encode()
    rejected(
        write_archive(tmp_path / "b.zip", entries, fix_checksums=False),
        settings,
        "does not match the manifest",
    )
    entries = dict(export["entries"])  # a PDF whose row disagrees (checksums fixed)
    edit_rows(
        entries,
        f"{nb}/tables/sources.jsonl",
        lambda rows: rows[0].update(sha256="f" * 64),
    )
    rejected(write_archive(tmp_path / "c.zip", entries), settings, "SHA-256")
    entries = dict(export["entries"])
    entries[pdf] = b"MZ\x90\x00 not a pdf" + b"\x00" * (len(entries[pdf]) - 14)
    edit_rows(entries, f"{nb}/tables/sources.jsonl", lambda rows: None)
    import hashlib

    sid = int(pdf.rsplit("/", 1)[1].split(".")[0])
    edit_rows(
        entries,
        f"{nb}/tables/sources.jsonl",
        lambda rows: [
            r.update(sha256=hashlib.sha256(entries[pdf]).hexdigest())
            for r in rows
            if r["id"] == sid
        ],
    )
    rejected(write_archive(tmp_path / "d.zip", entries), settings, "not a PDF")


def test_manifest_problems(export, settings, tmp_path):
    def with_manifest(**changes):
        entries = dict(export["entries"])
        manifest = json.loads(entries["manifest.json"])
        manifest.update(changes)
        entries["manifest.json"] = json.dumps(manifest).encode()
        return write_archive(tmp_path / "m.zip", entries)

    rejected(
        with_manifest(format="something-else"), settings, "Not a LiteNotebook archive"
    )
    rejected(with_manifest(export_format_version=0), settings, "invalid format version")
    rejected(
        with_manifest(export_format_version="1"), settings, "invalid format version"
    )
    with pytest.raises(UnsupportedArchiveError, match="newer LiteNotebook version"):
        check(with_manifest(export_format_version=99), settings)
    rejected(with_manifest(kind="everything"), settings, "unknown kind")
    rejected(with_manifest(notebooks=[]), settings, "do not match")
    entries = dict(export["entries"])
    del entries["manifest.json"]
    rejected(
        write_archive(tmp_path / "n.zip", entries, fix_checksums=False),
        settings,
        "manifest.json is missing",
    )


@pytest.mark.parametrize(
    "table, mutate, match",
    [
        ("notes", lambda r: r[0].update(secret="x"), "unexpected columns"),
        ("notes", lambda r: r[0].pop("title"), "unexpected columns"),
        ("notes", lambda r: r[0].update(revision="2"), "must be an integer"),
        ("notes", lambda r: r[0].update(title=None), "must not be empty"),
        ("notes", lambda r: r[0].update(title="nul\x00byte"), "must be text"),
        (
            "annotation_rects",
            lambda r: r[0].update(x0=float("inf")),
            "not valid JSON|number",
        ),
        ("chunks", lambda r: r[0].update(embedding="@@not-base64@@"), "base64"),
        ("chunks", lambda r: r[0].update(page_id=987654), "not in the archive"),
        (
            "messages",
            lambda r: r[1].update(conversation_id=55555),
            "not in the archive",
        ),
        ("notes", lambda r: r.append(dict(r[0])), "duplicate id"),
        (
            "sources",
            lambda r: r[0].update(status="processing"),
            "only finished sources",
        ),
    ],
)
def test_row_validation(export, settings, tmp_path, conn, table, mutate, match):
    entries = dict(export["entries"])
    edit_rows(entries, f"{export['nb']}/tables/{table}.jsonl", mutate)
    bad = write_archive(tmp_path / "rows.zip", entries)
    rejected(bad, settings, match)
    before = conn.execute("SELECT count(*) FROM notebooks").fetchone()[0]
    nothing_imported(conn, settings, bad, before)


# --- limits -------------------------------------------------------------------------


def test_size_and_count_limits(export, settings, tmp_path):
    base = Limits.from_settings(settings)
    path = export["path"]
    rejected(path, settings, "larger than the", replace(base, max_archive_bytes=100))
    rejected(path, settings, "more than", replace(base, max_files=5))
    rejected(path, settings, "per-file limit", replace(base, max_entry_bytes=500))
    rejected(path, settings, "expands beyond", replace(base, max_expanded_bytes=2000))


def test_decompression_bomb_is_stopped_by_limits(export, settings, tmp_path):
    bomb = b"\x00" * (64 * 1024 * 1024)  # 64 MiB of zeros compresses to ~64 KiB
    entries = {**export["entries"], f"{export['nb']}/tables/notes.jsonl": bomb}
    path = write_archive(tmp_path / "bomb.zip", entries)
    assert path.stat().st_size < 1024 * 1024
    small = replace(Limits.from_settings(settings), max_entry_bytes=8 * 1024 * 1024)
    rejected(path, settings, "per-file limit", small)
    lying = tmp_path / "lying.zip"  # header claims 10 bytes, stream holds more
    raw = bytearray(
        write_archive(tmp_path / "base.zip", export["entries"]).read_bytes()
    )
    with zipfile.ZipFile(tmp_path / "base.zip") as zf:
        info = zf.getinfo(f"{export['nb']}/notebook.json")
    header = info.header_offset
    # patch the uncompressed size in both the local header and the central directory
    import struct

    raw[header + 22 : header + 26] = struct.pack("<I", 10)
    cd = raw.rfind(b"PK\x01\x02", 0, len(raw))
    while cd != -1:
        name_len = struct.unpack("<H", raw[cd + 28 : cd + 30])[0]
        if raw[cd + 46 : cd + 46 + name_len] == info.filename.encode():
            raw[cd + 24 : cd + 28] = struct.pack("<I", 10)
            break
        cd = raw.rfind(b"PK\x01\x02", 0, cd)
    lying.write_bytes(bytes(raw))
    rejected(
        lying, settings, "larger than declared|corrupt|not a valid|Checksum|truncated"
    )


# --- metadata stays metadata --------------------------------------------------------


def test_hostile_original_filenames_are_only_metadata(conn, settings, tmp_path, export):
    entries = dict(export["entries"])
    evil = "..\\..\\C:\\Windows\\<script>alert(1)</script>.pdf"
    edit_rows(
        entries,
        f"{export['nb']}/tables/sources.jsonl",
        lambda rows: rows[0].update(original_filename=evil, title=evil),
    )
    path = write_archive(tmp_path / "names.zip", entries)
    _, [r] = import_archive(conn, settings, path, name="Names")
    sid, name, stored = conn.execute(
        "SELECT id, original_filename, stored_path FROM sources WHERE notebook_id = ? "
        "AND original_filename = ?",
        (r.notebook_id, evil),
    ).fetchone()
    assert name == evil and stored == f"notebooks/{r.notebook_id}/sources/{sid}.pdf"
    files = [p.name for p in settings.notebook_dir(r.notebook_id).rglob("*")]
    assert all("script" not in f and ".." not in f for f in files)
    import httpx
    from fastapi.testclient import TestClient

    from litenotebook.web.app import create_app

    app = create_app(
        settings,
        env={},
        llm_transport=httpx.MockTransport(
            lambda q: httpx.Response(200, json={"data": []})
        ),
    )
    with TestClient(app, base_url="http://127.0.0.1:5056") as c:
        html = c.get(f"/notebooks/{r.notebook_id}").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


# --- portable paths -----------------------------------------------------------------


def test_exported_names_are_portable(export):
    for name in export["entries"]:
        assert archive_name_problem(name) is None, name
        assert "\\" not in name and not name.startswith("/")
    keys = [portable_key(n) for n in export["entries"]]
    assert len(keys) == len(set(keys))


def test_portable_name_rules():
    assert archive_name_problem("notebooks/12/sources/3.pdf") is None
    for bad in ("COM1.json", "lpt9/x", "prn", "a/con.txt", "x/ y/z.", "a\x7fb"):
        assert archive_name_problem(bad), bad
    assert portable_key("Maths/A.pdf") == portable_key("maths/a.pdf")
    assert portable_key("Caf\u00e9") == portable_key("Cafe\u0301")  # NFC vs NFD
    assert download_name("Maths: Linear Ä/B?*") == "Maths-Linear-A-B"
    assert download_name("con") == "notebook" and download_name("???") == "notebook"
    assert download_name("..\\..\\evil") == "evil"


def test_instance_lock_and_exclusive_create(tmp_path):
    first, second = InstanceLock(tmp_path), InstanceLock(tmp_path)
    assert first.acquire() and not second.acquire() and in_use(tmp_path)
    first.release()
    assert not in_use(tmp_path) and second.acquire()
    second.release()
    with create_exclusive(tmp_path / "f.bin") as f:
        f.write(b"\r\n\n\x00")  # bytes are written verbatim (binary mode)
    assert (tmp_path / "f.bin").read_bytes() == b"\r\n\n\x00"
    with pytest.raises(FileExistsError):
        create_exclusive(tmp_path / "f.bin")


def test_no_linux_only_calls_in_phase10_code():
    from pathlib import Path

    root = Path(__file__).parents[1] / "src/litenotebook"
    files = [
        *root.joinpath("archive").glob("*.py"),
        *root.joinpath("platform_utils").glob("*.py"),
        root / "web/routes/backup.py",
    ]
    for path in files:
        text = path.read_text()
        for needle in (
            "/proc",
            "/run/media",
            "pkill",
            "systemctl",
            "os.fork",
            "/home/",
            "os.getuid",
            "os.chown",
            "signal.SIGKILL",
        ):
            assert needle not in text, (path.name, needle)
        if path.name != "filesystem.py":  # the platform layer may branch on the OS
            assert "fcntl" not in text and "msvcrt" not in text, path.name
