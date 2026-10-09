"""Phase 14: the optional Speech component — catalog, installs, engine
process handling and audio rendering. No real engine or voice is downloaded:
archives come from a local file server and the engine is a fake program."""

import io
import json
import os
import stat
import sys
import tarfile
import threading
import time
import wave
from pathlib import Path

import pytest

from ai_fixtures import FileServer, sha256
from audio_fixtures import FakeSpeech, install_fake_voices, write_tone
from conftest import process_gone
from litenotebook.ai import speech
from litenotebook.ai.catalog import (
    Asset,
    CatalogError,
    SpeechEngineEntry,
    VoiceEntry,
    load_speech,
)
from litenotebook.ai.download import Downloader
from litenotebook.ai.extract import UnsafeArchiveError
from litenotebook.platform_utils.system import with_library_dir
from litenotebook.study.audio_render import read_pcm, resample, trim_silence, wav_info

FAKE_ENGINE = """#!{python}
import math, os, struct, sys, time, wave
args = sys.argv[1:]
mode = os.environ.get("FAKE_TTS_MODE", "")
log = os.environ.get("FAKE_TTS_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{{os.getpid()}}\\t{{args[-1] if args else ''}}\\n")
if "--help" in args:
    print("Usage: sherpa-onnx-offline-tts --vits-model=... --vits-tokens=...")
    sys.exit(0)
if mode == "crash":
    sys.exit(3)
if mode == "hang":
    time.sleep(60)
out = next(a.split("=", 1)[1] for a in args if a.startswith("--output-filename="))
for a in args:
    if a.startswith("--vits-") and not os.path.exists(a.split("=", 1)[1]):
        sys.exit(4)
text = args[-1]
rate = 22050
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    n = int(rate * max(0.1, len(text) * 0.01))
    w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(i / 8)))
                           for i in range(n)))
"""


def tar_bz2(
    entries: dict[str, bytes | None],
    links: dict[str, str] | None = None,
    modes: dict[str, int] | None = None,
) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
                continue
            info.size = len(data)
            info.mode = (modes or {}).get(name, 0o644)
            tf.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)
    return buf.getvalue()


@pytest.fixture
def server():
    s = FileServer()
    yield s
    s.close()


@pytest.fixture
def downloader():
    d = Downloader(
        require_https=False, allowed_hosts=frozenset({"127.0.0.1"}), backoff_s=0.0
    )
    yield d
    d.close()


def engine_entry(server, body: bytes, **over) -> SpeechEngineEntry:
    url = server.add("/engine.tar.bz2", body)
    from litenotebook.platform_utils.system import machine_arch, os_name

    base = dict(
        id="sherpa-onnx-test",
        project="sherpa-onnx",
        version="1.13.8",
        os=os_name(),
        arch=machine_arch(),
        executable="eng/bin/sherpa-onnx-offline-tts",
        library_dir="eng/lib",
        assets=(Asset(url, len(body), sha256(body), "tar.bz2"),),
        license="Apache-2.0",
        license_url="https://x",
        license_note="",
        release_url="r",
        source_url="s",
        release_date="d",
        layout_verified=True,
        tested=True,
    )
    base.update(over)
    return SpeechEngineEntry(**base)


def voice_entry(server, body: bytes, vid="piper-test-voice") -> VoiceEntry:
    url = server.add(f"/{vid}.tar.bz2", body)
    return VoiceEntry(
        id=vid,
        display_name="Test voice",
        engine="sherpa-onnx",
        model_type="vits-piper",
        language="en",
        locale="en_US",
        quality="medium",
        sample_rate=22050,
        description="",
        license="PD",
        license_url="u",
        model_card_url="u",
        asset=Asset(url, len(body), sha256(body), "tar.bz2"),
    )


def engine_archive() -> bytes:
    exe = FAKE_ENGINE.format(python=sys.executable).encode()
    return tar_bz2(
        {"eng/": None, "eng/bin/sherpa-onnx-offline-tts": exe, "eng/lib/libx.so": b"x"},
        modes={"eng/bin/sherpa-onnx-offline-tts": 0o755},
    )


def voice_archive(*, tokens=True) -> bytes:
    files = {"v/en_US-test.int8.onnx": b"onnx", "v/espeak-ng-data/phontab": b"p"}
    if tokens:
        files["v/tokens.txt"] = b"a 1\n"
    return tar_bz2(files)


# --- catalog ----------------------------------------------------------------------


def test_shipped_speech_catalog_is_pinned_and_small():
    cat = load_speech()
    assert {e.os for e in cat.engines} == {"linux", "windows"}
    for e in cat.engines:
        assert e.license == "Apache-2.0" and e.version == "1.13.8"
        assert all(
            a.url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/")
            and len(a.sha256) == 64
            and a.format == "tar.bz2"
            for a in e.assets
        )
        assert "espeak-ng (GPL-3.0)" in e.license_note
    assert e.executable.endswith(".exe")  # the Windows entry
    assert len(cat.voices) == 3
    for v in cat.voices:
        assert "public domain" in v.license and v.sample_rate == 22050
        assert v.asset.size < 25_000_000
    assert set(cat.default_voices) == {"A", "B", "narrator"}
    assert cat.default_voices["A"] != cat.default_voices["B"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["voices"][0]["asset"].update(url="http://github.com/x.tar.bz2"),
        lambda d: d["voices"][0]["asset"].update(url="https://evil.example/x.tar.bz2"),
        lambda d: d["voices"][0]["asset"].update(sha256="nothex"),
        lambda d: d["voices"][0].update(id="../escape"),
        lambda d: d["engines"][0].update(executable="../../bin/sh"),
        lambda d: d["voices"].append(dict(d["voices"][0])),
        lambda d: d.update(default_voices={"A": "missing"}),
        lambda d: d["voices"][0]["asset"].update(format="exe"),
    ],
)
def test_bad_speech_catalogs_are_refused(mutate):
    from importlib import resources

    data = json.loads(
        resources.files("litenotebook.ai")
        .joinpath("catalog/speech.json")
        .read_text(encoding="utf-8")
    )
    mutate(data)
    with pytest.raises(CatalogError):
        load_speech(json.dumps(data))


# --- installing -------------------------------------------------------------------


def test_status_reads_files_only_and_starts_nothing(settings, monkeypatch):
    import litenotebook.ai.speech as sp

    monkeypatch.setattr(sp, "spawn", lambda *a, **k: pytest.fail("started a process"))
    st = speech.status(settings)
    assert st["installed"] is False and st["voices"] == [] and st["ready"] is False
    assert len(st["available_voices"]) == 3


@pytest.mark.posix_only
def test_install_engine_and_voice_then_synthesize(
    settings, server, downloader, tmp_path
):
    engine = speech.install_engine(
        settings, engine_entry(server, engine_archive()), downloader=downloader
    )
    assert Path(engine.executable).is_file() and engine.version == "1.13.8"
    voice = speech.install_voice(
        settings, voice_entry(server, voice_archive()), downloader=downloader
    )
    assert voice.model.endswith(".onnx") and Path(voice.data_dir).is_dir()
    st = speech.status(settings)
    assert st["ready"] and st["voice_ids"] == ["piper-test-voice"]
    assert not list(speech.downloads_dir(settings).glob("*.bz2"))  # archives removed
    provider = speech.provider_for(settings)
    out = tmp_path / "o.wav"
    provider.synthesize("Hello from the fake engine.", voice, out)
    assert wav_info(out).duration_s > 0.2
    result = speech.test_voice(settings, provider, voice.id)
    assert result["duration_s"] > 0 and result["voice"] == voice.id
    # repair keeps a working install
    again = speech.install_engine(
        settings, engine_entry(server, engine_archive()), downloader=downloader
    )
    assert again.id == engine.id
    assert (
        speech.remove_voice(settings, voice.id)
        and speech.voice(settings, voice.id) is None
    )
    assert speech.remove_voice(settings, "../../etc") is False
    assert speech.remove_engine(settings) and speech.installed_engine(settings) is None


def test_hostile_archives_install_nothing(settings, server, downloader):
    for body, error in (
        (tar_bz2({"../evil.sh": b"x"}), UnsafeArchiveError),
        (
            tar_bz2(
                {"eng/bin/sherpa-onnx-offline-tts": b"x"},
                links={"eng/lib/libc.so": "/etc/passwd"},
            ),
            UnsafeArchiveError,
        ),
        (b"BZh91AY&SY not really bzip2", UnsafeArchiveError),
        (tar_bz2({"eng/README": b"no program here"}), speech.SpeechError),
    ):
        with pytest.raises(error):
            speech.install_engine(
                settings,
                engine_entry(server, body),
                downloader=downloader,
                check_runs=False,
            )
        assert speech.installed_engine(settings) is None
        root = speech.speech_root(settings)
        assert not [p for p in root.glob(".staging-*")] if root.exists() else True
    with pytest.raises(speech.SpeechError, match="expected model"):
        speech.install_voice(
            settings,
            voice_entry(server, voice_archive(tokens=False)),
            downloader=downloader,
        )
    assert speech.installed_voices(settings) == []


def test_tampered_download_is_refused(settings, server, downloader):
    entry = engine_entry(server, engine_archive())
    body = bytearray(engine_archive())
    body[-1] ^= 0xFF  # same size, different bytes
    server.routes["/engine.tar.bz2"].body = bytes(body)
    with pytest.raises(Exception, match="published checksum"):
        speech.install_engine(settings, entry, downloader=downloader)
    assert speech.installed_engine(settings) is None


# --- the engine process -----------------------------------------------------------


def fake_engine(tmp_path: Path) -> speech.InstalledEngine:
    exe = tmp_path / "eng" / "bin" / "sherpa-onnx-offline-tts"
    exe.parent.mkdir(parents=True)
    (tmp_path / "eng" / "lib").mkdir()
    exe.write_text(FAKE_ENGINE.format(python=sys.executable), encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return speech.InstalledEngine(
        "e", "1", str(exe), str(tmp_path / "eng" / "lib"), 0, ""
    )


def fake_voice(settings) -> speech.InstalledVoice:
    install_fake_voices(settings, ("voice-a",))
    v = speech.voice(settings, "voice-a")
    assert v is not None
    return v


@pytest.mark.posix_only
def test_engine_crash_and_bad_text(settings, tmp_path, monkeypatch):
    provider = speech.SherpaProvider(fake_engine(tmp_path))
    voice = fake_voice(settings)
    monkeypatch.setenv("FAKE_TTS_MODE", "crash")
    with pytest.raises(speech.SpeechError, match="exit code 3"):
        provider.synthesize("Hello.", voice, tmp_path / "x.wav")
    monkeypatch.delenv("FAKE_TTS_MODE")
    with pytest.raises(speech.SpeechError, match="Nothing to say"):
        provider.synthesize("   ", voice, tmp_path / "x.wav")
    with pytest.raises(speech.SpeechError, match="too long"):
        provider.synthesize("x" * 2000, voice, tmp_path / "x.wav")


@pytest.mark.posix_only
def test_text_is_one_argument_never_a_flag_or_shell(settings, tmp_path, monkeypatch):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("FAKE_TTS_LOG", str(log))
    provider = speech.SherpaProvider(fake_engine(tmp_path))
    text = "--help; rm -rf / && echo $(whoami) `id` | cat"
    provider.synthesize(text, fake_voice(settings), tmp_path / "x.wav")
    spoken = log.read_text(encoding="utf-8").strip().split("\t", 1)[1]
    assert spoken == text.lstrip("-")  # passed literally, leading dashes removed
    assert (tmp_path / "x.wav").is_file()


@pytest.mark.posix_only
def test_cancel_stops_the_engine_by_handle_and_leaves_no_orphan(
    settings, tmp_path, monkeypatch
):
    log = tmp_path / "calls.log"
    monkeypatch.setenv("FAKE_TTS_LOG", str(log))
    monkeypatch.setenv("FAKE_TTS_MODE", "hang")
    provider = speech.SherpaProvider(fake_engine(tmp_path))
    cancel = threading.Event()
    threading.Timer(0.6, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(speech.SpeechCancelled):
        provider.synthesize(
            "Hello there.", fake_voice(settings), tmp_path / "x.wav", cancel
        )
    assert time.monotonic() - started < 10
    pid = int(log.read_text(encoding="utf-8").split("\t", 1)[0])
    assert pid == provider.last_pid and process_gone(pid)


def test_library_path_is_set_per_platform(tmp_path):
    lib = tmp_path / "lib dir ünï"
    linux = with_library_dir({"LD_LIBRARY_PATH": "/old"}, lib, system="linux")
    assert linux["LD_LIBRARY_PATH"] == f"{lib}:/old"
    win = with_library_dir({"PATH": r"C:\Windows"}, lib, system="windows")
    assert win["PATH"] == f"{lib};C:\\Windows" and "LD_LIBRARY_PATH" not in win
    assert with_library_dir({}, lib, system="linux")["LD_LIBRARY_PATH"] == str(lib)


# --- rendering --------------------------------------------------------------------


def test_wav_helpers(tmp_path):
    a = tmp_path / "a.wav"
    write_tone(a, 0.5, rate=16000)
    pcm, rate = read_pcm(a)
    assert rate == 16000 and pcm.size == 8000
    up = resample(pcm, 16000, 22050)
    assert abs(up.size - 11025) <= 1
    padded = __import__("numpy").concatenate(
        [
            __import__("numpy").zeros(4000, "int16"),
            pcm,
            __import__("numpy").zeros(4000, "int16"),
        ]
    )
    trimmed = trim_silence(padded, 16000)
    assert pcm.size <= trimmed.size < pcm.size + 2000
    stereo = tmp_path / "s.wav"
    with wave.open(str(stereo), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x10\x00\x30\x00" * 800)
    mono, r = read_pcm(stereo)
    assert r == 8000 and mono.size == 800 and int(mono[0]) == 32


def test_fake_speech_provider_contract(tmp_path, settings):
    install_fake_voices(settings)
    fs = FakeSpeech()
    out = tmp_path / "x.wav"
    fs.synthesize("hello", speech.voice(settings, "voice-a"), out)
    assert wav_info(out).frames > 0 and fs.calls == [("hello", "voice-a")]
    assert os.path.getsize(out) > 44
