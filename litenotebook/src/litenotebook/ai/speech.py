"""Speech: the optional third Local AI component (Phase 14).

The engine is sherpa-onnx's ``sherpa-onnx-offline-tts`` program, a pinned
release archive downloaded only when the user chooses to install speech. It
runs as an owned child process per synthesis (no shell, exact handles, no
speech ML stack in the web process). Voices are Piper voices (VITS / ONNX)
from the pinned speech catalog, each downloaded separately on request.

Layout under the data folder::

    speech/engine/<engine id>/      extracted engine + litenotebook-speech.json
    speech/voices/<voice id>/       extracted voice  + litenotebook-speech.json
    speech/tests/<voice id>.wav     the last "Test voice" sample

Install lifecycle (engine and voices alike)::

    disk preflight → download (.part, resumable, SHA-256 verified)
    → safe extraction into speech/.staging-<id> → expected files present
    → (engine) it starts and prints its usage → rename into place
    → metadata written last (an install without metadata does not exist)

Status pages read only files: nothing is started to show them.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from litenotebook.ai.catalog import (
    SpeechCatalog,
    SpeechEngineEntry,
    VoiceEntry,
    load_speech,
)
from litenotebook.ai.download import Downloader, Progress, check_space
from litenotebook.ai.extract import extract
from litenotebook.config import Settings
from litenotebook.platform_utils.filesystem import is_within, remove_tree, rename_dir
from litenotebook.platform_utils.processes import OwnedProcess, spawn, stop
from litenotebook.platform_utils.system import (
    machine_arch,
    os_name,
    with_library_dir,
)

METADATA = "litenotebook-speech.json"
PROBE_TIMEOUT_S = 20.0
MAX_TEXT_CHARS = 1200  # per synthesis call; longer turns are split by the caller
TEST_SENTENCE = (
    "This is a short test of local speech in LiteNotebook. Nothing leaves this "
    "computer."
)


class SpeechError(RuntimeError):
    """User-facing message (no paths, no tracebacks)."""


class SpeechCancelled(SpeechError):
    pass


@dataclass
class InstalledEngine:
    id: str
    version: str
    executable: str
    library_dir: str
    installed_at: float
    license: str


@dataclass
class InstalledVoice:
    id: str
    display_name: str
    model: str
    tokens: str
    data_dir: str
    sample_rate: int
    license: str
    installed_at: float


def speech_root(settings: Settings) -> Path:
    return settings.data_dir / "speech"


def downloads_dir(settings: Settings) -> Path:
    return settings.data_dir / "tmp" / "downloads"


def catalog() -> SpeechCatalog:
    return load_speech()


def engine_entry(cat: SpeechCatalog | None = None) -> SpeechEngineEntry | None:
    """The catalog engine for this computer (None: no build for this OS/CPU)."""
    return (cat or catalog()).engine_for(os_name(), machine_arch())


# --- reading what is installed (files only) ---------------------------------------


def _meta(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((path / METADATA).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def installed_engine(settings: Settings) -> InstalledEngine | None:
    root = speech_root(settings) / "engine"
    if not root.is_dir():
        return None
    for folder in sorted(root.iterdir()):
        data = _meta(folder)
        if data is None:
            continue
        exe = folder / str(data.get("executable", ""))
        lib = folder / str(data.get("library_dir", ""))
        if (
            not exe.is_file()
            or not is_within(exe, folder)
            or not is_within(lib, folder)
        ):
            continue
        return InstalledEngine(
            id=str(data["id"]),
            version=str(data.get("version", "")),
            executable=str(exe),
            library_dir=str(lib),
            installed_at=float(data.get("installed_at", 0)),
            license=str(data.get("license", "")),
        )
    return None


def installed_voices(settings: Settings) -> list[InstalledVoice]:
    root = speech_root(settings) / "voices"
    out: list[InstalledVoice] = []
    if not root.is_dir():
        return out
    for folder in sorted(root.iterdir()):
        data = _meta(folder)
        if data is None:
            continue
        files = {
            k: folder / str(data.get(k, "")) for k in ("model", "tokens", "data_dir")
        }
        if not all(is_within(p, folder) for p in files.values()):
            continue
        if not (files["model"].is_file() and files["tokens"].is_file()):
            continue
        out.append(
            InstalledVoice(
                id=str(data["id"]),
                display_name=str(data.get("display_name", data["id"])),
                model=str(files["model"]),
                tokens=str(files["tokens"]),
                data_dir=str(files["data_dir"]),
                sample_rate=int(data.get("sample_rate", 22050)),
                license=str(data.get("license", "")),
                installed_at=float(data.get("installed_at", 0)),
            )
        )
    return out


def voice(settings: Settings, voice_id: str) -> InstalledVoice | None:
    return next((v for v in installed_voices(settings) if v.id == voice_id), None)


def status(settings: Settings) -> dict[str, Any]:
    """For Settings and Study pages. Reads files only; starts nothing."""
    cat = catalog()
    engine = installed_engine(settings)
    voices = installed_voices(settings)
    entry = engine_entry(cat)
    have = {v.id for v in voices}
    return {
        "installed": engine is not None,
        "ready": engine is not None and bool(voices),
        "engine": asdict(engine) if engine else None,
        "engine_entry": entry,
        "voices": [asdict(v) for v in voices],
        "voice_ids": sorted(have),
        "available_voices": [v for v in cat.voices if v.id not in have],
        "catalog_voices": list(cat.voices),
        "default_voices": cat.default_voices,
        "supported": entry is not None,
        "note": cat.note,
    }


# --- installing -----------------------------------------------------------------------


def _download_and_extract(
    settings: Settings,
    assets: tuple[Any, ...],
    staging: Path,
    downloader: Downloader,
    say: Callable[[str, Progress | None], None],
    cancel: threading.Event | None,
) -> None:
    tmp = downloads_dir(settings)
    fetched: list[tuple[Path, str]] = []
    try:
        for asset in assets:
            name = asset.url.rsplit("/", 1)[1]
            label = f"Downloading {name}"
            say(label, None)

            def report(p: Progress, label: str = label) -> None:
                say(label, p)

            result = downloader.fetch(
                asset.url,
                tmp / name,
                sha256=asset.sha256,
                size=asset.size,
                progress=report,
                cancel=cancel,
            )
            fetched.append((result.path, asset.format))
        say("Extracting…", None)
        for archive, fmt in fetched:
            extract(archive, fmt, staging)
    finally:
        for archive, _ in fetched:
            with contextlib.suppress(OSError):
                archive.unlink(missing_ok=True)


def _swap_into_place(staging: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    old = None
    if final.exists():  # repair: keep the old copy until the new one is in place
        old = final.parent / f".old-{uuid.uuid4().hex[:12]}"
        rename_dir(final, old)
    try:
        rename_dir(staging, final)
    except BaseException:
        if old is not None:
            rename_dir(old, final)
        raise
    if old is not None:
        with contextlib.suppress(OSError):
            remove_tree(old)


def probe_engine(executable: Path, library_dir: Path) -> str | None:
    """None if the engine starts and prints its usage, else why not."""
    try:
        proc = spawn(
            [str(executable), "--help"],
            env=with_library_dir(os.environ, library_dir),
        )
    except OSError as e:
        return f"It could not be started ({e.strerror or e})"
    try:
        out, err = proc.proc.communicate(timeout=PROBE_TIMEOUT_S)
    except Exception:
        stop(proc, grace_s=1.0)
        return "It did not answer in time"
    text = (out + err).decode("utf-8", "replace").lower()
    if "vits" not in text and "offline-tts" not in text:
        return "It does not look like the sherpa-onnx speech program"
    return None


def install_engine(
    settings: Settings,
    entry: SpeechEngineEntry,
    *,
    downloader: Downloader,
    progress: Callable[[str, Progress | None], None] | None = None,
    cancel: threading.Event | None = None,
    check_runs: bool = True,
) -> InstalledEngine:
    def say(stage: str, value: Progress | None = None) -> None:
        if progress is not None:
            progress(stage, value)

    if entry.os != os_name():
        raise SpeechError(f"{entry.id} is for {entry.os}, not this computer")
    root = speech_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    check_space(root, entry.download_size * 4)
    staging = root / f".staging-{uuid.uuid4().hex[:12]}"
    try:
        _download_and_extract(settings, entry.assets, staging, downloader, say, cancel)
        exe, lib = staging / entry.executable, staging / entry.library_dir
        if not exe.is_file() or not is_within(exe, staging):
            raise SpeechError(
                "The speech engine archive does not contain the expected program; "
                "nothing was installed"
            )
        if os_name() != "windows":
            exe.chmod(exe.stat().st_mode | 0o111)
        if check_runs:
            say("Checking the speech engine…", None)
            problem = probe_engine(exe, lib)
            if problem:
                raise SpeechError(f"The speech engine does not run here: {problem}")
        (staging / METADATA).write_text(
            json.dumps(
                {
                    "id": entry.id,
                    "version": entry.version,
                    "executable": entry.executable,
                    "library_dir": entry.library_dir,
                    "installed_at": time.time(),
                    "license": entry.license,
                    "assets": [a.sha256 for a in entry.assets],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        engine_dir = root / "engine"
        for other in engine_dir.iterdir() if engine_dir.is_dir() else []:
            if other.name != entry.id and _meta(other) is not None:
                with contextlib.suppress(OSError):
                    remove_tree(other)  # one engine at a time
        _swap_into_place(staging, engine_dir / entry.id)
    except BaseException:
        with contextlib.suppress(OSError):
            remove_tree(staging)
        raise
    say("Installed", None)
    engine = installed_engine(settings)
    if engine is None:
        raise SpeechError("The speech engine was installed but could not be found")
    return engine


def _find_voice_files(staging: Path) -> tuple[Path, Path, Path]:
    models = [p for p in staging.rglob("*.onnx") if p.is_file()]
    tokens = [p for p in staging.rglob("tokens.txt") if p.is_file()]
    data = [p for p in staging.rglob("espeak-ng-data") if p.is_dir()]
    if len(models) != 1 or len(tokens) != 1 or len(data) != 1:
        raise SpeechError(
            "The voice archive does not have the expected model, tokens and "
            "pronunciation data; nothing was installed"
        )
    return models[0], tokens[0], data[0]


def install_voice(
    settings: Settings,
    entry: VoiceEntry,
    *,
    downloader: Downloader,
    progress: Callable[[str, Progress | None], None] | None = None,
    cancel: threading.Event | None = None,
) -> InstalledVoice:
    def say(stage: str, value: Progress | None = None) -> None:
        if progress is not None:
            progress(stage, value)

    root = speech_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    check_space(root, entry.asset.size * 4)
    staging = root / f".staging-{uuid.uuid4().hex[:12]}"
    try:
        _download_and_extract(
            settings, (entry.asset,), staging, downloader, say, cancel
        )
        model, tokens, data = _find_voice_files(staging)
        (staging / METADATA).write_text(
            json.dumps(
                {
                    "id": entry.id,
                    "display_name": entry.display_name,
                    "model": model.relative_to(staging).as_posix(),
                    "tokens": tokens.relative_to(staging).as_posix(),
                    "data_dir": data.relative_to(staging).as_posix(),
                    "sample_rate": entry.sample_rate,
                    "license": entry.license,
                    "installed_at": time.time(),
                    "sha256": entry.asset.sha256,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        _swap_into_place(staging, root / "voices" / entry.id)
    except BaseException:
        with contextlib.suppress(OSError):
            remove_tree(staging)
        raise
    say("Installed", None)
    found = voice(settings, entry.id)
    if found is None:
        raise SpeechError("The voice was installed but could not be found")
    return found


def remove_voice(settings: Settings, voice_id: str) -> bool:
    """Delete one installed voice (only inside LiteNotebook's speech folder)."""
    root = speech_root(settings) / "voices"
    folder = root / voice_id
    if not is_within(folder, root) or folder == root or _meta(folder) is None:
        return False
    remove_tree(folder)
    return True


def remove_engine(settings: Settings) -> bool:
    root = speech_root(settings) / "engine"
    removed = False
    for folder in root.iterdir() if root.is_dir() else []:
        if _meta(folder) is not None and is_within(folder, root):
            remove_tree(folder)
            removed = True
    return removed


# --- synthesis ------------------------------------------------------------------------


class SpeechProvider(Protocol):
    """Turns one piece of text into one WAV file with one voice."""

    name: str

    def synthesize(
        self,
        text: str,
        voice: InstalledVoice,
        out: Path,
        cancel: threading.Event | None = None,
    ) -> None: ...


class SherpaProvider:
    """``sherpa-onnx-offline-tts`` as an owned child, one call per text piece.
    The text is a single argument (no shell); the child is stopped by its
    exact handle on cancel or timeout."""

    name = "sherpa-onnx"

    def __init__(self, engine: InstalledEngine, threads: int = 2) -> None:
        self.engine = engine
        self.threads = max(1, threads)
        self.last_pid: int | None = None

    def synthesize(
        self,
        text: str,
        voice: InstalledVoice,
        out: Path,
        cancel: threading.Event | None = None,
    ) -> None:
        text = " ".join(text.split()).lstrip("-–— ")  # never parsed as a flag
        if not text:
            raise SpeechError("Nothing to say")
        if len(text) > MAX_TEXT_CHARS:
            raise SpeechError("Text piece too long for one synthesis call")
        argv = [
            self.engine.executable,
            f"--vits-model={voice.model}",
            f"--vits-tokens={voice.tokens}",
            f"--vits-data-dir={voice.data_dir}",
            f"--num-threads={self.threads}",
            f"--output-filename={out}",
            text,
        ]
        env = with_library_dir(os.environ, Path(self.engine.library_dir))
        try:
            proc = spawn(argv, env=env, pipes=True)
        except OSError as e:
            raise SpeechError(
                f"The speech engine could not be started ({e.strerror or e})"
            ) from e
        self.last_pid = proc.pid
        _wait(proc, cancel, timeout_s=max(30.0, len(text) / 5))
        if proc.proc.returncode != 0 or not _is_wav(out):
            raise SpeechError(
                f"The speech engine failed (exit code {proc.proc.returncode})"
            )


def _wait(proc: OwnedProcess, cancel: threading.Event | None, timeout_s: float) -> None:
    """Wait for the child, draining its output; stop it on cancel/timeout."""
    sink: list[bytes] = []

    def drain(stream: Any) -> None:
        with contextlib.suppress(OSError, ValueError):
            for chunk in iter(lambda: stream.read(65536), b""):
                if sum(len(c) for c in sink) < 65536:
                    sink.append(chunk)

    threads = [
        threading.Thread(target=drain, args=(s,), daemon=True)
        for s in (proc.stdout, proc.stderr)
        if s is not None
    ]
    for t in threads:
        t.start()
    if proc.stdin is not None:
        with contextlib.suppress(OSError):
            proc.stdin.close()
    deadline = time.monotonic() + timeout_s
    while proc.proc.poll() is None:
        if cancel is not None and cancel.is_set():
            stop(proc, grace_s=2.0)
            raise SpeechCancelled("Cancelled")
        if time.monotonic() > deadline:
            stop(proc, grace_s=2.0)
            raise SpeechError("The speech engine did not finish in time")
        time.sleep(0.05)
    stop(proc, grace_s=1.0)  # reaps, closes pipes
    for t in threads:
        t.join(timeout=1.0)


def _is_wav(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return len(head) == 12 and head[:4] == b"RIFF" and head[8:] == b"WAVE"


def provider_for(settings: Settings) -> SpeechProvider | None:
    engine = installed_engine(settings)
    return SherpaProvider(engine) if engine else None


def test_voice(
    settings: Settings, provider: SpeechProvider, voice_id: str
) -> dict[str, Any]:
    """Synthesize the fixed test sentence (no Qwen involved)."""
    found = voice(settings, voice_id)
    if found is None:
        raise SpeechError("That voice is not installed")
    folder = speech_root(settings) / "tests"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{found.id}.wav"
    tmp = folder / f".{uuid.uuid4().hex[:8]}.wav"
    started = time.perf_counter()
    try:
        provider.synthesize(TEST_SENTENCE, found, tmp)
        shutil.move(str(tmp), str(out))
    finally:
        tmp.unlink(missing_ok=True)
    from litenotebook.study.audio_render import wav_info

    info = wav_info(out)
    seconds = round(time.perf_counter() - started, 3)
    return {
        "voice": found.id,
        "seconds": seconds,
        "duration_s": info.duration_s,
        "sample_rate": info.sample_rate,
        "realtime_factor": round(info.duration_s / seconds, 1) if seconds else None,
    }
