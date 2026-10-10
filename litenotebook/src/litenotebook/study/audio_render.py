"""Turn a validated transcript into one WAV file with measured chapters.

Pipeline (one speech job at a time, in a background thread)::

    turns → citation markers and markup stripped → split into sentences
    → one engine call per piece (owned child process, see ai.speech)
    → mono 16-bit PCM, resampled to one rate if voices differ, edge
      silence trimmed → written as it goes (short pauses between turns,
      longer between sections) → duration measured from the file
    → chapter and turn timestamps from the actual sample counts

Output is WAV: playable by every browser and OS without an encoder or
ffmpeg (about 2.6 MiB per minute at 22,050 Hz). The engine only ever sees
validated script text — never source PDFs. A failure or cancel leaves the
transcript untouched and removes the partial file.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from litenotebook.db import connect
from litenotebook.platform_utils.filesystem import remove_tree, replace_file
from litenotebook.study import audio_store as store

if TYPE_CHECKING:
    from litenotebook.ai.speech import InstalledVoice, SpeechProvider
    from litenotebook.config import Settings

PCM = npt.NDArray[np.int16]
PIECE_CHARS = 400
GAP_PIECE_S = 0.12
GAP_TURN_S = 0.35
GAP_SECTION_S = 0.8
SILENCE_LEVEL = 300  # |sample| below this counts as silence at the edges
EDGE_PAD_S = 0.03

_LABELS = re.compile(r"\[\s*[SNM]\s*\d+(?:\s*[,;]\s*[SNM]?\s*\d+)*\s*\]", re.I)
_MARKUP = re.compile(r"[*_#`<>\[\]{}|~^]+")
_SENTENCES = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int
    frames: int

    @property
    def duration_s(self) -> float:
        return round(self.frames / self.sample_rate, 3) if self.sample_rate else 0.0


def wav_info(path: Path) -> WavInfo:
    with wave.open(str(path), "rb") as w:
        return WavInfo(
            w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        )


def read_pcm(path: Path) -> tuple[PCM, int]:
    """Mono int16 samples and the sample rate (8/16/32-bit PCM accepted)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    elif width == 4:
        data = (np.frombuffer(raw, dtype="<i4") >> 16).astype(np.int32)
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128) << 8
    else:
        raise ValueError(f"unsupported sample width {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int32)
    return np.clip(data, -32768, 32767).astype(np.int16), rate


def resample(samples: PCM, rate_from: int, rate_to: int) -> PCM:
    if rate_from == rate_to or samples.size == 0:
        return samples
    n = max(1, round(samples.size * rate_to / rate_from))
    x = np.linspace(0, samples.size - 1, n)
    return np.interp(x, np.arange(samples.size), samples).astype(np.int16)


def trim_silence(samples: PCM, rate: int) -> PCM:
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > SILENCE_LEVEL)
    if loud.size == 0:
        return samples[:0]
    pad = int(EDGE_PAD_S * rate)
    return samples[max(0, loud[0] - pad) : min(samples.size, loud[-1] + pad + 1)]


def speech_text(text: str) -> str:
    """What the engine reads: no citation markers ("[S1]" is never spoken),
    no markup, normal spacing, a closing punctuation mark."""
    text = _LABELS.sub(" ", text)
    text = _MARKUP.sub(" ", text)
    text = re.sub(r"\s+([.,!?;:…])", r"\1", " ".join(text.split()))
    if text and text[-1] not in ".!?…":
        text += "."
    return text


def split_for_speech(text: str, limit: int = PIECE_CHARS) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCES.split(text):
        while len(sentence) > limit:  # an overlong sentence: cut at a space
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


@dataclass
class SynthJob:
    settings: Settings
    notebook_id: int
    overview_id: int
    asset_id: int
    voices: dict[str, InstalledVoice]  # speaker → voice
    provider: SpeechProvider
    release: Any = None  # callable run when the job ends (frees the speech slot)
    cancel: threading.Event = field(default_factory=threading.Event)
    timings: dict[str, float] = field(default_factory=dict)


def run_synthesis(job: SynthJob) -> None:
    from litenotebook.ai.speech import SpeechCancelled, SpeechError

    conn = connect(job.settings.db_path)
    out = store.asset_path(job.settings, job.notebook_id, job.overview_id, job.asset_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.stem}.{uuid.uuid4().hex[:8]}.partial")
    scratch = out.parent / f".tmp-{job.asset_id}-{uuid.uuid4().hex[:8]}"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine_s = 0.0
    try:
        sections = store.sections(conn, job.overview_id)
        turns = [t for s in sections for t in s.turns]
        rate = max(v.sample_rate for v in job.voices.values())
        chapters: list[dict[str, Any]] = []
        turn_times: list[list[float]] = []
        written = 0
        done = 0
        with wave.open(str(partial), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)

            def silence(seconds: float) -> None:
                nonlocal written
                n = int(seconds * rate)
                w.writeframes(np.zeros(n, dtype="<i2").tobytes())
                written += n

            for s_index, section in enumerate(sections):
                if s_index:
                    silence(GAP_SECTION_S)
                chapters.append(
                    {
                        "section": section.position,
                        "title": section.title,
                        "start_s": round(written / rate, 3),
                    }
                )
                for t_index, turn in enumerate(section.turns):
                    if job.cancel.is_set():
                        raise SpeechCancelled("Cancelled")
                    store.set_synth_stage(
                        conn, job.overview_id, f"speaking:{done + 1}:{len(turns)}"
                    )
                    if t_index:
                        silence(GAP_TURN_S)
                    voice = job.voices[turn.speaker]
                    start = written
                    for p_index, piece in enumerate(
                        split_for_speech(speech_text(turn.text))
                    ):
                        target = scratch / f"{turn.id}-{p_index}.wav"
                        t0 = time.perf_counter()
                        job.provider.synthesize(piece, voice, target, job.cancel)
                        engine_s += time.perf_counter() - t0
                        samples, got = read_pcm(target)
                        samples = trim_silence(resample(samples, got, rate), rate)
                        target.unlink(missing_ok=True)
                        if p_index:
                            silence(GAP_PIECE_S)
                        w.writeframes(samples.astype("<i2").tobytes())
                        written += samples.size
                    turn_times.append(
                        [turn.id, round(start / rate, 3), round(written / rate, 3)]
                    )
                    done += 1
        info = wav_info(partial)  # measured from the file, not estimated
        digest = hashlib.sha256()
        with open(partial, "rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
        size = partial.stat().st_size
        replace_file(partial, out)
        total = round(time.perf_counter() - started, 3)
        store.finish_asset(
            conn,
            job.overview_id,
            job.asset_id,
            sample_rate=info.sample_rate,
            duration_s=info.duration_s,
            size_bytes=size,
            sha256=digest.hexdigest(),
            chapters=chapters,
            turn_times=turn_times,
            meta={
                "synthesis_s": total,
                "engine_s": round(engine_s, 3),
                "turns": len(turns),
                "realtime_factor": round(info.duration_s / total, 2) if total else None,
            },
        )
    except SpeechCancelled:
        store.end_asset(
            conn,
            job.overview_id,
            job.asset_id,
            status="cancelled",
            error="Cancelled. The transcript is kept.",
        )
    except (SpeechError, OSError, ValueError, wave.Error) as e:
        from litenotebook.platform_utils.filesystem import is_disk_full

        if isinstance(e, SpeechError):
            message = str(e)
        elif is_disk_full(e):
            message = "Not enough disk space to write the audio"
        else:
            message = "The audio could not be written"
        store.end_asset(
            conn,
            job.overview_id,
            job.asset_id,
            status="failed",
            error=f"{message}. The transcript is kept.",
        )
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Speech synthesis failed")
        store.end_asset(
            conn,
            job.overview_id,
            job.asset_id,
            status="failed",
            error="Speech synthesis failed; details are in the server log. "
            "The transcript is kept.",
        )
    finally:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            remove_tree(scratch)
        conn.close()
        if job.release is not None:
            job.release()


def transcript_text(
    title: str, sections: list[store.Section], evidence: dict[str, store.EvidenceRef]
) -> str:
    """Plain-text transcript with citation labels and an evidence list."""
    lines = [title, "=" * min(len(title), 80), ""]
    for s in sections:
        lines += [f"## {s.position}. {s.title}", ""]
        for t in s.turns:
            cites = "".join(f" [{r}]" for r in t.refs)
            lines.append(f"{t.speaker_label}: {t.text}{cites}")
        lines.append("")
    if evidence:
        lines += ["Evidence", "--------"]
        for e in evidence.values():
            where = f"{e.title}, page {e.page_no}" if e.kind == "source" else e.title
            lines.append(f"[{e.label}] {where}: {' '.join(e.excerpt.split())}")
    return "\n".join(lines).rstrip() + "\n"


def safe_filename(title: str, suffix: str) -> str:
    """A download name that is valid on Windows, macOS and Linux."""
    base = re.sub(r"[^\w\- ]+", "", title, flags=re.UNICODE).strip()
    base = re.sub(r"\s+", " ", base)[:60].strip(" .") or "audio-overview"
    if base.upper().split(".")[0] in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        base = f"_{base}"
    return f"{base}{suffix}"


def chapters_json(asset: store.Asset) -> str:
    return json.dumps(asset.chapters)
