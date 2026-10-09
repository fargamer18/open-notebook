"""Audio Overview vocabulary shared by generation, storage, speech and web.

Durations are targets: the script is sized by word count at an assumed
speaking rate, and only synthesized audio has a measured duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from litenotebook.study.model import EvidenceItem

FORMATS = ("two_person", "narrator", "revision", "deep_dive")
FORMAT_LABELS = {
    "two_person": "Two-person discussion",
    "narrator": "Single narrator",
    "revision": "Study revision",
    "deep_dive": "Deep dive",
}
FORMAT_HELP = {
    "two_person": "Two hosts: one explains, the other asks clarifying questions, "
    "challenges and summarises.",
    "narrator": "One narrator, concise explanation.",
    "revision": "One narrator, revision-oriented: definitions, relationships and key "
    "points.",
    "deep_dive": "Two hosts, longer and more detailed explanations.",
}
TWO_SPEAKER_FORMATS = frozenset({"two_person", "deep_dive"})
DURATIONS = (5, 10, 20)  # target minutes
DEFAULT_WPM = 150  # assumed speaking rate (configurable per overview)
SPEAKER_LABELS = {"A": "Host A", "B": "Host B", "narrator": "Narrator"}

# Script shape per target length (sections) and per-section word budget.
SECTIONS_FOR = {5: 3, 10: 5, 20: 8}
MAX_TURN_CHARS = 900  # one spoken turn; longer is a "giant turn"
MAX_TITLE_CHARS = 200
MAX_FOCUS_CHARS = 500

INTERRUPTED = (
    "Audio overview generation was interrupted (LiteNotebook was closed). "
    "Generate again."
)
SPEECH_INTERRUPTED = (
    "Speech synthesis was interrupted (LiteNotebook was closed). The transcript is "
    "kept; generate the audio again."
)


def speakers_for(fmt: str) -> tuple[str, ...]:
    return ("A", "B") if fmt in TWO_SPEAKER_FORMATS else ("narrator",)


def target_words(minutes: int, wpm: int = DEFAULT_WPM) -> int:
    return minutes * wpm


def estimate_seconds(words: int, wpm: int = DEFAULT_WPM) -> int:
    return round(words * 60 / max(1, wpm))


@dataclass
class OutlineSection:
    title: str
    points: list[str]
    refs: list[str]


@dataclass
class ScriptTurn:
    speaker: str
    text: str
    refs: list[str] = field(default_factory=list)


@dataclass
class ScriptSection:
    title: str
    turns: list[ScriptTurn]

    @property
    def refs(self) -> list[str]:
        out: list[str] = []
        for t in self.turns:
            for r in t.refs:
                if r not in out:
                    out.append(r)
        return out

    @property
    def words(self) -> int:
        return sum(len(t.text.split()) for t in self.turns)


@dataclass
class Script:
    title: str
    sections: list[ScriptSection]
    evidence: dict[str, EvidenceItem]  # label → snapshot (only labels used)

    @property
    def words(self) -> int:
        return sum(s.words for s in self.sections)
