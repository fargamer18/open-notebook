"""Quiz vocabulary shared by generation, storage and the web layer.

No heavy imports: rendering Study pages uses only this, SQLite and Jinja.
"""

from __future__ import annotations

from dataclasses import dataclass, field

QTYPES = ("single", "multiple", "true_false")
QTYPE_LABELS = {
    "single": "Single answer",
    "multiple": "Multiple answers",
    "true_false": "True / False",
}
DIFFICULTIES = ("easy", "medium", "hard", "exam")
DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
    "exam": "Exam-style",
}
# What each level asks of the model (stated in the prompt). Difficulty is a
# request to the model, not a psychometric calibration.
DIFFICULTY_RULES = {
    "easy": "Easy: terminology, direct relationships and basic comprehension of "
    "facts stated in the evidence.",
    "medium": "Medium: relationships between ideas, interpretation and moderate "
    "reasoning that the evidence supports.",
    "hard": "Hard: multi-step conceptual reasoning that combines facts stated in "
    "the evidence. Do not test exact wording.",
    "exam": "Exam-style: concise wording, application of the stated ideas where the "
    "evidence supports it, plausible distractors and short hints that do not "
    "lead to the answer.",
}
COUNTS = (5, 10, 15, 20)
LETTERS = "ABCD"
TRUE_FALSE = ("True", "False")
KINDS = ("source", "note", "chat")
KIND_PREFIX = {"source": "S", "note": "N", "chat": "M"}
KIND_LABELS = {"source": "Source", "note": "Note", "chat": "Chat"}

MAX_FOCUS_CHARS = 500
MAX_TITLE_CHARS = 200
MAX_QUESTION_CHARS = 400
MAX_OPTION_CHARS = 200
MAX_EXPLANATION_CHARS = 800
MAX_HINT_CHARS = 300
MAX_CONCEPTS = 3
MAX_CONCEPT_CHARS = 60
MAX_EVIDENCE_PER_QUESTION = 3
EXCERPT_CHARS = 1500  # stored evidence snapshot per item

INTERRUPTED = (
    "Quiz generation was interrupted (LiteNotebook was closed). Generate again."
)


@dataclass(frozen=True)
class EvidenceItem:
    """One labelled passage offered to the model (labels are per batch)."""

    label: str  # "S1", "N2", "M1" …
    kind: str  # source | note | chat
    text: str
    title: str  # file name / note title / conversation title
    source_id: int | None = None
    chunk_id: int | None = None
    page_no: int | None = None
    chunk_ord: int | None = None
    note_id: int | None = None
    message_id: int | None = None
    conversation_id: int | None = None


@dataclass
class Concept:
    key: str  # normalized identity, e.g. "row echelon form"
    label: str  # display text


@dataclass
class QuizQuestion:
    """A validated question, ready to save. ``correct`` holds 0-based option
    indexes; ``evidence`` is the supporting snapshot (at least one)."""

    qtype: str
    question: str
    options: list[str]
    correct: list[int]
    explanation: str
    hint: str
    evidence: list[EvidenceItem]
    concepts: list[Concept] = field(default_factory=list)
    difficulty: str = "medium"


@dataclass
class Material:
    """What the user selected for one quiz."""

    source_ids: list[int]
    include_notes: bool = False
    include_chat: bool = False
    focus: str = ""
