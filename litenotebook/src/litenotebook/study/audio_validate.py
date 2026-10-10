"""Server-side validation of model-written outlines and audio scripts.

Citations and source boundaries are strict: every cited label must be one
that was offered, from selected material. Grounding is deterministic and
deliberately less lexical than quiz answers (conversation paraphrases), but
a turn is removed when it

* names something the evidence never mentions ("the Black Sea"),
* contains a number the cited evidence does not (digits or 0–20 as words),
* is a declarative claim most of whose content words are not in the evidence
  ("chlorophyll absorbs light for photosynthesis" when the evidence only
  names chlorophyll),
* repeats or acts on instruction-like text from a document, advertises, or
  invents personal experiences,
* leaks markup, stage directions, option letters or model chatter.

Removing a turn never rewrites another; a section left without cited turns
is dropped. The generator reports what was removed and saves a shorter
script rather than an unsupported one.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from litenotebook.memory.concepts import meaningful, terms
from litenotebook.study.audio_model import (
    MAX_TURN_CHARS,
    OutlineSection,
    ScriptSection,
    ScriptTurn,
    speakers_for,
)
from litenotebook.study.model import EvidenceItem
from litenotebook.study.validate import (
    _INJECTION,
    _LETTER_REF,
    NUMBER_WORDS,
    QUIZ_STOPWORDS,
    STEM_CHARS,
    clean_text,
    has_html,
    number_values,
    split_sentences,
    support_text,
)

_LABEL_IN_TEXT = re.compile(r"\[\s*([SNM])\s*(\d+)\s*\]", re.IGNORECASE)
_ANY_LABEL = re.compile(
    r"\[\s*[SNM]\s*\d+(?:\s*[,;]\s*[SNM]?\s*\d+)*\s*\]", re.IGNORECASE
)
_SPEAKER_PREFIX = re.compile(
    r"^\s*(?:\*\*)?(?:host\s*[ab]|speaker\s*[ab12]|narrator|[ab])\s*(?:\*\*)?\s*:\s*",
    re.IGNORECASE,
)
_STAGE = re.compile(
    r"\[(?!\s*[SNM]\s*\d)[^\]]{0,40}\]|\((?:laughs?|laughing|pause[sd]?|music|sighs?|"
    r"chuckles?|intro|outro|applause|sound[^)]*|sfx[^)]*)\)|\*[^*]{1,40}\*",
    re.IGNORECASE,
)
_MARKDOWN = re.compile(r"[#`_]+|\*\*")
_CHATTER = re.compile(
    r"\bas an ai\b|\blanguage model\b|\bhere(?:'s| is) (?:the|your|a) (?:script|"
    r"outline|json)\b|\bjson\b|\bevidence label|\bthe labels?\b",
    re.IGNORECASE,
)
_ANECDOTE = re.compile(
    r"\b(?:i|we)\s+(?:remember|recall|used to|once|personally|grew up|studied|"
    r"learned this|tried this|visited)\b|\bwhen i was\b|\bin my (?:experience|"
    r"college|class|school|job|career|life)\b|\bmy (?:professor|teacher|students?|"
    r"colleagues?|friends?|family|kids?)\b",
    re.IGNORECASE,
)
_ADVERT = re.compile(
    r"\badvertis\w*|\bsponsor\w*|\bpromo(?:tion|tional)?\s*code|\bdiscount\b|"
    r"\bbuy (?:now|today|some)\b|\bour partner|\bbrought to you by\b|\bsubscribe\b",
    re.IGNORECASE,
)
_SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+|[\"“(]\s*)$")
_CAPITAL = re.compile(r"\b[A-Z][a-zA-Z'’-]{1,}\b")
_ACKS = frozenset(
    "yes yeah right exactly sure okay ok indeed absolutely true correct "
    "great interesting wow mm hmm got it makes sense i see".split()
)
# Words that carry conversation, not facts (on top of the concept stopwords).
CONVERSATIONAL = frozenset(
    """
    let lets unpack matters matter important interesting key distinction idea
    ideas point points means mean meaning basically essentially understand
    think look talk explain explains explained example sense right exactly
    great question questions simply words summary summarise summarize recap
    today episode discuss discussion cover covered topic topics overview welcome
    start begin begins next finally also really part whole thing things actually
    sure good okay clear picture step steps bit quick worth noting note notes
    remember helps help useful together means first second third last whats
    hosts host listeners listen dive deeper deep detail details detailed
    breakdown break down unpacking walk through look looking closely main core
    essential essentials basics fundamental fundamentals clarify clarifying
    connect connects connection relationship relationships overall short brief
    briefly quickly wrap wrapping takeaway takeaways review revising revision
    exam focus focuses focusing material sources source document documents
    passage passages evidence mentions mentioned describe described describes
    say says said tell tells told called name named known
    """.split()
)
NAME_ALLOW = frozenset(
    """
    I So And But Or Now Then Well Yes No Okay OK Right Exactly Let Let's Here
    There This That These Those It It's What Why How When Where Which Who The A
    An In On At For From With Without As Also Finally First Second Third Next
    Today Host Narrator Welcome Great Good Sure Absolutely Indeed Think Imagine
    Remember Notice Consider Together Overall Thanks Thank Hello Hi Mm Hmm
    """.split()
)
MIN_CLAIM_STEMS = 4
MAX_NOVEL_SHARE = 0.34
MAX_ACK_WORDS = 3


def _stems(text: str) -> set[str]:
    return {
        t[:STEM_CHARS]
        for t in terms(text)
        if meaningful(t)
        and t not in QUIZ_STOPWORDS
        and t not in CONVERSATIONAL
        and t not in NUMBER_WORDS
    }


def _labels(raw: Any) -> list[str]:
    out: list[str] = []
    for ref in raw if isinstance(raw, list) else [raw]:
        if isinstance(ref, str):
            label = ref.strip().strip("[]").replace(" ", "").upper()
            if label and label not in out:
                out.append(label)
    return out


def _words(text: str) -> set[str]:
    return {w.casefold() for w in re.findall(r"[^\W_]+", text)}


class Evidence:
    """The offered package, precomputed for the checks."""

    def __init__(
        self, package: Mapping[str, EvidenceItem], allowed_sources: Collection[int]
    ) -> None:
        self.package = dict(package)
        self.allowed_sources = set(allowed_sources)
        items = list(package.values())
        self.support = support_text(items)
        self.stems = _stems(self.support)
        self.words = _words(self.support)
        self.numbers = number_values(self.support)
        injected = " ".join(
            s
            for item in items
            for s in split_sentences(item.text)
            if _INJECTION.search(s)
        )
        # words that appear only in instruction-like sentences of a document
        self.injected = _stems(injected) - self.stems

    def valid(self, label: str) -> bool:
        item = self.package.get(label)
        if item is None:
            return False
        return item.kind != "source" or item.source_id in self.allowed_sources


def validate_outline(
    raw: Any, evidence: Evidence, wanted: int
) -> tuple[str, list[OutlineSection], list[str]]:
    """(title, sections, problems). Sections without a valid citation are
    dropped; unknown labels are removed."""
    problems: list[str] = []
    data = raw if isinstance(raw, dict) else {}
    title = clean_text(data.get("title"))[:120]
    if has_html(title):
        title = ""
    sections: list[OutlineSection] = []
    for item in data.get("sections") or []:
        if not isinstance(item, dict):
            problems.append("malformed outline section")
            continue
        heading = clean_text(item.get("title"))[:100]
        points = [
            p for p in (clean_text(x)[:200] for x in item.get("points") or []) if p
        ]
        refs = [r for r in _labels(item.get("refs")) if evidence.valid(r)]
        if not heading or has_html(heading) or any(has_html(p) for p in points):
            problems.append("outline section with markup or no title")
            continue
        if not refs:
            problems.append("outline section without valid evidence")
            continue
        sections.append(OutlineSection(heading, points[:4] or [heading], refs[:4]))
        if len(sections) == wanted:
            break
    return title, sections, problems


def _clean_turn(text: str) -> str:
    text = clean_text(text)
    for _ in range(2):
        text = _SPEAKER_PREFIX.sub("", text)
    text = _STAGE.sub(" ", text)
    text = _MARKDOWN.sub("", text)
    text = _LETTER_REF.sub("", text)
    return " ".join(text.split())


def _unsupported_name(text: str, evidence: Evidence) -> str | None:
    for m in _CAPITAL.finditer(text):
        word = m.group(0)
        if word in NAME_ALLOW or _SENTENCE_START.search(text[: m.start()]):
            continue
        if word.casefold() not in evidence.words:
            return word
    return None


def check_turn(
    raw: Any, evidence: Evidence, speakers: Sequence[str]
) -> ScriptTurn | str:
    """A cleaned turn, or why it was removed."""
    if not isinstance(raw, dict):
        return "malformed turn"
    speaker = clean_text(raw.get("speaker"))
    if speaker.lower() == "narrator":
        speaker = "narrator"
    elif speaker.upper() in ("A", "B"):
        speaker = speaker.upper()
    if speaker not in speakers:
        return "speaker not allowed"
    original = clean_text(raw.get("text"))
    refs = _labels(raw.get("refs"))
    for kind, number in _LABEL_IN_TEXT.findall(original):  # labels in the text
        label = f"{kind.upper()}{number}"
        if label not in refs:
            refs.append(label)
    if any(r not in evidence.package for r in refs):
        return "unsupported citation"
    if not all(evidence.valid(r) for r in refs):
        return "evidence outside selected sources"
    if has_html(original):
        return "html content"
    text = _clean_turn(_ANY_LABEL.sub(" ", original))
    if not text or not any(ch.isalpha() for ch in text):
        return "empty turn"
    if len(text) > MAX_TURN_CHARS:
        return "giant turn"
    if _CHATTER.search(text):
        return "model chatter"
    if _ANECDOTE.search(text):
        return "invented personal experience"
    if _ADVERT.search(text):
        return "advertising"
    stems = _stems(text)
    if evidence.injected & stems:
        return "repeats instruction-like document text"
    if name := _unsupported_name(text, evidence):
        return f"unsupported name ({name[:30]})"
    cited = support_text([evidence.package[r] for r in refs]) if refs else ""
    have = number_values(cited) if cited else evidence.numbers
    if number_values(text) - have:
        return "unsupported number"
    question = text.rstrip().endswith("?")
    if not question and len(stems) >= MIN_CLAIM_STEMS:
        novel = stems - evidence.stems
        if len(novel) >= 2 and len(novel) / len(stems) > MAX_NOVEL_SHARE:
            return "not supported by the evidence"
    return ScriptTurn(speaker, text, refs)


def is_ack(turn: ScriptTurn) -> bool:
    words = re.findall(r"[a-z']+", turn.text.lower())
    return len(words) <= MAX_ACK_WORDS and all(w in _ACKS for w in words)


def validate_script(
    raw_sections: Sequence[Any],
    evidence: Evidence,
    fmt: str,
    titles: Sequence[str],
) -> tuple[list[ScriptSection], list[dict[str, str]]]:
    """(sections, removed). ``titles`` are the outline titles, used when the
    model's section title is missing or unusable."""
    speakers = speakers_for(fmt)
    removed: list[dict[str, str]] = []
    out: list[ScriptSection] = []
    seen_text: set[str] = set()
    for index, raw in enumerate(raw_sections):
        if not isinstance(raw, dict):
            removed.append({"reason": "malformed section", "text": ""})
            continue
        title = clean_text(raw.get("title"))[:100]
        fallback = titles[index] if index < len(titles) else f"Part {index + 1}"
        if not title or has_html(title) or _ANY_LABEL.search(title):
            title = fallback
        turns: list[ScriptTurn] = []
        for raw_turn in raw.get("turns") or []:
            result = check_turn(raw_turn, evidence, speakers)
            text = raw_turn.get("text") if isinstance(raw_turn, dict) else ""
            if isinstance(result, str):
                removed.append({"reason": result, "text": str(text or "")[:200]})
                continue
            key = " ".join(result.text.lower().split())
            if key in seen_text:
                removed.append({"reason": "repeated turn", "text": result.text[:200]})
                continue
            if len(speakers) == 2 and result.speaker == "B" and is_ack(result):
                removed.append({"reason": "empty acknowledgement", "text": result.text})
                continue
            seen_text.add(key)
            turns.append(result)
        if not turns:
            removed.append({"reason": "section left empty", "text": title})
            continue
        if not any(t.refs for t in turns):
            removed.append({"reason": "section without evidence", "text": title})
            continue
        out.append(ScriptSection(title, turns))
    return out, removed


def one_sided(sections: Sequence[ScriptSection], fmt: str) -> bool:
    """Two-host formats where one host never speaks (or only once)."""
    if len(speakers_for(fmt)) != 2:
        return False
    counts = {"A": 0, "B": 0}
    for s in sections:
        for t in s.turns:
            counts[t.speaker] = counts.get(t.speaker, 0) + 1
    return min(counts.values()) < max(1, sum(counts.values()) // 6)
