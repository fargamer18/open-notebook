"""Server-side validation of model-generated quiz questions.

The model's output is never trusted. ``parse_output`` turns the raw text into
a list of question objects (salvaging complete objects from a truncated
reply); ``validate_question`` checks one object against the evidence that was
actually offered and either returns a ``QuizQuestion`` or a rejection reason.

Small, unambiguous defects are repaired (whitespace, "A) " option prefixes,
True/False casing and order, duplicate letters, a hint that gives the answer
away is dropped). Everything else is rejected — wrong structure, unknown or
out-of-scope evidence labels, duplicate options or questions, HTML, numbers
the evidence does not contain, or a question whose answer the cited evidence
does not support. The grounding test is lexical and conservative: it is a
gate against unsupported questions, not a proof of correctness.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from litenotebook.memory.concepts import meaningful, normalize_word, terms
from litenotebook.study.model import (
    LETTERS,
    MAX_CONCEPT_CHARS,
    MAX_CONCEPTS,
    MAX_EVIDENCE_PER_QUESTION,
    MAX_EXPLANATION_CHARS,
    MAX_HINT_CHARS,
    MAX_OPTION_CHARS,
    MAX_QUESTION_CHARS,
    QTYPES,
    TRUE_FALSE,
    Concept,
    EvidenceItem,
    QuizQuestion,
)


class MalformedOutputError(ValueError):
    """The reply contained no usable question list."""


@dataclass(frozen=True)
class Rejection:
    index: int  # position in the model's list (0-based)
    reason: str  # short, internal (kept in quiz meta, never shown verbatim)


# --- parsing ----------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_output(text: str) -> list[Any]:
    """The ``questions`` list from the model's reply.

    Accepts a JSON object ``{"questions": [...]}`` or a bare list. A reply cut
    off by the token limit keeps every question object that is complete.
    """
    text = _THINK.sub("", text or "").strip()
    text = _FENCE.sub("", text).strip()
    if not text:
        raise MalformedOutputError("empty reply")
    try:
        data = json.loads(text)
    except ValueError:
        salvaged = _salvage(text)
        if not salvaged:
            raise MalformedOutputError("reply is not valid JSON") from None
        return salvaged
    if isinstance(data, dict):
        data = data.get("questions")
    if not isinstance(data, list):
        raise MalformedOutputError("reply has no question list")
    return data


def _salvage(text: str) -> list[Any]:
    """Complete objects of a truncated ``"questions": [ {…}, {…}, {…`` reply."""
    start = text.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    out: list[Any] = []
    i = start + 1
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            break
        out.append(obj)
        i = end
    return out


# --- text helpers -----------------------------------------------------------------

_SPACE = re.compile(r"\s+")
_LABEL = re.compile(r"\[\s*[SNM]\s*\d+(?:\s*[,;]\s*[SNM]?\s*\d+)*\s*\]", re.IGNORECASE)
_OPTION_PREFIX = re.compile(r"^\s*(?:\(?[A-Da-d][\).:]|[A-Da-d]\s*[-–]\s)\s*")
_HTML = re.compile(
    r"<\s*/?\s*(?:script|style|iframe|object|embed|img|svg|math|a|div|span|p|br|b|i|"
    r"u|em|strong|html|head|body|form|input|button|link|meta|base|table|tr|td|ul|"
    r"ol|li|h[1-6]|pre|code|textarea|select|option|video|audio|source)\b[^>]*>?"
    r"|javascript\s*:|\bon[a-z]{3,}\s*=|<!--|&lt;\s*script",
    re.IGNORECASE,
)
_BANNED = re.compile(
    r"\baccording to (?:the |this |these )?(?:passage|passages|text|document|"
    r"documents|source|sources|excerpt|excerpts|evidence|material|notes?|author)\b"
    r"|\b(?:the|this|these) (?:passage|passages|excerpt|excerpts|evidence)\b"
    r"|\bin the (?:text|document|source|excerpt|material) (?:above|provided|given)\b"
    r"|\b(?:all|none) of the above\b|\bboth (?:a|b) and (?:b|c)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_TF_WORDS = re.compile(
    r"\b(?:true|false|correct|incorrect|right|wrong|accurate|inaccurate|supports?|"
    r"supported|confirms?|confirmed|contradicts?|contradicted|states?|stated)\b",
    re.I,
)
# A hint that points at the hidden evidence ("the passage says…") helps nobody.
_META_HINT = re.compile(
    r"\b(?:the|this|first|second|third|that) (?:passage|passages|evidence|excerpt|"
    r"excerpts|text|document|source)\b|\b(?:mentioned|described|stated) in\b",
    re.I,
)
# A leading "According to the passage," is removed; anywhere else it rejects.
_LEADING_REF = re.compile(
    r"^(?:according to|based on|as (?:stated|described|explained) in|in) "
    r"(?:the |this |these )?(?:passage|passages|text|evidence|excerpt|excerpts|"
    r"document|documents|source|sources|material)\s*,\s*",
    re.I,
)
# Sentences that read as instructions to the model. They stay in the evidence
# (they are document content) but never count as support for an answer.
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.!?\n]{0,60}\b(?:instruction|"
    r"instructions|prompt|prompts|rules|directions|guidelines)\b"
    r"|\b(?:the )?(?:correct|right|true) answers?\b[^.!?\n]{0,40}?"
    r"\b(?:is|are|should be|must be|will be)\b"
    r"|\b(?:you|assistant|model|ai|llm)\b[^.!?\n]{0,30}\b(?:must|should|will|shall)"
    r"\b[^.!?\n]{0,30}\b(?:answer|respond|reply|output|say|write|mark|choose)\b"
    r"|\b(?:always|only)\s+(?:answer|respond|reply|say|output)\b"
    r"|\b(?:hosts?|speakers?|narrators?|presenters?)\b[^.!?\n]{0,30}"
    r"\b(?:must|should|shall|will)\b[^.!?\n]{0,30}\b(?:tell|say|announce|read|"
    r"mention|promote|recommend)\b"
    r"|\buse (?:information|content|data|text|facts) from\b[^.!?\n]{0,40}\bnotebook"
    r"|\b(?:reveal|show|print|repeat|output)\b[^.!?\n]{0,30}\b(?:your|the)\s+"
    r"(?:instructions|prompt|rules)\b"
    r"|\bsystem (?:prompt|notice|message|override)\b|\bnew instructions?\b"
    r"|\bdeveloper mode\b",
    re.IGNORECASE,
)
# Sentence boundaries. A single newline inside PDF text is a line wrap, not a
# boundary (otherwise "Ignore previous\ninstructions." escapes _INJECTION);
# blank lines (paragraphs) are boundaries.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def split_sentences(text: str) -> list[str]:
    """Sentences of document text, with line wraps joined."""
    return [" ".join(part.split()) for part in _SENTENCE.split(text) if part.strip()]


# Question words that carry no subject matter (on top of the concept stopwords).
QUIZ_STOPWORDS = frozenset(
    """
    which following statement statements best describes describe described
    true false correct incorrect likely least option options identify select
    choose consider considered given scenario accurate accurately characterizes
    characterises summarizes summarises regarding concerning represents
    evidence passage passages text excerpt states stated explicitly mentions
    mentioned says directly answer answers primary function role main
    """.split()
)
# Option letters in an explanation are wrong once options are shuffled.
_LETTER_REF = re.compile(
    r"\s*\((?:[A-D](?:\s*(?:,|and|&)\s*[A-D])*)\)"
    r"|(?<=[.!?])\s*[^.!?]*\b(?i:option|options|answer|choice|choices|letter)s? "
    r"[A-D]\b[^.!?]*[.!?]"
    r"|^[^.!?]*\b(?i:option|options|answer|choice|choices|letter)s? "
    r"[A-D]\b[^.!?]*[.!?]\s*"
)
STEM_CHARS = 6
MIN_SUPPORT = 0.45  # share of the claim's content words found in the evidence


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFC", value).replace("\x00", "")
    value = "".join(
        ch for ch in value if ch in "\t\n" or unicodedata.category(ch)[0] != "C"
    )
    return _SPACE.sub(" ", value).strip()


def strip_labels(text: str) -> str:
    return _SPACE.sub(" ", _LABEL.sub("", text)).replace(" .", ".").strip()


def _norm(text: str) -> str:
    """Comparison form: case-folded words only."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _stems(text: str) -> set[str]:
    # Number words are checked as numbers (see number_values), not as words.
    return {
        t[:STEM_CHARS]
        for t in terms(text)
        if t not in QUIZ_STOPWORDS and t not in NUMBER_WORDS and meaningful(t)
    }


def _numbers(text: str) -> set[str]:
    out = set()
    for raw in _NUMBER.findall(text):
        value = raw.replace(",", "")
        if len(value.replace(".", "")) >= 2:  # single digits are not "facts"
            out.add(value)
    return out


# --- numeric grounding --------------------------------------------------------------
# Numbers in correct answers must be supported by the evidence. Values are
# compared after a small, deterministic normalisation: digit runs lose their
# thousands separators ("1,642" → "1642") and the words zero…twenty become
# their digits ("three" → "3"). Nothing else is interpreted — no ordinals,
# fractions, "dozen" or compound words ("twenty-one" yields 20 and 1, which
# then usually fail): an unrecognised form is rejected, never guessed.
NUMBER_WORDS = {
    word: str(value)
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)*|[.!?;:]")
_STOP = frozenset(".!?;:")  # a number's neighbours never cross these
NUMBER_WINDOW = 3  # words searched on each side of a number for its anchor


def _number_value(token: str) -> str | None:
    if token in _STOP:
        return None
    if token[0].isdigit():
        value = token.replace(",", "")
        if "." in value:
            value = value.rstrip("0").rstrip(".") or "0"
        return value.lstrip("0") or "0"
    return NUMBER_WORDS.get(token.casefold())


def number_values(text: str) -> set[str]:
    """Normalised numeric values in ``text`` (digits and the words 0–20)."""
    return {v for t in _TOKEN.findall(text) if (v := _number_value(t)) is not None}


def _neighbour(tokens: list[str], index: int, step: int) -> str | None:
    """Stem of the nearest meaningful word on one side of a number."""
    j, seen = index + step, 0
    while 0 <= j < len(tokens) and seen < NUMBER_WINDOW and tokens[j] not in _STOP:
        word = normalize_word(tokens[j])
        seen += 1
        if _number_value(tokens[j]) is None and meaningful(word):
            return word[:STEM_CHARS]
        j += step
    return None


def _anchor(tokens: list[str], index: int) -> tuple[int, str] | None:
    """What a number counts: the word after it ("three *stages*"), or, when
    nothing follows, the word before it ("*ended* in 2041")."""
    after = _neighbour(tokens, index, 1)
    if after is not None:
        return 1, after
    before = _neighbour(tokens, index, -1)
    return (-1, before) if before is not None else None


def contradicted_numbers(statement: str, support: str) -> bool:
    """True when every number of ``statement`` that the evidence lacks is
    contradicted by a different number in the evidence attached to the same
    word ("four stages" vs "exactly three stages"; "ended in 2050" vs "ended
    in 2041"). Purely positional: no reasoning about meaning."""
    have = number_values(support)
    s_tokens = _TOKEN.findall(statement)
    e_tokens = _TOKEN.findall(support)
    e_numbers = [
        (i, v) for i, t in enumerate(e_tokens) if (v := _number_value(t)) is not None
    ]
    missing = [
        (i, v)
        for i, t in enumerate(s_tokens)
        if (v := _number_value(t)) is not None and v not in have
    ]
    if not missing:
        return False
    for i, value in missing:
        anchor = _anchor(s_tokens, i)
        if anchor is None:
            return False
        step, stem = anchor
        if not any(
            other != value and _same_stem(_neighbour(e_tokens, j, step), stem)
            for j, other in e_numbers
        ):
            return False
    return True


def _same_stem(a: str | None, b: str | None) -> bool:
    """Equal stems, or one a prefix of the other ("end" / "ended")."""
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def supported_numbers(statement: str, support: str, question: str) -> bool:
    """Every number in a correct answer is in the evidence *attached to the same
    thing*. "three stages" needs a 3 (or "three") next to "stages" in the
    evidence. A bare number ("4") must be the only value the evidence attaches
    to a word of the question ("How many stages…?" ↔ "exactly three stages"),
    so an unrelated "tram line 4" never supports it; ambiguity rejects."""
    s_tokens = _TOKEN.findall(statement)
    e_tokens = _TOKEN.findall(support)
    e_numbers = [
        (j, v) for j, t in enumerate(e_tokens) if (v := _number_value(t)) is not None
    ]
    q_stems = _stems(question)
    for i, token in enumerate(s_tokens):
        value = _number_value(token)
        if value is None:
            continue
        anchor = _anchor(s_tokens, i)
        if anchor is not None:
            step, stem = anchor
            if not any(
                other == value and _same_stem(_neighbour(e_tokens, j, step), stem)
                for j, other in e_numbers
            ):
                return False
            continue
        candidates = set()
        for j, other in e_numbers:
            e_anchor = _anchor(e_tokens, j)
            if e_anchor and any(_same_stem(e_anchor[1], q) for q in q_stems):
                candidates.add(other)
        if candidates != {value}:
            return False
    return True


def support_text(items: Sequence[EvidenceItem]) -> str:
    """The cited evidence minus instruction-like sentences."""
    kept = []
    for item in items:
        for sentence in split_sentences(item.text):
            if not _INJECTION.search(sentence):
                kept.append(sentence)
    return " ".join(kept)


def has_html(text: str) -> bool:
    return bool(_HTML.search(text))


# --- one question -------------------------------------------------------------------

_TYPE_ALIASES = {
    "single": "single",
    "single_choice": "single",
    "single-choice": "single",
    "single_answer": "single",
    "mcq": "single",
    "multiple_choice": "single",
    "multiple": "multiple",
    "multi": "multiple",
    "multi_select": "multiple",
    "multiple_answer": "multiple",
    "multiple_answers": "multiple",
    "multiple_select": "multiple",
    "true_false": "true_false",
    "true/false": "true_false",
    "truefalse": "true_false",
    "tf": "true_false",
    "boolean": "true_false",
}


def _correct_indexes(raw: Any, options: list[str]) -> list[int] | None:
    values = raw if isinstance(raw, list) else [raw]
    out: list[int] = []
    for value in values:
        if not isinstance(value, str):
            return None
        v = value.strip().strip("[]().").strip()
        if len(v) == 1 and v.upper() in LETTERS[: len(options)]:
            index = LETTERS.index(v.upper())
        else:  # the option text itself
            matches = [i for i, o in enumerate(options) if _norm(o) == _norm(v)]
            if len(matches) != 1:
                return None
            index = matches[0]
        if index not in out:
            out.append(index)  # repeated letters: repaired
    return sorted(out)


def _concepts(raw: Any, grounded: set[str], fallback_text: str) -> list[Concept]:
    out: list[Concept] = []
    seen: set[str] = set()
    for value in raw if isinstance(raw, list) else []:
        label = clean_text(value).strip(" \"'.,;:")
        if not label or len(label) > MAX_CONCEPT_CHARS or has_html(label):
            continue
        words = [
            normalize_word(w) for w in re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", label)
        ]
        if not 1 <= len(words) <= 4 or not any(meaningful(w) for w in words):
            continue
        if not any(w[:STEM_CHARS] in grounded for w in words if meaningful(w)):
            continue  # a concept the evidence never mentions
        key = " ".join(words)
        if key in seen:
            continue
        seen.add(key)
        out.append(Concept(key, label))
        if len(out) == MAX_CONCEPTS:
            break
    if not out:  # deterministic fallback: the strongest phrase, if any
        for phrase in _top_phrases(fallback_text, grounded):
            out.append(Concept(phrase, phrase))
            break
    return out


def _top_phrases(text: str, grounded: set[str]) -> list[str]:
    from litenotebook.memory.concepts import extract_concepts

    candidates = extract_concepts(text, limit=6)
    return [
        c
        for c in candidates
        if all(w[:STEM_CHARS] in grounded for w in c.split())
        and not set(c.split()) & QUIZ_STOPWORDS
    ]


def validate_question(
    raw: Any,
    package: Mapping[str, EvidenceItem],
    *,
    allowed_types: Collection[str],
    allowed_sources: Collection[int],
    accepted: Sequence[QuizQuestion],
    difficulty: str,
) -> QuizQuestion | str:
    """A ``QuizQuestion`` or the reason it was rejected."""
    if not isinstance(raw, dict):
        return "not an object"
    qtype = _TYPE_ALIASES.get(clean_text(raw.get("type")).lower().replace(" ", "_"))
    if qtype is None or qtype not in QTYPES:
        return "invalid type"
    if qtype not in allowed_types:
        return "type not requested"

    question = strip_labels(clean_text(raw.get("question")))
    explanation = strip_labels(clean_text(raw.get("explanation")))
    hint = strip_labels(clean_text(raw.get("hint")))
    if _LEADING_REF.match(question):  # repaired: the clause is not needed
        question = _LEADING_REF.sub("", question, count=1)
        question = question[:1].upper() + question[1:]
    if not question:
        return "empty question"
    if len(question) > MAX_QUESTION_CHARS:
        return "question too long"
    explanation = _LETTER_REF.sub("", explanation).strip()  # repaired
    if not explanation:
        return "empty explanation"
    if len(explanation) > MAX_EXPLANATION_CHARS:
        return "explanation too long"

    raw_options = raw.get("options")
    if not isinstance(raw_options, list) or not all(
        isinstance(o, str) for o in raw_options
    ):
        if qtype != "true_false":
            return "options missing"
        raw_options = []
    options = [clean_text(o) for o in raw_options]
    if (
        qtype != "true_false"
        and options
        and all(_OPTION_PREFIX.match(o) for o in options)
    ):
        options = [_OPTION_PREFIX.sub("", o, count=1).strip() for o in options]
    options = [strip_labels(o) for o in options]

    if qtype == "true_false":
        if options and sorted(_norm(o) for o in options) != ["false", "true"]:
            return "invalid true/false options"
        correct_raw = raw.get("correct")
        picked = _correct_indexes(correct_raw, options) if options else None
        if picked is None:  # "True"/"False" given directly
            values = correct_raw if isinstance(correct_raw, list) else [correct_raw]
            words = {_norm(v) for v in values if isinstance(v, str)}
            if words not in ({"true"}, {"false"}):
                return "bad correct values"
            truth = "True" if words == {"true"} else "False"
        else:
            if len(picked) != 1:
                return "true/false needs exactly one answer"
            truth = "True" if _norm(options[picked[0]]) == "true" else "False"
        options = list(TRUE_FALSE)  # repaired: fixed wording and order
        correct = [TRUE_FALSE.index(truth)]
    else:
        if len(options) != 4:
            return "needs exactly 4 options"
        if any(not o for o in options):
            return "empty option"
        if any(len(o) > MAX_OPTION_CHARS for o in options):
            return "option too long"
        if len({_norm(o) for o in options}) != 4:
            return "duplicate options"
        picked = _correct_indexes(raw.get("correct"), options)
        if picked is None:
            return "bad correct values"
        if qtype == "single" and len(picked) != 1:
            return "single answer needs exactly one correct option"
        if qtype == "multiple" and not 2 <= len(picked) <= 3:
            return "multiple answers needs 2 or 3 correct options"
        correct = picked

    texts = [question, explanation, hint, *options]
    if any(has_html(t) for t in texts):
        return "html content"
    if any(_BANNED.search(t) for t in (question, *options)):
        return "banned phrasing"

    refs = raw.get("evidence")
    labels: list[str] = []
    for ref in refs if isinstance(refs, list) else [refs]:
        if isinstance(ref, str):
            label = ref.strip().strip("[]").replace(" ", "").upper()
            if label and label not in labels:
                labels.append(label)
    if not labels:
        return "missing evidence"
    if len(labels) > MAX_EVIDENCE_PER_QUESTION:
        labels = labels[:MAX_EVIDENCE_PER_QUESTION]
    if any(label not in package for label in labels):
        return "invalid citation"
    cited = [package[label] for label in labels]
    if any(
        item.kind == "source" and item.source_id not in allowed_sources
        for item in cited
    ):
        return "evidence outside selected sources"

    support = support_text(cited)
    grounded = _stems(support)
    correct_text = [options[i] for i in correct] if qtype != "true_false" else []
    claim = " ".join([question, *correct_text])
    claim_stems = _stems(claim)
    if not claim_stems:
        return "no content to ground"
    overlap = len(claim_stems & grounded)
    if overlap / len(claim_stems) < MIN_SUPPORT or overlap < min(2, len(claim_stems)):
        return "not supported by the evidence"
    # Each correct answer must be stated by the evidence, not just related to
    # it: no new content word in a short answer, one per four words beyond.
    statements = list(correct_text)
    if qtype == "true_false" and correct == [0]:
        statements.append(question)  # a true statement must be in the evidence
    for statement in statements:
        stems = _stems(statement)
        missing = len(stems - grounded)
        if stems and missing > (len(stems) - 1) // 4:
            return "correct answer not stated in the evidence"
    # Numeric grounding: every number in a correct answer (and in a statement
    # marked True) must be in the evidence; a statement marked False that
    # changes a number must be contradicted by the evidence.
    have = number_values(support)
    for statement in statements:
        context = question if statement is not question else ""
        if not supported_numbers(statement, support, context):
            return "unsupported number in a correct answer"
    if qtype == "true_false" and correct == [1] and number_values(question) - have:
        if not contradicted_numbers(question, support):
            return "false statement not contradicted by the evidence"
    evidence_numbers = _numbers(support)
    checked = [explanation, *correct_text]
    if not (qtype == "true_false" and correct == [1]):
        checked.append(question)  # a false statement may alter a number
    if any(n not in evidence_numbers for t in checked for n in _numbers(t)):
        return "unsupported number"

    if qtype == "single":
        right = len(options[correct[0]])
        others = [len(o) for i, o in enumerate(options) if i != correct[0]]
        average = sum(others) / len(others)
        if right > 2 * average and right - average > 30:
            return "correct option stands out by length"

    if hint:
        reveals = (
            len(hint) > MAX_HINT_CHARS
            or bool(_META_HINT.search(hint))
            or (qtype == "true_false" and bool(_TF_WORDS.search(hint)))
        )
        for option in correct_text:
            if len(_norm(option)) >= 3 and _norm(option) in _norm(hint):
                reveals = True
        if reveals:
            hint = ""  # repaired: the question stays, the hint goes

    key = _norm(question)
    words = set(key.split())
    for other in accepted:
        other_key = _norm(other.question)
        if other_key == key:
            return "duplicate question"
        other_words = set(other_key.split())
        union = words | other_words
        if union and len(words & other_words) / len(union) >= 0.8:
            return "near-duplicate question"

    return QuizQuestion(
        qtype=qtype,
        question=question,
        options=options,
        correct=correct,
        explanation=explanation,
        hint=hint,
        evidence=cited,
        concepts=_concepts(raw.get("concepts"), grounded, claim),
        difficulty=difficulty,
    )


def validate_batch(
    items: Sequence[Any],
    package: Mapping[str, EvidenceItem],
    *,
    allowed_types: Collection[str],
    allowed_sources: Collection[int],
    accepted: Sequence[QuizQuestion],
    difficulty: str,
    limit: int,
) -> tuple[list[QuizQuestion], list[Rejection]]:
    """Validate a reply's questions; at most ``limit`` are kept."""
    kept: list[QuizQuestion] = []
    rejected: list[Rejection] = []
    for index, raw in enumerate(items):
        if len(kept) >= limit:
            break
        result = validate_question(
            raw,
            package,
            allowed_types=allowed_types,
            allowed_sources=allowed_sources,
            accepted=[*accepted, *kept],
            difficulty=difficulty,
        )
        if isinstance(result, str):
            rejected.append(Rejection(index, result))
        else:
            kept.append(result)
    return kept, rejected
