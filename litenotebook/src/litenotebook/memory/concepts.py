"""Conservative, deterministic concept extraction (no NLP models).

Text is split into runs of meaningful words: punctuation and stopwords end a
run. Every meaningful word of a run is a candidate, and so is every pair of
adjacent words inside a run ("calibration system"). Candidates are scored by
frequency (phrases weigh more) and the top few are kept, so a long answer
cannot explode into dozens of graph nodes.

"What did we discuss about diagonalising matrix A?" → "diagonalising matrix",
"diagonalising", "matrix" — never "what", "did", "we" or "about".
"""

from __future__ import annotations

import re
import unicodedata

MIN_WORD_CHARS = 3
PHRASE_WEIGHT = 1.5
DEFAULT_LIMIT = 6

# Words split into runs; punctuation between words ends a run.
_SEGMENTS = re.compile(r"[^\w\s'’-]+|\s-\s|--+|\n")
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*(?:-[^\W_]+)*")
_CITATION = re.compile(r"\[\s*[SM]\s*\d+(?:\s*[,;]\s*[SM]?\s*\d+)*\s*\]", re.IGNORECASE)
# Markup that is never a concept: LaTeX commands (\frac, \mathbf) and the
# names of LaTeX environments (\begin{pmatrix}), URLs and e-mail addresses.
# They become run breaks, so "\frac{a}{b}" never yields "frac".
_NOT_WORDS = re.compile(
    r"\\(?:begin|end)\s*\{[^{}]*\}"
    r"|\\[A-Za-z]+"
    r"|\b(?:https?|ftp)://\S+|\bwww\.\S+"
    r"|\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
)

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each either else
    ever every few for from further get gets getting got had hadn't has hasn't
    have haven't having he her here hers herself him himself his how however i if
    in into is isn't it it's its itself just let let's like made make makes many
    may me might more most much must my myself no nor not now of off often on once
    one only or other others otherwise our ours ourselves out over own per please
    quite rather really same she should shouldn't since so some such than that
    that's the their theirs them themselves then there there's these they they're
    this those though through thus to too under until up upon us very via was
    wasn't we we're were weren't what what's when where where's whether which
    while who who's whom whose why will with within without won't would wouldn't
    yes yet you you're your yours yourself yourselves
    actually already always anything anyway around away back basically certain
    certainly come comes did done each else enough etc even exactly example
    first give given go goes going gone good great kind kinds know known last
    later least less lot lots maybe mean means meant need needs new next note
    okay ok part perhaps point possible probably put said say says see seem seems
    show shown simply something sometimes sort still sure take tell telling than
    thank thanks thing things think thought today try two use used uses using
    want wanted way ways well whatever yeah
    discuss discussed discussing discussion discussions earlier previous previously
    talk talked talking chat chatted conversation conversations question questions
    answer answered answers ask asked asking remember recall recalled call called
    explain explained explaining mentioned mention describe described tell told
    according source sources document documents notebook page pages pdf
    """.split()
)


def normalize_word(word: str) -> str:
    word = unicodedata.normalize("NFKC", word).casefold().replace("’", "'")
    if word.endswith("'s"):
        word = word[:-2]
    return word.replace("'", "")


def meaningful(word: str) -> bool:
    return (
        len(word) >= MIN_WORD_CHARS
        and word not in STOPWORDS
        and any(ch.isalpha() for ch in word)
        and not word.replace("-", "").isdigit()
    )


def runs(text: str) -> list[list[str]]:
    """Runs of consecutive meaningful, normalized words."""
    out: list[list[str]] = []
    text = _CITATION.sub(" ", text)
    text = _NOT_WORDS.sub("\n", text)
    for segment in _SEGMENTS.split(text):
        run: list[str] = []
        for raw in _WORD.findall(segment):
            word = normalize_word(raw)
            if meaningful(word):
                run.append(word)
            else:
                if run:
                    out.append(run)
                run = []
        if run:
            out.append(run)
    return out


def terms(text: str) -> list[str]:
    """Distinct meaningful words in order of first appearance."""
    return list(dict.fromkeys(w for run in runs(text) for w in run))


def extract_concepts(text: str, limit: int = DEFAULT_LIMIT) -> list[str]:
    scores: dict[str, float] = {}
    first: dict[str, int] = {}
    position = 0
    for run in runs(text):
        candidates: list[tuple[str, float]] = [(w, 1.0) for w in run]
        candidates += [
            (f"{a} {b}", PHRASE_WEIGHT)
            for a, b in zip(run, run[1:], strict=False)
            if a != b
        ]
        for candidate, weight in candidates:
            scores[candidate] = scores.get(candidate, 0.0) + weight
            first.setdefault(candidate, position)
            position += 1
    ranked = sorted(scores, key=lambda c: (-scores[c], first[c]))
    return ranked[:limit]
