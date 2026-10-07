import pytest

from litenotebook.rag.citations import format_citation, parse_labels, resolve
from litenotebook.retrieval.hybrid import SearchResult


def results(n=3):
    return [
        SearchResult(
            chunk_id=100 + i,
            source_id=7,
            source_filename=f"doc{i}.pdf",
            page_no=10 + i,
            ord=i,
            char_start=0,
            char_end=5,
            text="text",
            fused_score=0.1,
        )
        for i in range(1, n + 1)
    ]


@pytest.mark.parametrize(
    ("text", "labels"),
    [
        ("Rank is pivots [S1].", [1]),
        ("Both [S1][S2] agree.", [1, 2]),
        ("Grouped [S1, S2] here.", [1, 2]),
        ("Grouped [S3,S1; S2].", [3, 1, 2]),
        ("Shorthand [S1, 2].", [1, 2]),
        ("Lowercase [s2] and spaced [ S3 ].", [2, 3]),
        ("Repeat [S2] then [S2] again [S1].", [2, 1]),
        ("No citations here.", []),
        ("Not a label [1] or [Source 1] or S1.", []),
    ],
)
def test_parse_labels(text, labels):
    assert parse_labels(text) == labels


def test_resolve_uses_retrieval_metadata_not_model_text():
    report = resolve("The model says page 999 of fake.pdf [S2].", results())
    [c] = report.citations
    assert (c.label, c.chunk_id, c.source_filename, c.page_no, c.ord) == (
        2,
        102,
        "doc2.pdf",
        12,
        2,
    )
    assert format_citation(c) == "S2: doc2.pdf — page 12 — chunk 2"


def test_invalid_labels_are_rejected():
    report = resolve("Claims [S0] [S1] [S4, S2] [S99].", results(3))
    assert [c.label for c in report.citations] == [1, 2]
    assert report.invalid_labels == [0, 4, 99]


def test_no_context_means_every_label_is_invalid():
    report = resolve("[S1]", [])
    assert report.citations == [] and report.invalid_labels == [1]
