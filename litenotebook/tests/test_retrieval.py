import pytest

from conftest import make_pdf
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.notebooks import create_notebook
from litenotebook.retrieval.hybrid import (
    Retriever,
    SearchResult,
    dedupe,
    fts_query,
    rrf_fuse,
)

PAGES = [
    "Gaussian elimination reduces a matrix to row echelon form using pivots.",
    "The rank of a matrix equals the number of pivot columns in echelon form.",
    "A homogeneous system always has the trivial zero solution.",
    "Photosynthesis converts light energy into chemical energy in plants.",
]


@pytest.fixture
def retriever(conn, settings, embedder, store):
    return Retriever(conn, settings, embedder, store)


@pytest.fixture
def loaded(conn, settings, notebook, embedder, store, tmp_path):
    ingest_pdf(
        conn, settings, notebook, make_pdf(tmp_path / "la.pdf", PAGES), embedder, store
    )
    return notebook


def result(cid, text):
    return SearchResult(cid, 1, "f.pdf", 1, cid, 0, len(text), text, fused_score=1.0)


# --- FTS query sanitization ------------------------------------------------------


def test_fts_query_quotes_terms_and_drops_stop_words():
    assert fts_query("What is the rank of a matrix?") == '"rank" OR "matrix"'


def test_fts_query_keeps_stop_words_if_nothing_else():
    assert fts_query("what is it") == '"what" OR "is" OR "it"'


@pytest.mark.parametrize("text", ["", "   ", "?!*()", '""'])
def test_fts_query_none_for_no_words(text):
    assert fts_query(text) is None


MALFORMED = [
    'rank"',
    '"unbalanced',
    "NEAR(rank matrix)",
    "rank AND OR NOT",
    "text:rank",
    "rank*",
    "(((",
    "^rank",
    "rank - matrix",
    "{rank}",
    "'; DROP TABLE chunks; --",
    "a" * 5000,
    "ünïcödé ranks",
    "\x00rank",
]


@pytest.mark.parametrize("text", MALFORMED)
def test_malformed_queries_never_raise(loaded, retriever, text):
    retriever.lexical(loaded.id, text, 10)
    retriever.search(loaded.id, text)


def test_injection_attempt_leaves_db_intact(loaded, retriever, conn):
    retriever.search(loaded.id, "'; DROP TABLE chunks; --")
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 4


# --- fusion & dedup ---------------------------------------------------------------


def test_rrf_fuse_math():
    scores = rrf_fuse([[1, 2, 3], [3, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == 1


def test_rrf_rewards_agreement():
    scores = rrf_fuse([[10, 20], [20, 30]])
    assert scores[20] > scores[10] > scores[30]


def test_dedupe_drops_near_duplicates_keeps_order():
    a = "Gaussian elimination reduces a matrix to row echelon form"
    kept = dedupe(
        [
            result(1, a),
            result(2, a + " quickly"),
            result(3, "Rank equals pivots"),
            result(4, ""),
        ],
        threshold=0.8,
    )
    assert [r.chunk_id for r in kept] == [1, 3]


def test_dedupe_keeps_distinct_overlapping_text():
    kept = dedupe(
        [result(1, "one two three four"), result(2, "three four five six")], 0.8
    )
    assert len(kept) == 2


# --- end-to-end retrieval ---------------------------------------------------------


def test_lexical_ranks_matching_chunk_first(loaded, retriever, conn):
    hits = retriever.lexical(loaded.id, "photosynthesis", 5)
    text = conn.execute(
        "SELECT text FROM chunks WHERE id = ?", (hits[0][0],)
    ).fetchone()[0]
    assert "Photosynthesis" in text


def test_search_result_fields(loaded, retriever):
    results = retriever.search(loaded.id, "rank of a matrix pivot columns")
    top = results[0]
    assert "rank" in top.text
    assert top.source_filename == "la.pdf"
    assert top.page_no == 2 and top.ord == 1
    assert top.semantic_rank is not None and top.lexical_rank is not None
    assert top.fused_score == max(r.fused_score for r in results)
    assert [r.fused_score for r in results] == sorted(
        (r.fused_score for r in results), reverse=True
    )


def test_top_k(loaded, retriever):
    assert len(retriever.search(loaded.id, "matrix", top_k=2)) == 2


def test_search_is_scoped_to_one_notebook(conn, settings, embedder, store, tmp_path):
    a, b = create_notebook(conn, "a"), create_notebook(conn, "b")
    ingest_pdf(
        conn, settings, a, make_pdf(tmp_path / "a.pdf", PAGES[:3]), embedder, store
    )
    ingest_pdf(
        conn, settings, b, make_pdf(tmp_path / "b.pdf", PAGES[3:]), embedder, store
    )
    r = Retriever(conn, settings, embedder, store)

    for result_ in r.search(a.id, "photosynthesis light energy plants"):
        assert result_.source_filename == "a.pdf"
    assert {x.source_filename for x in r.search(b.id, "matrix rank pivots")} <= {
        "b.pdf"
    }
    assert r.lexical(a.id, "photosynthesis", 10) == []


def test_empty_notebook(conn, retriever):
    nb = create_notebook(conn, "empty")
    assert retriever.search(nb.id, "anything") == []
