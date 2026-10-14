"""Phase 13: Study pages, taking a quiz, review, safety and lazy loading."""

import re

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder
from litenotebook.ai.store import AIStore
from litenotebook.config import load_settings
from litenotebook.db import connect
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source
from litenotebook.study import store
from litenotebook.study.model import Concept, EvidenceItem, QuizQuestion
from litenotebook.web.app import create_app
from quiz_fixtures import (
    BASE,
    ORIGIN,
    PDF_A,
    FakeQuizLlama,
    build_notebook,
    correct_choice,
    make_ready_quiz,
    wrong_choice,
)


class CountingEmbedder(FakeEmbedder):
    created = 0

    def __init__(self):
        super().__init__()
        CountingEmbedder.created += 1


@pytest.fixture
def app_settings(tmp_path):
    return load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})


@pytest.fixture
def llama():
    return FakeQuizLlama()


@pytest.fixture
def client(app_settings, llama):
    CountingEmbedder.created = 0
    app = create_app(
        app_settings,
        env={},
        llm_transport=httpx.MockTransport(llama),
        embedder_factory=CountingEmbedder,
    )
    with TestClient(app, base_url=BASE) as c:
        yield c


@pytest.fixture
def setup(client, app_settings, tmp_path):
    nb_id, src = build_notebook(app_settings, tmp_path, sources={"Atlas.pdf": PDF_A})
    c = connect(app_settings.db_path)
    try:
        quiz_id = make_ready_quiz(c, nb_id, src["Atlas.pdf"])
    finally:
        c.close()
    return nb_id, src["Atlas.pdf"], quiz_id


def db(app_settings):
    return connect(app_settings.db_path)


def start(client, nb_id, quiz_id):
    r = client.post(
        f"/notebooks/{nb_id}/study/quizzes/{quiz_id}/start",
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert r.status_code == 303
    return int(r.headers["location"].rsplit("/", 1)[1])


def answer(client, nb_id, aid, q, choice):
    return client.post(
        f"/notebooks/{nb_id}/study/attempts/{aid}/answer",
        data={"q": str(q), "choice": [str(c) for c in choice]},
        headers=ORIGIN,
        follow_redirects=True,
    )


def test_navigation_has_study_between_notes_and_memory(client, setup):
    nb_id, _, _ = setup
    page = client.get(f"/notebooks/{nb_id}/study").text
    nav = re.search(r'<nav class="tabs" aria-label="Notebook">(.*?)</nav>', page, re.S)
    labels = re.findall(r">([A-Z][a-z]+)</a>", nav.group(1))
    assert labels[:6] == ["Sources", "Chat", "Notes", "Study", "Memory", "Search"]
    assert 'aria-current="page">Study' in nav.group(1)
    assert (
        client.get(f"/notebooks/{nb_id}?tab=study", follow_redirects=False).headers[
            "location"
        ]
        == f"/notebooks/{nb_id}/study"
    )
    palette = client.get(f"/palette.json?nb={nb_id}").json()["commands"]
    assert {"Study", "New quiz"} <= {c["label"] for c in palette}


def test_study_home_history_and_quiz_pages(client, setup):
    nb_id, _, quiz_id = setup
    home = client.get(f"/notebooks/{nb_id}/study")
    assert home.status_code == 200
    assert "Create quiz" in home.text and "Saved quiz" in home.text
    assert "Not started" in home.text and "Needs review" in home.text
    assert "mastered" not in home.text.lower().replace("not a measure of mastery", "")
    history = client.get(f"/notebooks/{nb_id}/study/quizzes")
    assert "Quiz history" in history.text and "Atlas.pdf" in history.text
    page = client.get(f"/notebooks/{nb_id}/study/quizzes/{quiz_id}")
    assert "Start quiz" in page.text and "Medium" in page.text
    new = client.get(f"/notebooks/{nb_id}/study/quizzes/new").text
    assert "<legend>Sources</legend>" in new and "<legend>Difficulty</legend>" in new
    assert 'name="sources" value=' in new and "checked" in new
    for label in ("Single answer", "Multiple answers", "True / False", "Exam-style"):
        assert label in new
    assert client.get(f"/notebooks/{nb_id}/study/quizzes/999").status_code == 404


def test_take_quiz_single_multiple_true_false_and_result(client, app_settings, setup):
    nb_id, _, quiz_id = setup
    aid = start(client, nb_id, quiz_id)
    q1 = client.get(f"/notebooks/{nb_id}/study/attempts/{aid}").text
    assert "Question 1 of 3" in q1 and "<fieldset" in q1 and "<legend" in q1
    assert 'type="radio" name="choice"' in q1 and "Submit answer" in q1
    assert "Hint" in q1 and "A harbour city." not in q1  # hint hidden until asked
    assert "Quillport is the capital." not in q1  # no explanation before answering
    assert "Open in PDF" not in q1  # evidence would give the answer away
    hint = client.post(
        f"/notebooks/{nb_id}/study/attempts/{aid}/hint",
        data={"q": "1"},
        headers=ORIGIN,
        follow_redirects=True,
    )
    assert "A harbour city." in hint.text
    c = db(app_settings)
    try:
        wrong = wrong_choice(c, aid, 1)
        multi = correct_choice(c, aid, 2)
        tf = correct_choice(c, aid, 3)
    finally:
        c.close()
    r = answer(client, nb_id, aid, 1, wrong)
    assert "✗</span> Incorrect" in r.text and 'id="feedback"' in r.text
    assert "Quillport is the capital." in r.text and "[S1]" in r.text
    assert "Atlas.pdf · page 1" in r.text and "Open in PDF" in r.text
    assert "hint used" in r.text
    assert 'name="choice"' not in r.text  # no accidental changes after submitting
    again = answer(
        client, nb_id, aid, 1, correct_choice(connect(app_settings.db_path), aid, 1)
    )
    assert "Incorrect" in again.text  # the first answer stands
    q2 = client.get(f"/notebooks/{nb_id}/study/attempts/{aid}?q=2").text
    assert 'type="checkbox" name="choice"' in q2 and "Select all that apply" in q2
    assert "Previous" in q2
    empty = answer(client, nb_id, aid, 2, [])
    assert "Choose one or more answers first." in empty.text
    r = answer(client, nb_id, aid, 2, multi)
    assert "✓</span> Correct" in r.text
    q3 = client.get(f"/notebooks/{nb_id}/study/attempts/{aid}?q=3").text
    assert "True" in q3 and "False" in q3 and "1–2 pick an option" in q3
    r = answer(client, nb_id, aid, 3, tf)
    assert "See result" in r.text
    result = client.get(f"/notebooks/{nb_id}/study/attempts/{aid}/result").text
    assert "2 / 3" in result and 'role="status"' in result
    assert (
        "<dt>Correct</dt><dd>2</dd>" in result
        and "<dt>Incorrect</dt><dd>1</dd>" in result
    )
    assert "Zorbania capital" in result  # needs review, from the missed question
    assert "Review mistakes" in result and "Retry mistakes (1)" in result
    review = client.get(
        f"/notebooks/{nb_id}/study/attempts/{aid}/review?only=mistakes"
    ).text
    assert "Which city is the capital" in review and "Which goods" not in review
    assert "your answer" in review and "correct answer" in review
    retry = client.post(
        f"/notebooks/{nb_id}/study/attempts/{aid}/retry-mistakes",
        headers=ORIGIN,
        follow_redirects=True,
    )
    assert "Question 1 of 1" in retry.text and "retrying mistakes" in retry.text
    page = client.get(f"/notebooks/{nb_id}/study/quizzes/{quiz_id}").text
    assert "Retry of mistakes" in page and "2/3" in page and "Resume" in page
    home = client.get(f"/notebooks/{nb_id}/study").text
    assert "Latest 2/3" in home and "Zorbania capital" in home


def test_resume_goes_to_the_first_unanswered_question(client, app_settings, setup):
    nb_id, _, quiz_id = setup
    aid = start(client, nb_id, quiz_id)
    c = db(app_settings)
    try:
        answer(client, nb_id, aid, 1, correct_choice(c, aid, 1))
    finally:
        c.close()
    r = client.get(f"/notebooks/{nb_id}/study/attempts/{aid}")
    assert "Question 2 of 3" in r.text and "1 of 3 answered" in r.text
    finish = client.post(
        f"/notebooks/{nb_id}/study/attempts/{aid}/finish",
        headers=ORIGIN,
        follow_redirects=True,
    )
    assert "1 / 3" in finish.text and "<dt>Skipped</dt><dd>2</dd>" in finish.text


def test_unanswered_questions_never_reveal_answers_in_review(client, setup):
    nb_id, _, quiz_id = setup
    aid = start(client, nb_id, quiz_id)
    review = client.get(f"/notebooks/{nb_id}/study/attempts/{aid}/review").text
    assert "correct answer" not in review and "Not answered yet" in review
    assert "Quillport is the capital." not in review


def test_reader_jump_and_deleted_source(client, app_settings, setup):
    nb_id, sid, quiz_id = setup
    aid = start(client, nb_id, quiz_id)
    c = db(app_settings)
    try:
        r = answer(client, nb_id, aid, 1, correct_choice(c, aid, 1))
        chunk = c.execute(
            "SELECT id FROM chunks WHERE source_id = ? ORDER BY ord LIMIT 1", (sid,)
        ).fetchone()[0]
    finally:
        c.close()
    link = f"/notebooks/{nb_id}/sources/{sid}/read?chunk={chunk}&amp;attempt={aid}"
    assert link in r.text
    reader = client.get(link.replace("&amp;", "&"))
    assert reader.status_code == 200 and "Back to quiz" in reader.text
    assert 'data-page="1"' in reader.text
    assert f'data-focus-kind="chunk" data-focus-id="{chunk}"' in reader.text
    assert f"/study/attempts/{aid}/review" in reader.text
    target = client.get(f"/notebooks/{nb_id}/sources/{sid}/chunks/{chunk}/highlight")
    hl = target.json()  # what reader.js draws: PDFium word geometry
    assert hl["page"] == 1 and hl["rects"] and hl["live"]
    c = db(app_settings)
    try:
        delete_source(c, app_settings, nb_id, sid, FaissStore(c, app_settings))
    finally:
        c.close()
    r = client.get(f"/notebooks/{nb_id}/study/attempts/{aid}?q=1")
    assert "Source deleted" in r.text and "Quillport" in r.text
    assert re.search(r"<button[^>]*disabled>Open in PDF</button>", r.text)
    assert "read?chunk=" not in r.text


def test_model_and_source_text_is_escaped(client, app_settings, setup):
    nb_id, sid, _ = setup
    evil = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    c = db(app_settings)
    try:
        quiz_id = store.create_quiz(
            c,
            nb_id,
            title=f"T {evil}",
            difficulty="hard",
            types=["single"],
            count=5,
            focus=evil,
            material={"sources": [{"id": sid, "filename": evil}]},
        )
        ev = EvidenceItem(
            "S1",
            "source",
            f"excerpt {evil}",
            f"file {evil}",
            source_id=sid,
            chunk_id=0,
            page_no=1,
        )
        store.save_questions(
            c,
            quiz_id,
            [
                QuizQuestion(
                    "single",
                    f"Q {evil}",
                    [f"A {evil}", "B", "C", "D"],
                    [0],
                    f"Explain {evil}",
                    f"Hint {evil}",
                    [ev],
                    [Concept("x", f"C {evil}")],
                )
            ],
            message=None,
            model=None,
            meta={},
        )
        aid = store.start_attempt(c, nb_id, quiz_id)
        store.record_hint(c, nb_id, aid, 1)
        answer(client, nb_id, aid, 1, [2])
        store.finish_attempt(c, nb_id, aid)
    finally:
        c.close()
    pages = [
        client.get(f"/notebooks/{nb_id}/study").text,
        client.get(f"/notebooks/{nb_id}/study/quizzes").text,
        client.get(f"/notebooks/{nb_id}/study/quizzes/{quiz_id}").text,
        client.get(f"/notebooks/{nb_id}/study/attempts/{aid}?q=1").text,
        client.get(f"/notebooks/{nb_id}/study/attempts/{aid}/result").text,
        client.get(f"/notebooks/{nb_id}/study/attempts/{aid}/review").text,
    ]
    for text in pages:
        assert "<script>alert" not in text and "<img src=x" not in text
    joined = "\n".join(pages)
    for field in ("T ", "Q ", "A ", "Explain ", "Hint ", "excerpt ", "file ", "C "):
        assert f"{field}&lt;script&gt;alert(1)&lt;/script&gt;" in joined, field


def test_delete_quiz_needs_confirmation(client, app_settings, setup):
    nb_id, _, quiz_id = setup
    aid = start(client, nb_id, quiz_id)
    page = client.get(f"/notebooks/{nb_id}/study/quizzes/{quiz_id}").text
    assert 'data-confirm-action="/notebooks/' in page and "all 1 attempt" in page
    noscript = client.get(
        f"/notebooks/{nb_id}/study/quizzes/{quiz_id}/delete", follow_redirects=True
    ).text
    assert "Delete this quiz permanently?" in noscript and 'name="confirm"' in noscript
    bad = client.post(
        f"/notebooks/{nb_id}/study/quizzes/{quiz_id}/delete",
        data={"confirm": "nope"},
        headers=ORIGIN,
    )
    assert bad.status_code == 422 and "not confirmed" in bad.text
    ok = client.post(
        f"/notebooks/{nb_id}/study/quizzes/{quiz_id}/delete",
        data={"confirm": str(quiz_id)},
        headers=ORIGIN,
        follow_redirects=True,
    )
    assert ok.status_code == 200 and "Quiz deleted" in ok.text
    assert client.get(f"/notebooks/{nb_id}/study/attempts/{aid}").status_code == 404


def test_pages_and_quiz_taking_wake_neither_minilm_nor_qwen(
    client, app_settings, setup, llama
):

    nb_id, sid, quiz_id = setup
    aid = start(client, nb_id, quiz_id)
    c = db(app_settings)
    try:
        for q in (1, 2, 3):
            answer(client, nb_id, aid, q, correct_choice(c, aid, q))
    finally:
        c.close()
    retry_target = start(client, nb_id, quiz_id)
    for url in (
        f"/notebooks/{nb_id}/study",
        f"/notebooks/{nb_id}/study/quizzes",
        f"/notebooks/{nb_id}/study/quizzes/new",
        f"/notebooks/{nb_id}/study/quizzes/{quiz_id}",
        f"/notebooks/{nb_id}/study/attempts/{aid}/result",
        f"/notebooks/{nb_id}/study/attempts/{aid}/review",
        f"/notebooks/{nb_id}/study/attempts/{retry_target}?q=2",
    ):
        assert client.get(url).status_code == 200, url
    assert CountingEmbedder.created == 0
    assert client.app.state.embedder.created == 0
    assert llama.paths == [] and llama.requests == []
    assert client.app.state.local_ai.runtime.running() is False


def test_existing_quizzes_work_with_ai_disabled(client, app_settings, setup, llama):
    nb_id, _, quiz_id = setup
    c = db(app_settings)
    try:
        AIStore(c).save_profile(AIStore(c).profile().__class__(type="none"))
    finally:
        c.close()
    aid = start(client, nb_id, quiz_id)
    c = db(app_settings)
    try:
        for q in (1, 2, 3):
            assert (
                answer(client, nb_id, aid, q, correct_choice(c, aid, q)).status_code
                == 200
            )
    finally:
        c.close()
    assert "3 / 3" in client.get(f"/notebooks/{nb_id}/study/attempts/{aid}/result").text
    assert llama.requests == []


def test_scoping_across_notebooks(client, app_settings, setup, tmp_path):
    nb_id, _, quiz_id = setup
    aid = start(client, nb_id, quiz_id)
    (tmp_path / "x").mkdir()
    other, _ = build_notebook(
        app_settings, tmp_path / "x", name="X", sources={"x.pdf": PDF_A}
    )
    assert client.get(f"/notebooks/{other}/study/quizzes/{quiz_id}").status_code == 404
    assert client.get(f"/notebooks/{other}/study/attempts/{aid}").status_code == 404
    r = client.post(
        f"/notebooks/{other}/study/attempts/{aid}/answer",
        data={"q": "1", "choice": ["1"]},
        headers=ORIGIN,
    )
    assert r.status_code == 404
    reader = client.get(f"/notebooks/{other}/sources/1/read?attempt={aid}")
    assert "Back to quiz" not in reader.text


def test_generation_status_page_polls_and_announces(client, app_settings, setup):
    nb_id, _, _ = setup
    c = db(app_settings)
    try:
        quiz_id = store.create_quiz(
            c,
            nb_id,
            title="G",
            difficulty="easy",
            types=["single"],
            count=5,
            focus="",
            material={},
        )
        store.set_stage(c, quiz_id, "generating:2:4")
    finally:
        c.close()
    page = client.get(f"/notebooks/{nb_id}/study/quizzes/{quiz_id}").text
    assert 'hx-trigger="every 1s"' in page and "request 2 of 4" in page
    assert 'id="quiz-announce"' in page and 'aria-live="polite"' in page
    assert "Delete quiz" not in page  # cannot delete while generating
    status = client.get(
        f"/notebooks/{nb_id}/study/quizzes/{quiz_id}/status",
        headers={"HX-Request": "true"},
    )
    assert status.status_code == 200 and "HX-Refresh" not in status.headers
    c = db(app_settings)
    try:
        store.finish_failed(c, quiz_id, "Cancelled.", status="cancelled")
    finally:
        c.close()
    done = client.get(
        f"/notebooks/{nb_id}/study/quizzes/{quiz_id}/status",
        headers={"HX-Request": "true"},
    )
    assert done.headers.get("HX-Refresh") == "true"
