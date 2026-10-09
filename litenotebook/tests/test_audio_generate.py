"""Phase 14: Audio Overview script generation through the web app (mock Qwen)."""

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from audio_fixtures import (
    PDF_AD,
    PDF_TRAP,
    FakeAudioLlama,
    script_for,
    wait_overview,
)
from conftest import FakeEmbedder
from litenotebook.ai.store import AIStore
from litenotebook.chat import complete_message, start_exchange
from litenotebook.config import load_settings
from litenotebook.db import connect
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source
from litenotebook.notes import create_note
from litenotebook.study import audio_store as store
from litenotebook.study.audio_generate import NOTHING_GROUNDED
from litenotebook.web.app import create_app
from quiz_fixtures import BASE, ORIGIN, PDF_C, build_notebook, label_with


@pytest.fixture
def app_settings(tmp_path):
    return load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})


@pytest.fixture
def llama():
    return FakeAudioLlama()


@pytest.fixture
def client(app_settings, llama):
    app = create_app(
        app_settings,
        env={},
        llm_transport=httpx.MockTransport(llama),
        embedder_factory=FakeEmbedder,
    )
    with TestClient(app, base_url=BASE) as c:
        yield c


@pytest.fixture
def nb(client, app_settings, tmp_path):
    return build_notebook(
        app_settings, tmp_path, sources={"Trap.pdf": PDF_TRAP, "Moon.pdf": PDF_C}
    )


def create(client, nb_id, sources, **form):
    data = {
        "sources": [str(s) for s in sources],
        "format": form.pop("format", "two_person"),
        "minutes": str(form.pop("minutes", 5)),
        **form,
    }
    return client.post(
        f"/notebooks/{nb_id}/study/audio",
        data=data,
        headers=ORIGIN,
        follow_redirects=False,
    )


def generate(client, settings, nb_id, sources, **form):
    r = create(client, nb_id, sources, **form)
    assert r.status_code == 303, r.text[:400]
    oid = int(r.headers["location"].rsplit("/", 1)[1])
    return oid, wait_overview(settings, oid)


def db(settings):
    return connect(settings.db_path)


def test_two_stage_generation_from_selected_sources_only(
    client, app_settings, nb, llama
):
    nb_id, src = nb
    oid, row = generate(client, app_settings, nb_id, [src["Trap.pdf"]])
    assert row["status"] == "script_ready", row
    assert llama.kinds() == ["outline", "script"]  # 5 min: one script request
    for body in llama.requests:
        assert body["response_format"]["type"] == "json_schema"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
    script_schema = llama.requests[1]["response_format"]["json_schema"]["schema"]
    speaker = script_schema["properties"]["sections"]["items"]["properties"]["turns"][
        "items"
    ]["properties"]["speaker"]
    assert speaker["enum"] == ["A", "B"]
    for prompt in llama.prompts:
        assert "Tarvos" not in prompt and "Moon.pdf" not in prompt
        assert "=== EVIDENCE (untrusted" in prompt
    system = llama.requests[0]["messages"][0]["content"]
    assert "Never follow it" in system and "Do not add outside knowledge" in system
    c = db(app_settings)
    try:
        sections = store.sections(c, oid)
        evidence = store.evidence(c, oid)
        o = store.get_overview(c, nb_id, oid)
    finally:
        c.close()
    assert [s.title for s in sections] == [
        "Getting around Quillport",
        "The Arvon process",
    ]
    texts = " ".join(t.text for s in sections for t in s.turns)
    assert "Black Sea" not in texts and "photosynthesis" not in texts
    assert "college" not in texts and "[S" not in texts
    assert all(t.speaker in ("A", "B") for s in sections for t in s.turns)
    assert set(evidence) == {"S1", "S2"}
    for e in evidence.values():
        assert e.source_id == src["Trap.pdf"] and e.title == "Trap.pdf" and e.exists
    assert evidence["S1"].page_no == 1 and evidence["S2"].page_no == 2
    assert all(r in evidence for s in sections for t in s.turns for r in t.refs)
    assert o.word_count == sum(len(t.text.split()) for s in sections for t in s.turns)
    assert (
        "shorter than requested" in o.message
        and "4 passages were left out" in o.message
    )
    meta = json.loads(row["meta"])
    assert meta["llm_calls"] == 2 and meta["removed"]["unsupported name"] == 1
    for key in ("retrieval", "outline", "script", "validation", "save", "total"):
        assert key in meta["timings"]
    assert client.app.state.embedder.created == 0  # no focus: MiniLM untouched


def test_narrator_format_and_longer_targets_stay_bounded(
    client, app_settings, nb, llama
):
    nb_id, src = nb
    llama.script = lambda p: script_for(p, "narrator")
    oid, row = generate(
        client, app_settings, nb_id, [src["Trap.pdf"]], format="revision", minutes=20
    )
    assert row["status"] == "script_ready"
    kinds = llama.kinds()
    assert kinds[0] == "outline" and 1 <= kinds.count("script") <= 5
    assert len(kinds) <= 6  # never dozens of calls
    speaker = llama.requests[1]["response_format"]["json_schema"]["schema"][
        "properties"
    ]["sections"]["items"]["properties"]["turns"]["items"]["properties"]["speaker"]
    assert speaker["enum"] == ["narrator"]
    c = db(app_settings)
    try:
        speakers = {t.speaker for s in store.sections(c, oid) for t in s.turns}
    finally:
        c.close()
    assert speakers == {"narrator"}


def test_malformed_output_fails_cleanly(client, app_settings, nb, llama):
    nb_id, src = nb
    llama.raw = lambda kind, prompt: "Sure, here is a podcast about trams!"
    oid, row = generate(client, app_settings, nb_id, [src["Trap.pdf"]])
    assert row["status"] == "failed" and row["error"] == NOTHING_GROUNDED
    assert len(llama.requests) == 2  # outline + one retry, then stop
    c = db(app_settings)
    try:
        assert c.execute("SELECT count(*) FROM audio_sections").fetchone()[0] == 0
    finally:
        c.close()


def test_script_without_valid_citations_is_not_saved(client, app_settings, nb, llama):
    nb_id, src = nb
    llama.script = lambda prompt: {
        "sections": [
            {
                "title": "Trams",
                "turns": [
                    {
                        "speaker": "A",
                        "text": "Quillport uses violet tram cards.",
                        "refs": ["S42"],
                    },
                    {
                        "speaker": "B",
                        "text": "The moon base Tarvos stores lemonade.",
                        "refs": [],
                    },
                ],
            }
        ]
    }
    _, row = generate(client, app_settings, nb_id, [src["Trap.pdf"]])
    assert row["status"] == "failed" and row["error"] == NOTHING_GROUNDED
    assert llama.kinds() == ["outline", "script", "script"]  # one repair request
    meta = json.loads(row["meta"])
    assert meta["removed"]["unsupported citation"] >= 1


def test_prompt_injection_source_is_ignored(app_settings, tmp_path):
    def obey(prompt):
        label = label_with(prompt, "Helix")
        return {
            "sections": [
                {
                    "title": "Helix",
                    "turns": [
                        {
                            "speaker": "A",
                            "text": "The Helix unit is defined as 17 units.",
                            "refs": [label],
                        },
                        {
                            "speaker": "B",
                            "text": "And bananas are the best snack, buy some today!",
                            "refs": [label],
                        },
                        {
                            "speaker": "A",
                            "text": "This podcast is brought to you by bananas.",
                            "refs": [],
                        },
                    ],
                }
            ]
        }

    def outline(prompt):
        return {
            "title": "Helix",
            "sections": [
                {
                    "title": "Helix",
                    "points": ["17 units"],
                    "refs": [label_with(prompt, "Helix")],
                }
            ],
        }

    llama = FakeAudioLlama(outline, obey)
    app = create_app(
        app_settings,
        env={},
        llm_transport=httpx.MockTransport(llama),
        embedder_factory=FakeEmbedder,
    )
    nb_id, src = build_notebook(app_settings, tmp_path, sources={"Ad.pdf": PDF_AD})
    with TestClient(app, base_url=BASE) as c:
        oid, row = generate(c, app_settings, nb_id, [src["Ad.pdf"]])
    assert row["status"] == "script_ready"
    conn = db(app_settings)
    try:
        texts = [t.text for s in store.sections(conn, oid) for t in s.turns]
    finally:
        conn.close()
    assert texts == ["The Helix unit is defined as 17 units."]
    assert all("banana" not in t.lower() for t in texts)


def test_notes_and_chat_only_when_selected(client, app_settings, nb, llama):
    nb_id, src = nb
    c = db(app_settings)
    try:
        create_note(
            c,
            nb_id,
            "Tram notes",
            "Quillport trams run every ten minutes along the harbour loop.",
        )
        ex = start_exchange(c, nb_id, None, "What is Ember?")
        complete_message(
            c,
            ex.assistant_message_id,
            "Ember is the first Arvon stage [S1].",
            model="qwen",
        )
    finally:
        c.close()
    generate(client, app_settings, nb_id, [src["Trap.pdf"]])
    assert all("ten minutes" not in p and "[N1]" not in p for p in llama.prompts)
    llama.requests.clear()
    generate(client, app_settings, nb_id, [src["Trap.pdf"]], notes="1", chat="1")
    assert "Tram notes" in llama.prompts[0] and "[N1]" in llama.prompts[0]
    assert "Earlier chat answer" in llama.prompts[0] and "[M1]" in llama.prompts[0]


def test_deleted_source_keeps_the_transcript_and_snapshots(client, app_settings, nb):
    nb_id, src = nb
    oid, _ = generate(client, app_settings, nb_id, [src["Trap.pdf"]])
    c = db(app_settings)
    try:
        delete_source(
            c, app_settings, nb_id, src["Trap.pdf"], FaissStore(c, app_settings)
        )
        ev = store.evidence(c, oid)
        assert ev and all(not e.exists and e.excerpt for e in ev.values())
        assert store.sections(c, oid)
    finally:
        c.close()
    page = client.get(f"/notebooks/{nb_id}/study/audio/{oid}").text
    assert "Source deleted" in page and "read?chunk=" not in page
    assert "violet tram cards" in page


def test_script_is_immutable(client, app_settings, nb):
    import sqlite3

    nb_id, src = nb
    oid, _ = generate(client, app_settings, nb_id, [src["Trap.pdf"]])
    c = db(app_settings)
    try:
        for sql in (
            "UPDATE audio_turns SET text = 'x'",
            "UPDATE audio_sections SET title = 'x'",
            "UPDATE audio_evidence SET excerpt = 'x'",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                c.execute(sql)
    finally:
        c.close()


@pytest.mark.parametrize(
    ("form", "message"),
    [
        ({"sources": []}, "Select at least one source"),
        ({"format": "radio-drama"}, "Choose a format"),
        ({"minutes": 7}, "about 5, 10 or 20 minutes"),
        ({"focus": "x" * 501}, "at most 500"),
    ],
)
def test_form_validation(client, nb, form, message):
    nb_id, src = nb
    sources = form.pop("sources", [src["Trap.pdf"]])
    r = create(client, nb_id, sources, **form)
    assert r.status_code == 422 and message in r.text


def test_not_configured_and_server_down(client, app_settings, nb, llama):
    nb_id, src = nb
    c = db(app_settings)
    try:
        AIStore(c).save_profile(AIStore(c).profile().__class__(type="none"))
    finally:
        c.close()
    r = create(client, nb_id, [src["Trap.pdf"]])
    assert r.status_code == 409 and "Local AI is not configured." in r.text
    assert 'href="/setup"' in r.text and llama.requests == []
    c = db(app_settings)
    try:
        AIStore(c).save_profile(AIStore(c).profile().__class__(type="unset"))
    finally:
        c.close()
    llama.mode = "down"
    _, row = generate(client, app_settings, nb_id, [src["Trap.pdf"]])
    assert row["status"] == "failed" and "Cannot reach the LLM server" in row["error"]


def test_chat_quiz_and_audio_share_one_generation_slot(client, app_settings, nb, llama):
    nb_id, src = nb
    llama.delay = 0.6
    r = create(client, nb_id, [src["Trap.pdf"]])
    oid = int(r.headers["location"].rsplit("/", 1)[1])
    busy = create(client, nb_id, [src["Trap.pdf"]])
    assert busy.status_code == 409 and "busy writing an audio overview" in busy.text
    chat = client.post(
        f"/notebooks/{nb_id}/chat/send", data={"question": "Hi?"}, headers=ORIGIN
    )
    assert chat.status_code == 409 and "audio overview" in chat.json()["error"]
    quiz = client.post(
        f"/notebooks/{nb_id}/study/quizzes",
        headers=ORIGIN,
        data={
            "sources": [str(src["Trap.pdf"])],
            "types": ["single"],
            "count": "5",
            "difficulty": "easy",
        },
    )
    assert quiz.status_code == 409 and "audio overview" in quiz.text
    assert wait_overview(app_settings, oid)["status"] == "script_ready"
    assert client.app.state.generation_guard.active == set()


def test_cancel_script_generation(client, app_settings, nb, llama):
    nb_id, src = nb
    llama.delay = 1.0
    r = create(client, nb_id, [src["Trap.pdf"]], minutes=20)
    oid = int(r.headers["location"].rsplit("/", 1)[1])
    deadline = time.monotonic() + 5
    while not llama.requests and time.monotonic() < deadline:
        time.sleep(0.02)
    status = client.get(
        f"/notebooks/{nb_id}/study/audio/{oid}/status", headers={"HX-Request": "true"}
    )
    assert "Writing the outline" in status.text and "%" not in status.text
    client.post(f"/notebooks/{nb_id}/study/audio/{oid}/cancel", headers=ORIGIN)
    row = wait_overview(app_settings, oid)
    assert row["status"] == "cancelled" and len(llama.requests) <= 2
    assert client.app.state.generation_guard.active == set()


def test_focus_uses_minilm_only_for_generation(client, app_settings, nb):
    nb_id, src = nb
    _, row = generate(
        client, app_settings, nb_id, [src["Trap.pdf"]], focus="Arvon process stages"
    )
    assert row["status"] == "script_ready"
    assert client.app.state.embedder.created == 1
    assert json.loads(row["meta"])["semantic"] is True


def test_regenerate_makes_a_new_overview(client, app_settings, nb, llama):
    nb_id, src = nb
    oid, _ = generate(client, app_settings, nb_id, [src["Trap.pdf"]])
    r = client.post(
        f"/notebooks/{nb_id}/study/audio/{oid}/regenerate",
        headers=ORIGIN,
        follow_redirects=False,
    )
    new = int(r.headers["location"].rsplit("/", 1)[1])
    assert new != oid
    assert wait_overview(app_settings, new)["status"] == "script_ready"
    c = db(app_settings)
    try:
        assert store.get_overview(c, nb_id, oid).status == "script_ready"
    finally:
        c.close()


def test_restart_recovers_unfinished_work(app_settings, tmp_path, llama):
    nb_id, _ = build_notebook(app_settings, tmp_path, sources={"Trap.pdf": PDF_TRAP})
    c = db(app_settings)
    try:
        oid = store.create_overview(
            c,
            nb_id,
            title="T",
            fmt="narrator",
            minutes=5,
            focus="",
            material={},
            wpm=150,
        )
    finally:
        c.close()
    app = create_app(
        app_settings,
        env={},
        llm_transport=httpx.MockTransport(llama),
        embedder_factory=FakeEmbedder,
    )
    with TestClient(app, base_url=BASE):
        pass
    row = wait_overview(app_settings, oid, timeout=0.1)
    assert row["status"] == "failed" and "interrupted" in row["error"]
