"""Phase 14: Audio Overview pages, speech synthesis, player, downloads,
lazy loading, safety and Settings → Speech (fake LLM, fake speech)."""

import json
import re
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from audio_fixtures import (
    PDF_TRAP,
    FakeAudioLlama,
    FakeSpeech,
    install_fake_engine,
    install_fake_voices,
    wait_overview,
)
from conftest import FakeEmbedder
from litenotebook.ai.store import AIStore
from litenotebook.config import load_settings
from litenotebook.db import connect
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source
from litenotebook.study import audio_store as store
from litenotebook.study.audio_model import Script, ScriptSection, ScriptTurn
from litenotebook.study.audio_render import safe_filename, wav_info
from litenotebook.study.model import EvidenceItem
from litenotebook.web.app import create_app
from quiz_fixtures import BASE, ORIGIN, build_notebook


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
    return FakeAudioLlama()


@pytest.fixture
def speaker():
    return FakeSpeech()


@pytest.fixture
def client(app_settings, llama, speaker):
    CountingEmbedder.created = 0
    provided = {"calls": 0}

    def provider(settings):
        from litenotebook.ai.speech import installed_engine

        provided["calls"] += 1
        return speaker if installed_engine(settings) else None

    app = create_app(
        app_settings,
        env={},
        llm_transport=httpx.MockTransport(llama),
        embedder_factory=CountingEmbedder,
        speech_provider=provider,
    )
    app.state.provided = provided
    with TestClient(app, base_url=BASE) as c:
        yield c


def saved_script(
    settings, nb_id, sid, fmt="two_person", title="Trams overview", texts=None
):
    """A script_ready overview saved directly (no LLM)."""
    c = connect(settings.db_path)
    try:
        chunk = c.execute(
            "SELECT id, page_no, text FROM chunks WHERE source_id = ? "
            "ORDER BY ord LIMIT 1",
            (sid,),
        ).fetchone()
        oid = store.create_overview(
            c,
            nb_id,
            title=title,
            fmt=fmt,
            minutes=5,
            focus="",
            material={"sources": [{"id": sid, "filename": "Trap.pdf"}]},
            wpm=150,
        )
        store.set_stage(c, oid, "generating_script")
        ev = EvidenceItem(
            "S1",
            "source",
            chunk["text"],
            "Trap.pdf",
            source_id=sid,
            chunk_id=chunk["id"],
            page_no=chunk["page_no"],
            chunk_ord=0,
        )
        a, b = ("A", "B") if fmt in ("two_person", "deep_dive") else ("narrator",) * 2
        texts = texts or [
            "Quillport uses violet tram cards.",
            "How many rides fit on one?",
            "Each card holds twelve rides.",
        ]
        script = Script(
            title,
            [
                ScriptSection(
                    "Trams",
                    [ScriptTurn(a, texts[0], ["S1"]), ScriptTurn(b, texts[1], [])],
                ),
                ScriptSection("Rides", [ScriptTurn(a, texts[2], ["S1"])]),
            ],
            {"S1": ev},
        )
        store.save_script(c, oid, script, message=None, model="test", meta={})
        return oid
    finally:
        c.close()


@pytest.fixture
def setup(client, app_settings, tmp_path):
    nb_id, src = build_notebook(app_settings, tmp_path, sources={"Trap.pdf": PDF_TRAP})
    oid = saved_script(app_settings, nb_id, src["Trap.pdf"])
    return nb_id, src["Trap.pdf"], oid


def page(client, nb_id, oid):
    return client.get(f"/notebooks/{nb_id}/study/audio/{oid}").text


def synth(client, nb_id, oid, **voices):
    data = {"voice_a": "voice-a", "voice_b": "voice-b", **voices}
    return client.post(
        f"/notebooks/{nb_id}/study/audio/{oid}/synthesize",
        data=data,
        headers=ORIGIN,
        follow_redirects=False,
    )


def test_study_home_lists_audio_and_pages_render(client, setup):
    nb_id, _, oid = setup
    home = client.get(f"/notebooks/{nb_id}/study").text
    assert "Audio Overviews" in home and "Create audio overview" in home
    assert "Trams overview" in home and "Transcript only" in home
    nav = re.search(r'<nav class="tabs" aria-label="Notebook">(.*?)</nav>', home, re.S)
    assert "Audio" not in nav.group(1)  # no new top-level tab
    listing = client.get(f"/notebooks/{nb_id}/study/audio").text
    assert "Two-person discussion" in listing and "~5 min target" in listing
    form = client.get(f"/notebooks/{nb_id}/study/audio/new").text
    for text in (
        "<legend>Format</legend>",
        "<legend>Length</legend>",
        "Single narrator",
        "Study revision",
        "Deep dive",
        "~20 min",
        "My notes",
        "Chat history",
    ):
        assert text in form
    view = page(client, nb_id, oid)
    assert "Host A:" in view and "Host B:" in view and "Transcript" in view
    assert "Local speech is not installed." in view and "Set up speech" in view
    assert "about 1 min at 150 words/min (estimate)" in view


def test_transcript_citations_open_the_reader_at_the_passage(
    client, setup, app_settings
):
    nb_id, sid, oid = setup
    view = page(client, nb_id, oid)
    c = connect(app_settings.db_path)
    try:
        chunk = c.execute(
            "SELECT id FROM chunks WHERE source_id = ? ORDER BY ord LIMIT 1", (sid,)
        ).fetchone()[0]
    finally:
        c.close()
    link = f"/notebooks/{nb_id}/sources/{sid}/read?chunk={chunk}&amp;audio={oid}"
    assert link in view and 'title="Trap.pdf, page 1"' in view
    reader = client.get(link.replace("&amp;", "&")).text
    assert f'data-focus-kind="chunk" data-focus-id="{chunk}"' in reader
    assert "Back to audio overview" in reader


def test_synthesis_two_voices_chapters_player_and_downloads(
    client, setup, app_settings, speaker
):
    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    view = page(client, nb_id, oid)
    assert (
        "Host A voice" in view and "Host B voice" in view and "Generate audio" in view
    )
    r = synth(client, nb_id, oid)
    assert r.status_code == 303
    row = wait_overview(app_settings, oid)
    assert row["status"] == "ready"
    assert [v for _, v in speaker.calls] == ["voice-a", "voice-b", "voice-a"]
    assert all("[" not in t for t, _ in speaker.calls)  # citations never spoken
    c = connect(app_settings.db_path)
    try:
        asset = store.list_assets(c, oid)[0]
        turns = [t for s in store.sections(c, oid) for t in s.turns]
    finally:
        c.close()
    path = store.asset_path(app_settings, nb_id, oid, asset.id)
    info = wav_info(path)
    assert asset.status == "ready" and asset.duration_s == info.duration_s > 0
    assert asset.size_bytes == path.stat().st_size and asset.sample_rate == 22050
    assert [ch["title"] for ch in asset.chapters] == ["Trams", "Rides"]
    assert asset.chapters[0]["start_s"] == 0
    # chapter 2 starts after the measured end of chapter 1's last turn + a pause
    assert asset.chapters[1]["start_s"] > asset.turn_times[1][2]
    assert [t[0] for t in asset.turn_times] == [t.id for t in turns]
    assert asset.voices == {"A": "voice-a", "B": "voice-b"}
    view = page(client, nb_id, oid)
    assert f'src="/notebooks/{nb_id}/study/audio/{oid}/audio/{asset.id}"' in view
    assert 'data-seek="0.0"' in view and "Chapters" in view and "Download audio" in view
    assert 'id="turn-times"' in view
    audio = client.get(
        f"/notebooks/{nb_id}/study/audio/{oid}/audio/{asset.id}",
        headers={"Range": "bytes=0-43"},
    )
    assert audio.status_code == 206 and audio.content[:4] == b"RIFF"
    assert audio.headers["content-type"] == "audio/wav"
    full = client.get(
        f"/notebooks/{nb_id}/study/audio/{oid}/audio/{asset.id}?download=1"
    )
    disposition = full.headers["content-disposition"]
    assert (
        disposition.startswith("attachment;") and "Trams%20overview.wav" in disposition
    )
    txt = client.get(f"/notebooks/{nb_id}/study/audio/{oid}/transcript.txt")
    assert 'filename="Trams overview.txt"' in txt.headers["content-disposition"]
    assert "Host A: Quillport uses violet tram cards. [S1]" in txt.text
    assert "[S1] Trap.pdf, page 1:" in txt.text
    home = client.get(f"/notebooks/{nb_id}/study").text
    assert "Audio ready" in home
    # re-synthesizing the same script makes a new asset; the transcript is unchanged
    synth(client, nb_id, oid, voice_a="voice-b", voice_b="voice-a")
    wait_overview(app_settings, oid)
    c = connect(app_settings.db_path)
    try:
        assert len(store.list_assets(c, oid)) == 2
        assert [t.text for s in store.sections(c, oid) for t in s.turns] == [
            t.text for t in turns
        ]
    finally:
        c.close()
    assert "Audio versions" in page(client, nb_id, oid)


def test_one_voice_needs_an_explicit_fallback(client, setup, app_settings, speaker):
    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings, ("voice-a",))
    view = page(client, nb_id, oid)
    assert "Only one voice is installed" in view
    r = synth(client, nb_id, oid, voice_b="voice-a")
    assert r.status_code == 422 and "same voice" in r.text and speaker.calls == []
    r = synth(client, nb_id, oid, voice_b="voice-a", same_voice="1")
    assert r.status_code == 303
    assert wait_overview(app_settings, oid)["status"] == "ready"
    r = synth(client, nb_id, oid, voice_a="nope")
    assert r.status_code == 422 and "installed voice" in r.text


def test_narrator_uses_one_voice(client, app_settings, tmp_path, speaker):
    nb_id, src = build_notebook(app_settings, tmp_path, sources={"Trap.pdf": PDF_TRAP})
    oid = saved_script(app_settings, nb_id, src["Trap.pdf"], fmt="narrator")
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    view = page(client, nb_id, oid)
    assert "Narrator voice" in view and "Host B voice" not in view
    client.post(
        f"/notebooks/{nb_id}/study/audio/{oid}/synthesize",
        data={"voice_narrator": "voice-b"},
        headers=ORIGIN,
    )
    assert wait_overview(app_settings, oid)["status"] == "ready"
    assert {v for _, v in speaker.calls} == {"voice-b"}


def test_engine_failure_keeps_the_transcript(client, setup, app_settings, speaker):
    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    speaker.fail_at = 2
    synth(client, nb_id, oid)
    row = wait_overview(app_settings, oid)
    assert row["status"] == "script_ready"
    c = connect(app_settings.db_path)
    try:
        asset = store.list_assets(c, oid)[0]
        assert store.sections(c, oid)  # transcript intact
    finally:
        c.close()
    assert asset.status == "failed" and "exit code 1" in asset.error
    assert "transcript is kept" in asset.error
    folder = store.overview_dir(app_settings, nb_id, oid)
    assert not list(folder.glob("*.wav")) and not list(folder.glob(".*"))  # no partials
    assert "exit code 1" in page(client, nb_id, oid)
    assert not client.app.state.speech_lock.locked()


def test_cancel_synthesis_keeps_transcript(client, setup, app_settings, speaker):
    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    speaker.delay = 1.0
    synth(client, nb_id, oid)
    deadline = time.monotonic() + 5
    while not speaker.calls and time.monotonic() < deadline:
        time.sleep(0.02)
    status = client.get(
        f"/notebooks/{nb_id}/study/audio/{oid}/status", headers={"HX-Request": "true"}
    ).text
    assert "Generating audio: turn 1 of 3" in status and "Cancel" in status
    busy = synth(client, nb_id, oid)
    assert busy.status_code == 409  # one speech job at a time
    client.post(f"/notebooks/{nb_id}/study/audio/{oid}/cancel", headers=ORIGIN)
    row = wait_overview(app_settings, oid)
    assert row["status"] == "script_ready"
    c = connect(app_settings.db_path)
    try:
        assert store.list_assets(c, oid)[0].status == "cancelled"
    finally:
        c.close()
    assert len(speaker.calls) <= 2
    assert not client.app.state.speech_lock.locked()


def test_restart_during_synthesis_keeps_transcript(app_settings, tmp_path, llama):
    nb_id, src = build_notebook(app_settings, tmp_path, sources={"Trap.pdf": PDF_TRAP})
    oid = saved_script(app_settings, nb_id, src["Trap.pdf"])
    c = connect(app_settings.db_path)
    try:
        aid = store.start_asset(c, oid, "x", {"A": "a", "B": "b"})
        partial = store.asset_path(app_settings, nb_id, oid, aid)
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"half")
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
    assert row["status"] == "script_ready" and not partial.exists()
    c = connect(app_settings.db_path)
    try:
        assert store.list_assets(c, oid)[0].status == "failed"
        assert store.sections(c, oid)
    finally:
        c.close()


def test_history_transcript_and_playback_start_nothing(
    client, setup, app_settings, llama, speaker
):
    import sys

    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    synth(client, nb_id, oid)
    wait_overview(app_settings, oid)
    calls_after_synth = len(speaker.calls)
    provided = client.app.state.provided["calls"]
    c = connect(app_settings.db_path)
    try:
        aid = store.list_assets(c, oid)[0].id
    finally:
        c.close()
    for url in (
        f"/notebooks/{nb_id}/study",
        f"/notebooks/{nb_id}/study/audio",
        f"/notebooks/{nb_id}/study/audio/new",
        f"/notebooks/{nb_id}/study/audio/{oid}",
        f"/notebooks/{nb_id}/study/audio/{oid}/transcript.txt",
        f"/notebooks/{nb_id}/study/audio/{oid}/audio/{aid}",
    ):
        assert client.get(url).status_code == 200, url
    assert CountingEmbedder.created == 0 and client.app.state.embedder.created == 0
    assert llama.paths == [] and len(speaker.calls) == calls_after_synth
    assert client.app.state.provided["calls"] == provided  # no provider was built
    assert client.app.state.local_ai.runtime.running() is False
    assert "sherpa_onnx" not in sys.modules and "onnxruntime" not in sys.modules


def test_existing_audio_works_with_ai_disabled(client, setup, app_settings):
    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    synth(client, nb_id, oid)
    wait_overview(app_settings, oid)
    c = connect(app_settings.db_path)
    try:
        AIStore(c).save_profile(AIStore(c).profile().__class__(type="none"))
        aid = store.list_assets(c, oid)[0].id
    finally:
        c.close()
    assert "<audio" in page(client, nb_id, oid)
    assert (
        client.get(f"/notebooks/{nb_id}/study/audio/{oid}/audio/{aid}").status_code
        == 200
    )


def test_titles_and_transcripts_are_escaped(client, app_settings, tmp_path):
    nb_id, src = build_notebook(app_settings, tmp_path, sources={"Trap.pdf": PDF_TRAP})
    evil = "<script>alert(1)</script>"
    oid = saved_script(
        app_settings,
        nb_id,
        src["Trap.pdf"],
        title=f"T {evil}",
        texts=[f"A {evil}", f"B {evil}", f"C {evil}"],
    )
    for url in (
        f"/notebooks/{nb_id}/study",
        f"/notebooks/{nb_id}/study/audio",
        f"/notebooks/{nb_id}/study/audio/{oid}",
    ):
        text = client.get(url).text
        assert "<script>alert" not in text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    txt = client.get(f"/notebooks/{nb_id}/study/audio/{oid}/transcript.txt")
    assert txt.headers["content-type"].startswith("text/plain")
    assert 'filename="T scriptalert1script.txt"' in txt.headers["content-disposition"]


def test_safe_download_names():
    assert (
        safe_filename("Lecture 1: Eigen/vectors?*", ".wav")
        == "Lecture 1 Eigenvectors.wav"
    )
    assert safe_filename("CON", ".txt") == "_CON.txt"
    assert safe_filename("...", ".wav") == "audio-overview.wav"
    assert safe_filename("Ünïcødé — notes", ".txt") == "Ünïcødé notes.txt"
    assert len(safe_filename("x" * 300, ".wav")) <= 64


def test_paths_come_from_ids_only(app_settings):
    p = store.asset_path(app_settings, 3, 7, 11)
    assert p.parts[-3:] == ("audio", "7", "11.wav")
    with pytest.raises(ValueError):
        store.asset_path(app_settings, 3, "7/../../x", 11)  # type: ignore[arg-type]


def test_delete_needs_confirmation_and_removes_files(client, setup, app_settings):
    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    synth(client, nb_id, oid)
    wait_overview(app_settings, oid)
    folder = store.overview_dir(app_settings, nb_id, oid)
    assert list(folder.glob("*.wav"))
    assert 'data-confirm-action="/notebooks/' in page(client, nb_id, oid)
    bad = client.post(
        f"/notebooks/{nb_id}/study/audio/{oid}/delete",
        data={"confirm": "x"},
        headers=ORIGIN,
    )
    assert bad.status_code == 422 and folder.exists()
    ok = client.post(
        f"/notebooks/{nb_id}/study/audio/{oid}/delete",
        data={"confirm": str(oid)},
        headers=ORIGIN,
        follow_redirects=True,
    )
    assert "Audio overview deleted" in ok.text and not folder.exists()
    assert client.get(f"/notebooks/{nb_id}/study/audio/{oid}").status_code == 404


def test_deleted_source_keeps_audio(client, setup, app_settings):
    nb_id, sid, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    synth(client, nb_id, oid)
    wait_overview(app_settings, oid)
    c = connect(app_settings.db_path)
    try:
        delete_source(c, app_settings, nb_id, sid, FaissStore(c, app_settings))
    finally:
        c.close()
    view = page(client, nb_id, oid)
    assert "<audio" in view and "Source deleted" in view
    assert re.search(r"<button[^>]*disabled>Open in PDF</button>", view)


def test_scoping_across_notebooks(client, setup, app_settings, tmp_path):
    nb_id, _, oid = setup
    (tmp_path / "x").mkdir()
    other, _ = build_notebook(
        app_settings, tmp_path / "x", name="X", sources={"x.pdf": PDF_TRAP}
    )
    for url in (
        f"/notebooks/{other}/study/audio/{oid}",
        f"/notebooks/{other}/study/audio/{oid}/transcript.txt",
        f"/notebooks/{other}/study/audio/{oid}/audio/1",
    ):
        assert client.get(url).status_code == 404, url


def test_settings_speech_section_install_flow_and_voice_test(
    client, app_settings, speaker
):
    settings_page = client.get("/settings/ai").text
    assert 'id="speech"' in settings_page and "Not installed" in settings_page
    assert (
        "Install speech engine" in settings_page
        and "Kristin (US English)" in settings_page
    )
    assert "espeak-ng (GPL-3.0)" in settings_page
    install_fake_engine(app_settings)
    install_fake_voices(app_settings, ("voice-a",))
    settings_page = client.get("/settings/ai").text
    assert "Ready" in settings_page and "Test voice" in settings_page
    r = client.post(
        "/settings/ai/test-voice",
        data={"voice_id": "voice-a"},
        headers={**ORIGIN, "HX-Request": "true"},
    )
    assert r.status_code == 200
    job_id = re.search(r"/ai/jobs/([0-9a-f]+)", r.text)
    deadline = time.monotonic() + 10
    while True:
        job = client.get(
            f"/ai/jobs/{job_id.group(1)}", headers={"Accept": "application/json"}
        ).json()
        if job["state"] != "running" or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert job["state"] == "done", job
    assert speaker.calls[-1][0].startswith("This is a short test of local speech")
    # the sample route only serves catalog voices; voice-a is not one
    assert client.get("/settings/ai/speech/sample/voice-a").status_code == 404
    assert client.get("/settings/ai/speech/sample/..%2F..%2Fetc").status_code == 404
    r = client.post(
        "/settings/ai/remove-voice", data={"voice_id": "voice-a"}, headers=ORIGIN
    )
    assert "not confirmed" in r.text
    r = client.post(
        "/settings/ai/remove-voice",
        data={"voice_id": "voice-a", "confirm": "voice-a"},
        headers=ORIGIN,
        follow_redirects=True,
    )
    assert (
        r.status_code == 200
        and "No voices installed" in client.get("/settings/ai").text
    )
    components = client.get("/settings/ai").text
    assert 'Speech</strong> <span class="muted">(optional)</span>' in components


def test_generate_audio_without_speech_explains_setup(client, setup):
    nb_id, _, oid = setup
    r = synth(client, nb_id, oid)
    assert r.status_code == 409 and "Local speech is not installed." in r.text


def test_wizard_does_not_ask_for_speech(client):
    for step in (
        "welcome",
        "hardware",
        "embeddings",
        "generation",
        "model",
        "test",
        "ready",
    ):
        text = client.get(f"/setup?step={step}").text
        assert "Install speech" not in text


def test_audio_json_chapters_are_valid(client, setup, app_settings):
    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    synth(client, nb_id, oid)
    wait_overview(app_settings, oid)
    view = page(client, nb_id, oid)
    raw = re.search(
        r'<script type="application/json" id="turn-times">(.*?)</script>', view, re.S
    ).group(1)
    times = json.loads(raw)
    assert len(times) == 3 and all(t[1] < t[2] for t in times)


def test_csp_allows_same_origin_media_only(client, setup):
    nb_id, _, oid = setup
    csp = client.get(f"/notebooks/{nb_id}/study/audio/{oid}").headers[
        "content-security-policy"
    ]
    assert "media-src 'self'" in csp and "default-src 'none'" in csp


def test_export_page_shows_audio_size(client, setup, app_settings):
    nb_id, _, oid = setup
    install_fake_engine(app_settings)
    install_fake_voices(app_settings)
    synth(client, nb_id, oid)
    wait_overview(app_settings, oid)
    text = client.get(f"/notebooks/{nb_id}/export").text
    assert "Audio Overviews" in text and "of WAV files" in text and "Quizzes" in text
