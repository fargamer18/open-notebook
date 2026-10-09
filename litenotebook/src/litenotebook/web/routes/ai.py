"""First-run AI setup wizard and Settings → Local AI.

Page views only read stored state: nothing here scans hardware, starts
llama-server, loads MiniLM or touches the network unless the user presses a
button for it. Long actions (downloads, installs, starting, tests) run as
background jobs that the page polls; the same ``LocalAI`` methods back the
``litenotebook ai …`` CLI.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from litenotebook.ai import catalog as cat
from litenotebook.ai import embeddings as emb
from litenotebook.ai.download import DownloadError, Progress, human
from litenotebook.ai.jobs import Job, JobManager
from litenotebook.ai.runtimes import discover_existing
from litenotebook.ai.service import LocalAI, SetupError, platform_name
from litenotebook.ai.store import SETUP_STEPS, AIStore
from litenotebook.db import connect
from litenotebook.platform_utils.system import open_folder
from litenotebook.web.deps import app_settings, get_db, is_htmx, redirect, render
from litenotebook.web.navigation import notify

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]
RATING_LABELS = {
    "recommended": "runs well",
    "good": "runs",
    "slow": "slow here",
    "not_recommended": "not recommended",
}
STEP_TITLES = {
    "welcome": "Welcome",
    "hardware": "Your computer",
    "embeddings": "Semantic search",
    "generation": "AI runtime",
    "model": "Model",
    "test": "Test",
    "ready": "Ready",
}


def _ai(request: Request) -> LocalAI:
    service: LocalAI = request.app.state.local_ai
    return service


def _jobs(request: Request) -> JobManager:
    jobs: JobManager = request.app.state.ai_jobs
    return jobs


def _legacy(request: Request, conn: sqlite3.Connection) -> Any:
    from litenotebook.services.settings import SettingsService

    return SettingsService(conn, request.app.state.env).effective()


def _run_job(
    request: Request,
    kind: str,
    label: str,
    work: Callable[[sqlite3.Connection, Job], Any],
) -> Job:
    """Each job gets its own connection: it outlives the request."""
    db_path = app_settings(request).db_path

    def run(job: Job) -> Any:
        conn = connect(db_path)
        try:
            return work(conn, job)
        finally:
            conn.close()

    return _jobs(request).start(kind, label, run)


def _after_job(request: Request, job: Job, back: str) -> Response:
    if is_htmx(request):
        return render(request, "partials/ai_job.html", {"job": job, "back": back})
    return redirect(f"{back}{'&' if '?' in back else '?'}job={job.id}")


def _error(request: Request, message: str, back: str) -> Response:
    if is_htmx(request):
        response = HTMLResponse(
            f'<p class="error" role="alert">{_escape(message)}</p>', status_code=422
        )
        return notify(response, message, "error")
    return _page_for(request, back, error=message)


def _escape(text: str) -> str:
    from markupsafe import escape

    return str(escape(text))


def _page_for(request: Request, back: str, **extra: Any) -> HTMLResponse:
    conn = connect(app_settings(request).db_path)
    try:
        if back.startswith("/setup"):
            step = back.partition("step=")[2].partition("&")[0] or None
            return _wizard(request, conn, step, status_code=422, **extra)
        return _settings_page(request, conn, status_code=422, **extra)
    finally:
        conn.close()


# --- shared view model ----------------------------------------------------------------


def _common(request: Request, conn: sqlite3.Connection) -> dict[str, Any]:
    service = _ai(request)
    overview = service.overview(conn)
    hardware = overview["hardware"]
    options = service.runtime_options(conn) if hardware else []
    # Conservative: preselect only builds whose archive layout was verified.
    suitable = [
        o["entry"] for o in options if o["suitable"] and o["entry"].layout_verified
    ]
    preferred = next((r for r in suitable if r.backend == "cuda"), None) or next(
        iter(suitable), None
    )
    return {
        **overview,
        "platform": platform_name(),
        "steps": SETUP_STEPS,
        "step_titles": STEP_TITLES,
        "rating_labels": RATING_LABELS,
        "runtime_catalog": options,
        "preferred_runtime": preferred.id if preferred else None,
        "model_catalog": {m.id: m for m in cat.load_models()},
        "embedding_entry": emb.entry(),
        "active_jobs": {j.kind: j for j in _jobs(request).active()},
        "human": human,
    }


def _job_from_query(request: Request) -> Job | None:
    job_id = request.query_params.get("job")
    return _jobs(request).get(job_id) if job_id else None


# --- setup wizard ---------------------------------------------------------------------


def _wizard(
    request: Request,
    conn: sqlite3.Connection,
    step: str | None,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    store = AIStore(conn)
    state = store.setup()
    step = step if step in SETUP_STEPS else state.next_step()
    context = _common(request, conn)
    if step in ("model", "ready") and context["hardware"]:
        context["fits"] = {f.model_id: f for f in _ai(request).recommendations(conn)}
    if step == "generation":
        context["existing"] = discover_existing()
    context.update(step=step, state=state, job=_job_from_query(request), error=None)
    context.update(extra)
    return render(request, "setup.html", context, status_code=status_code)


@router.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request, conn: Db, step: str | None = None) -> HTMLResponse:
    return _wizard(request, conn, step)


@router.post("/setup/{step}/skip", response_model=None)
def setup_skip(request: Request, conn: Db, step: str) -> Response:
    if step not in SETUP_STEPS or step == "ready":
        return redirect("/setup")
    service = _ai(request)
    if step == "embeddings":
        service.skip_embeddings(conn)
    elif step == "generation":
        service.skip_generation(conn)
    else:
        AIStore(conn).mark(step, skipped=True)
    return redirect("/setup")


@router.post("/setup/welcome", response_model=None)
def setup_welcome(conn: Db) -> Response:
    AIStore(conn).mark("welcome")
    return redirect("/setup?step=hardware")


@router.post("/setup/hardware", response_model=None)
def setup_hardware(
    request: Request, conn: Db, action: Annotated[str, Form()] = "continue"
) -> Response:
    if action == "scan":
        _ai(request).hardware(conn, rescan=True)
        return redirect("/setup?step=hardware")
    if AIStore(conn).hardware() is None:
        _ai(request).hardware(conn, rescan=True)
    AIStore(conn).mark("hardware")
    return redirect("/setup?step=embeddings")


@router.post("/setup/embeddings", response_model=None)
def setup_embeddings(request: Request, conn: Db) -> Response:
    if emb.status()["installed"]:
        AIStore(conn).mark("embeddings")
        AIStore(conn).save_embeddings({"state": "installed"})
        return redirect("/setup?step=generation")
    return _install_embeddings(request, "/setup?step=embeddings")


@router.post("/setup/generation", response_model=None)
def setup_generation(
    request: Request,
    conn: Db,
    choice: Annotated[str, Form()] = "",
    runtime_id: Annotated[str, Form()] = "",
    path: Annotated[str, Form()] = "",
) -> Response:
    back = "/setup?step=generation"
    service = _ai(request)
    try:
        if choice == "managed":
            return _install_runtime(request, runtime_id, back)
        if choice == "existing":
            service.use_existing_runtime(conn, path.strip())
        elif choice == "external":
            service.use_external_server(conn)
            return redirect("/setup?step=test")
        elif choice == "none":
            service.skip_generation(conn)
            return redirect("/setup?step=ready")
        else:
            return _error(request, "Choose how LiteNotebook should run AI.", back)
    except SetupError as e:
        return _error(request, str(e), back)
    return redirect("/setup?step=model")


@router.post("/setup/model", response_model=None)
def setup_model(
    request: Request,
    conn: Db,
    choice: Annotated[str, Form()] = "",
    model_id: Annotated[str, Form()] = "",
    path: Annotated[str, Form()] = "",
    copy_file: Annotated[str, Form(alias="copy")] = "",
) -> Response:
    back = "/setup?step=model"
    try:
        if choice == "install":
            return _install_model(request, model_id, back)
        if choice == "import":
            return _import_model(request, path, bool(copy_file), back)
        if choice == "select":
            _ai(request).select_model(conn, model_id)
            return redirect("/setup?step=test")
    except SetupError as e:
        return _error(request, str(e), back)
    return _error(request, "Choose a model.", back)


@router.post("/setup/test", response_model=None)
def setup_test(request: Request) -> Response:
    return _test(request, "/setup?step=test", benchmark=False)


@router.post("/setup/finish", response_model=None)
def setup_finish(conn: Db) -> Response:
    AIStore(conn).mark("ready")
    return redirect("/settings/ai")


@router.post("/setup/reset", response_model=None)
def setup_reset(conn: Db) -> Response:
    AIStore(conn).reset_setup()  # never removes models, runtimes or notebooks
    return redirect("/setup")


# --- jobs ----------------------------------------------------------------------------


def _install_runtime(request: Request, runtime_id: str, back: str) -> Response:
    try:
        entry = cat.runtime(runtime_id)
    except cat.CatalogError:
        return _error(request, "Choose a llama.cpp build from the list.", back)

    def work(conn: sqlite3.Connection, job: Job) -> Any:
        def progress(stage: str, value: Progress | None) -> None:
            job.update(stage, value)

        try:
            _ai(request).install_runtime(
                conn, entry.id, progress=progress, cancel=job.cancel
            )
        except DownloadError as e:
            raise SetupError(str(e)) from e
        return {"next": "/setup?step=model"}

    kind = "NVIDIA CUDA" if entry.backend == "cuda" else "CPU"
    label = f"Installing llama.cpp {entry.version} ({kind})"
    return _after_job(request, _run_job(request, "runtime", label, work), back)


def _install_model(request: Request, model_id: str, back: str) -> Response:
    try:
        entry = cat.model(model_id)
    except cat.CatalogError:
        return _error(request, "Choose a model from the list.", back)

    def work(conn: sqlite3.Connection, job: Job) -> Any:
        job.update("Downloading")
        try:
            _ai(request).install_model(
                conn,
                entry.id,
                progress=lambda p: job.update("Downloading", p),
                cancel=job.cancel,
            )
        except DownloadError as e:
            raise SetupError(str(e)) from e
        return {"next": "/setup?step=test"}

    label = f"Downloading {entry.display_name} ({human(entry.size)})"
    return _after_job(request, _run_job(request, "model", label, work), back)


def _import_model(request: Request, path: str, copy: bool, back: str) -> Response:
    path = path.strip()
    if not path:
        return _error(request, "Enter the full path of a .gguf file.", back)

    def work(conn: sqlite3.Connection, job: Job) -> Any:
        job.update("Copying and checking the file" if copy else "Checking the file")
        _ai(request).import_model(conn, path, copy=copy, cancel=job.cancel)
        return {"next": "/setup?step=test"}

    return _after_job(
        request, _run_job(request, "model", "Adding a GGUF model", work), back
    )


def _install_embeddings(request: Request, back: str) -> Response:
    entry = emb.entry()

    def work(conn: sqlite3.Connection, job: Job) -> Any:
        def progress(message: dict[str, Any]) -> None:
            if message.get("file"):
                job.update(
                    f"Downloading {message['file']} "
                    f"({message.get('index')}/{message.get('total')})"
                )

        _ai(request).install_embeddings(conn, progress=progress, cancel=job.cancel)
        job.update("Loading MiniLM in its worker process")
        result = emb.test(request.app.state.embedder)
        return {"next": "/setup?step=generation", "test": result}

    label = f"Installing {entry.display_name} ({human(entry.size)})"
    return _after_job(request, _run_job(request, "embeddings", label, work), back)


def _test(request: Request, back: str, *, benchmark: bool) -> Response:
    def work(conn: sqlite3.Connection, job: Job) -> Any:
        outcome = _ai(request).test(
            conn, _legacy(request, conn), progress=job.update, benchmark=benchmark
        )
        if not outcome.ok:
            raise SetupError(outcome.message)
        return {
            "next": "/setup?step=ready" if back.startswith("/setup") else None,
            "outcome": outcome.as_dict(),
        }

    label = "Running a short benchmark" if benchmark else "Testing the configuration"
    return _after_job(request, _run_job(request, "test", label, work), back)


@router.get("/ai/jobs/{job_id}", response_model=None)
def job_status(request: Request, job_id: str, back: str = "/settings/ai") -> Response:
    job = _jobs(request).get(job_id)
    if job is None:
        return JSONResponse({"error": "No such job"}, status_code=404)
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(job.as_dict())
    if not back.startswith("/"):
        back = "/settings/ai"
    return render(request, "partials/ai_job.html", {"job": job, "back": back})


@router.post("/ai/jobs/{job_id}/cancel", response_model=None)
def job_cancel(request: Request, job_id: str, back: str = "/settings/ai") -> Response:
    _jobs(request).cancel(job_id)
    if not back.startswith("/"):
        back = "/settings/ai"
    job = _jobs(request).get(job_id)
    if is_htmx(request) and job is not None:
        return render(request, "partials/ai_job.html", {"job": job, "back": back})
    return redirect(back)


# --- Settings → Local AI -------------------------------------------------------------


def _settings_page(
    request: Request, conn: sqlite3.Connection, *, status_code: int = 200, **extra: Any
) -> HTMLResponse:
    context = _common(request, conn)
    if context["hardware"]:
        context["fits"] = {f.model_id: f for f in _ai(request).recommendations(conn)}
    context.update(job=_job_from_query(request), error=None, confirm_remove=None)
    context.update(extra)
    return render(request, "settings_ai.html", context, status_code=status_code)


@router.get("/settings/ai", response_class=HTMLResponse)
def settings_ai(request: Request, conn: Db, remove: str | None = None) -> HTMLResponse:
    target = AIStore(conn).model(remove) if remove else None
    return _settings_page(request, conn, confirm_remove=target)


@router.post("/settings/ai/{action}", response_model=None)
def settings_ai_action(
    request: Request,
    conn: Db,
    action: str,
    model_id: Annotated[str, Form()] = "",
    runtime_id: Annotated[str, Form()] = "",
    path: Annotated[str, Form()] = "",
    copy_file: Annotated[str, Form(alias="copy")] = "",
    confirm: Annotated[str, Form()] = "",
    autostart: Annotated[str, Form()] = "",
    voice_id: Annotated[str, Form()] = "",
) -> Response:
    back = "/settings/ai"
    service = _ai(request)
    try:
        if action in ("start", "restart"):

            def start(c: sqlite3.Connection, job: Job) -> Any:
                if action == "restart":
                    service.stop()
                job.update("Starting llama-server")
                status, changes = service.start_with_fallback(c, progress=job.update)
                return {"status": status, "changes": changes}

            return _after_job(
                request, _run_job(request, "test", "Starting local AI", start), back
            )
        if action == "stop":
            service.stop()
            return _done(request, "Local AI stopped")
        if action in ("test", "benchmark"):
            return _test(request, back, benchmark=action == "benchmark")
        if action == "select":
            service.select_model(conn, model_id)
            return _done(request, "Model selected")
        if action == "remove":
            # the confirmation dialog (or no-JS page) sends the id as "confirm"
            if not confirm or (model_id and confirm != model_id):
                return _error(request, "Removal not confirmed.", back)
            removed = service.remove_model(conn, confirm)
            return _done(
                request,
                "Model file deleted" if removed else "Model removed from the list",
            )
        if action == "import":
            return _import_model(request, path, bool(copy_file), back)
        if action == "install-runtime":
            return _install_runtime(request, runtime_id, back)
        if action == "use-runtime":
            service.use_existing_runtime(conn, path.strip())
            return _done(request, "llama-server checked and selected")
        if action == "install-model":
            return _install_model(request, model_id, back)
        if action == "install-embeddings":
            return _install_embeddings(request, back)
        if action == "rescan":
            service.hardware(conn, rescan=True)
            return _done(request, "Hardware scanned")
        if action == "autostart":
            store = AIStore(conn)
            profile = store.profile()
            profile.autostart = bool(autostart)
            store.save_profile(profile)
            return _done(request, "Saved")
        if action == "install-embedding-runtime":
            return _install_embedding_runtime(request, path.strip(), back)
        if action in SPEECH_ACTIONS:
            return _speech_action(request, conn, action, voice_id, confirm, back)
        if action == "open-models-folder":
            folder = service.overview(conn)["models_dir"]
            folder.mkdir(parents=True, exist_ok=True)
            opened = open_folder(folder)
            return _done(
                request,
                "Opened the models folder" if opened else f"Models folder: {folder}",
            )
    except SetupError as e:
        return _error(request, str(e), back)
    return _error(request, "Unknown action.", back)


def _install_embedding_runtime(request: Request, path: str, back: str) -> Response:
    """Packaged builds: install the optional MiniLM worker runtime from the
    release archive the user downloaded (checked against this build's
    manifest: exact file name, size and SHA-256)."""
    from litenotebook.packaging import RuntimeInstallError, install_runtime_archive

    if not path:
        return _error(request, "Enter the path of the runtime archive.", back)
    settings = app_settings(request)

    def work(c: sqlite3.Connection, job: Job) -> Any:
        try:
            install_runtime_archive(settings, Path(path), progress=job.update)
        except RuntimeInstallError as e:
            raise SetupError(str(e)) from e
        return {"next": back}

    return _after_job(
        request,
        _run_job(
            request, "embedding-runtime", "Installing the semantic search runtime", work
        ),
        back,
    )


SPEECH_ACTIONS = (
    "install-speech",
    "add-voice",
    "remove-voice",
    "test-voice",
    "remove-speech",
)


def _speech_action(
    request: Request,
    conn: sqlite3.Connection,
    action: str,
    voice_id: str,
    confirm: str,
    back: str,
) -> Response:
    """Speech is optional and installed only from here (or the Audio page's
    "Set up speech" link, which leads here)."""
    from litenotebook.ai import speech

    settings = app_settings(request)
    service = _ai(request)
    back = "/settings/ai#speech"
    if action == "install-speech":
        entry = speech.engine_entry()
        if entry is None:
            return _error(request, "No speech engine build for this computer.", back)

        def install(c: sqlite3.Connection, job: Job) -> Any:
            engine = service.install_speech_engine(
                progress=job.update, cancel=job.cancel
            )
            return {"speech": {"engine": engine.id}}

        label = f"Installing the speech engine ({human(entry.download_size)})"
        return _after_job(request, _run_job(request, "speech", label, install), back)
    if action == "add-voice":
        try:
            entry_v = speech.catalog().voice(voice_id)
        except cat.CatalogError:
            return _error(request, "Unknown voice.", back)

        def add(c: sqlite3.Connection, job: Job) -> Any:
            installed = service.install_voice(
                voice_id, progress=job.update, cancel=job.cancel
            )
            return {"speech": {"voice": installed.id}}

        label = f"Adding the voice {entry_v.display_name} ({human(entry_v.asset.size)})"
        return _after_job(request, _run_job(request, "speech", label, add), back)
    if action == "remove-voice":
        if not voice_id or confirm != voice_id:
            return _error(request, "Removal not confirmed.", back)
        removed = speech.remove_voice(settings, voice_id)
        return _done(
            request, "Voice removed" if removed else "That voice is not installed"
        )
    if action == "remove-speech":
        if confirm != "speech":
            return _error(request, "Removal not confirmed.", back)
        speech.remove_engine(settings)
        return _done(request, "Speech engine removed (voices kept)")
    # test-voice
    provider = request.app.state.speech_provider(settings)
    if provider is None or speech.voice(settings, voice_id) is None:
        return _error(request, "Install the speech engine and this voice first.", back)

    def test(c: sqlite3.Connection, job: Job) -> Any:
        job.update("Speaking the test sentence")
        try:
            result = speech.test_voice(settings, provider, voice_id)
        except speech.SpeechError as e:
            raise SetupError(str(e)) from e
        return {
            "speech": {
                "test": result,
                "sample": f"/settings/ai/speech/sample/{voice_id}",
            }
        }

    return _after_job(
        request, _run_job(request, "speech", "Testing the voice", test), back
    )


@router.get("/settings/ai/speech/sample/{voice_id}", response_model=None)
def speech_sample(request: Request, voice_id: str) -> Response:
    """The last "Test voice" recording (ids from the catalog only)."""
    from fastapi.responses import FileResponse

    from litenotebook.ai import speech

    known = {v.id for v in speech.catalog().voices}
    if voice_id not in known:
        return JSONResponse({"error": "Unknown voice"}, status_code=404)
    path = speech.speech_root(app_settings(request)) / "tests" / f"{voice_id}.wav"
    if not path.is_file():
        return JSONResponse({"error": "No sample yet"}, status_code=404)
    return FileResponse(
        path, media_type="audio/wav", headers={"Cache-Control": "no-store"}
    )


def _done(request: Request, message: str) -> Response:
    if is_htmx(request):
        response = Response(status_code=204, headers={"HX-Refresh": "true"})
        return notify(response, message)
    return redirect("/settings/ai", "ai-updated")
