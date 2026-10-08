"""Settings persistence, precedence, validation and connection checks."""

import httpx
import pytest

from litenotebook.services.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    RAG_TOP_K,
    SettingsError,
    SettingsService,
    check_connection,
)

SECRET = "sk-test-SECRET-4242"


def test_defaults(conn):
    s = SettingsService(conn, env={}).effective()
    assert s.llm_base_url == "http://127.0.0.1:8080/v1"
    assert s.llm_model == "Qwen/Qwen3-4B-GGUF:Q4_K_M"
    assert s.llm_api_key == "" and s.rag_top_k == 8
    assert set(s.sources.values()) == {"default"}


def test_persistence_in_sqlite(conn):
    SettingsService(conn, env={}).update(
        llm_base_url="http://localhost:9000/v1/", llm_model="m", rag_top_k="5"
    )
    s = SettingsService(conn, env={}).effective()  # fresh service, same DB
    assert (s.llm_base_url, s.llm_model, s.rag_top_k) == (
        "http://localhost:9000/v1",
        "m",
        5,
    )
    assert s.sources[LLM_BASE_URL] == "db"
    stored = dict(conn.execute("SELECT key, value FROM settings").fetchall())
    assert stored["rag.top_k"] == "5"


def test_environment_overrides_db_and_is_not_persisted(conn):
    service = SettingsService(conn, env={})
    service.update(llm_base_url="http://localhost:9000/v1")
    env = {
        "LITENOTEBOOK_LLM_BASE_URL": "http://127.0.0.1:7000/v1",
        "LITENOTEBOOK_LLM_API_KEY": SECRET,
        "LITENOTEBOOK_TOP_K": "3",
    }
    s = SettingsService(conn, env=env).effective()
    assert (
        s.llm_base_url == "http://127.0.0.1:7000/v1"
        and s.sources[LLM_BASE_URL] == "env"
    )
    assert s.llm_api_key == SECRET and s.sources[LLM_API_KEY] == "env"
    assert s.rag_top_k == 3 and s.sources[RAG_TOP_K] == "env"
    values = [r[0] for r in conn.execute("SELECT value FROM settings")]
    assert SECRET not in values and "http://127.0.0.1:7000/v1" not in values


def test_invalid_environment_value(conn):
    with pytest.raises(SettingsError, match="LITENOTEBOOK_TOP_K"):
        SettingsService(conn, env={"LITENOTEBOOK_TOP_K": "999"}).effective()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_base_url", "ftp://host/v1"),
        ("llm_base_url", "http://user:pw@127.0.0.1:8080/v1"),
        ("llm_base_url", "http://127.0.0.1:8080/v1?x=1"),
        ("llm_base_url", "not a url"),
        ("llm_base_url", ""),
        ("rag_top_k", "0"),
        ("rag_top_k", "21"),
        ("rag_top_k", "eight"),
        ("llm_model", "x" * 201),
        ("llm_model", "bad\nmodel"),
        ("llm_api_key", "has space"),
    ],
)
def test_validation(conn, field, value):
    with pytest.raises(SettingsError):
        SettingsService(conn, env={}).update(**{field: value})


def test_invalid_update_is_all_or_nothing(conn):
    service = SettingsService(conn, env={})
    with pytest.raises(SettingsError) as exc:
        service.update(llm_model="good", rag_top_k="0")
    assert set(exc.value.errors) == {"rag.top_k"}
    assert service.effective().llm_model == "Qwen/Qwen3-4B-GGUF:Q4_K_M"


def test_api_key_kept_when_blank_and_cleared_explicitly(conn):
    service = SettingsService(conn, env={})
    service.update(llm_api_key=SECRET)
    assert service.update(llm_api_key="").llm_api_key == SECRET
    assert service.update(llm_api_key=None).llm_api_key == SECRET
    assert service.update(clear_api_key=True).llm_api_key == ""


def test_api_key_never_in_repr_or_public(conn):
    s = SettingsService(conn, env={}).update(llm_api_key=SECRET)
    assert SECRET not in repr(s)
    assert SECRET not in str(s.public())
    assert s.public()["api_key_configured"] is True


def _transport(models=("Qwen/Qwen3-4B-GGUF:Q4_K_M",), status=200, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        assert request.url.path.endswith("/models")
        return httpx.Response(status, json={"data": [{"id": m} for m in models]})

    return httpx.MockTransport(handler)


def test_check_connection_success_sends_api_key(conn):
    s = SettingsService(conn, env={}).update(llm_api_key=SECRET)
    seen = []
    report = check_connection(s, transport=_transport(seen=seen))
    assert report.ok and report.model_found
    assert report.message == "Connected — Qwen/Qwen3-4B-GGUF:Q4_K_M available"
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}"
    assert seen[0].method == "GET"  # discovery only, never generation


def test_check_connection_model_missing(conn):
    s = SettingsService(conn, env={}).effective()
    report = check_connection(s, transport=_transport(models=("other-model",)))
    assert not report.ok and report.reachable and report.model_found is False
    assert "not found" in report.message and "other-model" in report.message


def test_check_connection_no_model_selected(conn):
    s = SettingsService(conn, env={}).update(llm_model="")
    report = check_connection(s, transport=_transport())
    assert report.ok and report.model_found is None


def test_check_connection_unreachable(conn):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    report = check_connection(
        SettingsService(conn, env={}).effective(), transport=httpx.MockTransport(refuse)
    )
    assert not report.ok and not report.reachable
    assert "llama-server" in report.message


def test_check_connection_http_error_is_reachable(conn):
    report = check_connection(
        SettingsService(conn, env={}).effective(), transport=_transport(status=500)
    )
    assert not report.ok and report.reachable and "HTTP 500" in report.message


@pytest.mark.parametrize("body", [[{"id": "x"}], {"data": "nope"}, {"object": "list"}])
def test_check_connection_non_openai_models_response(conn, body):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    report = check_connection(
        SettingsService(conn, env={}).effective(), transport=transport
    )
    assert not report.ok and report.reachable and "/v1" in report.message
