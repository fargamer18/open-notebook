"""CLI additions in Phase 2: `serve` and settings-aware `llm check`."""

import sys
import types

import pytest

from litenotebook import cli
from litenotebook.db import open_db
from litenotebook.services.settings import ConnectionReport, SettingsService


@pytest.fixture
def fake_uvicorn(monkeypatch):
    calls = []
    module = types.SimpleNamespace(run=lambda app, **kw: calls.append((app, kw)))
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    return calls


def test_serve_defaults_to_loopback_5056(tmp_path, fake_uvicorn, capsys):
    assert cli.main(["--data-dir", str(tmp_path), "serve"]) == 0
    [(app, kw)] = fake_uvicorn
    assert (kw["host"], kw["port"]) == ("127.0.0.1", 5056)
    assert kw["proxy_headers"] is False
    assert app.state.settings.data_dir == tmp_path
    assert "http://127.0.0.1:5056" in capsys.readouterr().out


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_serve_refuses_non_loopback(tmp_path, fake_uvicorn, capsys, host):
    assert cli.main(["--data-dir", str(tmp_path), "serve", "--host", host]) == 1
    assert fake_uvicorn == []
    assert "loopback" in capsys.readouterr().err


def test_serve_custom_port(tmp_path, fake_uvicorn):
    assert cli.main(["--data-dir", str(tmp_path), "serve", "--port", "6060"]) == 0
    assert fake_uvicorn[0][1]["port"] == 6060


def test_llm_check_uses_saved_settings(tmp_path, monkeypatch, capsys):
    conn = open_db(tmp_path / "litenotebook.sqlite3")
    SettingsService(conn, env={}).update(llm_base_url="http://localhost:9100/v1")
    conn.close()
    seen = {}

    def fake_check(settings):
        seen["url"] = settings.llm_base_url
        return ConnectionReport(
            True, True, ["m"], "m", True, "Connected — m available", 1.0
        )

    monkeypatch.setattr(cli, "check_connection", fake_check)
    assert cli.main(["--data-dir", str(tmp_path), "llm", "check"]) == 0
    assert seen["url"] == "http://localhost:9100/v1"
    assert "Connected — m available" in capsys.readouterr().out
