"""Phase 6: the isolated embedding worker process and its parent-side manager.

These tests start real child processes using the deterministic test backend
(LITENOTEBOOK_EMBED_WORKER_BACKEND=test-fake), so they need no MiniLM model.
"""

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from litenotebook.embed import protocol as p
from litenotebook.embed._test_backend import fake_vectors
from litenotebook.embed.embedder import (
    BLOB_BYTES,
    EmbeddingError,
    EmbeddingModelUnavailableError,
    to_blob,
)
from litenotebook.embed.worker_client import (
    EmbeddingBusyError,
    EmbeddingWorkerError,
    WorkerEmbedder,
)


def process_gone(pid, timeout=10.0):
    """True once ``pid`` no longer exists and is not a zombie."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state == "Z":  # a zombie still counts as a leak
            time.sleep(0.05)
            continue
        time.sleep(0.05)
    return False


def pids(directory):
    return sorted(int(f.read_text()) for f in Path(directory).glob("worker-*.pid"))


@pytest.fixture
def make(tmp_path):
    created = []

    def factory(mode="", delay=0.0, **kw):
        env = {
            "LITENOTEBOOK_EMBED_WORKER_BACKEND": "test-fake",
            "LITENOTEBOOK_EMBED_WORKER_TEST_DIR": str(tmp_path),
            "LITENOTEBOOK_EMBED_WORKER_TEST_MODE": mode,
            "LITENOTEBOOK_EMBED_WORKER_TEST_DELAY": str(delay),
        }
        kw.setdefault("idle_seconds", 60)
        w = WorkerEmbedder(env=env, **kw)
        created.append(w)
        return w

    yield factory
    for w in created:
        w.close()


# --- startup & correctness ---------------------------------------------------------


def test_lazy_spawn_and_exactly_one_worker(make, tmp_path):
    w = make()
    assert w.status()["status"] == "stopped" and w.pid is None
    assert pids(tmp_path) == [] and not w.loaded
    vectors = w.encode(["hello world"])
    assert pids(tmp_path) == [w.pid] and w.spawns == 1 and w.loaded
    assert w.info["protocol"] == p.PROTOCOL_VERSION and w.info["dim"] == 384
    assert w.info["pid"] == w.pid and w.info["model"] == w.model_name
    np.testing.assert_array_equal(vectors, fake_vectors(["hello world"]))
    w.encode(["again"])
    assert w.spawns == 1 and len(pids(tmp_path)) == 1


def test_vectors_are_valid_ordered_and_blob_compatible(make):
    w = make(batch_size=7)
    texts = [f"text number {i} about matrices" for i in range(30)]
    vectors = w.encode(texts)
    assert vectors.shape == (30, 384) and vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(vectors, fake_vectors(texts))  # order kept
    assert all(len(to_blob(v)) == BLOB_BYTES for v in vectors)
    assert w.encode([]).shape == (0, 384)


def test_concurrent_first_requests_share_one_worker(make, tmp_path):
    w = make(mode="slow_load", delay=0.5)
    barrier = threading.Barrier(8)
    results, errors = {}, []

    def ask(i):
        barrier.wait()
        try:
            results[i] = w.encode([f"caller {i} unique words {i * 7}"])
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert errors == [] and w.spawns == 1 and len(pids(tmp_path)) == 1
    for i, vec in results.items():  # nobody got someone else's answer
        np.testing.assert_array_equal(
            vec, fake_vectors([f"caller {i} unique words {i * 7}"])
        )


def test_state_transitions(make):
    w = make(mode="slow_load", delay=0.6, idle_seconds=0.6)
    seen = set()
    t = threading.Thread(target=w.encode, args=(["x"],))
    t.start()
    while t.is_alive():
        seen.add(w.status()["status"])
        time.sleep(0.02)
    assert "starting" in seen and w.status()["status"] == "ready"
    assert "idle_seconds" in w.status() and w.status()["pid"] == w.pid
    deadline = time.monotonic() + 5
    while w.status()["status"] != "stopped" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert w.status() == {
        "status": "stopped",
        "idle_timeout_seconds": 0.6,
        "waiting": 0,
    }


def test_busy_state_while_embedding(make):
    w = make(mode="slow_embed", delay=0.6)
    w.load()
    t = threading.Thread(target=w.encode, args=(["slow"],))
    t.start()
    time.sleep(0.2)
    assert w.status()["status"] == "busy"
    t.join()
    assert w.status()["status"] == "ready"


# --- lifecycle -----------------------------------------------------------------------


def test_idle_timeout_exits_process_and_next_request_respawns(make, tmp_path):
    w = make(idle_seconds=0.5)
    w.encode(["one"])
    first = w.pid
    assert process_gone(first, timeout=5)  # the PROCESS exits, not just GC
    assert w.status()["status"] == "stopped" and w.pid is None
    vec = w.encode(["two"])
    assert w.pid != first and w.spawns == 2 and len(pids(tmp_path)) == 2
    np.testing.assert_array_equal(vec, fake_vectors(["two"]))


def test_idle_timer_never_kills_an_active_request(make):
    w = make(mode="slow_embed", delay=1.2, idle_seconds=0.3)
    vec = w.encode(["long running"])  # longer than the idle timeout
    np.testing.assert_array_equal(vec, fake_vectors(["long running"]))
    assert w.spawns == 1


def test_close_reaps_and_is_idempotent(make):
    w = make()
    w.encode(["x"])
    pid = w.pid
    w.close()
    w.close()
    assert process_gone(pid, timeout=5) and w.status()["status"] == "stopped"
    with pytest.raises(EmbeddingWorkerError, match="shut down"):
        w.encode(["after close"])


def test_worker_exits_on_stdin_eof(tmp_path):
    env = {**os.environ, "LITENOTEBOOK_EMBED_WORKER_BACKEND": "test-fake"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "litenotebook.embed.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    proc.stdin.write(p.dumps(p.request(1, "hello")))
    proc.stdin.flush()
    assert json.loads(proc.stdout.readline())["ok"] is True
    proc.stdin.close()  # what happens when the parent dies
    assert proc.wait(timeout=10) == 0


def test_worker_dies_with_an_abruptly_killed_parent(tmp_path):
    code = """
import os, sys
from litenotebook.embed.worker_client import WorkerEmbedder
w = WorkerEmbedder(env={"LITENOTEBOOK_EMBED_WORKER_BACKEND": "test-fake"})
w.encode(["x"])
print(w.pid, flush=True)
os._exit(0)   # no cleanup at all
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    child = int(out.stdout.strip())
    assert process_gone(child, timeout=10), "worker outlived its parent"


# --- failures ------------------------------------------------------------------------


def test_missing_model_is_reported_not_retried(make, tmp_path):
    w = make(mode="model_missing")
    with pytest.raises(EmbeddingModelUnavailableError, match="not available locally"):
        w.encode(["x"])
    assert w.spawns == 1 and w.status()["status"] == "failed"
    assert all(process_gone(pid, 5) for pid in pids(tmp_path))


def test_crash_before_ready_retries_once_then_fails(make, tmp_path):
    w = make(mode="exit_before_ready")
    with pytest.raises(EmbeddingWorkerError, match="stopped unexpectedly"):
        w.encode(["x"])
    assert w.spawns == 2
    assert all(process_gone(pid, 5) for pid in pids(tmp_path))


def test_crash_during_request_is_retried_once(make, tmp_path):
    w = make(mode="crash_once")
    vec = w.encode(["survives one crash"])
    np.testing.assert_array_equal(vec, fake_vectors(["survives one crash"]))
    assert w.spawns == 2 and process_gone(pids(tmp_path)[0], 5)


def test_repeated_crash_surfaces_and_later_request_recovers(make, tmp_path):
    w = make(mode="crash_on_embed")
    with pytest.raises(EmbeddingWorkerError, match=r"exit code 9\)\. Retry\."):
        w.encode(["x"])
    assert w.spawns == 2 and w.status()["status"] == "failed"
    assert "exit code 9" in w.status()["last_error"]
    w.extra_env["LITENOTEBOOK_EMBED_WORKER_TEST_MODE"] = ""  # the problem went away
    np.testing.assert_array_equal(w.encode(["ok now"]), fake_vectors(["ok now"]))
    assert w.spawns == 3
    assert all(process_gone(pid, 5) for pid in pids(tmp_path)[:2])


def test_request_timeout_kills_the_worker(make, tmp_path):
    w = make(mode="hang_on_embed", request_timeout=1.0)
    started = time.monotonic()
    with pytest.raises(EmbeddingWorkerError, match="did not answer within 1 s"):
        w.encode(["x"])
    assert time.monotonic() - started < 15
    assert all(process_gone(pid, 10) for pid in pids(tmp_path))


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("garbage", "invalid reply"),
        ("wrong_id", "answered the wrong request"),
        ("bad_vectors", "expected 1 vectors of 384 dims"),
    ],
)
def test_malformed_replies_are_rejected(make, tmp_path, mode, message):
    w = make(mode=mode)
    with pytest.raises(EmbeddingWorkerError, match=message):
        w.encode(["x"])
    assert w.spawns == 2  # one retry on a fresh worker, then a clean error
    assert all(process_gone(pid, 5) for pid in pids(tmp_path))


def test_externally_killed_worker_is_replaced(make, tmp_path):
    w = make()
    w.encode(["x"])
    victim = w.pid
    os.kill(victim, signal.SIGKILL)
    time.sleep(0.2)
    np.testing.assert_array_equal(w.encode(["y"]), fake_vectors(["y"]))
    assert w.pid != victim and w.spawns == 2
    assert process_gone(victim, 5)


def test_oversized_text_is_a_structured_error_not_a_crash(make):
    w = make()
    with pytest.raises(EmbeddingWorkerError, match="at most 256 texts"):
        w._embed_batch(["x"] * 300, 1)
    assert w.spawns == 1 and w.status()["status"] == "ready"


# --- scheduling ---------------------------------------------------------------------


def test_waiting_requests_are_bounded(make):
    w = make(mode="slow_embed", delay=0.8, max_waiting=2)
    w.load()
    holder = threading.Thread(target=w.encode, args=(["holder"],))
    holder.start()
    time.sleep(0.2)
    waiters = [threading.Thread(target=w.encode, args=([f"w{i}"],)) for i in range(2)]
    for t in waiters:
        t.start()
    time.sleep(0.2)
    with pytest.raises(EmbeddingBusyError, match="busy"):
        w.encode(["one too many"])
    for t in [holder, *waiters]:
        t.join(10)


def test_queries_jump_ahead_of_ingestion_batches(make, tmp_path):
    w = make(mode="slow_embed", delay=0.25, batch_size=1)
    w.load()
    background = threading.Thread(target=w.encode, args=([f"bg{i}" for i in range(4)],))
    background.start()
    time.sleep(0.4)  # bg0 running, bg1.. waiting
    w.encode(["query"])  # interactive: a single text
    background.join(10)
    order = (tmp_path / "order.log").read_text().split()
    assert order[0] == "bg0" and order.index("query") < order.index("bg3")
    assert sorted(order) == sorted(["bg0", "bg1", "bg2", "bg3", "query"])


# --- protocol ----------------------------------------------------------------------


def test_protocol_validation():
    with pytest.raises(p.ProtocolError, match="version"):
        p.loads(b'{"v": 99, "id": 1}')
    with pytest.raises(p.ProtocolError, match="object"):
        p.loads(b"[1, 2]")
    with pytest.raises(p.ProtocolError, match="id"):
        p.loads(b'{"v": 1}')
    with pytest.raises(p.ProtocolError, match="JSON"):
        p.loads(b"\xff not json")
    good = p.encode_vectors(fake_vectors(["a", "b"]))
    assert p.decode_vectors(good, 2).shape == (2, 384)
    for bad, msg in [
        ({**good, "shape": [3, 384]}, "expected 2 vectors"),
        ({**good, "dtype": "<f8"}, "dtype"),
        ({**good, "data": "%%%"}, "base64"),
        ({**good, "data": good["data"][:-8]}, "length|base64"),
        ("nope", "not an object"),
    ]:
        with pytest.raises(p.ProtocolError, match=msg):
            p.decode_vectors(bad, 2)


def test_worker_serve_rejects_bad_requests(monkeypatch):
    from litenotebook.embed import worker
    from litenotebook.embed._test_backend import FakeBackend

    monkeypatch.setenv("LITENOTEBOOK_EMBED_WORKER_BACKEND", "test-fake")
    lines = [
        b"garbage\n",
        p.dumps(p.request(1, "embed", texts=["x"])),  # before hello
        p.dumps(p.request(2, "hello")),
        p.dumps({"v": 1, "id": 3, "op": "explode"}),
        p.dumps(p.request(4, "embed", texts="not a list")),
        p.dumps(p.request(5, "ping")),
        p.dumps(p.request(6, "shutdown")),
        p.dumps(p.request(7, "ping")),  # never answered: the worker stopped
    ]
    out = io.BytesIO()
    assert worker.serve(io.BytesIO(b"".join(lines)), out, FakeBackend()) == 0
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [(r["id"], r["ok"]) for r in replies] == [
        (-1, False),
        (1, False),
        (2, True),
        (3, False),
        (4, False),
        (5, True),
        (6, True),
    ]
    assert replies[0]["error"]["type"] == "bad_request"
    assert replies[3]["error"]["message"] == "unknown op 'explode'"


def test_embedding_error_type_maps_to_embedding_error(make):
    w = make()
    with pytest.raises(EmbeddingError):
        w._embed_batch(["x"] * 300, 1)
