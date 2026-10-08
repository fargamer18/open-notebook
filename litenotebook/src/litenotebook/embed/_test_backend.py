"""Deterministic stand-in for MiniLM, used ONLY by tests.

Selected with LITENOTEBOOK_EMBED_WORKER_BACKEND=test-fake. It produces the same
hashed bag-of-words vectors as the test suite's FakeEmbedder and can simulate
failures (LITENOTEBOOK_EMBED_WORKER_TEST_MODE) so the parent's crash, timeout
and protocol handling can be tested without the real model. It deliberately
writes noise to stdout to prove that stdout noise cannot corrupt the protocol.
"""

from __future__ import annotations

import os
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from litenotebook.config import EMBED_DIM
from litenotebook.embed.embedder import EmbeddingModelUnavailableError


def fake_vectors(texts: list[str]) -> npt.NDArray[np.float32]:
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            h = zlib.crc32(word.encode())
            out[row, h % EMBED_DIM] += 1.0 if (h >> 16) & 1 else -1.0
        norm = np.linalg.norm(out[row])
        if norm == 0:
            out[row, 0], norm = 1.0, 1.0
        out[row] /= norm
    return out


class FakeBackend:
    def __init__(self) -> None:
        self.mode = os.environ.get("LITENOTEBOOK_EMBED_WORKER_TEST_MODE", "")
        directory = os.environ.get("LITENOTEBOOK_EMBED_WORKER_TEST_DIR")
        self.dir = Path(directory) if directory else None
        self.delay = float(os.environ.get("LITENOTEBOOK_EMBED_WORKER_TEST_DELAY", "0"))
        if self.dir:
            (self.dir / f"worker-{os.getpid()}.pid").write_text(
                str(os.getpid()), encoding="utf-8"
            )

    def _noise(self) -> None:
        print("stray print from a chatty library")  # noqa: T201
        os.write(1, b"raw C-level write to fd 1\n")
        sys.stdout.flush()

    def load(self) -> dict[str, Any]:
        self._noise()
        if self.mode == "slow_load":
            time.sleep(self.delay)
        if self.mode == "exit_before_ready":
            os._exit(3)
        if self.mode == "model_missing":
            raise EmbeddingModelUnavailableError()
        return {"load_s": 0.0, "device": "cpu", "torch": None, "torch_cuda": None}

    def encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self._noise()
        if self.dir:
            with open(self.dir / "order.log", "a", encoding="utf-8") as log:
                log.write((texts[0] if texts else "") + "\n")
        if self.mode == "crash_on_embed":
            os._exit(9)
        if self.mode == "crash_once" and self.dir:
            marker = self.dir / "crashed-once"
            if not marker.exists():
                marker.write_text("1", encoding="utf-8")
                os._exit(9)
        if self.mode == "hang_on_embed":
            time.sleep(3600)
        if self.mode == "slow_embed":
            time.sleep(self.delay)
        return fake_vectors(texts)
