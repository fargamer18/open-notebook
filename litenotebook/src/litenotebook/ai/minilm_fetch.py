"""Child process that downloads MiniLM: ``python -m litenotebook.ai.minilm_fetch``.

Run only by an explicit "Install MiniLM" action. It is a separate process so
the web server never imports the download stack, and so the normal
local-only rule (``HF_HUB_OFFLINE``) stays in force everywhere else; the
parent enables network access for this child alone. Progress and the result
are JSON lines on stdout. Files go to the standard Hugging Face cache, where
the embedding worker already looks.
"""

from __future__ import annotations

import json
import sys


def emit(**message: object) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> int:
    from litenotebook.ai.catalog import load_embeddings

    entry = load_embeddings()[0]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        emit(error="huggingface_hub is not installed")
        return 2
    total = len(entry.files)
    for index, f in enumerate(entry.files, 1):
        emit(stage="downloading", file=f.path, index=index, total=total, size=f.size)
        try:
            hf_hub_download(entry.repository, f.path, revision="main")
        except Exception as e:  # network or hub errors: report and stop
            name = type(e).__name__
            offline = "Connect" in name or "Offline" in name or "Timeout" in name
            emit(
                error="No internet connection"
                if offline
                else f"Download failed ({name})",
                offline=offline,
            )
            return 1
    emit(ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
