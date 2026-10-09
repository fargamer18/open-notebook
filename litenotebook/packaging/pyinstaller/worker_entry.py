"""Entry point of the packaged embedding worker (litenotebook.embed.worker).

A separate executable so torch / sentence-transformers never enter the web
app's process; it speaks the same stdin/stdout protocol as
``python -m litenotebook.embed.worker``.
"""

from litenotebook.embed.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
