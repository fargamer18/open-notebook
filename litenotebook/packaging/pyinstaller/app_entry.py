"""Entry point of the packaged LiteNotebook app (see litenotebook.desktop)."""

import sys

from litenotebook.desktop import entry

if __name__ == "__main__":
    raise SystemExit(entry(sys.argv[1:]))
