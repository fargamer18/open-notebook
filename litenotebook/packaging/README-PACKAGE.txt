LiteNotebook (pre-release)
==========================

Start it
  Windows:  Start menu → LiteNotebook (or LiteNotebook.exe in this folder)
  Linux:    ./litenotebook  (or the AppImage). To add it to your application
            menu once:  ./litenotebook --install-desktop-entry

LiteNotebook opens in your web browser at http://127.0.0.1:<port> and listens
only on this computer. Quit it from Settings → Diagnostics → Quit LiteNotebook.

Your data
  Notebooks, settings, models, logs and generated audio live in your user data
  folder, never inside this program folder:
    Windows: %LOCALAPPDATA%\LiteNotebook   (logs: %LOCALAPPDATA%\LiteNotebook\Logs)
    Linux:   ~/.local/share/LiteNotebook   (logs: ~/.local/state/LiteNotebook/logs)
  Updating or uninstalling LiteNotebook does not delete it.

Optional components (never included here; installed only when you choose)
  - Semantic search runtime (PyTorch + sentence-transformers): the separate
    "LiteNotebook-embedding-runtime-<version>-<platform>" download. Unpack it as
    the "embedding-runtime" folder next to this program, or install it from
    Settings → Local AI. Needed to index new PDFs.
  - MiniLM embedding model, llama.cpp + a Qwen model, local speech + voices:
    from the setup wizard or Settings → Local AI.

Licenses: see THIRD_PARTY_LICENSES.txt/md. PyMuPDF (used to read PDFs) is
AGPL-3.0 or commercially licensed by Artifex.
