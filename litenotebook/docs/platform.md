# LiteNotebook on Linux and Windows

LiteNotebook is one Python application for Linux and Windows 11. Everything
that genuinely differs between the two lives in `src/litenotebook/platform_utils/`;
the rest of the code uses `pathlib` and the standard library and is checked by
`tests/test_platform.py` to stay free of OS-specific calls.

> Status (0.9.2):
>
> - **Linux:** Built, Automated tests passed, Manually verified.
> - **Windows:** **Prepared** only. Windows rules are *simulated* in tests
>   on Linux; the app is not yet Built, Automated-tested or Manually
>   verified on real Windows. See [windows.md](windows.md) for the status
>   words and [windows-test-plan.md](windows-test-plan.md).

## Where data lives

| Mode | When | Data | Config | Cache |
|---|---|---|---|---|
| Custom | `LITENOTEBOOK_DATA_DIR` is set | that folder | `<data>/config` | `<data>/cache` |
| Development | running from a source checkout (`pyproject.toml` next to `src/litenotebook`) | `<checkout>/data` | `<data>/config` | `<data>/cache` |
| Platform default | an installed copy, or `LITENOTEBOOK_MODE=user` | see below | see below | see below |

Platform defaults:

| OS | Data | Config | Cache |
|---|---|---|---|
| Linux | `$XDG_DATA_HOME/LiteNotebook` (`~/.local/share/LiteNotebook`) | `$XDG_CONFIG_HOME/LiteNotebook` | `$XDG_CACHE_HOME/LiteNotebook` |
| Windows | `%LOCALAPPDATA%\LiteNotebook` | same | `%LOCALAPPDATA%\LiteNotebook\Cache` |
| macOS (unsupported, best effort) | `~/Library/Application Support/LiteNotebook` | same | `~/Library/Caches/LiteNotebook` |

`LITENOTEBOOK_CONFIG_DIR` / `LITENOTEBOOK_CACHE_DIR` override config and cache.
**Data is never moved between these locations automatically.** To move to the
platform default, make a full backup (Settings → Backup & Restore) and restore
it, or move the folder yourself while LiteNotebook is closed.

`litenotebook diagnostics platform` (or Settings → Diagnostics) shows which mode
and folders are in use.

## Setting environment variables

LiteNotebook only reads `os.environ`; how you set variables depends on the shell.

| Shell | One command | Session |
|---|---|---|
| Linux (bash/zsh) | `LITENOTEBOOK_DATA_DIR=~/notes litenotebook serve` | `export LITENOTEBOOK_DATA_DIR=~/notes` |
| PowerShell | — | `$env:LITENOTEBOOK_DATA_DIR = "$env:USERPROFILE\Notes"` |
| cmd.exe | — | `set LITENOTEBOOK_DATA_DIR=%USERPROFILE%\Notes` |

`%VAR%` (Windows), `$VAR` (Linux) and `~` inside `LITENOTEBOOK_DATA_DIR` are
expanded by LiteNotebook itself, so quoting styles do not matter.

Running from source: `python -m litenotebook.cli serve --open` works on both
systems (`--open` uses Python's `webbrowser`, never `xdg-open`/`start`).

## What is handled per OS

| Concern | Linux | Windows | Where |
|---|---|---|---|
| One process per data folder | `fcntl.flock` on `<data>/.instance.lock` | `msvcrt.locking` on a byte range of the same file | `filesystem.InstanceLock` |
| Crashed holder | lock released by the OS; the file left behind is ignored | same | |
| Directory fsync after rename | `fsync` on the directory | not possible / not needed (NTFS journals metadata) — skipped | `filesystem.fsync_dir` |
| Replacing a file | `os.replace` (atomic) | `os.replace` (atomic, same volume); sharing violations retried | `filesystem.replace_file` |
| Renaming a directory | atomic rename | atomic rename; refused while a file inside is open — only done when nothing has it open | `filesystem.rename_dir` |
| Deleting read-only files | — | read-only attribute cleared first | `filesystem.remove_tree` |
| Binary pipes / files | — | `O_BINARY` / `msvcrt.setmode`, so `\n` is never rewritten | `filesystem.create_exclusive`, `set_binary_mode` |
| Child processes | argument lists; graceful → SIGTERM → SIGKILL on the exact handle; optional own process group | no console window, own process group, kill-on-close **Job Object** (no orphan `python.exe` even if LiteNotebook crashes); graceful → TerminateProcess → TerminateJobObject | `processes.spawn` / `stop` |
| Finding executables | `shutil.which` | `shutil.which` (PATHEXT) + `.exe` suffix | `system.find_executable`, `executable_name` |
| Hardware summary | `/proc/cpuinfo`, `/proc/meminfo` | `GlobalMemoryStatusEx`, registry CPU name | `system.system_info` |

Rules that hold everywhere: no `shell=True`, no killing by process name, no
`fork`, no POSIX-only permission semantics, no deleting files that are still
open, explicit UTF-8 for every text file and the worker's JSON-lines protocol,
path containment by resolved components (never string prefixes), and user
file names are display metadata only — stored files are always `<id>.pdf`.

## Restore and file locking

A restore is staged in `<data>.restore-staging`, checked, then switched in by
two directory renames when no LiteNotebook process has the data folder open
(normally at the next start). The sequence is crash-recoverable from what is
on disk; see the table in `src/litenotebook/archive/restore.py`. The previous
data folder is kept as `<data>.pre-restore-<time>`.

## Continuous integration

A Linux + Windows matrix (ruff, mypy, pytest; model/LLM tests skip) is ready
in `ci/litenotebook-ci.yml`. It is not active because this repository's
`.github/workflows` belongs to the legacy Open Notebook project; copy the file
there (or into LiteNotebook's own repository) to enable it.
