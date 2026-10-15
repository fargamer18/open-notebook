"""Write install-litenotebook.sh for one Linux release.

    python tools/linux_installer.py DIST     # uses DIST/build-report.json

The installer template (packaging/linux/install-litenotebook.sh.in) gets the
version and the exact name, size and SHA-256 of the core archive and of the
optional embedding runtime, so the script installs only those files.
tools/build.py calls render() during a Linux build.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "packaging" / "linux" / "install-litenotebook.sh.in"
SCRIPT_NAME = "install-litenotebook.sh"
_SAFE = re.compile(r"^[A-Za-z0-9._+-]*$")


def render(
    version: str,
    core: dict[str, Any],
    runtime: dict[str, Any] | None,
    template: Path = TEMPLATE,
) -> str:
    """core/runtime: {"file", "size", "sha256"}; core may give "dir" (the archive's
    top folder, default: the file name without .tar.gz)."""
    core_dir = core.get("dir") or str(core["file"]).removesuffix(".tar.gz")
    values = {
        "VERSION": version,
        "CORE_FILE": core["file"],
        "CORE_SIZE": core["size"],
        "CORE_SHA256": core["sha256"],
        "CORE_DIR": core_dir,
        "RUNTIME_FILE": runtime["file"] if runtime else "",
        "RUNTIME_SIZE": runtime["size"] if runtime else "",
        "RUNTIME_SHA256": runtime["sha256"] if runtime else "",
    }
    text = template.read_text("utf-8")
    for key, value in values.items():
        value = str(value)
        if not _SAFE.match(value):  # values land inside double quotes in bash
            raise ValueError(
                f"{key} has characters the installer cannot embed: {value}"
            )
        text = text.replace(f"@@{key}@@", value)
    if "@@" in text.split("\nset -euo pipefail", 1)[1].split("say() {", 1)[0]:
        raise ValueError("unfilled placeholder in the installer template")
    return text


def write(
    target: Path, version: str, core: dict[str, Any], runtime: dict[str, Any] | None
) -> Path:
    target.write_text(render(version, core, runtime), encoding="utf-8", newline="\n")
    target.chmod(0o755)
    return target


def from_dist(dist: Path) -> Path:
    report = json.loads((dist / "build-report.json").read_text("utf-8"))
    version = report["version"]
    by_name = {a["file"]: a for a in report["artifacts"]}
    core = by_name[f"LiteNotebook-{version}-{report['platform']}.tar.gz"]
    runtime = report.get("embedding_runtime")
    return write(dist / SCRIPT_NAME, version, core, runtime)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(from_dist(Path(sys.argv[1])))
