"""Versioned catalogs of runtimes, generation models and embedding models.

The data lives in ``ai/catalog/{runtimes,models,embeddings}.json`` (shipped
with the package), separate from the logic that uses it. Every entry is
validated on load: pinned HTTPS URLs on an allowlisted host, lowercase
SHA-256, positive sizes, portable archive paths. A catalog with a newer
``catalog_version`` is refused rather than half-understood.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from litenotebook.platform_utils.paths import archive_name_problem

CATALOG_VERSION = 1
SOURCE_HOSTS = frozenset({"github.com", "huggingface.co"})
# Where pinned downloads may redirect to (CDNs of the same publishers).
REDIRECT_HOST_SUFFIXES = (
    "github.com",
    ".githubusercontent.com",
    "huggingface.co",
    ".huggingface.co",
    ".hf.co",
    "hf.co",
    ".xethub.hf.co",
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
OSES = ("linux", "windows")
ARCHES = ("x86_64", "arm64")
BACKENDS = ("cpu", "cuda")


class CatalogError(ValueError):
    pass


def normalize_arch(machine: str) -> str:
    m = machine.lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return m


def host_allowed(url: str, *, redirect: bool = False) -> bool:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        return False
    host = parts.hostname.lower()
    if not redirect:
        return host in SOURCE_HOSTS
    return any(
        host == s.lstrip(".") or host.endswith(s) for s in REDIRECT_HOST_SUFFIXES
    )


@dataclass(frozen=True)
class Asset:
    url: str
    size: int
    sha256: str
    format: str  # zip | tar.gz | file


@dataclass(frozen=True)
class RuntimeEntry:
    id: str
    project: str
    version: str
    os: str
    arch: str
    backend: str
    executable: str
    assets: tuple[Asset, ...]
    license: str
    license_url: str
    release_url: str
    source_url: str
    release_date: str
    layout_verified: bool
    tested: bool
    requires: dict[str, Any] = field(default_factory=dict)

    @property
    def download_size(self) -> int:
        return sum(a.size for a in self.assets)


@dataclass(frozen=True)
class ModelEntry:
    id: str
    display_name: str
    family: str
    publisher: str
    parameter_class: str
    size_class: str
    quantization: str
    filename: str
    repository: str
    revision: str
    url: str
    sha256: str
    size: int
    architecture: dict[str, int]
    recommended_contexts: tuple[int, ...]
    capabilities: tuple[str, ...]
    alias: str
    runtime_args: tuple[str, ...]
    model_page: str
    license: str
    license_url: str
    min_runtime_version: str
    notes: str


@dataclass(frozen=True)
class EmbeddingFile:
    path: str
    size: int
    sha256: str | None


@dataclass(frozen=True)
class EmbeddingEntry:
    id: str
    display_name: str
    repository: str
    revision: str
    publisher: str
    license: str
    license_url: str
    model_page: str
    dimension: int
    files: tuple[EmbeddingFile, ...]
    size: int
    purpose: str


def _need(entry: dict[str, Any], key: str, kind: type, where: str) -> Any:
    value = entry.get(key)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise CatalogError(f"{where}: {key!r} must be {kind.__name__}")
    return value


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA.match(value):
        raise CatalogError(f"{where}: invalid SHA-256")
    return value


def _url(value: Any, where: str) -> str:
    if not isinstance(value, str) or not host_allowed(value):
        raise CatalogError(
            f"{where}: download URL must be HTTPS on {sorted(SOURCE_HOSTS)}"
        )
    return value


def _read(kind: str, raw: str | None) -> dict[str, Any]:
    text = (
        raw
        if raw is not None
        else (
            resources.files("litenotebook.ai")
            .joinpath(f"catalog/{kind}.json")
            .read_text(encoding="utf-8")
        )
    )
    try:
        data = json.loads(text)
    except ValueError as e:
        raise CatalogError(f"{kind} catalog is not valid JSON") from e
    if not isinstance(data, dict) or data.get("kind") != kind:
        raise CatalogError(f"not a {kind} catalog")
    version = data.get("catalog_version")
    if not isinstance(version, int) or version < 1:
        raise CatalogError(f"{kind} catalog has an invalid version")
    if version > CATALOG_VERSION:
        raise CatalogError(
            f"The {kind} catalog was made for a newer LiteNotebook (version {version})"
        )
    return data


def load_runtimes(raw: str | None = None) -> list[RuntimeEntry]:
    data = _read("runtimes", raw)
    out: list[RuntimeEntry] = []
    ids: set[str] = set()
    for i, e in enumerate(data.get("runtimes") or []):
        where = f"runtime #{i}"
        if not isinstance(e, dict):
            raise CatalogError(f"{where}: not an object")
        rid = _need(e, "id", str, where)
        if rid in ids:
            raise CatalogError(f"{where}: duplicate id {rid}")
        ids.add(rid)
        if (
            e.get("os") not in OSES
            or e.get("arch") not in ARCHES
            or e.get("backend") not in BACKENDS
        ):
            raise CatalogError(f"{rid}: unsupported os/arch/backend")
        exe = _need(e, "executable", str, rid)
        if archive_name_problem(exe):
            raise CatalogError(f"{rid}: executable path is not a safe relative path")
        assets = []
        for a in _need(e, "assets", list, rid):
            if not isinstance(a, dict) or a.get("format") not in ("zip", "tar.gz"):
                raise CatalogError(f"{rid}: bad asset")
            size = _need(a, "size", int, rid)
            if size <= 0:
                raise CatalogError(f"{rid}: asset size must be positive")
            assets.append(
                Asset(
                    _url(a.get("url"), rid),
                    size,
                    _sha(a.get("sha256"), rid),
                    a["format"],
                )
            )
        if not assets:
            raise CatalogError(f"{rid}: no assets")
        out.append(
            RuntimeEntry(
                id=rid,
                project=_need(e, "project", str, rid),
                version=_need(e, "version", str, rid),
                os=e["os"],
                arch=e["arch"],
                backend=e["backend"],
                executable=exe,
                assets=tuple(assets),
                license=_need(e, "license", str, rid),
                license_url=_need(e, "license_url", str, rid),
                release_url=_need(e, "release_url", str, rid),
                source_url=_need(e, "source_url", str, rid),
                release_date=_need(e, "release_date", str, rid),
                layout_verified=bool(e.get("layout_verified")),
                tested=bool(e.get("tested")),
                requires=dict(e.get("requires") or {}),
            )
        )
    return out


def load_models(raw: str | None = None) -> list[ModelEntry]:
    data = _read("models", raw)
    out: list[ModelEntry] = []
    ids: set[str] = set()
    for i, e in enumerate(data.get("models") or []):
        where = f"model #{i}"
        if not isinstance(e, dict):
            raise CatalogError(f"{where}: not an object")
        mid = _need(e, "id", str, where)
        if mid in ids:
            raise CatalogError(f"{where}: duplicate id {mid}")
        ids.add(mid)
        filename = _need(e, "filename", str, mid)
        if (
            archive_name_problem(filename)
            or "/" in filename
            or not filename.endswith(".gguf")
        ):
            raise CatalogError(f"{mid}: filename must be a plain .gguf name")
        arch = _need(e, "architecture", dict, mid)
        for k in ("layers", "kv_heads", "head_dim", "max_context"):
            if not isinstance(arch.get(k), int) or arch[k] <= 0:
                raise CatalogError(
                    f"{mid}: architecture.{k} must be a positive integer"
                )
        size = _need(e, "size", int, mid)
        if size <= 0:
            raise CatalogError(f"{mid}: size must be positive")
        out.append(
            ModelEntry(
                id=mid,
                display_name=_need(e, "display_name", str, mid),
                family=_need(e, "family", str, mid),
                publisher=_need(e, "publisher", str, mid),
                parameter_class=_need(e, "parameter_class", str, mid),
                size_class=_need(e, "size_class", str, mid),
                quantization=_need(e, "quantization", str, mid),
                filename=filename,
                repository=_need(e, "repository", str, mid),
                revision=_need(e, "revision", str, mid),
                url=_url(e.get("url"), mid),
                sha256=_sha(e.get("sha256"), mid),
                size=size,
                architecture=dict(arch),
                recommended_contexts=tuple(
                    int(c) for c in e.get("recommended_contexts") or (4096,)
                ),
                capabilities=tuple(e.get("capabilities") or ()),
                alias=_need(e, "alias", str, mid),
                runtime_args=tuple(str(a) for a in e.get("runtime_args") or ()),
                model_page=_need(e, "model_page", str, mid),
                license=_need(e, "license", str, mid),
                license_url=_need(e, "license_url", str, mid),
                min_runtime_version=str(e.get("min_runtime_version") or ""),
                notes=str(e.get("notes") or ""),
            )
        )
    return out


def load_embeddings(raw: str | None = None) -> list[EmbeddingEntry]:
    data = _read("embeddings", raw)
    out = []
    for i, e in enumerate(data.get("embeddings") or []):
        where = f"embedding #{i}"
        eid = _need(e, "id", str, where)
        files = []
        for f in _need(e, "files", list, eid):
            path = _need(f, "path", str, eid)
            if archive_name_problem(path):
                raise CatalogError(f"{eid}: unsafe file path {path!r}")
            sha = f.get("sha256")
            files.append(
                EmbeddingFile(
                    path,
                    _need(f, "size", int, eid),
                    _sha(sha, eid) if sha is not None else None,
                )
            )
        out.append(
            EmbeddingEntry(
                id=eid,
                display_name=_need(e, "display_name", str, eid),
                repository=_need(e, "repository", str, eid),
                revision=_need(e, "revision", str, eid),
                publisher=_need(e, "publisher", str, eid),
                license=_need(e, "license", str, eid),
                license_url=_need(e, "license_url", str, eid),
                model_page=_need(e, "model_page", str, eid),
                dimension=_need(e, "dimension", int, eid),
                files=tuple(files),
                size=_need(e, "size", int, eid),
                purpose=str(e.get("purpose") or ""),
            )
        )
    return out


def runtime(entry_id: str, entries: list[RuntimeEntry] | None = None) -> RuntimeEntry:
    for e in entries if entries is not None else load_runtimes():
        if e.id == entry_id:
            return e
    raise CatalogError(f"Unknown runtime {entry_id!r}")


def model(entry_id: str, entries: list[ModelEntry] | None = None) -> ModelEntry:
    for e in entries if entries is not None else load_models():
        if e.id == entry_id:
            return e
    raise CatalogError(f"Unknown model {entry_id!r}")


def runtimes_for(
    os_kind: str, arch: str, entries: list[RuntimeEntry] | None = None
) -> list[RuntimeEntry]:
    return [
        e
        for e in (entries if entries is not None else load_runtimes())
        if e.os == os_kind and e.arch == normalize_arch(arch)
    ]


# --- speech (Phase 14) -------------------------------------------------------------

SPEECH_FORMATS = ("tar.bz2", "tar.gz", "zip")


@dataclass(frozen=True)
class SpeechEngineEntry:
    id: str
    project: str
    version: str
    os: str
    arch: str
    executable: str  # relative to the extracted archive
    library_dir: str  # shared libraries the executable needs
    assets: tuple[Asset, ...]
    license: str
    license_url: str
    license_note: str
    release_url: str
    source_url: str
    release_date: str
    layout_verified: bool
    tested: bool

    @property
    def download_size(self) -> int:
        return sum(a.size for a in self.assets)


@dataclass(frozen=True)
class VoiceEntry:
    id: str
    display_name: str
    engine: str
    model_type: str
    language: str
    locale: str
    quality: str
    sample_rate: int
    description: str
    license: str
    license_url: str
    model_card_url: str
    asset: Asset


@dataclass(frozen=True)
class SpeechCatalog:
    engines: tuple[SpeechEngineEntry, ...]
    voices: tuple[VoiceEntry, ...]
    default_voices: dict[str, str]
    note: str

    def engine_for(self, os_kind: str, arch: str) -> SpeechEngineEntry | None:
        return next(
            (
                e
                for e in self.engines
                if e.os == os_kind and e.arch == normalize_arch(arch)
            ),
            None,
        )

    def voice(self, voice_id: str) -> VoiceEntry:
        for v in self.voices:
            if v.id == voice_id:
                return v
        raise CatalogError(f"Unknown voice {voice_id!r}")


def _speech_asset(a: Any, where: str) -> Asset:
    if not isinstance(a, dict) or a.get("format") not in SPEECH_FORMATS:
        raise CatalogError(f"{where}: invalid asset")
    size = _need(a, "size", int, where)
    if size <= 0:
        raise CatalogError(f"{where}: invalid size")
    return Asset(
        _url(a.get("url"), where), size, _sha(a.get("sha256"), where), a["format"]
    )


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,79}$")


def load_speech(raw: str | None = None) -> SpeechCatalog:
    data = _read("speech", raw)
    engines: list[SpeechEngineEntry] = []
    for i, e in enumerate(data.get("engines") or []):
        where = f"speech engine #{i}"
        if not isinstance(e, dict):
            raise CatalogError(f"{where}: not an object")
        eid = _need(e, "id", str, where)
        if not _SAFE_ID.match(eid):
            raise CatalogError(f"{where}: unsafe id")
        for key in ("executable", "library_dir"):
            if archive_name_problem(_need(e, key, str, eid)):
                raise CatalogError(f"{eid}: unsafe {key}")
        engines.append(
            SpeechEngineEntry(
                id=eid,
                project=_need(e, "project", str, eid),
                version=_need(e, "version", str, eid),
                os=_need(e, "os", str, eid),
                arch=normalize_arch(_need(e, "arch", str, eid)),
                executable=e["executable"],
                library_dir=e["library_dir"],
                assets=tuple(
                    _speech_asset(a, eid) for a in _need(e, "assets", list, eid)
                ),
                license=_need(e, "license", str, eid),
                license_url=_need(e, "license_url", str, eid),
                license_note=str(e.get("license_note") or ""),
                release_url=_need(e, "release_url", str, eid),
                source_url=_need(e, "source_url", str, eid),
                release_date=_need(e, "release_date", str, eid),
                layout_verified=bool(e.get("layout_verified")),
                tested=bool(e.get("tested")),
            )
        )
    voices: list[VoiceEntry] = []
    for i, v in enumerate(data.get("voices") or []):
        where = f"voice #{i}"
        if not isinstance(v, dict):
            raise CatalogError(f"{where}: not an object")
        vid = _need(v, "id", str, where)
        if not _SAFE_ID.match(vid):
            raise CatalogError(f"{where}: unsafe id")
        voices.append(
            VoiceEntry(
                id=vid,
                display_name=_need(v, "display_name", str, vid),
                engine=_need(v, "engine", str, vid),
                model_type=_need(v, "model_type", str, vid),
                language=_need(v, "language", str, vid),
                locale=_need(v, "locale", str, vid),
                quality=_need(v, "quality", str, vid),
                sample_rate=_need(v, "sample_rate", int, vid),
                description=str(v.get("description") or ""),
                license=_need(v, "license", str, vid),
                license_url=_need(v, "license_url", str, vid),
                model_card_url=_need(v, "model_card_url", str, vid),
                asset=_speech_asset(v.get("asset"), vid),
            )
        )
    ids = [e.id for e in engines] + [v.id for v in voices]
    if len({i.casefold() for i in ids}) != len(ids):  # ids are folder names
        raise CatalogError("speech catalog has duplicate ids")
    defaults = data.get("default_voices") or {}
    known = {v.id for v in voices}
    if not isinstance(defaults, dict) or any(d not in known for d in defaults.values()):
        raise CatalogError("speech catalog default voices are unknown")
    return SpeechCatalog(
        tuple(engines), tuple(voices), dict(defaults), str(data.get("note") or "")
    )
