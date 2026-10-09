"""Generation models: catalog installs, imported GGUF files, selection.

* ``install_model`` downloads a catalog model into the models folder
  (verified SHA-256; only a verified file is ever registered).
* ``import_model`` uses a GGUF the user already has: by default it is only
  *referenced* where it is (nothing moved or copied); copying into the models
  folder is optional. The header is checked (``GGUF`` magic, version, sane
  counts) and its metadata read without loading tensors.
* ``remove_model`` deletes a file only when LiteNotebook owns it (downloaded
  or copied it) *and* it is inside the models folder; referenced files are
  only forgotten, never deleted.

A file in a Hugging Face cache is stored under its SHA-256 (the blob name),
so an existing download can be matched to the catalog without re-hashing
gigabytes.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from litenotebook.ai.catalog import ModelEntry, load_models
from litenotebook.ai.download import Downloader, Progress, check_space
from litenotebook.ai.store import AIStore, ModelRecord
from litenotebook.config import Settings
from litenotebook.platform_utils.filesystem import (
    COPY_BUFFER_BYTES,
    create_exclusive,
    is_within,
    replace_file,
)

GGUF_MAGIC = b"GGUF"
_SHA_NAME = re.compile(r"^[0-9a-f]{64}$")
MAX_KV = 4096
MAX_STRING = 1 << 20


class ModelError(ValueError):
    pass


def models_dir(settings: Settings, store: AIStore) -> Path:
    custom = store.models_dir()
    return Path(custom) if custom else settings.data_dir / "models"


# --- GGUF header --------------------------------------------------------------------


@dataclass
class GGUFInfo:
    version: int
    tensor_count: int
    kv_count: int
    metadata: dict[str, Any]

    @property
    def architecture(self) -> dict[str, int] | None:
        """layers / KV heads / head dim from metadata, if present."""
        arch = self.metadata.get("general.architecture")
        if not isinstance(arch, str):
            return None
        get = self.metadata.get
        layers = get(f"{arch}.block_count")
        heads = get(f"{arch}.attention.head_count")
        kv = get(f"{arch}.attention.head_count_kv", heads)
        dim = get(f"{arch}.attention.key_length")
        if dim is None and isinstance(get(f"{arch}.embedding_length"), int) and heads:
            dim = get(f"{arch}.embedding_length") // heads
        ctx = get(f"{arch}.context_length", 4096)
        if (
            isinstance(layers, int)
            and isinstance(kv, int)
            and isinstance(dim, int)
            and min(layers, kv, dim) > 0
        ):
            return {
                "layers": layers,
                "kv_heads": kv,
                "head_dim": dim,
                "max_context": ctx if isinstance(ctx, int) else 4096,
            }
        return None


_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _read(f: IO[bytes], n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ModelError("The GGUF header is truncated")
    return data


def _string(f: IO[bytes]) -> str:
    (length,) = struct.unpack("<Q", _read(f, 8))
    if length > MAX_STRING:
        raise ModelError("The GGUF header contains an implausible string")
    return _read(f, length).decode("utf-8", "replace")


def _value(f: IO[bytes], kind: int) -> Any:
    if kind in _SCALARS:
        fmt = _SCALARS[kind]
        return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]
    if kind == 8:
        return _string(f)
    if kind == 9:  # array: skipped (tokenizer vocabularies are huge)
        (inner,) = struct.unpack("<I", _read(f, 4))
        (count,) = struct.unpack("<Q", _read(f, 8))
        if count > 50_000_000:
            raise ModelError("The GGUF header contains an implausible array")
        if inner in _SCALARS:
            f.seek(struct.calcsize(_SCALARS[inner]) * count, 1)
        elif inner == 8:
            for _ in range(count):
                (length,) = struct.unpack("<Q", _read(f, 8))
                if length > MAX_STRING:
                    raise ModelError("The GGUF header contains an implausible string")
                f.seek(length, 1)
        else:
            raise ModelError("Unsupported GGUF metadata")
        return None
    raise ModelError("Unsupported GGUF metadata type")


def read_gguf(path: Path) -> GGUFInfo:
    try:
        with open(path, "rb") as f:
            if _read(f, 4) != GGUF_MAGIC:
                raise ModelError("This is not a GGUF model file")
            (version,) = struct.unpack("<I", _read(f, 4))
            if version not in (2, 3):
                raise ModelError(f"Unsupported GGUF version {version}")
            tensors, kv_count = struct.unpack("<QQ", _read(f, 16))
            if tensors > 1_000_000 or kv_count > MAX_KV:
                raise ModelError("The GGUF header has implausible counts")
            metadata: dict[str, Any] = {}
            for _ in range(kv_count):
                key = _string(f)
                (kind,) = struct.unpack("<I", _read(f, 4))
                value = _value(f, kind)
                if value is not None:
                    metadata[key] = value
    except OSError as e:
        raise ModelError(f"The model file cannot be read ({e.strerror or e})") from e
    return GGUFInfo(version, tensors, kv_count, metadata)


def known_sha256(path: Path) -> str | None:
    """SHA-256 known without hashing: a Hugging Face cache blob's own name."""
    with contextlib.suppress(OSError):
        target = path.resolve()
        if target.parent.name == "blobs" and _SHA_NAME.match(target.name):
            return target.name
    return None


def sha256_file(path: Path, cancel: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(COPY_BUFFER_BYTES):
            if cancel is not None and cancel.is_set():
                raise ModelError("Cancelled")
            digest.update(chunk)
    return digest.hexdigest()


# --- registry operations ------------------------------------------------------------


def _record_from_entry(
    entry: ModelEntry, path: Path, *, owned: bool, verified: bool = True
) -> ModelRecord:
    return ModelRecord(
        id=entry.id,
        display_name=f"{entry.display_name} · {entry.quantization}",
        path=str(path),
        owned=owned,
        size=entry.size,
        catalog_id=entry.id,
        sha256=entry.sha256,
        verified=verified,
        quantization=entry.quantization,
        alias=entry.alias,
        architecture=dict(entry.architecture),
        runtime_args=list(entry.runtime_args),
        added_at=time.time(),
    )


def _register(store: AIStore, record: ModelRecord) -> ModelRecord:
    records = [r for r in store.models() if r.id != record.id]
    records.append(record)
    store.save_models(records)
    return record


def install_model(
    settings: Settings,
    store: AIStore,
    entry: ModelEntry,
    *,
    downloader: Downloader,
    progress: Callable[[Progress], None] | None = None,
    cancel: threading.Event | None = None,
) -> ModelRecord:
    folder = models_dir(settings, store)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / entry.filename
    downloader.fetch(
        entry.url,
        dest,
        sha256=entry.sha256,
        size=entry.size,
        progress=progress,
        cancel=cancel,
    )
    return _register(store, _record_from_entry(entry, dest, owned=True))


def import_model(
    settings: Settings,
    store: AIStore,
    path: Path,
    *,
    copy: bool = False,
    cancel: threading.Event | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ModelRecord:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ModelError("That file does not exist")
    info = read_gguf(source)
    size = source.stat().st_size
    # A Hugging Face cache blob is named after its SHA-256: that identifies the
    # catalog entry for free, but is not a verification (we did not hash it).
    sha = known_sha256(source)
    hashed = False
    entry = next((m for m in load_models() if sha and m.sha256 == sha), None)
    if entry is None:  # not identifiable for free: compare against catalog sizes first
        same_size = [m for m in load_models() if m.size == size]
        if same_size:
            sha = sha256_file(source, cancel)
            hashed = True
            entry = next((m for m in same_size if m.sha256 == sha), None)
    target = source
    if copy:
        folder = models_dir(settings, store)
        folder.mkdir(parents=True, exist_ok=True)
        check_space(folder, size)
        target = folder / source.name
        if target.exists():
            raise ModelError(
                f"A model named {source.name} is already in the models folder"
            )
        part = target.with_name(target.name + ".part")
        part.unlink(missing_ok=True)
        done = 0
        with open(source, "rb") as src, create_exclusive(part) as out:
            while chunk := src.read(COPY_BUFFER_BYTES):
                if cancel is not None and cancel.is_set():
                    out.close()
                    part.unlink(missing_ok=True)
                    raise ModelError("Cancelled")
                out.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, size)
        replace_file(part, target)
    if entry is not None:
        if entry.size != size:
            raise ModelError("The file is not the complete model (wrong size)")
        record = _record_from_entry(entry, target, owned=copy, verified=hashed)
        return _register(store, record)
    name = str(info.metadata.get("general.name") or source.stem)
    slug = re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-")[:48] or "model"
    return _register(
        store,
        ModelRecord(
            id=f"local-{slug}",
            display_name=f"{name} (imported)",
            path=str(target),
            owned=copy,
            size=size,
            sha256=sha,
            verified=False,
            alias=f"local/{slug}",
            architecture=info.architecture,
            runtime_args=["--jinja"],
            added_at=time.time(),
        ),
    )


def remove_model(settings: Settings, store: AIStore, model_id: str) -> bool:
    """Forget a model; delete its file only if LiteNotebook owns it. Returns
    True if a file was deleted. The caller stops a running runtime first."""
    record = store.model(model_id)
    if record is None:
        raise ModelError("Unknown model")
    deleted = False
    path = Path(record.path)
    folder = models_dir(settings, store)
    if record.owned and is_within(path, folder) and path.is_file():
        path.unlink()
        deleted = True
    store.save_models([r for r in store.models() if r.id != model_id])
    return deleted


def model_status(record: ModelRecord) -> str:
    path = Path(record.path)
    if not path.is_file():
        return "missing"
    if path.stat().st_size != record.size:
        return "changed"
    return "ok"
