"""Archive format version 1: layout, row schemas and limits."""

from __future__ import annotations

import re
from dataclasses import dataclass

from litenotebook.config import Settings

FORMAT_NAME = "litenotebook-archive"
FORMAT_VERSION = 2  # independent of the database schema version
# Version history: 1 = Phase 10 (sources, chats, notes, annotations, memory);
# 2 = Phase 13 adds the Study tables below. Version 1 archives are still
# imported: tables introduced later are simply absent (no quizzes).
KINDS = ("notebook", "backup")
MANIFEST = "manifest.json"
CHECKSUMS = "checksums.json"
SETTINGS = "settings.json"
EMBEDDING_DTYPE = "float32-le"
NEWER_VERSION_MESSAGE = (
    "This backup was created by a newer LiteNotebook version. Update LiteNotebook "
    "to import it."
)

# Settings a full backup may carry. API keys and anything machine-specific
# beyond the LLM endpoint are never exported.
BACKUP_SETTINGS = ("llm.base_url", "llm.model", "rag.top_k")

INT, TEXT, REAL, BLOB = "int", "text", "real", "blob"


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    nullable: bool = False


def _cols(spec: str) -> tuple[Column, ...]:
    """``"id:int title:text page_count:int? ..."`` → columns (``?`` = nullable)."""
    out = []
    for item in spec.split():
        name, kind = item.split(":")
        out.append(Column(name, kind.rstrip("?"), kind.endswith("?")))
    return tuple(out)


NOTEBOOK = _cols(
    "id:int name:text description:text embed_model:text embed_dim:int "
    "created_at:text updated_at:text"
)

# Exported tables in import order (parents before children). Columns are an
# allowlist: storage paths, progress counters and derived tables never leave.
TABLES: dict[str, tuple[Column, ...]] = {
    "sources": _cols(
        "id:int title:text original_filename:text mime:text sha256:text "
        "size_bytes:int page_count:int? status:text stage:text ocr:int "
        "error:text? timings:text? attempts:int created_at:text started_at:text? "
        "finished_at:text? updated_at:text"
    ),
    "pages": _cols("id:int source_id:int page_no:int text:text method:text"),
    "chunks": _cols(
        "id:int source_id:int page_id:int page_no:int ord:int char_start:int "
        "char_end:int heading:text? text:text n_tokens:int embedding:blob?"
    ),
    "conversations": _cols("id:int title:text created_at:text updated_at:text"),
    "messages": _cols(
        "id:int conversation_id:int role:text content:text status:text "
        "reply_to:int? model:text? error:text? meta:text? created_at:text "
        "completed_at:text?"
    ),
    "message_citations": _cols(
        "id:int message_id:int label:int ordinal:int chunk_id:int source_id:int "
        "source_filename:text page_no:int chunk_ord:int chunk_text:text"
    ),
    "memory_outcomes": _cols(
        "id:int message_id:int outcome:text correction:text? created_at:text "
        "updated_at:text"
    ),
    "annotations": _cols(
        "id:int source_id:int page_no:int annotation_type:text selected_text:text "
        "comment:text color:text created_at:text updated_at:text"
    ),
    "annotation_rects": _cols(
        "id:int annotation_id:int page_no:int x0:real y0:real x1:real y1:real "
        "ordinal:int"
    ),
    "reading_state": _cols("source_id:int current_page:int zoom:real updated_at:text"),
    "notes": _cols(
        "id:int title:text content:text revision:int created_at:text updated_at:text"
    ),
    "note_links": _cols(
        "id:int note_id:int link_type:text target_key:text source_id:int? "
        "chunk_id:int? message_id:int? conversation_id:int? memory_key:text? "
        "annotation_id:int? label:text excerpt:text page_no:int? chunk_ord:int? "
        "details:text created_at:text"
    ),
    "message_memory_refs": _cols(
        "id:int message_id:int label:int ordinal:int node_key:text kind:text "
        "provenance:text conversation_id:int? conversation_title:text "
        "question:text excerpt:text outcome:text? correction:text? sources:text "
        "remembered_at:text"
    ),
    "quizzes": _cols(
        "id:int title:text status:text stage:text difficulty:text "
        "question_types:text requested_count:int question_count:int focus:text "
        "material:text message:text? error:text? model:text? meta:text? "
        "created_at:text updated_at:text completed_at:text?"
    ),
    "quiz_questions": _cols(
        "id:int quiz_id:int position:int qtype:text question:text explanation:text "
        "hint:text difficulty:text concepts:text"
    ),
    "quiz_question_options": _cols(
        "id:int question_id:int position:int text:text is_correct:int"
    ),
    "quiz_evidence": _cols(
        "id:int question_id:int ordinal:int kind:text source_id:int? chunk_id:int? "
        "page_no:int? chunk_ord:int? note_id:int? message_id:int? "
        "conversation_id:int? label:text excerpt:text"
    ),
    "quiz_attempts": _cols(
        "id:int quiz_id:int mode:text parent_attempt_id:int? status:text total:int "
        "score:int? started_at:text completed_at:text?"
    ),
    "quiz_attempt_items": _cols(
        "id:int attempt_id:int question_id:int position:int selected:text? "
        "is_correct:int? hint_used:int answered_at:text?"
    ),
    "study_concept_stats": _cols(
        "id:int concept_key:text label:text times_seen:int correct_count:int "
        "incorrect_count:int streak:int last_result:text? last_seen:text?"
    ),
}
# The first archive format version that contains each table (default 1).
TABLE_SINCE = {
    "quizzes": 2,
    "quiz_questions": 2,
    "quiz_question_options": 2,
    "quiz_evidence": 2,
    "quiz_attempts": 2,
    "quiz_attempt_items": 2,
    "study_concept_stats": 2,
}


def tables_for(version: int) -> list[str]:
    """Tables an archive of this format version contains, in import order."""
    return [t for t in TABLES if TABLE_SINCE.get(t, 1) <= version]


_NB = r"notebooks/(?P<nb>[1-9]\d{0,11})"
ENTRY_PATTERNS = {
    "manifest": re.compile(rf"^{re.escape(MANIFEST)}$"),
    "checksums": re.compile(rf"^{re.escape(CHECKSUMS)}$"),
    "settings": re.compile(rf"^{re.escape(SETTINGS)}$"),
    "notebook": re.compile(rf"^{_NB}/notebook\.json$"),
    "table": re.compile(rf"^{_NB}/tables/(?P<table>[a-z_]+)\.jsonl$"),
    "source": re.compile(rf"^{_NB}/sources/(?P<sid>[1-9]\d{{0,11}})\.pdf$"),
}


def notebook_prefix(original_id: int) -> str:
    return f"notebooks/{int(original_id)}"


class ArchiveError(ValueError):
    """The archive cannot be used. The message is safe to show the user."""


class UnsupportedArchiveError(ArchiveError):
    pass


@dataclass(frozen=True)
class Limits:
    max_archive_bytes: int
    max_expanded_bytes: int
    max_entry_bytes: int
    max_files: int
    max_json_bytes: int = 64 * 1024 * 1024  # manifest / checksums / one JSON doc
    max_line_bytes: int = 32 * 1024 * 1024  # one JSONL row

    @classmethod
    def from_settings(cls, settings: Settings) -> Limits:
        mib = 1024 * 1024
        return cls(
            max_archive_bytes=settings.archive_max_mb * mib,
            max_expanded_bytes=settings.archive_max_expanded_mb * mib,
            max_entry_bytes=settings.archive_max_entry_mb * mib,
            max_files=settings.archive_max_files,
        )
