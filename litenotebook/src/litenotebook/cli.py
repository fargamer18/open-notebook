"""LiteNotebook command line (Phase 1 vertical slice).

python -m litenotebook.cli notebook create demo
python -m litenotebook.cli ingest --notebook demo document.pdf
python -m litenotebook.cli search --notebook demo "question"
python -m litenotebook.cli ask --notebook demo "question"
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from litenotebook.config import ConfigError, Settings, load_settings
from litenotebook.db import open_db
from litenotebook.embed.embedder import EmbeddingError, MiniLMEmbedder
from litenotebook.index.faiss_store import FaissStore, IndexMismatchError
from litenotebook.ingest.chunker import ChunkingError
from litenotebook.ingest.pipeline import IngestError, ingest_pdf
from litenotebook.llm.client import LLMClient, LLMError
from litenotebook.notebooks import (
    Notebook,
    NotebookError,
    create_notebook,
    get_notebook_by_name,
)
from litenotebook.rag.answer import answer_question
from litenotebook.rag.citations import format_citation
from litenotebook.rag.prompts import source_header
from litenotebook.retrieval.hybrid import Retriever
from litenotebook.services.settings import (
    SettingsError,
    SettingsService,
    check_connection,
)

USER_ERRORS = (
    ConfigError,
    SettingsError,
    NotebookError,
    IngestError,
    LLMError,
    IndexMismatchError,
    ChunkingError,
    EmbeddingError,
)


def _print_timings(timings: dict[str, float]) -> None:
    print("Timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()))


def cmd_notebook_create(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    nb = create_notebook(conn, args.name, args.description)
    print(f"Created notebook {nb.name!r} (id {nb.id})")
    return 0


def cmd_notebook_list(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    rows = conn.execute(
        "SELECT n.id, n.name, "
        "(SELECT count(*) FROM sources s WHERE s.notebook_id = n.id) AS sources, "
        "(SELECT count(*) FROM chunks c WHERE c.notebook_id = n.id) AS chunks "
        "FROM notebooks n ORDER BY n.name"
    ).fetchall()
    if not rows:
        print("No notebooks yet.")
    for r in rows:
        print(
            f"{r['id']:>4}  {r['name']}  ({r['sources']} sources, {r['chunks']} chunks)"
        )
    return 0


def cmd_ingest(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    nb = get_notebook_by_name(conn, args.notebook)
    embedder, store = MiniLMEmbedder(), FaissStore(conn, settings)
    failures = 0
    for path in args.files:
        try:
            report = ingest_pdf(conn, settings, nb, path, embedder, store, ocr=args.ocr)
        except IngestError as e:
            print(f"error: {path}: {e}", file=sys.stderr)
            failures += 1
            continue
        print(
            f"Ingested {report.filename}: {report.pages} pages "
            f"({report.pages_ocr} OCR, {report.pages_empty} empty), "
            f"{report.chunks} chunks (source {report.source_id})"
        )
        for warning in report.warnings:
            print(f"  warning: {warning}")
        if args.timings:
            _print_timings(report.timings)
    return 1 if failures else 0


def cmd_search(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    nb = get_notebook_by_name(conn, args.notebook)
    retriever = Retriever(conn, settings, MiniLMEmbedder(), FaissStore(conn, settings))
    results = retriever.search(nb.id, args.query, args.k)
    if not results:
        print("No results.")
    for i, r in enumerate(results, 1):
        sem = (
            f"sem #{r.semantic_rank} ({r.semantic_score:.3f})"
            if r.semantic_rank
            else "sem -"
        )
        lex = (
            f"bm25 #{r.lexical_rank} ({r.lexical_score:.2f})"
            if r.lexical_rank
            else "bm25 -"
        )
        print(f"{source_header(i, r)}  [chunk_id {r.chunk_id}]")
        print(f"     fused {r.fused_score:.4f} · {sem} · {lex}")
        print(f"     {r.snippet()}\n")
    return 0


def cmd_ask(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    nb = get_notebook_by_name(conn, args.notebook)
    effective = SettingsService(conn).effective()
    retriever = Retriever(conn, settings, MiniLMEmbedder(), FaissStore(conn, settings))
    with LLMClient(
        effective.llm_base_url,
        effective.llm_model,
        api_key=effective.llm_api_key or None,
        timeout_s=settings.llm_timeout_s,
    ) as llm:
        answer = answer_question(
            args.question, nb.id, retriever, llm, top_k=args.k or effective.rag_top_k
        )

    if args.show_context:
        print("Context:")
        for i, r in enumerate(answer.results, 1):
            print(f"  {source_header(i, r)}")
        print()
    print(answer.text.strip())
    print()
    if answer.citations.citations:
        print("Sources:")
        for citation in answer.citations.citations:
            print(f"  {format_citation(citation)}")
    elif answer.uncited:
        print("warning: the answer cites no sources; treat it as unverified")
    if answer.citations.invalid_labels:
        labels = ", ".join(f"S{n}" for n in answer.citations.invalid_labels)
        print(f"warning: ignored citations not in the supplied context: {labels}")
    if args.timings:
        if answer.llm:
            print(
                f"LLM: {answer.llm.prompt_tokens} prompt + "
                f"{answer.llm.completion_tokens} completion tokens"
            )
        _print_timings(answer.timings)
    return 0


def cmd_index_check(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    nb = get_notebook_by_name(conn, args.notebook)
    status = FaissStore(conn, settings).check(nb.id)
    print(
        f"SQLite embedded chunks: {status.sqlite_count}; FAISS vectors: "
        f"{status.index_count if status.index_count is not None else '-'}"
    )
    if status.ok:
        print("Index OK")
        return 0
    for problem in status.problems:
        print(f"problem: {problem}")
    print("Run `index rebuild` to recreate the index from SQLite.")
    return 1


def cmd_index_rebuild(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    nb = get_notebook_by_name(conn, args.notebook)
    index = FaissStore(conn, settings).rebuild(nb.id)
    print(f"Rebuilt index for {nb.name!r}: {index.ntotal} vectors")
    return 0


def cmd_llm_check(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    effective = SettingsService(conn).effective()
    report = check_connection(effective)
    print(report.message)
    return 0 if report.ok else 1


def _notebook_arg(conn: sqlite3.Connection, value: str) -> Notebook:
    """A notebook given by name or numeric id."""
    from litenotebook.notebooks import get_notebook

    if value.strip().isdigit():
        try:
            return get_notebook(conn, int(value))
        except NotebookError:
            pass
    return get_notebook_by_name(conn, value)


def cmd_memory_rebuild(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    from litenotebook.memory.graph import check, rebuild

    nb = _notebook_arg(conn, args.notebook)
    synced = rebuild(conn, nb.id)
    report = check(conn, nb.id)
    print(
        f"Rebuilt memory for {nb.name!r} from {synced} exchange(s): "
        f"{report.total_nodes} nodes, {report.total_edges} edges "
        f"(outcomes and corrections kept: {sum(report.outcomes.values())})"
    )
    return 0 if report.ok else 1


def cmd_memory_check(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    from litenotebook.memory.graph import check

    nb = _notebook_arg(conn, args.notebook)
    report = check(conn, nb.id)
    nodes = ", ".join(f"{k} {v}" for k, v in sorted(report.nodes.items())) or "none"
    edges = ", ".join(f"{k} {v}" for k, v in sorted(report.edges.items())) or "none"
    outcomes = (
        ", ".join(f"{k} {v}" for k, v in sorted(report.outcomes.items())) or "none"
    )
    print(f"Memory for {nb.name!r}")
    print(f"  nodes ({report.total_nodes}): {nodes}")
    print(f"  edges ({report.total_edges}): {edges}")
    print(f"  outcomes: {outcomes}")
    print(f"  exchanges in graph: {report.synced_exchanges}/{report.exchanges}")
    if report.ok:
        print("Memory OK")
        return 0
    for problem in report.problems:
        print(f"  problem: {problem}")
    print("Run `litenotebook memory rebuild` to re-derive the graph.")
    return 1


def cmd_memory_search(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    from litenotebook.memory.search import search_memory

    nb = _notebook_arg(conn, args.notebook)
    found = search_memory(conn, nb.id, args.query, include_dead_ends=True, limit=args.k)
    if not found.items:
        print("No memory matches.")
        return 0
    for item in found.items:
        outcome = f" · {item.outcome}" if item.outcome else ""
        print(
            f"#{item.rank}  {item.kind:<10}  score {item.score:.3f}  "
            f"{item.hops} hop(s)  {item.trust}{outcome}"
        )
        print(f"     conversation: {item.conversation_title}")
        print(f"     question: {item.question}")
        if item.excerpt:
            print(
                f"     {'correction' if item.kind == 'correction' else 'answer'}: "
                f"{item.excerpt[:200]}"
            )
        for src in item.sources:
            gone = " (source deleted)" if src["deleted"] else ""
            print(f"     cited: {src['filename']} p.{src['page']}{gone}")
    t = found.timings
    print(
        f"({len(found.seeds)} seed(s), {found.reached} node visit(s); "
        f"seed {t.get('seed', 0) * 1000:.2f} ms, "
        f"traverse {t.get('traverse', 0) * 1000:.2f} ms, "
        f"total {t.get('total', 0) * 1000:.2f} ms)"
    )
    return 0


def cmd_serve(
    args: argparse.Namespace, conn: sqlite3.Connection, settings: Settings
) -> int:
    import uvicorn

    from litenotebook.web.app import create_app

    conn.close()  # the web app opens its own per-request connections
    settings = replace(
        settings, host=args.host or settings.host, port=args.port or settings.port
    )  # re-validated: loopback only
    print(f"LiteNotebook on http://{settings.host}:{settings.port}  (Ctrl+C to stop)")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info" if args.verbose else "warning",
        proxy_headers=False,
        server_header=False,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litenotebook",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, help="override LITENOTEBOOK_DATA_DIR")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    notebook = sub.add_parser("notebook", help="manage notebooks")
    nb_sub = notebook.add_subparsers(dest="notebook_command", required=True)
    create = nb_sub.add_parser("create", help="create a notebook")
    create.add_argument("name")
    create.add_argument("--description", default="")
    create.set_defaults(func=cmd_notebook_create)
    nb_sub.add_parser("list", help="list notebooks").set_defaults(
        func=cmd_notebook_list
    )

    ingest = sub.add_parser("ingest", help="ingest PDF files into a notebook")
    ingest.add_argument("--notebook", required=True)
    ingest.add_argument(
        "--ocr",
        action="store_true",
        help="OCR pages without a text layer (needs tesseract)",
    )
    ingest.add_argument("--timings", action="store_true")
    ingest.add_argument("files", nargs="+", type=Path)
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="hybrid search within one notebook")
    search.add_argument("--notebook", required=True)
    search.add_argument("-k", type=int, default=None, help="results to show")
    search.add_argument("query")
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="answer a question from a notebook's sources")
    ask.add_argument("--notebook", required=True)
    ask.add_argument("-k", type=int, default=None, help="sources to give the LLM")
    ask.add_argument("--show-context", action="store_true")
    ask.add_argument("--timings", action="store_true")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    index = sub.add_parser("index", help="inspect or rebuild a notebook's FAISS index")
    index_sub = index.add_subparsers(dest="index_command", required=True)
    for name, func in (("check", cmd_index_check), ("rebuild", cmd_index_rebuild)):
        p = index_sub.add_parser(name)
        p.add_argument("--notebook", required=True)
        p.set_defaults(func=func)

    llm = sub.add_parser("llm", help="LLM server utilities")
    llm_sub = llm.add_subparsers(dest="llm_command", required=True)
    llm_sub.add_parser("check", help="check the LLM server").set_defaults(
        func=cmd_llm_check
    )

    memory = sub.add_parser("memory", help="graph memory of earlier conversations")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    for name, func, help_ in (
        ("rebuild", cmd_memory_rebuild, "re-derive the graph (keeps corrections)"),
        ("check", cmd_memory_check, "report consistency statistics"),
    ):
        p = memory_sub.add_parser(name, help=help_)
        p.add_argument("--notebook", required=True, help="notebook name or id")
        p.set_defaults(func=func)
    msearch = memory_sub.add_parser("search", help="debug graph-memory retrieval")
    msearch.add_argument("--notebook", required=True, help="notebook name or id")
    msearch.add_argument("-k", type=int, default=6, help="results to show")
    msearch.add_argument("query")
    msearch.set_defaults(func=cmd_memory_search)

    serve = sub.add_parser("serve", help="run the local web app")
    serve.add_argument("--host", help="loopback address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port (default 5056)")
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = load_settings()
        if args.data_dir:
            settings = replace(settings, data_dir=args.data_dir.expanduser())
        conn = open_db(settings.db_path)
        try:
            return int(args.func(args, conn, settings))
        finally:
            conn.close()
    except USER_ERRORS as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
