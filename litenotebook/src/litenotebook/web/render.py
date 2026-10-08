"""Safe rendering of assistant answers.

Model text is untrusted: every character is HTML-escaped. Only reference
groups whose labels were validated against what the model was actually given
(persisted as ``message_citations`` / ``message_memory_refs``) become links,
and those links are built entirely by the application. Anything else,
including "[S99]", "[M99]" or "<script>", stays text.
"""

from __future__ import annotations

from markupsafe import Markup, escape

from litenotebook.chat import Message
from litenotebook.rag.citations import ref_spans


def _link(url: str, css: str, title: str, text: str) -> str:
    return (
        f'<a class="{css}" href="{url}" hx-get="{url}" hx-target="#evidence" '
        f'hx-swap="innerHTML" title="{escape(title)}">{escape(text)}</a>'
    )


def render_answer(message: Message, notebook_id: int) -> Markup:
    text = message.content
    nb = int(notebook_id)
    out: list[str] = []
    pos = 0
    for span in ref_spans(text):
        out.append(str(escape(text[pos : span.start])))
        parts = []
        for kind, label in span.refs:
            if kind == "S" and (citation := message.citation(label)) is not None:
                parts.append(
                    _link(
                        f"/notebooks/{nb}/chat/citations/{citation.id}",
                        "cite" + ("" if citation.chunk_exists else " cite-deleted"),
                        f"{citation.source_filename} — page {citation.page_no}"
                        + ("" if citation.chunk_exists else " (source deleted)"),
                        f"S{label}",
                    )
                )
            elif kind == "M" and (ref := message.memory_ref(label)) is not None:
                parts.append(
                    _link(
                        f"/notebooks/{nb}/memory/refs/{ref.id}",
                        "cite cite-memory",
                        f"Memory: {ref.trust} — {ref.conversation_title}",
                        f"M{label}",
                    )
                )
            else:
                parts.append(str(escape(f"{kind}{label}")))
        out.append("[" + ", ".join(parts) + "]")
        pos = span.end
    out.append(str(escape(text[pos:])))
    return Markup("".join(out))
