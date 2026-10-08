"""Safe rendering of assistant answers.

Model text is untrusted: every character is HTML-escaped. Only citation
groups whose labels were validated against the retrieval context (persisted
as ``message_citations``) become links, and those links are built entirely by
the application. Anything else, including "[S99]" or "<script>", stays text.
"""

from __future__ import annotations

import re

from markupsafe import Markup, escape

from litenotebook.chat import Message

_GROUP = re.compile(r"\[\s*S\s*\d+(?:\s*[,;]\s*S?\s*\d+)*\s*\]", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")


def render_answer(message: Message, notebook_id: int) -> Markup:
    text = message.content
    out: list[str] = []
    pos = 0
    for match in _GROUP.finditer(text):
        out.append(str(escape(text[pos : match.start()])))
        links = []
        for number in _NUMBER.findall(match.group(0)):
            citation = message.citation(int(number))
            if citation is None:
                links.append(str(escape(f"S{int(number)}")))
                continue
            url = f"/notebooks/{int(notebook_id)}/chat/citations/{citation.id}"
            state = "" if citation.chunk_exists else " cite-deleted"
            title = escape(
                f"{citation.source_filename} — page {citation.page_no}"
                + ("" if citation.chunk_exists else " (source deleted)")
            )
            links.append(
                f'<a class="cite{state}" href="{url}" hx-get="{url}" '
                f'hx-target="#evidence" hx-swap="innerHTML" title="{title}">'
                f"S{citation.label}</a>"
            )
        out.append("[" + ", ".join(links) + "]")
        pos = match.end()
    out.append(str(escape(text[pos:])))
    return Markup("".join(out))
