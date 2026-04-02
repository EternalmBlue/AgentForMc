from __future__ import annotations

from agent_for_mc.domain.models import SemanticMemoryDoc


def format_semantic_memory_docs(
    docs: list[SemanticMemoryDoc],
    *,
    preview_chars: int,
) -> str:
    if not docs:
        return "No matching semantic memories were found."

    parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        preview = " ".join(doc.memory_text.split())
        if preview_chars > 0:
            preview = preview[:preview_chars]
        parts.append(
            f"[{index}] server={doc.server_id} plugin={doc.plugin_name}\n"
            f"type={doc.memory_type}\n"
            f"relation={doc.relation_type}\n"
            f"reason={doc.match_reason}\n"
            f"distance={doc.distance:.6f}\n"
            f"preview={preview}"
        )
    return "\n\n".join(parts)
